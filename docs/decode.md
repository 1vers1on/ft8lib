# `ft8lib.decode` — waveform to messages

Source: [`src/ft8lib/decode.py`](../src/ft8lib/decode.py). Ports of WSJT-X's
receive chain (`sync8.f90`, `ft8_downsample.f90`, `sync8d.f90`, the metric
section of `ft8b.f90`; `getcandidates4.f90`, `ft4_baseline.f90`,
`ft4_downsample.f90`, `get_ft4_bitmetrics.f90`, `ft4_decode.f90`), plus the
a-priori (AP) decoding logic from both `ft8b.f90` and `ft4_decode.f90`
(`ncontest=0` paths).

## Result type

```python
@dataclass
class Decode:
    message: str
    snr: float   # dB in a 2500 Hz reference bandwidth
    dt: float    # seconds, relative to the nominal 0.5 s period start
    freq: float  # audio Hz of the lowest (Costas) tone
    sync: float  # coarse-sync strength score (not directly comparable to WSJT-X's)
    mode: str = "FT8"
    ap: int = 0  # a-priori type used, 0..6 (see "A-priori decoding" below)
```

## Pipeline overview

Both `decode_ft8` and `decode_ft4` follow the same shape, differing in the
DSP details of each stage:

```
audio (12 kHz, any length/scale, zero-padded/truncated to one period)
    │  _prepare()
    ▼
candidate search  ──────────────  coarse (freq, dt) peaks where a Costas-like
    │                              sync pattern correlates strongly
    ▼
per candidate:
    downsample to baseband at the mode's low rate (200 Hz FT8, 666.67 Hz FT4)
    │
    ▼
fine time/frequency sync  ──────  refine dt (± symbols) and freq (± few Hz)
    │                              against the exact Costas arrays
    ▼
extract per-symbol complex spectra (one FFT bin group per tone)
    │
    ▼
hard-decision sync sanity check  ─ reject candidates whose Costas peaks
    │                              don't land where expected
    ▼
soft bit metrics  ──────────────  several LLR estimates per bit, from
    │                              noncoherent single/double/triple-symbol
    │                              combining (defeats phase/freq slips)
    ▼
LDPC decode174_91()  ────────────  belief propagation, then OSD fallback,
    │  (+ a-priori passes at depth 3)   tried against each metric variant
    ▼
CRC-verified 77-bit message → unpack77() → text, SNR estimate from tone power
    │
    ▼
results dict keyed by message text (keeps the best-sync duplicate)
    │
    ▼
multi-pass subtraction: regenerate + subtract each new decode from the
    audio, then repeat the whole search (up to `npasses` times) so signals
    that were masked by stronger ones become visible
```

## Public API

### `decode_ft8(audio, freq_min=200.0, freq_max=4000.0, syncmin=1.3, max_candidates=120, hashes=None, depth=3, mycall="", dxcall="", npasses=3) -> List[Decode]`

Decodes a 15 s, 12 kHz period (`FT8.NMAX = 180000` samples; shorter arrays
are zero-padded, longer ones truncated). Returns decodes sorted by
frequency, one per distinct message.

### `decode_ft4(audio, freq_min=200.0, freq_max=4000.0, syncmin=1.18, max_candidates=100, hashes=None, depth=3, mycall="", dxcall="", npasses=3) -> List[Decode]`

Same contract for a 7.5 s period (`FT4.NMAX = 72576` samples, up to 6.048 s
of real audio before padding).

Shared parameters:

- `freq_min` / `freq_max` — audio frequency search window in Hz.
- `syncmin` — minimum normalized coarse-sync strength to keep a candidate
  (WSJT-X defaults: 1.3 for FT8, 1.18 for FT4).
- `max_candidates` — cap on candidates tried per subtraction pass.
- `hashes` — an optional [`HashTable`](../src/ft8lib/pack.py) (share it
  across calls so `<hashed>` nonstandard callsigns resolve once seen).
- `depth` — how much decoding effort to spend per candidate:

  | depth | BP | OSD | a-priori (AP) | subtraction passes |
  |---|---|---|---|---|
  | 1 | yes | no | no | forced to 1 |
  | 2 | yes | yes (`max_osd=2`) | no | up to `npasses` |
  | 3 (default) | yes | yes | yes | up to `npasses` |

- `mycall`, `dxcall` — station callsigns. Only used at `depth=3`; they
  unlock the deeper AP types that assume a directed QSO (see below). Plain
  "CQ" AP decoding needs neither.
- `npasses` — max signal-subtraction iterations (ignored, forced to 1, when
  `depth < 2`).

## FT8 stages

### Coarse candidate search — `_ft8_symbol_spectra`, `_ft8_candidates` (`sync8.f90`)

`_ft8_symbol_spectra` computes the power spectrum of the audio in 1/4-symbol
steps (`FT8.NSTEP = NSPS/4`), giving fine time resolution for the sync
search: array `s[bin, step]`.

`_ft8_candidates` correlates this against FT8's three Costas 7-tone sync
blocks (at symbol offsets 0, 36, 72). For every candidate (frequency bin,
time lag) it sums the power landing exactly on the expected Costas tones
(`t`) and compares it to the total power summed over a ±6-bin neighborhood
around those tones (`t0`, via the precomputed `Q` array — a sliding
7-bin-wide sum of the padded spectrum `P`). The ratio `t/t0` (`sync_abc`) is
the "how much of the local energy is exactly where the sync tones should
be" score; `sync_bc` recomputes it from blocks B+C only, which helps when a
message starts a little late and block A is corrupted or absent.

Both scores are searched over a coarse ±13-step lag window first
(`lag_lo`/`jpeak`/`red`) and then the full range (`jpeak2`/`red2`), so a
strong candidate found only in the wide search doesn't get discarded. Scores
are normalized by the 40th-percentile score across all frequency bins
(`base`/`base2` — a robust noise-floor estimate) before thresholding against
`syncmin`. Near-duplicate candidates (within 4 Hz and 0.04 s) are
de-duplicated, keeping the stronger one, and the result is capped at
`max_candidates`, strongest first.

### Downsampling — `_FT8Downsampler` (`ft8_downsample.f90`)

For each candidate frequency `f0`, `_FT8Downsampler.__call__` extracts a
~3200-sample complex baseband segment at 200 S/s (`NDOWN = 60`) centered on
`f0`, tapering the edges of the frequency-domain window with a raised-cosine
to avoid ringing, and applying a linear phase ramp (`np.roll`) to re-center
the spectrum before the inverse FFT. The full-resolution FFT of the entire
15 s signal (`self.cx`) is computed once per subtraction pass and reused for
every candidate that pass.

### Fine sync — `_ft8_sync8d` (`sync8d.f90`)

`_ft8_sync8d(cd0, i0, ctwk)` correlates the downsampled signal at a trial
start sample `i0` against the exact Costas waveforms (`_FT8_CSYNC`,
precomputed 32-sample-per-symbol complex exponentials for each of the 7
Costas tones) across all three sync blocks, returning total sync power.
`ctwk` optionally applies a frequency-offset twiddle factor so the same
function serves both the time-refinement and frequency-refinement searches.
This is dispatched to the C kernel (`_kernels.sync8d`) when available, with
an equivalent pure-Python loop as fallback.

`decode_ft8`'s per-candidate loop uses `_ft8_sync8d` three times:
1. ±10 samples around the coarse `dt` to refine start time.
2. ±5 half-Hz steps to refine frequency (re-downsampling at the corrected
   frequency after this).
3. ±4 samples again at the corrected frequency, since frequency correction
   shifts the optimal start sample slightly.

### Symbol extraction and sync sanity check

79 per-symbol 32-point FFTs are taken at the refined start (`cs`, keeping
only the first 8 bins = the 8 possible tones); `s8` is their magnitude. A
candidate is discarded if its hard-decision tone (`argmax` of `s8`) matches
the expected Costas tone in 6 or fewer of the 21 sync positions — that's the
threshold below which the "signal" is almost certainly noise that happened
to correlate.

### Soft bit metrics — `_ft8_softbits` (metric section of `ft8b.f90`)

Rather than a single LLR per bit, WSJT-X (and this port) compute *five*
independent metrics and try LDPC decoding against each in turn, because
different combining strategies are robust to different failure modes
(residual frequency error, one Costas block being clobbered by another
signal, etc.):

- **bmeta** — single-symbol (noncoherent) metric: for each of the 3 bits
  encoded in each 8-FSK symbol, compare the max spectral power over the 4
  tone combinations consistent with bit=1 against the max over the 4
  consistent with bit=0.
- **bmetb** — two-symbol combining: sums adjacent symbol pairs' spectra
  (still noncoherently, via `_FT8_COMBOS[2]`) before the same max/max
  comparison, which averages out symbol-to-symbol noise at the cost of
  coarser time resolution.
- **bmetc** — three-symbol combining, same idea, most averaging.
- **bmetd** — bmeta normalized by `(mx1+mx0)`-style total energy instead of
  the corpus-wide sigma, making it more robust when local SNR varies a lot
  in frequency (bin-dependent noise).
- **bmete** — per-bit, picks whichever of bmeta/b/c has the largest
  magnitude at that bit position; a hedge that lets each bit borrow the
  most confident of the three combining strategies.

All are Gray-coded across 8 tones via `_FT8_COMBOS` (built once at import
time by `_combo_tables()`), scaled by `LLR_SCALE = 2.83` and normalized to
unit variance by `_normalize_bmet` (subtract mean, divide by empirical
sigma) so the LLR magnitudes are on a consistent scale for the LDPC decoder
regardless of the input signal's absolute power.

`decode_ft8` builds `attempts = [(llra,None,0), (llrb,...), (llrc,...), (llrd,...), (llre,...)]`
and tries `decode174_91` against each in order, stopping at the first
success (`max_osd = 2 if depth >= 2 else -1`, i.e. OSD fallback enabled from
depth 2 up — see [`ldpc.py`](../src/ft8lib/ldpc.py) for the BP+OSD hybrid
itself).

### A-priori (AP) decoding — `_APInfo`, `_ft8_ap_passes`

At `depth=3`, before falling through the five plain metric passes fail,
`decode_ft8` also tries decoding with parts of the 174-bit LLR vector
*forced* to known values — useful because a large fraction of real FT8
traffic follows a few fixed templates, and forcing those bits to
near-certain LLRs (`apmag`, set to 1.1x the strongest ordinary LLR) gives
the decoder far more margin on the remaining unknown bits.

`_APInfo(mycall, dxcall)` precomputes which AP types are worth trying:

| type | payload template | needs |
|---|---|---|
| 1 | `CQ ??? ???` | nothing |
| 2 | `MyCall ??? ???` | `mycall` |
| 3 | `MyCall DxCall ???` | `mycall` + `dxcall` |
| 4 | `MyCall DxCall RRR` | `mycall` + `dxcall` |
| 5 | `MyCall DxCall 73` | `mycall` + `dxcall` |
| 6 | `MyCall DxCall RR73` | `mycall` + `dxcall` |

Types 2–6 require that `pack77(f"{mycall} {dummy} RR73")` round-trips
exactly as a standard (i3=1) message — nonstandard/hashed calls can't be
predicted bit-for-bit, so AP falls back to type 1 only. `bits58`/`apsym`
cache the ±1 LLR-sign form of the first 58 bits (the two callsigns) so
`_ft8_ap_passes` doesn't need to repack per candidate.

`_ft8_ap_passes` yields `(llr, apmask, iaptype)` for each viable type,
against both the bmeta and bmetc base metrics (`for base in (llra, llrc)`):
`apmask` marks which of the 174 LLR positions are "known" (passed to
`decode174_91`'s `apmask`, which both excludes them from BP updates and
blocks them from OSD's error-pattern search — see `bp_decode`/`osd_decode`
in `ldpc.py`), and the corresponding `llrz` entries are forced to
`±apmag`. Bits 74:77 encode the 3-bit `i3` message type field, forced to
`001` (i3=1) for types 1–3, since AP only predicts standard messages.

### FT4's version — `_ft4_ap_passes`

Structurally identical, but FT4 scrambles its 77 payload bits with
`FT4.RVEC` *before* LDPC encoding (see `encode.py`), so every AP bit pattern
here is pre-XORed with the matching slice of `RVEC` before being turned into
an LLR sign. Only types 1, 2, 3 and 6 exist for FT4 (no separate RRR/73
tails — `ncontest=0` in `ft4_decode.f90` only distinguishes CQ / directed /
directed+RR73).

### Message validation and SNR

After a successful LDPC decode, `message_i3n3` rejects a few nonsensical
`(i3,n3)` combinations that shouldn't reach this point (defensive check
against a false decode with a lucky CRC), then `unpack77` renders the text
and resolves any `<hashed>` calls via `hashes`.

SNR is estimated the same way WSJT-X does: `itone` (the correct tone at each
of the 79 symbol positions, recovered from the decoded bits via
`ft8_tones_from_bits`) picks out the signal power `xsig` at those exact
bins, while `xnoi` samples power 4 tones away (out-of-band) as a noise
reference; `10*log10(xsig/xnoi - 1) - 27.0` converts the resulting SNR-like
ratio into the dB-in-2500-Hz convention WSJT-X reports, with `-25 dB` as a
floor for pathological ratios.

### Multi-pass subtraction

Each successful new decode (`prev is None`, i.e. not just a duplicate found
via a different AP pass) is queued in `new_signals`; after all candidates in
a pass are tried, [`subtract_ft8`](../src/ft8lib/subtract.py) regenerates
each one's exact GFSK waveform and coherently subtracts it from `dd` in
place. The whole candidate search then re-runs against the cleaned-up
audio, for up to `npasses` iterations — this is what lets a strong CQ and a
much weaker reply on a nearby frequency both decode from the same period.

## FT4 stages

FT4's DSP differs from FT8's mainly because it has no dedicated fine-sync
correlator function analogous to `sync8d.f90` — WSJT-X folds sync search
directly into a stride-2 grid search — and needs an explicit spectral
baseline fit since its shorter symbols give a noisier raw spectrum.

### Candidate search — `_ft4_baseline`, `_ft4_candidates` (`ft4_baseline.f90`, `getcandidates4.f90`)

`_ft4_candidates` computes a Nuttall-windowed average power spectrum
(`savg`) over the whole period, smooths it with a 15-bin moving average
(`savsm`), then flattens the smoothed spectrum against a slowly-varying
noise floor fit by `_ft4_baseline`.

`_ft4_baseline` fits that floor as a degree-4 polynomial in dB, but robustly:
it splits the search band into 10 segments, keeps only the bottom 10th
percentile of points in each segment (i.e. points *below* any signal peaks,
which is where the true noise floor lives), and fits the polynomial through
those. Peaks in the resulting normalized spectrum (local maxima at least
`syncmin`) become candidates, with a parabolic interpolation (`delta`) for
sub-bin frequency precision. `f_offset` accounts for `getcandidates4.f90`
reporting the frequency of the block's *first* tone rather than its center.

### Downsampling — `_FT4Downsampler` (`ft4_downsample.f90`)

Same idea as FT8's downsampler (frequency-domain window + IFFT to baseband
at a lower rate, 666.67 S/s here), but the window shape here is an explicit
trapezoid (`iwt` samples raised-cosine in, `iwf` flat, `iwt` raised-cosine
out) built once per period and reused for every candidate frequency via
`np.roll`-style index shifting rather than recomputed each time.

### Sync search — `_ft4_sync_templates`, `_ft4_sync_search`

FT4 has no separate "coarse then fine" correlator; instead
`_ft4_sync_search` runs a brute-force grid search over start sample and
frequency offset directly, correlating a stride-2 (every other baseband
sample, `_FT8_NSS`→32 samples/symbol effectively downsampled to 16 taps)
version of the signal against precomputed Costas waveform templates for all
four sync blocks (`_FT4_TEMPLATES`, one per `COSTAS_A..D`). `decode_ft4`
calls it twice: once over a coarse grid (±12 Hz in 3 Hz steps, start times
in 4-sample steps across the whole search range), then again over a tight
window around the best coarse hit. Candidates whose best sync power is
below `1.2` are dropped.

### Bit metrics — `_ft4_bitmetrics` (`get_ft4_bitmetrics.f90`)

Same noncoherent 1/2/4-symbol combining idea as FT8's `_ft8_softbits`, but
producing 3 metric columns instead of 5 (FT4's 4-tone Gray map only needs 2
bits/symbol, so the combinatorics are smaller: `_FT4_GRAY`/`_ONE8` vs. FT8's
`_FT8_GRAY`/`_ONE9`). A hard-decision Costas sync check
(`nsync < 8` out of 16 expected matches) gates out bad candidates the same
way FT8's `nsync <= 6` check does.

After bit metrics, a second, independent sync sanity check compares hard
bit decisions against `ft4_decode.f90`'s expected sync-word bit patterns at
each of the four Costas positions (the `ns` count); fewer than 20/32
matches rejects the candidate. This mirrors WSJT-X catching candidates that
passed the continuous sync-power test but decode to garbage bits.

### Decode loop — `decode_ft4`

Structurally the same as `decode_ft8`'s: build `llr_cols` (3 metrics ×
`LLR_SCALE`), add AP attempts at depth 3, try `decode174_91` against each.
The one FT4-specific step is un-scrambling: `decode174_91` returns
`scrambled77` (the LDPC message bits, still XORed with `RVEC`), so
`message77 = (scrambled77 + FT4.RVEC) % 2` must run before `unpack77`.

SNR here follows `ft4_decode.f90`'s empirical formula from the *candidate*
peak strength (`10*log10(strength - 1.0) - 14.8`, floored at -21 dB) rather
than FT8's post-decode tone-power ratio, since FT4's shorter symbols make a
post-decode measurement noisier.

Subtraction (`subtract_ft4`) and the pass-repeat loop work identically to
FT8's.

## Performance and implementation notes

- The hot inner loops (`_ft8_sync8d`, and `bp_decode`/`osd_decode` in
  `ldpc.py`) dispatch to a compiled C extension (`_kernels` module,
  `_ckernels.c`) when available, with numerically-equivalent pure-numpy
  fallbacks used only if the extension failed to build. Belief propagation
  must stay double precision with exact `atanh`/`tanh` — a single-precision
  or polynomial-approximated variant was tried and made BP converge ~3x
  less often, cascading into far more (slow) OSD fallback calls.
- Decode thresholds (10-trial AWGN sweeps, `tools/measure_decode_threshold.py`):
  FT8 ~100% down to −19 dB, FT4 down to −16 dB (WSJT-X itself, with the same
  OSD+AP machinery, reaches −21 dB / −17.5 dB — the gap is mostly algorithm
  tuning headroom, not a missing feature).
- `depth` and `npasses` are the two knobs to trade decode completeness
  against runtime: `depth=1` skips OSD, AP, and multi-pass subtraction
  entirely for a fast single sweep; `depth=3, npasses=3` (the default) is
  what WSJT-X does by default in normal (non-contest) operation.

## Fortran correspondence

| Python | Fortran |
|---|---|
| `_ft8_candidates` | `lib/ft8/sync8.f90` |
| `_FT8Downsampler` | `lib/ft8/ft8_downsample.f90` |
| `_ft8_sync8d` | `lib/ft8/sync8d.f90` |
| `_ft8_softbits`, AP logic, message loop | `lib/ft8/ft8b.f90` |
| `_ft4_candidates` | `lib/ft4/getcandidates4.f90` |
| `_ft4_baseline` | `lib/ft4/ft4_baseline.f90` |
| `_FT4Downsampler` | `lib/ft4/ft4_downsample.f90` |
| `_ft4_bitmetrics` | `lib/ft4/get_ft4_bitmetrics.f90` |
| `_ft4_sync_search`, AP logic, message loop | `lib/ft4/ft4_decode.f90` |
| `decode174_91` (LDPC BP+OSD) | see [`ldpc.py`](../src/ft8lib/ldpc.py) |
| `subtract_ft8`/`subtract_ft4` | see [`subtract.py`](../src/ft8lib/subtract.py) |
