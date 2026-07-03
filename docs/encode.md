# `ft8lib.encode` — message to waveform

Source: [`src/ft8lib/encode.py`](../src/ft8lib/encode.py). Ports of WSJT-X
`lib/ft8/genft8.f90`, `lib/ft4/genft4.f90`, `lib/ft8/gen_ft8wave.f90` and
`lib/ft4/gen_ft4wave.f90`.

## Pipeline

```
message string
    │  pack77()                          (pack.py)
    ▼
77 payload bits
    │  encode174_91()                    (ldpc.py)
    ▼
174-bit LDPC codeword (91 message bits [77 + 14-bit CRC] + 83 parity bits)
    │  ft8_tones_from_bits() / ft4_tones_from_bits()
    ▼
79 (FT8) or 103 (FT4) channel symbols, each a tone index
    │  gen_ft8wave() / gen_ft4wave()  →  _gfsk_phase_increments()
    ▼
float64 audio waveform at 12000 S/s
```

`encode_ft8` / `encode_ft4` drive the whole pipeline; the intermediate
functions are exported for callers that want tones or codewords directly
(the CLI and the test suite use them this way).

## Public API

### `encode_ft8(message, f0=1000.0, hashes=None) -> np.ndarray`

Packs `message` with [`pack77`](../src/ft8lib/pack.py), LDPC-encodes it, maps
it to 79 tones, and synthesizes a GFSK waveform. Returns 151680 samples
(12.64 s at 12 kHz). By WSJT-X convention transmission starts 0.5 s into the
15 s T/R cycle — the returned waveform does *not* include that leading
silence, so callers must pad it themselves (see the README example).

### `encode_ft4(message, f0=1000.0, hashes=None) -> np.ndarray`

Same shape of pipeline for FT4: 103 tones, 60480 samples (5.04 s), meant to
start 0.5 s into the 7.5 s cycle.

Both raise `ValueError` if the message can't be encoded — `_check_roundtrip`
immediately unpacks the freshly packed bits and rejects anything whose
unpack fails. Note this checks only that the bits *unpack successfully*,
not that the text matches the input: a message that doesn't parse as any
structured type falls back to free text (truncated to 13 characters), so
e.g. `"CQ K1ABC XX99"` encodes as `"CQ K1ABC"` rather than raising.

`hashes` is an optional [`HashTable`](../src/ft8lib/pack.py) — pass the same
table you use for decoding if the message contains a `<hashed>` nonstandard
callsign that needs to resolve consistently with prior traffic.

## Bits → tones

### `ft8_tones_from_bits(msgbits) -> np.ndarray` (79 tones, 0–7)

FT8 interleaves three 7-symbol Costas sync arrays (`FT8.COSTAS = (3,1,4,0,6,5,2)`)
with the 58 data symbols, three codeword bits per symbol via
`FT8.GRAYMAP = (0,1,3,2,5,6,4,7)` (Gray-coded 8-FSK: adjacent tones differ by
one bit, so a single-tone slip costs the minimum number of bit errors):

```
symbol:  0        7                  36       43                 72       79
        ├─Costas─┼───29 data syms───┼─Costas─┼───29 data syms────┼─Costas─┤
tones:   3 1 4 0 6 5 2   (data)      3 1 4 0 6 5 2   (data)       3 1 4 0 6 5 2
```

### `ft4_tones_from_bits(msgbits) -> np.ndarray` (103 tones, 0–3)

FT4 first XOR-scrambles the 77 bits with `FT4.RVEC` (a fixed 77-bit vector;
this whitens the payload so that BP decoding sees more balanced LLR
statistics — see `decode.py`'s `_ft4_ap_passes` for the matching
descramble), *then* LDPC-encodes, then Gray-maps pairs of codeword bits with
map `{00:0, 01:1, 11:3, 10:2}`, and interleaves four different 4-symbol
Costas arrays (`COSTAS_A..D`) between four blocks of ~29 data symbols:

```
symbol:  0    4        33   37        66   70        99  103
        ├─A──┼─29 data─┼─B──┼─29 data─┼─C──┼─29 data─┼─D──┤
```

## Tones → waveform (GFSK synthesis)

Both modes use continuous-phase Gaussian-filtered FSK: the instantaneous
frequency at each instant is the tone value convolved with a Gaussian pulse
(`gfsk_pulse` in `protocol.py`, parameterized by the mode's `BT` — the
Gaussian filter's bandwidth-time product, 2.0 for FT8, 1.0 for FT4) rather
than switching sharply between tones, which keeps the transmitted spectrum
compact.

`_gfsk_phase_increments(itone, nsps, bt, f0, fsample, extend_edges)` builds
the per-sample frequency waveform:

1. Each symbol contributes a 3-symbol-wide Gaussian pulse of height
   `tone * dphi_peak` centered on that symbol, summed with its neighbors'
   pulse tails (`pulse` is precomputed for a 3-symbol window via
   `gfsk_pulse(bt, t)`).
2. `extend_edges=True` (FT8 only) mirrors the first/last tone outward by one
   symbol so the pulse has something to convolve with at the very start/end
   — FT4's own symbol layout instead reserves real ramp symbols (see below).
3. A constant `2*pi*f0*dt` term shifts the whole signal up to the audio
   carrier frequency `f0`.

`gen_ft8wave` / `gen_ft4wave` then integrate (`cumsum`) the phase increments
to get instantaneous phase, take `sin` (or `exp(1j*phase)` if
`complex_output=True`, used internally by `subtract.py` for coherent
subtraction), and apply a raised-cosine amplitude ramp over the first/last
`nsps/8` (FT8) or full symbol (FT4) samples to avoid a step discontinuity at
turn-on/turn-off. FT4's waveform is 2 symbols longer than its 103 data/sync
symbols (`NN2 = NN + 2`) specifically to hold that ramp; FT8's ramp instead
eats into the first/last real symbol's duration.

## Constants reference

Both modes' timing/DSP constants live in [`protocol.py`](../src/ft8lib/protocol.py):

| | FT8 | FT4 |
|---|---|---|
| samples/symbol (`NSPS`) | 1920 | 576 |
| tones | 8 | 4 |
| tone spacing | 6.25 Hz | 20.83 Hz |
| symbol period | 0.16 s | 0.048 s |
| channel symbols (`NN`) | 79 | 103 |
| T/R cycle | 15 s | 7.5 s |
| GFSK `BT` | 2.0 | 1.0 |
| waveform length | 151680 samples (12.64 s) | 60480 samples (5.04 s) |

All at `SAMPLE_RATE = 12000` S/s.

## Fortran correspondence

| Python | Fortran |
|---|---|
| `ft8_tones_from_bits` | `lib/ft8/genft8.f90` |
| `ft4_tones_from_bits` | `lib/ft4/genft4.f90` |
| `gen_ft8wave` | `lib/ft8/gen_ft8wave.f90` |
| `gen_ft4wave` | `lib/ft4/gen_ft4wave.f90` |
| `gfsk_pulse` (protocol.py) | `lib/ft2/gfsk_pulse.f90` |
