# ft8lib usage guide

How to encode and decode FT8, FT4 and WSPR with the public API: batch and
streaming decoding, realtime operation, callsign hash tables, and the
command-line tool. For *how the DSP works* (pipelines, Fortran-source
correspondence), see [`encode.md`](encode.md) and [`decode.md`](decode.md).

Everything documented here is importable from the top-level package:

```python
import ft8lib
```

## Contents

- [Fundamentals: sample rate, periods, timing](#fundamentals-sample-rate-periods-timing)
- [Encoding](#encoding)
  - [FT8 and FT4](#ft8-and-ft4)
  - [WSPR](#wspr)
  - [Lower-level building blocks](#lower-level-building-blocks)
- [Reading and writing WAV files](#reading-and-writing-wav-files)
- [Decoding](#decoding)
  - [The `Decode` result](#the-decode-result)
  - [FT8 and FT4 batch decoding](#ft8-and-ft4-batch-decoding)
  - [Streaming results from one period](#streaming-results-from-one-period)
  - [Live audio: `RealtimeDecoder`](#live-audio-realtimedecoder)
  - [WSPR decoding](#wspr-decoding)
- [Callsign hash tables](#callsign-hash-tables)
- [Command-line interface](#command-line-interface)
- [Performance notes](#performance-notes)

## Fundamentals: sample rate, periods, timing

All audio in and out of the library is real-valued at
`ft8lib.SAMPLE_RATE = 12000` samples/s. Encoders return `float64` arrays
scaled to ±1; decoders accept any amplitude scale (float or integer PCM
converted to float both work — only relative levels matter).

Each mode lives on a fixed transmit/receive cycle aligned to UTC:

|                              | FT8                  | FT4          | WSPR                |
|------------------------------|----------------------|--------------|---------------------|
| T/R period                   | 15 s                 | 7.5 s        | 120 s               |
| cycle starts at              | :00 :15 :30 :45      | every 7.5 s  | even UTC minutes    |
| transmission starts          | 0.5 s into the cycle | 0.5 s        | 1 s                 |
| transmission length          | 12.64 s              | 5.04 s       | 110.6 s             |
| waveform samples             | 151 680              | 60 480       | 1 327 104           |
| modulation                   | 8-GFSK               | 4-GFSK       | 4-FSK               |
| tone spacing                 | 6.25 Hz              | 20.83 Hz     | 1.4648 Hz           |
| occupied bandwidth           | ~50 Hz               | ~83 Hz       | ~6 Hz               |
| payload                      | 77 bits              | 77 bits      | 50 bits             |
| decoder input (max used)     | 15 s (180 000)       | 6.048 s (72 576) | ~123 s          |

These constants are available programmatically on `ft8lib.FT8`,
`ft8lib.FT4` and `ft8lib.WSPR` (e.g. `FT8.NMAX`, `WSPR.PERIOD`,
`FT4.TONE_SPACING`); see [`protocol.py`](../src/ft8lib/protocol.py).

Two conventions to keep in mind:

- **Encoders return only the transmission**, not the whole period. To
  build a full T/R period, place the waveform at the conventional start
  offset (0.5 s for FT8/FT4, 1 s for WSPR) in a zero array of
  `MODE.NMAX` samples.
- **Decoders assume the array starts at the cycle boundary.** The
  reported `dt` of each decode is relative to the nominal start; feeding
  audio that begins mid-cycle shifts every `dt` accordingly (and signals
  can fall outside the searched time window entirely).

## Encoding

### FT8 and FT4

```python
import numpy as np
import ft8lib

wave = ft8lib.encode_ft8("CQ K1ABC FN42", f0=1500.0)      # 151680 samples
wave4 = ft8lib.encode_ft4("W9XYZ K1ABC RR73", f0=1500.0)  # 60480 samples

# Full 15-s FT8 period with the conventional 0.5 s leading delay:
period = np.zeros(ft8lib.FT8.NMAX)
period[6000:6000 + len(wave)] = wave
```

`encode_ft8(message, f0=1000.0, hashes=None)` /
`encode_ft4(message, f0=1000.0, hashes=None)`

- `message` — the text to send (case-insensitive; whitespace collapsed).
- `f0` — audio frequency of the lowest tone, in Hz. Default 1000. Keep the
  signal inside the receiver's passband: `f0` between ~200 and
  ~3950 Hz (FT8) / ~3900 Hz (FT4) with the usual 200–4000 Hz decode range.
- `hashes` — an optional [`HashTable`](#callsign-hash-tables); pass one when
  the message contains a `<bracketed>` hashed callsign that must resolve
  consistently with earlier traffic.

Both raise `ValueError` only when the message cannot be packed at all
(empty, or free text containing characters outside the free-text
alphabet). A message that doesn't parse as any structured type **falls
back to free text truncated to 13 characters**, and a malformed field can
be silently dropped — `"CQ K1ABC XX99"` (invalid grid) encodes as
`"CQ K1ABC"`, and `"HELLO WORLD THIS IS LONG"` as `"HELLO WORLD T"`. To
guarantee the over-the-air message is exactly what you asked for,
round-trip it yourself before transmitting:

```python
bits = ft8lib.pack77(message)
sent, ok = ft8lib.unpack77(bits, nrx=0)
assert ok and sent == " ".join(message.upper().split())
```

Supported message types (same as WSJT-X): standard messages (types 1/2,
including `/R` and `/P` suffixes), free text up to 13 characters (type
0.0), DXpedition (0.1), ARRL Field Day (0.3/0.4), telemetry (0.5),
WSPR-style (0.6), ARRL RTTY roundup (3), nonstandard/hashed calls (4), and
EU VHF contest (5). Some examples:

```python
ft8lib.encode_ft8("CQ K1ABC FN42")          # CQ with grid
ft8lib.encode_ft8("K1ABC W9XYZ R-08")       # signal report
ft8lib.encode_ft8("W9XYZ K1ABC RR73")       # QSO close
ft8lib.encode_ft8("TNX 73 GL")              # free text (max 13 chars)
ft8lib.encode_ft8("CQ PJ4/K1ABC")           # nonstandard call (type 4)
ft8lib.encode_ft8("123456789ABCDE")         # telemetry: <=18 hex digits,
                                            # first digit <= 7
```

### WSPR

```python
wave = ft8lib.encode_wspr("K1ABC FN42 37")   # 1327104 samples, 110.6 s

# Full 2-minute period with the conventional 1 s leading delay:
period = np.zeros(ft8lib.WSPR.NMAX)
period[12000:12000 + len(wave)] = wave
```

`encode_wspr(message, f0=1500.0)` — `f0` is the *center* frequency of the
4-FSK signal, conventionally in the 1400–1600 Hz audio window (WSPR
sub-bands are only 200 Hz wide).

WSPR has exactly three message forms:

| type | form                      | fields                                    |
|------|---------------------------|-------------------------------------------|
| 1    | `K1ABC FN42 33`           | standard call, 4-char grid, power in dBm  |
| 2    | `PJ4/K1ABC 37`            | compound call (prefix or suffix), power   |
| 3    | `<PJ4/K1ABC> FK52UD 37`   | *hashed* call, 6-char grid, power         |

The power field is dBm and must be one of the legal WSPR values:
**0–60, ending in 0, 3 or 7** (0, 3, 7, 10, 13, … 57, 60). Anything else
raises `ValueError` — the channel encoding cannot represent other values,
so e.g. `"K1ABC FN42 36"` would silently come out as a different power
and is rejected instead.

Type 3 transmits only a 15-bit hash of the callsign; a receiver can only
display the call if it has previously decoded the matching type 1 or 2
transmission (see [`WsprHashTable`](#callsign-hash-tables)). The usual
compound-call practice is to alternate type 2 and type 3 transmissions.

### Lower-level building blocks

`encode_ft8`/`encode_ft4`/`encode_wspr` chain together steps that are all
exported individually, for callers who want the intermediate forms:

```python
bits = ft8lib.pack77("CQ K1ABC FN42")        # 77-bit uint8 array
msg, ok = ft8lib.unpack77(bits)              # back to text
cw = ft8lib.encode174_91(bits)               # 174-bit LDPC codeword

tones = ft8lib.ft8_tones_from_bits(bits)     # 79 channel symbols, 0-7
tones4 = ft8lib.ft4_tones_from_bits(bits)    # 103 channel symbols, 0-3
toneswspr = ft8lib.wspr_tones_from_message("K1ABC FN42 37")  # 162, 0-3

wave = ft8lib.gen_ft8wave(tones, f0=1500.0)
wave4 = ft8lib.gen_ft4wave(tones4, f0=1500.0)
wavew = ft8lib.gen_wsprwave(toneswspr, f0=1500.0)

zwave = ft8lib.gen_ft8wave(tones, f0=1500.0, complex_output=True)
```

Notes:

- The `gen_*wave` functions accept `fsample` to synthesize at other sample
  rates, and `complex_output=True` to get the analytic (complex) signal
  instead of its real part — useful for IQ transmitters or simulation.
- `pack77` also takes `i3_hint`/`n3_hint` to force a specific WSJT-X
  message type when the text is ambiguous (e.g. `i3_hint=0, n3_hint=5`
  forces telemetry).
- WSPR's source coding is exposed as `pack_wspr(message)` (11-byte data
  vector: 50 payload bits + encoder flush) and
  `unpack_wspr(data, hashes)` → `(message, callsign, ok)`.

## Reading and writing WAV files

The library works on numpy arrays; converting to and from WAV is left to
the caller (the [CLI](#command-line-interface) does it for you). With only
the standard library:

```python
import wave
import numpy as np

def write_wav(path, audio):                       # audio: float64 in +/-1
    pcm = (np.clip(audio, -1.0, 1.0) * 32767 * 0.9).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(12000)
        w.writeframes(pcm.tobytes())

def read_wav(path):                               # 16-bit mono, 12 kHz
    with wave.open(path, "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float64)
```

Recordings at other sample rates must be resampled to 12 kHz first
(`scipy.signal.resample_poly`, or use the CLI which resamples
automatically). No amplitude normalization is needed for decoding.

## Decoding

### The `Decode` result

Every decoder returns `ft8lib.Decode` objects:

```python
@dataclass
class Decode:
    message: str    # decoded text, e.g. "CQ K1ABC FN42"
    snr: float      # estimated SNR in a 2500 Hz reference bandwidth (dB)
    dt: float       # time offset in seconds relative to the nominal start
    freq: float     # audio Hz: lowest tone (FT8/FT4) or center (WSPR)
    sync: float     # sync-quality score (mode-specific scale)
    mode: str       # "FT8", "FT4" or "WSPR"
    ap: int         # a-priori type used, 0 = none, 1..6 as in WSJT-X
    drift: float    # linear frequency drift over the transmission (WSPR, Hz)
```

`str(result)` formats a WSJT-X-style line:

```
FT8 -15 dB  DT +0.00 s  1500.0 Hz  CQ K1ABC FN42
```

`dt = 0` corresponds to a transmission starting exactly on time: 0.5 s
into the supplied audio array for FT8/FT4, 1 s for WSPR (the modes'
nominal transmit offsets, matching WSJT-X's conventions). `ap` is nonzero
only for FT8/FT4 decodes found with a-priori information; `drift` is
meaningful only for WSPR.

### FT8 and FT4 batch decoding

```python
for result in ft8lib.decode_ft8(audio_15s):
    print(result)

for result in ft8lib.decode_ft4(audio_7p5s):
    print(result)
```

`decode_ft8(audio, freq_min=200.0, freq_max=4000.0, syncmin=1.3,
max_candidates=120, hashes=None, depth=3, mycall="", dxcall="",
npasses=3)` — and `decode_ft4` with the same signature (defaults
`syncmin=1.18`, `max_candidates=100`). Both return a list of `Decode`
sorted by frequency, one entry per distinct message.

- `audio` — one receive period at 12 kHz. FT8: 180 000 samples (15 s);
  FT4: 72 576 samples (6.048 s — an FT4 transmission is over by ~5.5 s, so
  a full 7.5 s array is fine and simply gets truncated). Shorter arrays
  are zero-padded.
- `freq_min`, `freq_max` — audio-frequency search window in Hz.
  Narrowing it around a known signal speeds up decoding and removes
  false-candidate competition.
- `syncmin` — minimum coarse-sync score to accept a candidate. Lower
  values try more (weaker) candidates; the defaults match WSJT-X.
- `max_candidates` — cap on candidates tried per pass.
- `hashes` — a [`HashTable`](#callsign-hash-tables), shared across calls so
  hashed callsigns resolve.
- `depth` — decoding effort per candidate:

  | depth | BP | OSD | a-priori (AP) | subtraction passes |
  |-------|----|-----|---------------|--------------------|
  | 1     | ✓  | –   | –             | 1 (forced)         |
  | 2     | ✓  | ✓   | –             | up to `npasses`    |
  | 3 (default) | ✓ | ✓ | ✓          | up to `npasses`    |

- `mycall`, `dxcall` — your callsign and the station you are working.
  Only used at `depth=3`: they unlock the deeper AP hypotheses
  (`MyCall ???`, `MyCall DxCall RRR/73/RR73`), which is how WSJT-X digs
  replies addressed to you out of the noise. Plain `CQ ??? ???` AP needs
  neither. Decodes found this way carry the AP type in `result.ap`.
- `npasses` — maximum subtraction passes (each pass subtracts the signals
  decoded so far and re-searches, so weak signals hidden under strong
  ones emerge). Ignored at `depth=1`.

Typical receiving-station setup:

```python
table = ft8lib.HashTable(mycall="K1ABC")
results = ft8lib.decode_ft8(audio, hashes=table,
                            mycall="K1ABC", dxcall="W9XYZ")
```

For the fastest possible sweep (e.g. scanning many recordings for strong
signals only), use `depth=1`.

### Streaming results from one period

`decode_ft8_stream` / `decode_ft4_stream` take the same parameters as the
batch functions but are generators, yielding each unique message as soon
as it decodes — strongest candidates first, later subtraction passes
last — instead of returning a frequency-sorted list at the end:

```python
for result in ft8lib.decode_ft8_stream(audio_15s, mycall="K1ABC"):
    print(result)          # appears as soon as it's found
```

### Live audio: `RealtimeDecoder`

For continuous operation on a live audio stream, `RealtimeDecoder`
accepts arbitrary-size sample blocks and runs decode attempts part-way
through each receive period, on the WSJT-X schedule (FT8: at 11.8, 13.5
and 14.7 s into the cycle; FT4: at 5.6 and 6.048 s) — so most messages
appear seconds before the period ends. Cycle rollover is automatic.

```python
rt = ft8lib.RealtimeDecoder("FT8", mycall="K1ABC")   # or "FT4"
for block in audio_blocks:            # e.g. sound-card capture callback
    for result in rt.feed(block):     # new decodes from this block
        print(rt.cycle, result)
rt.flush()                            # decode a partial final period
```

- **Start feeding at a cycle boundary** (:00/:15/:30/:45 for FT8). The
  decoder counts time from the first sample fed; use `rt.reset()` to
  re-align. `rt.t` is the current position (seconds) in the cycle,
  `rt.cycle` the 0-based period counter.
- `feed(samples)` returns the messages newly decoded as a result of that
  block (usually an empty list; non-empty when the block crossed a
  scheduled attempt time). Blocks may span cycle boundaries. Each unique
  message is reported once per cycle.
- `flush()` runs one off-schedule decode over whatever is buffered —
  useful when a recording ends mid-cycle.
- Constructor keywords are those of `decode_ft8`/`decode_ft4`
  (`freq_min`, `freq_max`, `syncmin`, `max_candidates`, `hashes`,
  `depth`, `mycall`, `dxcall`, `npasses`) plus `attempt_times`, an
  iterable of seconds-into-cycle to override the attempt schedule.
- The hash table (`rt.hashes`) persists across cycles, so a hashed call
  heard in one period resolves in later ones.

A complete example with the `sounddevice` package (not a dependency):

```python
import queue
import sounddevice as sd
import ft8lib

q: "queue.Queue" = queue.Queue()
rt = ft8lib.RealtimeDecoder("FT8", mycall="K1ABC")

def callback(indata, frames, time_info, status):
    q.put(indata[:, 0].copy())

# start the stream exactly on a 15-s boundary
with sd.InputStream(samplerate=12000, channels=1, callback=callback):
    while True:
        for result in rt.feed(q.get()):
            print(f"cycle {rt.cycle}: {result}")
```

`decode_realtime(blocks, mode="FT8", **kwargs)` wraps the same loop as a
generator over an iterable of blocks, flushing at the end.

`RealtimeDecoder` supports FT8 and FT4 only; for WSPR, collect the full
2-minute period and call `decode_wspr`.

### WSPR decoding

```python
table = ft8lib.WsprHashTable()
for result in ft8lib.decode_wspr(audio_2min, hashes=table):
    print(result)
# WSPR  -19 dB  DT +0.10 s  1500.1 Hz  K1ABC FN42 37
print(result.drift)   # linear frequency drift in Hz over the 110.6 s
```

`decode_wspr(audio, freq_min=1390.0, freq_max=1610.0, hashes=None,
deep=False, quick=False, npasses=3, subtraction=True, maxcycles=10000,
bias=0.45)` — returns a list of `Decode` sorted by frequency. This is a
port of WSJT-X's `wsprd`; the parameter ↔ `wsprd` flag mapping is noted
below.

- `audio` — the 2-minute receive period at 12 kHz (transmissions start
  1 s into the even UTC minute). Shorter arrays are zero-padded; only the
  first ~123 s are used.
- `freq_min`, `freq_max` — audio search range in Hz. The decoder is
  hard-limited to 1350–1650 Hz (the downconverter is centered on
  1500 Hz); the default ±110 Hz window matches `wsprd`.
- `hashes` — a [`WsprHashTable`](#callsign-hash-tables), kept across calls
  so type 3 `<CALL>` messages resolve once the matching type 1/2 has been
  heard.
- `deep` — additionally try candidates below the spectral-peak threshold
  (`wsprd -d`); finds a few more marginal signals at a large runtime cost
  on busy bands.
- `quick` — skip the time-jitter search around each candidate
  (`wsprd -q`); faster, slightly less sensitive.
- `npasses` — decoding passes. Each pass subtracts the signals already
  decoded and re-searches:
  - `1` — single pass, no subtraction retry (≈ `wsprd -s`);
  - `2` — adds one subtraction+retry pass (≈ `wsprd -B`, i.e. still no
    block demodulation);
  - `3` (default) — the final pass switches to noncoherent *block*
    demodulation of 4-symbol sequences, which pulls out the weakest
    signals.
- `subtraction` — set `False` to disable decoded-signal subtraction
  entirely (subtraction is what lets overlapping signals both decode).
- `maxcycles` — Fano sequential-decoder timeout in cycles per bit
  (`wsprd -C`). Raise it (e.g. `31000`) for a little more sensitivity on
  the weakest signals, at the cost of time spent on undecodable
  candidates.
- `bias` — Fano metric bias (`wsprd -z`, default 0.45). Rarely worth
  touching.

Results: `freq` is the audio center frequency in Hz, `dt` the start-time
offset in seconds relative to the nominal 1 s transmit start (`dt = 0`
means on time, as `wsprd` reports it), `snr` the estimated SNR in
2500 Hz, `drift` the linear frequency drift in Hz over the transmission,
and `mode` is `"WSPR"`. Two decodes of the same callsign within 4 Hz are
treated as duplicates and reported once.

Ported from `wsprd`: the full sync/demodulation chain, Fano decoding,
multi-pass subtraction and block demodulation. *Not* ported: the Jelinek
stack decoder and the OSD deep search, so sensitivity at the extreme
margin is slightly below `wsprd` with those options enabled.

## Callsign hash tables

Both protocol families transmit some callsigns only as hashes, so
decoding them requires state carried across receive periods. Create one
table at startup and pass it to every encode/decode call.

### FT8/FT4: `HashTable`

Nonstandard callsigns (e.g. `PJ4/K1ABC`) appear in some message types as
10/12/22-bit hashes, displayed as `<PJ4/K1ABC>` once known and `<...>`
otherwise.

```python
table = ft8lib.HashTable(mycall="K1ABC", dxcall="PJ4/W9XYZ")
table.set_station("K1ABC", "PJ4/W9XYZ")   # update during operation
table.save("PJ4/DL1XYZ")                  # prime a known call manually

results = ft8lib.decode_ft8(audio, hashes=table)
```

The table fills itself from full (unhashed) calls seen in decoded
traffic; `set_station` additionally lets hashes of your own call and the
current DX call resolve immediately, exactly as WSJT-X does. The same
table can be passed to `encode_ft8`/`encode_ft4` so transmitted hashed
calls are consistent with what the other side expects.

### WSPR: `WsprHashTable`

Type 3 WSPR messages carry a 15-bit hash instead of the callsign:

```python
table = ft8lib.WsprHashTable()

ft8lib.decode_wspr(period1, hashes=table)  # decodes "PJ4/K1ABC 37" (type 2)
ft8lib.decode_wspr(period2, hashes=table)  # "<PJ4/K1ABC> FK52UD 37" resolves

table.save("PJ4/K1ABC")                    # or prime it manually
```

Without a table entry the callsign renders as `<...>`. This mirrors the
`hashtab` file `wsprd` keeps between runs. Note that `HashTable` and
`WsprHashTable` are distinct types with different hash functions — FT8
and WSPR state don't mix.

## Command-line interface

The `ft8lib` console script covers simple encode/decode workflows without
writing Python:

```bash
# Encode to a 12 kHz 16-bit mono WAV
ft8lib encode "CQ K1ABC FN42" cq.wav --mode ft8 --freq 1500
ft8lib encode "K1ABC FN42 37" beacon.wav --mode wspr

# --full-period pads to a complete T/R period, with the nominal leading
# delay (0.5 s; 1 s for WSPR) — ready to be played out on a cycle boundary
ft8lib encode "CQ K1ABC FN42" cq.wav -m ft8 -f 1500 --full-period

# Decode a recording (any sample rate / bit depth; resampled to 12 kHz,
# first channel of multi-channel files)
ft8lib decode 210701_133000.wav --mode ft8
ft8lib decode 210701_1330.wav -m wspr --freq-min 1400 --freq-max 1600
```

`decode` prints one WSJT-X-style line per message and exits nonzero if
nothing decoded. `--freq-min`/`--freq-max` narrow the search window;
other decoder parameters use their Python defaults.

## Performance notes

- The decode hot paths (LDPC belief propagation, OSD, fine sync, WSPR
  demodulation and Fano decoding) run in a small C extension compiled at
  install time. If it isn't available, the library transparently falls
  back to slower pure-numpy implementations of the same algorithms —
  identical results, just slower.
- `depth` and `npasses` are the FT8/FT4 speed/completeness knobs:
  `depth=1` is a fast single sweep; the default `depth=3, npasses=3`
  matches WSJT-X's normal operation.
- For WSPR, `quick=True` and `npasses=1` give the fast sweep;
  `deep=True` and a larger `maxcycles` trade time for the last fraction
  of a dB.
- Measured AWGN decode thresholds (single signal, 10 trials/point,
  `tools/measure_decode_threshold.py`): FT8 ~100% down to −19 dB, FT4 to
  −16 dB (WSJT-X reaches −21 / −17.5 dB; the gap is tuning headroom, not
  missing features).
- Narrowing `freq_min`/`freq_max` to the band segment you care about is
  the cheapest speedup for all three decoders.
