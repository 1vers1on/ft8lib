# ft8lib

FT8, FT4 and WSPR amateur-radio digital modes in Python.

`ft8lib` is a Python port of the FT8/FT4 protocol implementation from
[WSJT-X](https://wsjt.sourceforge.io/wsjtx.html): message packing (77-bit),
CRC-14, LDPC(174,91) forward error correction, GFSK waveform synthesis, and a
full receive chain (candidate sync search, downconversion, fine time/frequency
sync, noncoherent multi-symbol soft demodulation, belief-propagation decoding).
It also includes a WSPR encoder and a port of the `wsprd` decoder (sync,
noncoherent demodulation, Fano sequential decoding, signal subtraction).

## Installation

```bash
pip install .
```

The only runtime dependency is numpy.  The decode hot paths (LDPC belief
propagation, ordered-statistics decoding, fine sync) are compiled from a
small C extension at install time; if the extension is unavailable the
library transparently falls back to slower pure-numpy implementations of
the same algorithms.

## Usage

### Python API

```python
import numpy as np
import ft8lib

# --- Encode a message to an audio waveform (float64, 12000 S/s) ---
wave = ft8lib.encode_ft8("CQ K1ABC FN42", f0=1500.0)   # 12.64 s
wave4 = ft8lib.encode_ft4("W9XYZ K1ABC RR73", f0=1500.0)  # 5.04 s

# Transmission conventionally starts 0.5 s into the 15 s (FT8) or
# 7.5 s (FT4) cycle:
period = np.zeros(ft8lib.FT8.NMAX)
period[6000:6000 + len(wave)] = wave

# --- Decode a receive period (12000 S/s audio, any amplitude scale) ---
for result in ft8lib.decode_ft8(period):
    print(result)
# FT8 -15 dB  DT +0.00 s  1500.0 Hz  CQ K1ABC FN42

for result in ft8lib.decode_ft4(audio_7p5s):
    print(result)

# --- WSPR: 2-minute periods, 110.6 s transmissions near 1500 Hz ---
wave = ft8lib.encode_wspr("K1ABC FN42 37")
for result in ft8lib.decode_wspr(audio_2min):
    print(result)
```

Each decode is a `ft8lib.Decode` with fields `message`, `snr` (dB in 2500 Hz),
`dt` (seconds relative to the nominal 0.5 s start), `freq` (Hz), `sync`, and
`mode`.

### Realtime decoding

To decode live audio, feed a `RealtimeDecoder` blocks of samples as they
arrive (starting at a cycle boundary).  Like WSJT-X, it runs decode attempts
part-way through the receive period (FT8: 11.8, 13.5 and 14.7 s; FT4: 5.6
and 6.05 s), so most messages are reported seconds before the period ends,
and it rolls over automatically at each cycle boundary:

```python
rt = ft8lib.RealtimeDecoder("FT8", mycall="K1ABC")
for block in audio_blocks:            # e.g. sound-card capture callback
    for result in rt.feed(block):     # new decodes, as soon as they appear
        print(rt.cycle, result)
rt.flush()                            # decode a partial final period
```

`ft8lib.decode_realtime(blocks, mode="FT8")` wraps this as a generator over
an iterable of blocks.  For a single already-captured period there are also
`decode_ft8_stream` / `decode_ft4_stream`, generator versions of the batch
decoders that yield each message as soon as it decodes instead of returning
the full list at the end.

Lower-level building blocks are exposed too:

```python
bits = ft8lib.pack77("CQ K1ABC FN42")        # 77-bit numpy array
msg, ok = ft8lib.unpack77(bits)              # back to text
tones = ft8lib.ft8_tones_from_bits(bits)     # 79 channel symbols (0-7)
codeword = ft8lib.encode174_91(bits)         # 174-bit LDPC codeword
```

Hashed nonstandard callsigns (`<PJ4/K1ABC>`) resolve through a
`ft8lib.HashTable`, which you can keep across decode cycles and prime with
`table.set_station(mycall, dxcall)` — the same behavior as WSJT-X.

### Command line

```bash
# Encode to a WAV file (12 kHz, 16-bit mono)
ft8lib encode "CQ K1ABC FN42" cq.wav --mode ft8 --freq 1500 --full-period

# Decode a recording (any sample rate; resampled to 12 kHz)
ft8lib decode 210701_133000.wav --mode ft8
```

## Message types supported

Standard messages (types 1/2 incl. `/R` and `/P`), free text (0.0),
DXpedition mode (0.1), ARRL Field Day (0.3/0.4), telemetry (0.5), WSPR-style
(0.6), ARRL RTTY roundup (3), nonstandard/hashed calls (4), and EU VHF
contest (5).

## Performance

Decode thresholds measured with additive white Gaussian noise (10 trials per
point, single signal):

| Mode | ~100% decode down to | WSJT-X (with OSD+AP) |
|------|----------------------|----------------------|
| FT8  | −19 dB               | −21 dB               |
| FT4  | −16 dB               | −17.5 dB             |

The port implements the full WSJT-X machinery — all five metric passes,
ordered-statistics decoding (OSD), a-priori (AP) decoding and multi-pass
signal subtraction; the remaining 1–2 dB is algorithm tuning headroom. A
full 15 s FT8 period with a busy band decodes in well under a second on a
typical desktop.

## Documentation

A comprehensive usage guide for the whole public API — encoding, decoding,
realtime operation, WSPR, hash tables and the CLI — lives in
[`docs/usage.md`](docs/usage.md).

## Development

Detailed docs on the encode and decode pipelines, with Fortran-source
correspondence tables, live in [`docs/encode.md`](docs/encode.md) and
[`docs/decode.md`](docs/decode.md).

The LDPC tables in `src/ft8lib/_tables.py` are generated from the WSJT-X
Fortran sources by `tools/gen_tables.py` (expects the `wsjtx` source tree in
the repository root).

Run the tests with:

```bash
pip install -e .[test]
pytest
```

## License

GPL-3.0-or-later. This library is a derivative work of WSJT-X,
Copyright (C) 2001-2024 by Joe Taylor, K1JT, and the WSJT-X Development Team,
licensed under the GNU GPL v3. See `LICENSE`.
