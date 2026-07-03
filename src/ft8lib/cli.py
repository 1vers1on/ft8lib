"""Command-line interface: encode messages to WAV, decode WAV recordings."""

from __future__ import annotations

import argparse
import sys
import wave as wavmod

import numpy as np

from . import __version__, decode_ft4, decode_ft8, encode_ft4, encode_ft8
from .protocol import FT4, FT8, SAMPLE_RATE


def _read_wav(path: str) -> np.ndarray:
    with wavmod.open(path, "rb") as w:
        nch = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    if width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    elif width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float64)
    elif width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0
    else:
        raise SystemExit(f"unsupported WAV sample width: {width} bytes")
    if nch > 1:
        data = data.reshape(-1, nch)[:, 0]
    if rate != SAMPLE_RATE:
        n_out = int(round(len(data) * SAMPLE_RATE / rate))
        data = np.interp(
            np.arange(n_out) * rate / SAMPLE_RATE, np.arange(len(data)), data
        )
    return data


def _write_wav(path: str, audio: np.ndarray) -> None:
    scaled = np.clip(audio, -1.0, 1.0)
    pcm = (scaled * 32767 * 0.9).astype(np.int16)
    with wavmod.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ft8lib", description="FT8/FT4 encoder and decoder"
    )
    parser.add_argument("--version", action="version", version=f"ft8lib {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="encode a message to a WAV file")
    enc.add_argument("message", help='message text, e.g. "CQ K1ABC FN42"')
    enc.add_argument("output", help="output WAV path (12 kHz, 16-bit mono)")
    enc.add_argument("-m", "--mode", choices=["ft8", "ft4"], default="ft8")
    enc.add_argument("-f", "--freq", type=float, default=1500.0,
                     help="audio frequency in Hz (default 1500)")
    enc.add_argument("--full-period", action="store_true",
                     help="pad to a full T/R period with 0.5 s leading delay")

    dec = sub.add_parser("decode", help="decode a WAV recording")
    dec.add_argument("input", help="input WAV path (any rate; resampled to 12 kHz)")
    dec.add_argument("-m", "--mode", choices=["ft8", "ft4"], default="ft8")
    dec.add_argument("--freq-min", type=float, default=200.0)
    dec.add_argument("--freq-max", type=float, default=4000.0)

    args = parser.parse_args(argv)

    if args.command == "encode":
        wave = (encode_ft8 if args.mode == "ft8" else encode_ft4)(
            args.message, f0=args.freq
        )
        if args.full_period:
            period = FT8.NMAX if args.mode == "ft8" else FT4.NMAX
            padded = np.zeros(period)
            start = SAMPLE_RATE // 2
            padded[start:start + len(wave)] = wave
            wave = padded
        _write_wav(args.output, wave)
        print(f"wrote {args.output} ({len(wave) / SAMPLE_RATE:.2f} s, "
              f"{args.mode.upper()} at {args.freq:.1f} Hz)")
        return 0

    audio = _read_wav(args.input)
    results = (decode_ft8 if args.mode == "ft8" else decode_ft4)(
        audio, freq_min=args.freq_min, freq_max=args.freq_max
    )
    for r in results:
        print(r)
    if not results:
        print("no decodes", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
