#!/usr/bin/env python3
"""Profile the FT8/FT4 decode path on synthetic multi-signal audio.

Builds a busy receive period (several simultaneous transmissions at
different frequencies/SNRs), decodes it, and reports where the time goes
via cProfile. Useful for finding hot spots before/after an optimization
without needing a real recording.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ft8lib import FT4, FT8, decode_ft4, decode_ft8, encode_ft4, encode_ft8

CALLSIGNS = [
    "K1ABC", "W9XYZ", "JA1XYZ", "VK2ABC", "DL1ABC", "N5ABC", "W1XYZ",
    "EA1XYZ", "VE3ABC", "VK2DEF", "KA1ABC", "N0XYZ", "W2DEF", "PA3ABC",
    "G4XYZ", "F5ABC", "OK1XYZ", "SM0ABC", "ON4XYZ", "HB9ABC",
]
GRIDS = ["FN42", "EN91", "JO62", "IN80", "QF56", "JJ00", "GG66", "PM95"]


def _make_message(rng: np.random.Generator, i: int) -> str:
    a, b = CALLSIGNS[i % len(CALLSIGNS)], CALLSIGNS[(i + 7) % len(CALLSIGNS)]
    kind = i % 4
    if kind == 0:
        return f"CQ {a} {rng.choice(GRIDS)}"
    if kind == 1:
        return f"{a} {b} {rng.integers(-20, 20):+03d}"
    if kind == 2:
        return f"{a} {b} RR73"
    return f"{a} {b} 73"


def _build_audio(mode: str, nsignals: int, snr_low: float, snr_high: float,
                  freq_min: float, freq_max: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nmax = FT8.NMAX if mode == "ft8" else FT4.NMAX
    encode = encode_ft8 if mode == "ft8" else encode_ft4
    sigma = 1.0
    audio = rng.normal(0.0, sigma, nmax)
    freqs = np.linspace(freq_min, freq_max, nsignals, endpoint=False)
    freqs = freqs + rng.uniform(-20, 20, nsignals)
    start = nmax // 20
    for i in range(nsignals):
        msg = _make_message(rng, i)
        snr = rng.uniform(snr_low, snr_high)
        amp = sigma * np.sqrt(10 ** (snr / 10) * 2 * 2500 / 6000)
        try:
            wave = amp * encode(msg, f0=float(freqs[i]))
        except ValueError:
            continue
        audio[start:start + len(wave)] += wave
    return audio


def _decode(mode: str, audio: np.ndarray, freq_min: float, freq_max: float,
            depth: int, npasses: int):
    fn = decode_ft8 if mode == "ft8" else decode_ft4
    return fn(audio, freq_min=freq_min, freq_max=freq_max, depth=depth,
              npasses=npasses)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="profile_decode.py",
        description="Profile decode_ft8/decode_ft4 on a synthetic busy band.",
    )
    parser.add_argument("--mode", choices=["ft8", "ft4"], default="ft4")
    parser.add_argument("--nsignals", type=int, default=15,
                        help="number of simultaneous transmissions")
    parser.add_argument("--snr-low", type=float, default=-16.0)
    parser.add_argument("--snr-high", type=float, default=-4.0)
    parser.add_argument("--freq-min", type=float, default=300.0)
    parser.add_argument("--freq-max", type=float, default=2900.0)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--npasses", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5,
                        help="decode calls to include in the profiled run")
    parser.add_argument("--top", type=int, default=25,
                        help="number of profiler rows to print")
    parser.add_argument("--sort", default="cumulative",
                        choices=["cumulative", "tottime", "calls"],
                        help="pstats sort key")
    args = parser.parse_args(argv)

    audio = _build_audio(args.mode, args.nsignals, args.snr_low, args.snr_high,
                         args.freq_min, args.freq_max, args.seed)

    t0 = time.perf_counter()
    results = _decode(args.mode, audio, args.freq_min, args.freq_max,
                      args.depth, args.npasses)
    elapsed = time.perf_counter() - t0
    print(f"{args.mode.upper()}: {args.nsignals} signals injected, "
          f"{len(results)} decoded, single call took {elapsed:.3f}s")
    for r in sorted(results, key=lambda r: r.freq):
        print(f"  {r}")

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(args.repeat):
        _decode(args.mode, audio, args.freq_min, args.freq_max, args.depth,
               args.npasses)
    profiler.disable()

    print(f"\ncProfile over {args.repeat} calls, sorted by {args.sort}:")
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.sort_stats(args.sort)
    stats.print_stats(args.top)
    print(buf.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
