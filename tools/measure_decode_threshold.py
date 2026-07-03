#!/usr/bin/env python3
"""Measure FT8/FT4 decode thresholds with synthetic AWGN.

The script encodes a known-good message, embeds it in a receive period, adds
Gaussian noise at a range of SNR values, and reports the weakest SNR that
still decodes reliably.

By default the success criterion is 100% decode rate across the requested
number of trials. Use ``--target-rate`` to relax that threshold.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ft8lib import FT4, FT8, decode_ft4, decode_ft8, encode_ft4, encode_ft8


DEFAULT_MESSAGES = {
    "ft8": "CQ K1ABC FN42",
    "ft4": "W9XYZ K1ABC RR73",
}


@dataclass
class SweepPoint:
    snr_db: float
    hits: int
    trials: int

    @property
    def rate(self) -> float:
        return self.hits / self.trials if self.trials else 0.0


def _embed(wave: np.ndarray, nmax: int, snr_db: float, rng: np.random.Generator,
           start: int = 6000) -> np.ndarray:
    """Embed a waveform in white Gaussian noise at the requested SNR."""
    sigma = np.sqrt(1.2 / 10 ** (snr_db / 10))
    audio = rng.normal(0.0, sigma, nmax)
    audio[start:start + len(wave)] += wave
    return audio


def _decode(mode: str, audio: np.ndarray, freq_min: float, freq_max: float,
            depth: int, mycall: str, dxcall: str, npasses: int):
    if mode == "ft8":
        return decode_ft8(
            audio,
            freq_min=freq_min,
            freq_max=freq_max,
            depth=depth,
            mycall=mycall,
            dxcall=dxcall,
            npasses=npasses,
        )
    return decode_ft4(
        audio,
        freq_min=freq_min,
        freq_max=freq_max,
        depth=depth,
        mycall=mycall,
        dxcall=dxcall,
        npasses=npasses,
    )


def _encode(mode: str, message: str, f0: float):
    if mode == "ft8":
        return encode_ft8(message, f0=f0)
    return encode_ft4(message, f0=f0)


def _period_length(mode: str) -> int:
    return FT8.NMAX if mode == "ft8" else FT4.NMAX


def _sweep_snr(mode: str, message: str, f0: float, start_db: float,
               stop_db: float, step_db: float, trials: int, seed: int,
               freq_min: float, freq_max: float, depth: int,
               mycall: str, dxcall: str, npasses: int) -> list[SweepPoint]:
    wave = _encode(mode, message, f0)
    nmax = _period_length(mode)
    points = []
    snr_values = np.arange(start_db, stop_db + step_db / 2.0, step_db)
    for snr_db in snr_values:
        hits = 0
        for trial in range(trials):
            rng = np.random.default_rng(seed + trial)
            audio = _embed(wave, nmax, float(snr_db), rng)
            results = _decode(
                mode,
                audio,
                freq_min=freq_min,
                freq_max=freq_max,
                depth=depth,
                mycall=mycall,
                dxcall=dxcall,
                npasses=npasses,
            )
            hits += any(result.message == message for result in results)
        points.append(SweepPoint(float(snr_db), hits, trials))
    return points


def _threshold(points: Iterable[SweepPoint], target_rate: float) -> SweepPoint | None:
    best = None
    for point in points:
        if point.rate >= target_rate:
            best = point
    return best


def _measure_one(mode: str, message: str, args) -> int:
    points = _sweep_snr(
        mode=mode,
        message=message,
        f0=args.freq,
        start_db=args.start_db,
        stop_db=args.stop_db,
        step_db=args.step_db,
        trials=args.trials,
        seed=args.seed,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        depth=args.depth,
        mycall=args.mycall,
        dxcall=args.dxcall,
        npasses=args.npasses,
    )
    threshold = _threshold(points, args.target_rate)

    label = mode.upper()
    print(f"{label} {message}")
    for point in points:
        print(f"  {point.snr_db:6.1f} dB  {point.hits:2d}/{point.trials:<2d}  ({point.rate:5.1%})")
    if threshold is None:
        print(f"  no SNR reached the target rate of {args.target_rate:0.1%}")
        return 1
    print(f"  threshold: {threshold.snr_db:.1f} dB at {threshold.rate:0.1%} success")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="measure_decode_threshold.py",
        description="Measure the minimum SNR needed for FT8/FT4 decode success.",
    )
    parser.add_argument("--mode", choices=["ft8", "ft4", "both"], default="both")
    parser.add_argument("--message", help="message to encode and test")
    parser.add_argument("--freq", type=float, default=1500.0,
                        help="audio tone frequency in Hz (default 1500)")
    parser.add_argument("--start-db", type=float, default=-25.0,
                        help="start of SNR sweep (dB)")
    parser.add_argument("--stop-db", type=float, default=-10.0,
                        help="end of SNR sweep (dB)")
    parser.add_argument("--step-db", type=float, default=1.0,
                        help="SNR step size (dB)")
    parser.add_argument("--trials", type=int, default=10,
                        help="trials per SNR point")
    parser.add_argument("--target-rate", type=float, default=1.0,
                        help="decode success rate required for the threshold")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed offset for the noise generator")
    parser.add_argument("--freq-min", type=float, default=200.0)
    parser.add_argument("--freq-max", type=float, default=4000.0)
    parser.add_argument("--depth", type=int, default=3,
                        help="decoder depth passed through to ft8lib")
    parser.add_argument("--mycall", default="", help="my call sign for AP decoding")
    parser.add_argument("--dxcall", default="", help="DX call sign for AP decoding")
    parser.add_argument("--npasses", type=int, default=3,
                        help="maximum subtraction passes")
    args = parser.parse_args(argv)

    if args.mode == "both" and args.message:
        parser.error("--message cannot be combined with --mode both; choose one mode or omit --message")

    modes = [args.mode] if args.mode != "both" else ["ft8", "ft4"]
    exit_code = 0
    for mode in modes:
        message = args.message or DEFAULT_MESSAGES[mode]
        exit_code |= _measure_one(mode, message, args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())