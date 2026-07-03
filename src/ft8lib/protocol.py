"""Protocol constants for FT8, FT4 and WSPR.

From WSJT-X lib/ft8/ft8_params.f90, lib/ft4/ft4_params.f90, genft8.f90,
genft4.f90 and lib/wsprd/wspr_params.f90.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 12000  # Hz, native sample rate of both modes


class FT8:
    NAME = "FT8"
    KK = 91                 # information bits (77 + CRC14)
    ND = 58                 # data symbols
    NS = 21                 # sync symbols (3 x Costas 7x7)
    NN = NS + ND            # total channel symbols (79)
    NSPS = 1920             # samples per symbol at 12000 S/s
    NZ = NSPS * NN          # samples in full waveform (151680)
    NMAX = 15 * 12000       # samples in a receive period (180000)
    NFFT1 = 2 * NSPS        # FFT length for symbol spectra
    NH1 = NFFT1 // 2
    NSTEP = NSPS // 4       # coarse time-sync step
    NHSYM = NMAX // NSTEP - 3
    NDOWN = 60              # downsample factor
    TONE_SPACING = SAMPLE_RATE / NSPS   # 6.25 Hz
    SYMBOL_PERIOD = NSPS / SAMPLE_RATE  # 0.16 s
    PERIOD = 15.0           # T/R cycle (s)
    NTONES = 8
    BT = 2.0                # Gaussian filter bandwidth-time product

    COSTAS = (3, 1, 4, 0, 6, 5, 2)
    GRAYMAP = (0, 1, 3, 2, 5, 6, 4, 7)


class FT4:
    NAME = "FT4"
    KK = 91
    ND = 87                 # data symbols
    NS = 16                 # sync symbols (4 x 4)
    NN = NS + ND            # sync + data symbols (103)
    NN2 = NN + 2            # incl. ramp up/down symbols (105)
    NSPS = 576              # samples per symbol at 12000 S/s
    NZ = NSPS * NN
    NZ2 = NSPS * NN2        # samples in the shaped waveform (60480)
    NMAX = 21 * 3456        # samples in a receive period (72576)
    NFFT1 = 2304
    NH1 = NFFT1 // 2
    NSTEP = NSPS
    NHSYM = (NMAX - NFFT1) // NSTEP
    NDOWN = 18
    TONE_SPACING = SAMPLE_RATE / NSPS   # 20.833 Hz
    SYMBOL_PERIOD = NSPS / SAMPLE_RATE  # 0.048 s
    PERIOD = 7.5            # T/R cycle (s)
    NTONES = 4
    BT = 1.0

    COSTAS_A = (0, 1, 3, 2)
    COSTAS_B = (1, 0, 2, 3)
    COSTAS_C = (2, 3, 1, 0)
    COSTAS_D = (3, 2, 0, 1)
    GRAYMAP = (0, 1, 3, 2)

    # 77-bit scrambling vector applied before FEC encoding (genft4.f90)
    RVEC = np.array(
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
         1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
         1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1,
         1, 1, 0, 0, 0, 1, 0, 1],
        dtype=np.uint8,
    )


class WSPR:
    NAME = "WSPR"
    NN = 162                # channel symbols
    NSPS = 8192             # samples per symbol at 12000 S/s
    NZ = NSPS * NN          # samples in full waveform (1327104, 110.6 s)
    NMAX = 120 * SAMPLE_RATE   # samples in a receive period (2 minutes)
    NDOWN = 32              # downsample factor: decoder runs at 375 S/s
    FS_DEC = SAMPLE_RATE / NDOWN        # 375.0
    NSPS_DEC = NSPS // NDOWN            # 256 samples/symbol when decoding
    NPTS_DEC = 46080        # downsampled samples the decoder works on
    TONE_SPACING = SAMPLE_RATE / NSPS   # 1.4648 Hz
    SYMBOL_PERIOD = NSPS / SAMPLE_RATE  # 0.6827 s
    PERIOD = 120.0          # T/R cycle (s)
    NTONES = 4
    POLY1 = 0xF2D05351      # Layland-Lushbaugh convolutional code, K=32 r=1/2
    POLY2 = 0xE4613C47

    # 162-bit pseudo-random sync vector (pr3 in wsprd.c)
    SYNC = np.array(
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
         0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
         1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
         0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
         0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0,
         0, 0],
        dtype=np.uint8,
    )


_erf = np.vectorize(__import__("math").erf, otypes=[np.float64])


def gfsk_pulse(bt: float, t: np.ndarray) -> np.ndarray:
    """Gaussian-filtered frequency pulse (lib/ft2/gfsk_pulse.f90)."""
    c = np.pi * np.sqrt(2.0 / np.log(2.0))
    return 0.5 * (_erf(c * bt * (t + 0.5)) - _erf(c * bt * (t - 0.5)))
