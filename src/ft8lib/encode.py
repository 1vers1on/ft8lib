"""FT8/FT4 encoders: message -> channel symbols -> GFSK audio waveform.

Ports of WSJT-X lib/ft8/genft8.f90, lib/ft4/genft4.f90,
lib/ft8/gen_ft8wave.f90 and lib/ft4/gen_ft4wave.f90.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .ldpc import encode174_91
from .pack import HashTable, pack77, unpack77
from .protocol import FT4, FT8, SAMPLE_RATE, gfsk_pulse


def ft8_tones_from_bits(msgbits) -> np.ndarray:
    """77 payload bits -> 79 channel symbols (tones 0-7)."""
    codeword = encode174_91(msgbits)
    itone = np.zeros(FT8.NN, dtype=np.int64)
    icos = np.array(FT8.COSTAS)
    itone[0:7] = icos
    itone[36:43] = icos
    itone[72:79] = icos
    graymap = np.array(FT8.GRAYMAP)
    k = 7
    for j in range(FT8.ND):          # 58 data symbols, 3 bits each
        if j == 29:
            k += 7                   # skip the middle Costas array
        i = 3 * j
        idx = codeword[i] * 4 + codeword[i + 1] * 2 + codeword[i + 2]
        itone[k] = graymap[idx]
        k += 1
    return itone


def ft4_tones_from_bits(msgbits) -> np.ndarray:
    """77 payload bits -> 103 channel symbols (tones 0-3)."""
    bits = (np.asarray(msgbits, dtype=np.uint8) + FT4.RVEC) % 2
    codeword = encode174_91(bits)
    itmp = np.zeros(FT4.ND, dtype=np.int64)
    for i in range(FT4.ND):
        two = int(codeword[2 * i]) * 2 + int(codeword[2 * i + 1])
        itmp[i] = {0: 0, 1: 1, 2: 3, 3: 2}[two]  # gray map 00,01,11,10
    itone = np.zeros(FT4.NN, dtype=np.int64)
    itone[0:4] = FT4.COSTAS_A
    itone[4:33] = itmp[0:29]
    itone[33:37] = FT4.COSTAS_B
    itone[37:66] = itmp[29:58]
    itone[66:70] = FT4.COSTAS_C
    itone[70:99] = itmp[58:87]
    itone[99:103] = FT4.COSTAS_D
    return itone


def _gfsk_phase_increments(itone, nsps: int, bt: float, f0: float,
                           fsample: float, extend_edges: bool) -> np.ndarray:
    """Smoothed frequency waveform dphi of length (nsym+2)*nsps."""
    nsym = len(itone)
    twopi = 2.0 * np.pi
    dt = 1.0 / fsample
    hmod = 1.0
    t = (np.arange(1, 3 * nsps + 1) - 1.5 * nsps) / nsps
    pulse = gfsk_pulse(bt, t)
    dphi_peak = twopi * hmod / nsps
    dphi = np.zeros((nsym + 2) * nsps)
    for j, tone in enumerate(itone):
        ib = j * nsps
        dphi[ib:ib + 3 * nsps] += dphi_peak * pulse * tone
    if extend_edges:
        # dummy symbols at either end with the edge tone values (FT8)
        dphi[0:2 * nsps] += dphi_peak * itone[0] * pulse[nsps:3 * nsps]
        dphi[nsym * nsps:(nsym + 2) * nsps] += dphi_peak * itone[-1] * pulse[0:2 * nsps]
    dphi += twopi * f0 * dt
    return dphi


def gen_ft8wave(itone, f0: float = 1000.0, fsample: float = SAMPLE_RATE,
                bt: float = FT8.BT, complex_output: bool = False) -> np.ndarray:
    """Generate the FT8 waveform for 79 tones. Length = 79*1920 samples at 12 kHz."""
    itone = np.asarray(itone)
    nsym = len(itone)
    nsps = int(round(FT8.NSPS * fsample / SAMPLE_RATE))
    nwave = nsym * nsps
    dphi = _gfsk_phase_increments(itone, nsps, bt, f0, fsample, extend_edges=True)
    # skip the leading dummy symbol
    phi = np.concatenate([[0.0], np.cumsum(dphi[nsps:nsps + nwave - 1])]) % (2 * np.pi)
    wave = np.exp(1j * phi) if complex_output else np.sin(phi)
    # envelope shaping of the first and last symbols
    nramp = int(round(nsps / 8.0))
    ramp = (1.0 - np.cos(2 * np.pi * np.arange(nramp) / (2.0 * nramp))) / 2.0
    wave[:nramp] *= ramp
    k1 = nsym * nsps - nramp
    wave[k1:k1 + nramp] *= ramp[::-1]
    return wave


def gen_ft4wave(itone, f0: float = 1000.0, fsample: float = SAMPLE_RATE,
                complex_output: bool = False) -> np.ndarray:
    """Generate the FT4 waveform for 103 tones. Length = 105*576 samples at 12 kHz."""
    itone = np.asarray(itone)
    nsym = len(itone)
    nsps = int(round(FT4.NSPS * fsample / SAMPLE_RATE))
    nwave = (nsym + 2) * nsps
    dphi = _gfsk_phase_increments(itone, nsps, FT4.BT, f0, fsample, extend_edges=False)
    phi = np.concatenate([[0.0], np.cumsum(dphi[: nwave - 1])]) % (2 * np.pi)
    wave = np.exp(1j * phi) if complex_output else np.sin(phi)
    # ramp-up / ramp-down over one full symbol
    ramp = (1.0 - np.cos(2 * np.pi * np.arange(nsps) / (2.0 * nsps))) / 2.0
    wave[:nsps] *= ramp
    k1 = (nsym + 1) * nsps
    wave[k1:k1 + nsps] *= ramp[::-1]
    return wave


def encode_ft8(message: str, f0: float = 1000.0,
               hashes: Optional[HashTable] = None) -> np.ndarray:
    """Encode a message string to an FT8 audio waveform (float64, 12 kHz).

    The returned waveform is 12.64 s long (151680 samples); transmission
    normally starts 0.5 s into the 15-s cycle.
    """
    bits = pack77(message, hashes=hashes)
    _check_roundtrip(bits, hashes)
    return gen_ft8wave(ft8_tones_from_bits(bits), f0=f0)


def encode_ft4(message: str, f0: float = 1000.0,
               hashes: Optional[HashTable] = None) -> np.ndarray:
    """Encode a message string to an FT4 audio waveform (float64, 12 kHz).

    The returned waveform is 5.04 s long (60480 samples); transmission
    normally starts 0.5 s into the 7.5-s cycle.
    """
    bits = pack77(message, hashes=hashes)
    _check_roundtrip(bits, hashes)
    return gen_ft4wave(ft4_tones_from_bits(bits), f0=f0)


def _check_roundtrip(bits, hashes) -> None:
    _, ok = unpack77(bits, nrx=0, hashes=hashes)
    if not ok:
        raise ValueError("message cannot be encoded (bad message)")
