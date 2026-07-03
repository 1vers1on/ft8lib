"""Coherent signal subtraction for multi-pass decoding.

Ports of WSJT-X lib/ft8/subtractft8.f90 and lib/ft4/subtractft4.f90.

A decoded signal is regenerated as a complex reference waveform; the
received data is mixed against it and low-pass filtered to estimate the
signal's time-varying complex amplitude, and the reconstructed signal is
subtracted from the data.  Weaker signals hidden underneath then become
decodable on the next pass.
"""

from __future__ import annotations

import numpy as np

from .encode import gen_ft4wave, gen_ft8wave
from .protocol import FT4, FT8, SAMPLE_RATE


def _lowpass_kernel_fft(nfilt: int, nfft: int) -> np.ndarray:
    """FFT of the normalized cos^2 smoothing window, centered at sample 0."""
    j = np.arange(-nfilt // 2, nfilt // 2 + 1)
    window = np.cos(np.pi * j / nfilt) ** 2
    window /= window.sum()
    cw = np.zeros(nfft)
    cw[: nfilt + 1] = window
    cw = np.roll(cw, -(nfilt // 2))
    return np.fft.fft(cw)


_FT8_NFILT = 4000
_FT4_NFILT = 1400
_ft8_cw = None
_ft8_endcorr = None
_ft4_cw = None


def _ft8_filter():
    global _ft8_cw, _ft8_endcorr
    if _ft8_cw is None:
        _ft8_cw = _lowpass_kernel_fft(_FT8_NFILT, FT8.NMAX)
        j = np.arange(-_FT8_NFILT // 2, _FT8_NFILT // 2 + 1)
        window = np.cos(np.pi * j / _FT8_NFILT) ** 2
        sumw = window.sum()
        # correction for the filter hanging off the frame edges
        half = window[_FT8_NFILT // 2:]  # offsets 0..NFILT/2 from center
        tail = np.cumsum(half[::-1])[::-1] / sumw
        _ft8_endcorr = 1.0 / (1.0 - tail)
    return _ft8_cw, _ft8_endcorr


def subtract_ft8(dd: np.ndarray, itone, f0: float, tstart: float) -> None:
    """Subtract one decoded FT8 signal from dd (modified in place).

    tstart is the absolute signal start time in seconds (nominally ~0.5).
    """
    nframe = FT8.NZ
    cwfft, endcorr = _ft8_filter()
    cref = gen_ft8wave(itone, f0=f0, complex_output=True)
    nstart = int(tstart * SAMPLE_RATE)

    camp = np.zeros(FT8.NMAX, dtype=complex)
    a = max(0, -nstart)
    b = min(nframe, FT8.NMAX - nstart)
    if b <= a:
        return
    camp[a:b] = dd[nstart + a:nstart + b] * np.conj(cref[a:b])

    cfilt = np.fft.ifft(np.fft.fft(camp) * cwfft)
    ncorr = _FT8_NFILT // 2 + 1
    cfilt[:ncorr] *= endcorr
    cfilt[nframe - 1:nframe - 1 - ncorr:-1] *= endcorr

    dd[nstart + a:nstart + b] -= 2.0 * (cfilt[a:b] * cref[a:b]).real


def _ft4_filter():
    global _ft4_cw
    if _ft4_cw is None:
        _ft4_cw = _lowpass_kernel_fft(_FT4_NFILT, FT4.NMAX)
    return _ft4_cw


def subtract_ft4(dd: np.ndarray, itone, f0: float, tstart: float) -> None:
    """Subtract one decoded FT4 signal from dd (modified in place).

    tstart is the absolute start time of the 103 data/sync symbols; the
    generated waveform includes a one-symbol ramp-up before that point.
    """
    nframe = FT4.NZ2
    cwfft = _ft4_filter()
    cref = gen_ft4wave(itone, f0=f0, complex_output=True)
    nstart = int(tstart * SAMPLE_RATE) - FT4.NSPS

    camp = np.zeros(FT4.NMAX, dtype=complex)
    a = max(0, -nstart)
    b = min(nframe, FT4.NMAX - nstart)
    if b <= a:
        return
    camp[a:b] = dd[nstart + a:nstart + b] * np.conj(cref[a:b])

    cfilt = np.fft.ifft(np.fft.fft(camp) * cwfft)
    dd[nstart + a:nstart + b] -= 2.0 * (cfilt[a:b] * cref[a:b]).real
