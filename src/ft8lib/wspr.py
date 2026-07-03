"""WSPR encoder and decoder.

A port of the WSJT-X wsprd decoder (lib/wsprd/wsprd.c, by K9AN and K1JT)
and of the WSPR channel encoder (lib/wsprd/wsprsim_utils.c / fano.c /
lib/genwspr.f90).

The transmit chain is: 50-bit source coding (wspr_pack.py), K=32 r=1/2
convolutional encoding (Layland-Lushbaugh polynomials), bit-reversal
interleaving, and 4-FSK modulation against the 162-bit sync vector at
1.4648 baud -- 110.6 s per transmission.

The receive chain mirrors wsprd.c: downconvert 1500 Hz +/- 187.5 Hz to
complex baseband at 375 S/s; find spectral-peak candidates; coarse and
fine sync (time / frequency / linear drift) against the sync vector;
noncoherent (multi-)symbol demodulation to 162 soft symbols; deinterleave;
Fano sequential decoding; unpack, with up to three passes of decoded-signal
subtraction.  The Jelinek stack decoder and the OSD deep search of wsprd
are not ported.

Correspondence to wsprd.c:

    _downsample_wspr              readwavfile (12 kHz real input path)
    _wspr_spectra / _candidates   main: windowed FFTs, smoothed spectrum
    _coarse_sync                  main: shift/drift/freq estimation loop
    _sync_and_demod               sync_and_demodulate (modes 0 and 1)
    _ncsd                         noncoherent_sequence_detection
    _fano_decode                  fano.c fano()
    _subtract_wspr                subtract_signal2
    decode_wspr                   main decode loop
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from . import _kernels
from ._tables import WSPR_METRIC_TABLE
from .decode import Decode
from .protocol import SAMPLE_RATE, WSPR
from .wspr_pack import WsprHashTable, pack_wspr, unpack_wspr

_DT = 1.0 / WSPR.FS_DEC                    # 375 S/s sample period
_DF = WSPR.FS_DEC / WSPR.NSPS_DEC          # tone spacing at 375 S/s (1.4648)
_DF_TONES = (np.arange(4) - 1.5) * _DF     # the four FSK tone offsets
_NPTS = WSPR.NPTS_DEC                      # 46080 baseband samples
_NFFT_TD = _NPTS * WSPR.NDOWN              # 1474560 audio samples (122.9 s)

# decoder tuning constants (wsprd.c main)
_SYMFAC = 50                # soft-symbol normalizing factor
_MINSYNC1 = 0.10            # first sync limit
_MINRMS = 52.0 * (_SYMFAC / 64.0)   # plausible-demodulation test
_IIFAC = 8                  # step size in the final DT peak-up
_DELTA = 60                 # Fano threshold step

_METRIC = np.array(WSPR_METRIC_TABLE)


def _fano_mettab(bias: float) -> np.ndarray:
    """Integer Fano metric table for a given bias (wsprd.c main)."""
    scaled0 = 10.0 * (_METRIC - bias)
    scaled1 = 10.0 * (_METRIC[::-1] - bias)
    # C round(): halfway cases away from zero
    tab = np.stack([scaled0, scaled1])
    return np.where(tab >= 0, np.floor(tab + 0.5),
                    np.ceil(tab - 0.5)).astype(np.int32)


# ---------------------------------------------------------------------------
# Channel encoding
# ---------------------------------------------------------------------------

def _bitreverse8(i: int) -> int:
    return int(f"{i:08b}"[::-1], 2)


# position in the interleaved frame of each sequential symbol
_ILEAVE = np.array([j for j in (_bitreverse8(i) for i in range(256)) if j < 162])


def _interleave(sym: np.ndarray) -> np.ndarray:
    out = np.empty_like(sym)
    out[_ILEAVE] = sym
    return out


def _deinterleave(sym: np.ndarray) -> np.ndarray:
    return sym[_ILEAVE]


def _conv_encode(data: np.ndarray) -> np.ndarray:
    """First 162 output bits of the K=32 r=1/2 encoder (fano.c encode())."""
    bits = np.unpackbits(np.asarray(data, dtype=np.uint8))
    out = np.zeros(162, dtype=np.uint8)
    state = 0
    for i in range(81):
        state = ((state << 1) | int(bits[i])) & 0xFFFFFFFF
        out[2 * i] = bin(state & WSPR.POLY1).count("1") & 1
        out[2 * i + 1] = bin(state & WSPR.POLY2).count("1") & 1
    return out


def wspr_tones_from_message(message: str) -> np.ndarray:
    """Message string -> 162 channel symbols (tones 0-3)."""
    data = pack_wspr(message)
    chanbits = _interleave(_conv_encode(data))
    return (2 * chanbits + WSPR.SYNC).astype(np.int64)


def gen_wsprwave(itone, f0: float = 1500.0, fsample: float = SAMPLE_RATE,
                 complex_output: bool = False) -> np.ndarray:
    """Generate the WSPR waveform for 162 tones (wspr_wav.f90).

    Phase-continuous 4-FSK centered on f0, tone spacing 1.4648 Hz.
    Length 162 * 8192 samples (110.6 s) at 12 kHz.
    """
    itone = np.asarray(itone)
    nsps = int(round(WSPR.NSPS * fsample / SAMPLE_RATE))
    dphi_sym = 2.0 * np.pi * (f0 + (itone - 1.5) * WSPR.TONE_SPACING) / fsample
    dphi = np.repeat(dphi_sym, nsps)
    phi = np.cumsum(dphi) % (2 * np.pi)
    return np.exp(1j * phi) if complex_output else np.sin(phi)


def encode_wspr(message: str, f0: float = 1500.0) -> np.ndarray:
    """Encode a WSPR message to an audio waveform (float64, 12 kHz).

    The returned waveform is 110.6 s long; transmission conventionally
    starts 1 s into the even 2-minute cycle.  Accepted message forms are
    ``"K1ABC FN42 33"``, ``"PJ4/K1ABC 37"`` and ``"<PJ4/K1ABC> FK52UD 37"``.
    """
    data = pack_wspr(message)
    scratch = WsprHashTable()
    sent, _, ok = unpack_wspr(data, scratch)
    if not ok and not message.lstrip().startswith("<"):
        raise ValueError(f"message cannot be encoded: {message!r}")
    if not message.lstrip().startswith("<"):
        if " ".join(sent.split()) != " ".join(message.upper().split()):
            raise ValueError(
                f"message encodes as {' '.join(sent.split())!r}, "
                f"not {message!r} (illegal power level?)")
    chanbits = _interleave(_conv_encode(data))
    return gen_wsprwave(2 * chanbits + WSPR.SYNC, f0=f0)


# ---------------------------------------------------------------------------
# Decoder front end
# ---------------------------------------------------------------------------

def _downsample_wspr(dd: np.ndarray) -> np.ndarray:
    """12 kHz real audio -> 46080 complex samples at 375 S/s, 1500 Hz at DC.

    Port of the .wav branch of readwavfile in wsprd.c.
    """
    x = np.zeros(_NFFT_TD)
    n = min(len(dd), _NFFT_TD)
    x[:n] = dd[:n]
    spec = np.fft.rfft(x)
    df = SAMPLE_RATE / _NFFT_TD
    i0 = int(1500.0 / df + 0.5)
    nh2 = _NPTS // 2
    idx = i0 + np.arange(_NPTS)
    idx[nh2 + 1:] -= _NPTS
    return np.fft.ifft(spec[idx]) * _NPTS / 1000.0


def _wspr_spectra(c: np.ndarray) -> np.ndarray:
    """Power spectra over 2-symbol windows stepped by half symbols.

    Returns ps indexed [bin, step]; bin j is frequency (j - 256) * 375/512 Hz.
    """
    nffts = 4 * (_NPTS // 512) - 1
    w = np.sin(0.006147931 * np.arange(512))
    idx = np.arange(nffts)[:, None] * 128 + np.arange(512)
    # the last windows run past the data; wsprd reads zero padding there
    cpad = np.concatenate([c, np.zeros((nffts - 1) * 128 + 512 - _NPTS,
                                       dtype=c.dtype)])
    spec = np.fft.fft(cpad[idx] * w, axis=1)
    return np.fft.fftshift(spec.real ** 2 + spec.imag ** 2, axes=1).T.copy()


class _Cand:
    __slots__ = ("freq", "snr", "shift", "drift", "sync")

    def __init__(self, freq, snr):
        self.freq = freq
        self.snr = snr
        self.shift = 0
        self.drift = 0.0
        self.sync = 0.0


def _candidates(ps: np.ndarray, fmin: float, fmax: float,
                more_candidates: bool) -> List[_Cand]:
    """Spectral-peak candidate search (wsprd.c main)."""
    df = WSPR.FS_DEC / 512  # 0.7324 Hz bin spacing of the 512-point FFTs
    psavg = ps.sum(axis=1)
    # smooth with a 7-point boxcar, keep +/- 150 Hz (411 bins)
    smspec = np.convolve(psavg, np.ones(7), mode="full")[6:][48:48 + 411]
    noise_level = np.sort(smspec)[122]  # 30th percentile
    if noise_level <= 0:
        return []
    min_snr = 10.0 ** (-8.0 / 10.0)
    snr_scaling_factor = 26.3
    smspec = smspec / noise_level - 1.0
    smspec[smspec < min_snr] = 0.1 * min_snr

    cands = []
    for j in range(1, 410):
        if (smspec[j] > smspec[j - 1] and smspec[j] > smspec[j + 1]
                and len(cands) < 200):
            cands.append(_Cand((j - 205) * df,
                               10 * np.log10(smspec[j]) - snr_scaling_factor))
    if more_candidates:
        for j in range(0, 411, 3):
            if smspec[j] > min_snr and len(cands) < 200:
                cands.append(_Cand((j - 205) * df,
                                   10 * np.log10(smspec[j]) - snr_scaling_factor))

    cands = [c for c in cands if fmin <= c.freq <= fmax]
    cands.sort(key=lambda c: -c.snr)
    return cands


def _coarse_sync(ps: np.ndarray, cand: _Cand, maxdrift: int) -> None:
    """Coarse estimates of shift, drift and frequency (wsprd.c main)."""
    df = WSPR.FS_DEC / 512
    nffts = ps.shape[1]
    if0 = int(cand.freq / df + 256)
    k = np.arange(162)
    k0s = np.arange(-10, 22)
    kindex = k0s[:, None] + 2 * k[None, :]          # (32, 162)
    kvalid = (kindex >= 0) & (kindex < nffts)
    kindex = np.clip(kindex, 0, nffts - 1)
    sgn = (2.0 * WSPR.SYNC - 1.0)

    ifrs = np.arange(if0 - 2, if0 + 3)
    drifts = np.arange(-maxdrift, maxdrift + 1)
    sync_all = np.full((len(ifrs), len(k0s), len(drifts)), -np.inf)
    for a, ifr in enumerate(ifrs):
        for b, idrift in enumerate(drifts):
            ifd = (ifr + (k - 81.0) / 81.0 * idrift / (2.0 * df)).astype(int)
            p = np.sqrt(ps[ifd[None, :, None] + np.array([-3, -1, 1, 3]),
                           kindex[:, :, None]])       # (32, 162, 4)
            p = np.where(kvalid[:, :, None], p, 0.0)
            ss = (sgn * ((p[:, :, 1] + p[:, :, 3])
                         - (p[:, :, 0] + p[:, :, 2]))).sum(axis=1)
            pw = p.sum(axis=(1, 2))
            with np.errstate(invalid="ignore", divide="ignore"):
                sync_all[a, :, b] = np.where(pw > 0, ss / pw, -np.inf)

    a, b, d = np.unravel_index(np.argmax(sync_all), sync_all.shape)
    cand.shift = 128 * (int(k0s[b]) + 1)
    cand.drift = float(drifts[d])
    cand.freq = (float(ifrs[a]) - 256) * df
    cand.sync = float(sync_all[a, b, d])


# ---------------------------------------------------------------------------
# Fine sync and demodulation (sync_and_demodulate / n.c.s.d. in wsprd.c)
# ---------------------------------------------------------------------------

def _tone_correlations(c: np.ndarray, f0: float, drift: float, lag: int):
    """Per-symbol correlations of c against the 4 tones: (4, 162) complex.

    Also returns the per-symbol tone phase advance factors (162, 4).
    """
    i_sym = np.arange(162)
    fp = f0 + (drift / 2.0) * (i_sym - 81.0) / 81.0
    tones = fp[:, None] + _DF_TONES[None, :]                  # (162, 4)
    w = np.exp(-1j * 2 * np.pi * _DT
               * tones[:, :, None] * np.arange(256))          # (162, 4, 256)
    k = lag + i_sym[:, None] * 256 + np.arange(256)
    valid = (k > 0) & (k < _NPTS)
    seg = np.where(valid, c[np.clip(k, 0, _NPTS - 1)], 0.0)   # (162, 256)
    z = np.einsum("is,its->ti", seg, w)                       # (4, 162)
    cf = np.exp(1j * 2 * np.pi * _DT * tones * 256)           # (162, 4)
    return z, cf


def _sync_and_demod_py(c, f1, ifmin, ifmax, fstep,
                       shift1, lagmin, lagmax, lagstep, drift1):
    """Modes 0/1 of sync_and_demodulate: best (sync, freq, lag)."""
    best = (-1e30, f1, shift1)
    for ifreq in range(ifmin, ifmax + 1):
        f0 = f1 + ifreq * fstep
        for lag in range(lagmin, lagmax + 1, lagstep):
            z, _ = _tone_correlations(c, f0, drift1, lag)
            p = np.abs(z)                                     # (4, 162)
            totp = p.sum()
            if totp <= 0:
                continue
            cmet = (p[1] + p[3]) - (p[0] + p[2])
            ss = float(np.where(WSPR.SYNC == 1, cmet, -cmet).sum() / totp)
            if ss > best[0]:
                best = (ss, f0, lag)
    return best


def _ncsd_py(c, f1, shift1, drift1, symfac, nblock, bitbybit):
    """Noncoherent block demodulation -> 162 soft symbols (0..255)."""
    z, cf = _tone_correlations(c, f1, drift1, shift1)
    nseq = 1 << nblock
    fsymb = np.zeros(162)
    for i in range(0, 162, nblock):
        p = np.zeros(nseq)
        for j in range(nseq):
            x = 0.0 + 0.0j
            wc = 1.0 + 0.0j
            for ib in range(nblock):
                bbit = (j >> (nblock - 1 - ib)) & 1
                itone = int(WSPR.SYNC[i + ib]) + 2 * bbit
                x += z[itone, i + ib] * np.conj(wc)
                wc *= cf[i + ib, itone]
            p[j] = abs(x)
        for ib in range(nblock):
            imask = 1 << (nblock - 1 - ib)
            sel = (np.arange(nseq) & imask) != 0
            xm1 = p[sel].max()
            xm0 = p[~sel].max()
            fsymb[i + ib] = xm1 - xm0
            if bitbybit:
                fsymb[i + ib] /= max(xm1, xm0)
    return _normalize_symbols(fsymb, symfac)


def _normalize_symbols(fsymb: np.ndarray, symfac: int) -> np.ndarray:
    fsum = fsymb.mean()
    f2sum = (fsymb * fsymb).mean()
    fac = np.sqrt(max(f2sum - fsum * fsum, 0.0))
    if not fac > 0:
        fac = 1.0
    scaled = np.clip(symfac * fsymb / fac, -128.0, 127.0)
    return (scaled + 128.0).astype(np.uint8)


def _sync_and_demod(c, f1, ifmin, ifmax, fstep,
                    shift1, lagmin, lagmax, lagstep, drift1):
    if _kernels.HAVE_FAST:
        return _kernels.wspr_sync_demod(c, float(f1), int(ifmin), int(ifmax),
                                        float(fstep), int(lagmin), int(lagmax),
                                        int(lagstep), float(drift1))
    return _sync_and_demod_py(c, f1, ifmin, ifmax, fstep,
                              shift1, lagmin, lagmax, lagstep, drift1)


def _ncsd(c, f1, shift1, drift1, symfac, nblock, bitbybit):
    if _kernels.HAVE_FAST:
        return _kernels.wspr_ncsd(c, float(f1), int(shift1), float(drift1),
                                  int(symfac), int(nblock), int(bitbybit))
    return _ncsd_py(c, f1, shift1, drift1, symfac, nblock, bitbybit)


# ---------------------------------------------------------------------------
# Fano sequential decoder (fano.c)
# ---------------------------------------------------------------------------

def _fano_py(symbols, mettab, delta, maxcycles):
    """Pure-Python Fano decoder; port of fano() in fano.c (nbits=81)."""
    nbits = 81
    met0 = mettab[0].tolist()
    met1 = mettab[1].tolist()
    s0 = symbols[0::2]
    s1 = symbols[1::2]
    metrics = [
        (met0[a] + met0[b], met0[a] + met1[b],
         met1[a] + met0[b], met1[a] + met1[b])
        for a, b in zip(s0.tolist(), s1.tolist())
    ]
    poly1, poly2 = WSPR.POLY1, WSPR.POLY2

    def branch_sym(state):
        return (((bin(state & poly1).count("1") & 1) << 1)
                | (bin(state & poly2).count("1") & 1))

    tail = nbits - 31
    enc = [0] * (nbits + 1)
    gam = [0] * (nbits + 1)
    tm0 = [0] * (nbits + 1)
    tm1 = [0] * (nbits + 1)
    cur = [0] * (nbits + 1)

    lsym = branch_sym(0)
    m0 = metrics[0][lsym]
    m1 = metrics[0][3 ^ lsym]
    if m0 > m1:
        tm0[0], tm1[0] = m0, m1
    else:
        tm0[0], tm1[0] = m1, m0
        enc[0] = 1
    t = 0
    node = 0
    maxtotal = maxcycles * nbits
    i = 1
    while i <= maxtotal:
        ngamma = gam[node] + (tm0[node] if cur[node] == 0 else tm1[node])
        if ngamma >= t:
            if gam[node] < t + delta:
                while ngamma >= t + delta:
                    t += delta
            gam[node + 1] = ngamma
            enc[node + 1] = enc[node] << 1
            node += 1
            if node == nbits:
                break
            lsym = branch_sym(enc[node] & 0xFFFFFFFF)
            if node >= tail:
                tm0[node] = metrics[node][lsym]
            else:
                m0 = metrics[node][lsym]
                m1 = metrics[node][3 ^ lsym]
                if m0 > m1:
                    tm0[node], tm1[node] = m0, m1
                else:
                    tm0[node], tm1[node] = m1, m0
                    enc[node] += 1
            cur[node] = 0
        else:
            while True:
                if node == 0 or gam[node - 1] < t:
                    t -= delta
                    if cur[node] != 0:
                        cur[node] = 0
                        enc[node] ^= 1
                    break
                node -= 1
                if node < tail and cur[node] != 1:
                    cur[node] = 1
                    enc[node] ^= 1
                    break
        i += 1
    data = np.zeros(11, dtype=np.uint8)
    for b in range(nbits >> 3):
        data[b] = enc[7 + 8 * b] & 0xFF
    if i >= maxtotal:
        return False, data, gam[node], i + 1
    return True, data, gam[node], i + 1


def _fano_decode(symbols, mettab, maxcycles):
    if _kernels.HAVE_FAST:
        ok, data, metric, cycles = _kernels.wspr_fano(
            symbols, mettab, _DELTA, int(maxcycles))
        return bool(ok), np.frombuffer(data, dtype=np.uint8), metric, cycles
    return _fano_py(symbols, mettab, _DELTA, maxcycles)


# ---------------------------------------------------------------------------
# Decoded-signal subtraction (subtract_signal2 in wsprd.c)
# ---------------------------------------------------------------------------

def _subtract_wspr(c, f0, shift0, drift0, channel_symbols):
    """Subtract the coherent component of a decoded signal, in place."""
    nsym, nsps = 162, 256
    nfilt = 360
    nsig = nsym * nsps
    nc2 = 45000

    cs = np.asarray(channel_symbols, dtype=np.float64)
    i_sym = np.arange(nsym)
    dphi_sym = 2 * np.pi * _DT * (
        f0 + (drift0 / 2.0) * (i_sym - 81.0) / 81.0 + (cs - 1.5) * _DF)
    dphi = np.repeat(dphi_sym, nsps)
    phi = np.cumsum(dphi) - dphi                 # phase before each sample
    ref = np.exp(1j * phi)

    k = shift0 + np.arange(nsig)
    valid = (k > 0) & (k < _NPTS)
    ci = np.zeros(nc2, dtype=complex)
    ci[nfilt:nfilt + nsig][valid] = c[k[valid]] * np.conj(ref[valid])

    w = np.sin(np.pi * np.arange(nfilt) / (nfilt - 1))
    w /= w.sum()
    cf = np.zeros(nc2, dtype=complex)
    conv = np.convolve(ci, w)
    cf[nfilt // 2:nc2 - nfilt // 2] = conv[nfilt // 2 + nfilt // 2 - 1:
                                           nc2 - nfilt // 2 + nfilt // 2 - 1]

    partialsum = np.cumsum(w)
    norm = np.ones(nsig)
    edge = np.arange(nfilt // 2)
    norm[:nfilt // 2] = partialsum[nfilt // 2 + edge]
    norm[nsig - nfilt // 2:] = partialsum[nfilt // 2 + edge[::-1]]

    c[k[valid]] -= (cf[nfilt:nfilt + nsig] * ref / norm)[valid]


# ---------------------------------------------------------------------------
# Main decoder
# ---------------------------------------------------------------------------

def decode_wspr(audio, freq_min: float = 1390.0, freq_max: float = 1610.0,
                hashes: Optional[WsprHashTable] = None, deep: bool = False,
                quick: bool = False, npasses: int = 3,
                subtraction: bool = True, maxcycles: int = 10000,
                bias: float = 0.45) -> List[Decode]:
    """Decode all WSPR signals in a 2-minute, 12 kHz audio array.

    Parameters
    ----------
    audio : array-like, real audio at 12000 S/s covering the receive
        period (transmissions start 1 s into the even UTC minute).
    freq_min, freq_max : audio-frequency search range in Hz.  WSPR signals
        sit in 1400-1600 Hz; wsprd's default range is 1500 +/- 110 Hz.
    hashes : optional WsprHashTable, kept across calls so type 3
        ``<CALL>`` messages can be resolved.
    deep : also try candidates below the local-peak threshold (wsprd -d).
    quick : skip the time-jitter search (wsprd -q).
    npasses : decoding passes; passes subtract decoded signals and retry,
        and the third pass uses block demodulation (1 = wsprd -s,
        2 = wsprd -B, 3 = default).
    maxcycles : Fano timeout, cycles per bit (wsprd -C).
    bias : Fano metric bias (wsprd -z).

    Returns a list of Decode results sorted by frequency: ``freq`` is the
    audio frequency in Hz, ``dt`` the start-time offset relative to the
    nominal 1 s, ``snr`` the estimated SNR in 2500 Hz, and ``drift`` the
    linear frequency drift in Hz over the transmission.
    """
    hashes = hashes if hashes is not None else WsprHashTable()
    dd = np.asarray(audio, dtype=np.float64).ravel()
    if not np.any(dd):
        return []
    c = _downsample_wspr(dd)
    mettab = _fano_mettab(bias)
    fmin = max(freq_min - 1500.0, -150.0)
    fmax = min(freq_max - 1500.0, 150.0)

    results = {}
    uniques = []                       # (callsign, freq) of printed decodes
    ndecodes_pass = 0
    ipass = 0
    while ipass < npasses:
        if ipass == 1 and ndecodes_pass == 0 and npasses > 2:
            ipass = 2
            if ipass >= npasses:
                break
        if ipass < 2:
            nblocksize, maxdrift, minsync2 = 1, 4, 0.12
        else:
            nblocksize, maxdrift, minsync2 = 4, 0, 0.10
        ndecodes_pass = 0

        ps = _wspr_spectra(c)
        cands = _candidates(ps, fmin, fmax, deep)
        for cand in cands:
            _coarse_sync(ps, cand, maxdrift)

        # refine shift and frequency estimates using sync as the metric
        for cand in cands:
            f1, drift1, shift1, sync1 = (cand.freq, cand.drift,
                                         cand.shift, cand.sync)
            lagmin, lagmax, lagstep = shift1 - 128, shift1 + 128, 64
            sync1, f1, shift1 = _sync_and_demod(
                c, f1, 0, 0, 0.0, shift1, lagmin, lagmax, lagstep, drift1)
            sync1, f1, shift1 = _sync_and_demod(
                c, f1, -2, 2, 0.25, shift1, shift1, shift1, lagstep, drift1)
            if ipass < 2:
                syncp, fp_, sp_ = _sync_and_demod(
                    c, f1, 0, 0, 0.0, shift1, shift1, shift1, lagstep,
                    drift1 + 0.5)
                syncm, fm_, sm_ = _sync_and_demod(
                    c, f1, 0, 0, 0.0, shift1, shift1, shift1, lagstep,
                    drift1 - 0.5)
                if syncp > sync1:
                    drift1, sync1, f1, shift1 = drift1 + 0.5, syncp, fp_, sp_
                elif syncm > sync1:
                    drift1, sync1, f1, shift1 = drift1 - 0.5, syncm, fm_, sm_
            if sync1 > _MINSYNC1:
                lagmin, lagmax, lagstep = shift1 - 32, shift1 + 32, 16
                sync1, f1, shift1 = _sync_and_demod(
                    c, f1, 0, 0, 0.0, shift1, lagmin, lagmax, lagstep, drift1)
                sync1, f1, shift1 = _sync_and_demod(
                    c, f1, -2, 2, 0.05, shift1, shift1, shift1, lagstep,
                    drift1)
                cand.freq, cand.shift = f1, shift1
                cand.drift, cand.sync = drift1, sync1

        # merge candidates that converged to the same signal
        wat: List[_Cand] = []
        for cand in cands:
            dupe_of = None
            for prev in wat:
                if (abs(cand.freq - prev.freq) < 0.05
                        and abs(cand.shift - prev.shift) < 16):
                    dupe_of = prev
                    break
            if dupe_of is not None:
                if cand.sync > dupe_of.sync:
                    wat[wat.index(dupe_of)] = cand
            elif cand.sync > minsync2:
                wat.append(cand)

        # demodulate and decode each candidate
        for cand in wat:
            f1, shift1, drift1 = cand.freq, cand.shift, cand.drift
            decoded = False
            data = None
            ib = 1
            while ib <= nblocksize and not decoded:
                blocksize, bitmetric = (ib, 0) if ib < 4 else (1, 1)
                idt = 0
                while not decoded and idt <= 128 // _IIFAC:
                    ii = (idt + 1) // 2
                    if idt % 2 == 1:
                        ii = -ii
                    ii *= _IIFAC
                    symbols = _ncsd(c, f1, shift1 + ii, drift1,
                                    _SYMFAC, blocksize, bitmetric)
                    y = symbols.astype(np.float64) - 128.0
                    if np.sqrt((y * y).mean()) > _MINRMS:
                        sym_d = np.ascontiguousarray(_deinterleave(symbols))
                        decoded, data, _metric, _cycles = _fano_decode(
                            sym_d, mettab, maxcycles)
                    idt += 1
                    if quick:
                        break
                ib += 1

            if not decoded:
                continue
            ndecodes_pass += 1
            message, callsign, ok = unpack_wspr(data, hashes)
            if subtraction and ok:
                try:
                    chan_syms = wspr_tones_from_message(message)
                except (ValueError, KeyError):
                    chan_syms = None
                if chan_syms is not None:
                    _subtract_wspr(c, f1, shift1, drift1, chan_syms)
            if not ok:
                continue
            dupe = any(callsign == pc and abs(f1 - pf) < 4.0
                       for pc, pf in uniques)
            if not dupe:
                uniques.append((callsign, f1))
                msg = " ".join(message.split())
                results[msg] = Decode(
                    message=msg, snr=round(cand.snr), dt=shift1 * _DT - 0.5,
                    freq=1500.0 + f1, sync=cand.sync, mode="WSPR",
                    drift=drift1)
        ipass += 1

    return sorted(results.values(), key=lambda r: r.freq)
