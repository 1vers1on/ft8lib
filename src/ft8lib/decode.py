from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

from . import _kernels
from .encode import ft4_tones_from_bits, ft8_tones_from_bits
from .ldpc import bp_decode, decode174_91
from .pack import HashTable, message_i3n3, pack77, unpack77
from .protocol import FT4, FT8, SAMPLE_RATE
from .subtract import subtract_ft4, subtract_ft8

BP_ITERATIONS = 30
LLR_SCALE = 2.83


@dataclass
class Decode:
    """One decoded message."""

    message: str
    snr: float          # estimated SNR in 2500 Hz (dB)
    dt: float           # time offset relative to the nominal start (s)
    freq: float         # audio frequency of lowest tone (FT8/FT4) or center (WSPR)
    sync: float         # sync-quality score
    mode: str = "FT8"
    ap: int = 0         # a-priori type used (0 = none; 1..6 as in WSJT-X)
    drift: float = 0.0  # linear frequency drift over the transmission (WSPR, Hz)

    def __str__(self):
        apstr = f"  a{self.ap}" if self.ap else ""
        return (f"{self.mode} {self.snr:+3.0f} dB  DT {self.dt:+5.2f} s  "
                f"{self.freq:7.1f} Hz  {self.message}{apstr}")


# ---------------------------------------------------------------------------
# A-priori (AP) decoding support (ft8b.f90 / ft4_decode.f90, ncontest=0)
# ---------------------------------------------------------------------------

# First 29 payload bits of a "CQ ..." standard message, and the last 19 bits
# of standard messages ending in RRR / 73 / RR73 (data statements in ft8b.f90)
_MCQ = np.array([0] * 26 + [1, 0, 0], dtype=np.int64)
_MRRR = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                 dtype=np.int64)
_M73 = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                dtype=np.int64)
_MRR73 = np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
                  dtype=np.int64)
_AP_TAILS = {4: _MRRR, 5: _M73, 6: _MRR73}


class _APInfo:
    """A-priori bit patterns derived from the station's own and DX callsigns."""

    def __init__(self, mycall: str = "", dxcall: str = ""):
        self.types = [1]          # "CQ ??? ???" needs no station info
        self.apsym = None         # +/-1 form of the first 58 payload bits
        self.bits58 = None
        mycall = (mycall or "").strip().upper()
        dxcall = (dxcall or "").strip().upper()
        if len(mycall) < 3:
            return
        dummy = dxcall if len(dxcall) >= 3 else mycall
        msg = f"{mycall} {dummy} RR73"
        scratch = HashTable()
        bits = pack77(msg, hashes=scratch)
        i3, _ = message_i3n3(bits)
        sent, ok = unpack77(bits, nrx=0, hashes=scratch)
        if i3 != 1 or not ok or sent != msg:
            return                # nonstandard call: no AP symbols
        self.bits58 = bits[:58].astype(np.int64)
        self.apsym = 2 * self.bits58 - 1
        self.types.append(2)
        if len(dxcall) >= 3:
            self.types.extend([3, 4, 5, 6])


def _ft8_ap_passes(llra, llrc, ap: _APInfo):
    """Yield (llr, apmask, iaptype) for the FT8 AP passes (ft8b.f90)."""
    apmag = float(np.abs(llra).max()) * 1.1
    for iaptype in ap.types:
        if iaptype >= 2 and ap.apsym is None:
            continue
        for base in (llra, llrc):
            llrz = base.copy()
            apmask = np.zeros(174, dtype=np.int64)
            if iaptype == 1:                    # CQ ??? ???
                apmask[0:29] = 1
                llrz[0:29] = apmag * (2 * _MCQ - 1)
                apmask[74:77] = 1
                llrz[74:76] = -apmag
                llrz[76] = apmag
            elif iaptype == 2:                  # MyCall ??? ???
                apmask[0:29] = 1
                llrz[0:29] = apmag * ap.apsym[0:29]
                apmask[74:77] = 1
                llrz[74:76] = -apmag
                llrz[76] = apmag
            elif iaptype == 3:                  # MyCall DxCall ???
                apmask[0:58] = 1
                llrz[0:58] = apmag * ap.apsym
                apmask[74:77] = 1
                llrz[74:76] = -apmag
                llrz[76] = apmag
            else:                               # MyCall DxCall RRR|73|RR73
                apmask[0:77] = 1
                llrz[0:58] = apmag * ap.apsym
                llrz[58:77] = apmag * (2 * _AP_TAILS[iaptype] - 1)
            yield llrz, apmask, iaptype


def _ft4_ap_passes(llrc, ap: _APInfo):
    """Yield (llr, apmask, iaptype) for the FT4 AP passes (ft4_decode.f90).

    FT4 payloads are scrambled with RVEC before FEC, so the a-priori bit
    patterns are scrambled accordingly.  Only types 1, 2, 3 and 6 exist.
    """
    rvec = FT4.RVEC.astype(np.int64)
    apmag = float(np.abs(llrc).max()) * 1.1
    mcq = 2 * ((_MCQ + rvec[0:29]) % 2) - 1
    apbits = None
    if ap.bits58 is not None:
        apbits = 2 * ((ap.bits58 + rvec[0:58]) % 2) - 1
    mrr73 = 2 * ((_MRR73 + rvec[58:77]) % 2) - 1
    for iaptype in ap.types:
        if iaptype in (4, 5):
            continue
        if iaptype >= 2 and apbits is None:
            continue
        llrz = llrc.copy()
        apmask = np.zeros(174, dtype=np.int64)
        if iaptype == 1:
            apmask[0:29] = 1
            llrz[0:29] = apmag * mcq
        elif iaptype == 2:
            apmask[0:29] = 1
            llrz[0:29] = apmag * apbits[0:29]
        elif iaptype == 3:
            apmask[0:58] = 1
            llrz[0:58] = apmag * apbits
        else:                                   # 6: MyCall DxCall RR73
            apmask[0:77] = 1
            llrz[0:58] = apmag * apbits
            llrz[58:77] = apmag * mrr73
        yield llrz, apmask, iaptype


def _normalize_bmet(bmet: np.ndarray) -> np.ndarray:
    av = bmet.mean()
    var = (bmet * bmet).mean() - av * av
    sigma = np.sqrt(var) if var > 0 else np.sqrt((bmet * bmet).mean())
    if sigma == 0:
        return bmet
    return bmet / sigma


def _prepare(audio, nmax: int) -> np.ndarray:
    dd = np.asarray(audio, dtype=np.float64).ravel()
    if dd.size < nmax:
        dd = np.concatenate([dd, np.zeros(nmax - dd.size)])
    return dd[:nmax]


# ===========================================================================
# FT8
# ===========================================================================

_FT8_GRAY = np.array(FT8.GRAYMAP)


def _ft8_symbol_spectra(dd: np.ndarray) -> np.ndarray:
    """Power spectra of 1/4-symbol steps: array indexed [bin, step] (bin 1-based)."""
    nstep, nsps, nfft1, nh1, nhsym = FT8.NSTEP, FT8.NSPS, FT8.NFFT1, FT8.NH1, FT8.NHSYM
    idx = np.arange(nhsym)[:, None] * nstep + np.arange(nsps)[None, :]
    frames = dd[idx] / 300.0
    spec = np.fft.rfft(frames, n=nfft1, axis=1)
    s = (spec.real ** 2 + spec.imag ** 2)[:, 1:nh1 + 1].T  # rows: bins 1..NH1
    return s


def _ft8_candidates(dd: np.ndarray, nfa: float, nfb: float, syncmin: float,
                    max_candidates: int) -> List[tuple]:
    """Coarse sync search. Returns [(freq, dt, sync), ...]. Port of sync8.f90."""
    nh1, nhsym = FT8.NH1, FT8.NHSYM
    jz = 62
    tstep = FT8.NSTEP / SAMPLE_RATE       # 0.04 s
    df = SAMPLE_RATE / FT8.NFFT1          # 3.125 Hz
    jstrt = int(0.5 / tstep)              # 12 (Fortran truncation)

    s = _ft8_symbol_spectra(dd)

    # padded array indexed by [bin, m] with bin==row, m==col (both 1-based)
    pad_t = 400
    P = np.zeros((nh1 + 16, nhsym + 2 * pad_t))
    P[1:nh1 + 1, pad_t:pad_t + nhsym] = s
    # frequency-summed spectra: Q[i, m] = sum_k P[i+2k, m], k=0..6
    Q = np.zeros_like(P)
    for k in range(7):
        Q[: nh1 + 2, :] += P[2 * k: 2 * k + nh1 + 2, :]

    ia = max(1, int(round(nfa / df)))
    ib = min(nh1 - 13, int(round(nfb / df)))
    if ib <= ia:
        return []
    nbins = ib - ia + 1
    jrange = np.arange(-jz, jz + 1)

    blocks = {}
    for boff, key in ((0, "a"), (144, "b"), (288, "c")):
        t = np.zeros((nbins, 2 * jz + 1))
        t0 = np.zeros((nbins, 2 * jz + 1))
        for n in range(7):
            c0 = pad_t - 1 + jstrt + 4 * n + boff  # column of m for j=0 (0-based)
            rows = ia + 2 * FT8.COSTAS[n]
            t += P[rows:rows + nbins, c0 - jz:c0 + jz + 1]
            t0 += Q[ia:ia + nbins, c0 - jz:c0 + jz + 1]
        blocks[key] = (t, t0)

    ta, t0a = blocks["a"]
    tb, t0b = blocks["b"]
    tc, t0c = blocks["c"]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ta + tb + tc
        t0 = (t0a + t0b + t0c - t) / 6.0
        sync_abc = t / t0
        t2 = tb + tc
        t02 = (t0b + t0c - t2) / 6.0
        sync_bc = t2 / t02
    sync2d = np.fmax(np.nan_to_num(sync_abc, nan=0.0, posinf=0.0, neginf=0.0),
                     np.nan_to_num(sync_bc, nan=0.0, posinf=0.0, neginf=0.0))

    mlag = 13
    lag_lo = sync2d[:, jz - mlag:jz + mlag + 1]
    jpeak = lag_lo.argmax(axis=1) - mlag
    red = lag_lo.max(axis=1)
    jpeak2 = sync2d.argmax(axis=1) - jz
    red2 = sync2d.max(axis=1)

    npctile = int(round(0.40 * nbins))
    if npctile < 1:
        return []
    base = np.sort(red)[npctile - 1]
    base2 = np.sort(red2)[npctile - 1]
    if base <= 0 or base2 <= 0:
        return []
    red = red / base
    red2 = red2 / base2

    order = np.argsort(-red, kind="stable")
    cands = []
    for n in order[:1000]:
        if red[n] >= syncmin and np.isfinite(red[n]):
            cands.append([(ia + n) * df, (jpeak[n] - 0.5) * tstep, red[n]])
        if jpeak2[n] != jpeak[n] and red2[n] >= syncmin and np.isfinite(red2[n]):
            cands.append([(ia + n) * df, (jpeak2[n] - 0.5) * tstep, red2[n]])

    # keep only the strongest of near-duplicate (freq, dt) candidates
    for i in range(len(cands)):
        for j in range(i):
            if (abs(cands[i][0] - cands[j][0]) < 4.0
                    and abs(cands[i][1] - cands[j][1]) < 0.04):
                if cands[i][2] >= cands[j][2]:
                    cands[j][2] = 0.0
                else:
                    cands[i][2] = 0.0
    cands = [c for c in cands if c[2] >= syncmin]
    cands.sort(key=lambda c: -c[2])
    return [tuple(c) for c in cands[:max_candidates]]


class _FT8Downsampler:
    """Mix a candidate frequency to baseband at 200 S/s (ft8_downsample.f90)."""

    NFFT1 = 192000
    NFFT2 = 3200

    def __init__(self, dd: np.ndarray):
        x = np.zeros(self.NFFT1)
        x[: FT8.NMAX] = dd
        self.cx = np.fft.rfft(x)
        self.taper = 0.5 * (1.0 + np.cos(np.arange(101) * np.pi / 100))

    def __call__(self, f0: float) -> np.ndarray:
        df = SAMPLE_RATE / self.NFFT1
        baud = SAMPLE_RATE / FT8.NSPS
        i0 = int(round(f0 / df))
        it = min(int(round((f0 + 8.5 * baud) / df)), self.NFFT1 // 2)
        ib = max(1, int(round((f0 - 1.5 * baud) / df)))
        k = it - ib + 1
        c1 = np.zeros(self.NFFT2, dtype=complex)
        c1[:k] = self.cx[ib:it + 1]
        c1[:101] *= self.taper[::-1]
        c1[k - 101:k] *= self.taper
        c1 = np.roll(c1, -(i0 - ib))
        c1 = np.fft.ifft(c1) * self.NFFT2 / np.sqrt(float(self.NFFT1) * self.NFFT2)
        return c1


_FT8_NP2 = 2812
_FT8_CSYNC = np.exp(
    1j * 2 * np.pi * np.array(FT8.COSTAS)[:, None] * np.arange(32)[None, :] / 32.0
)  # (7, 32) Costas waveforms
_FT8_CSYNC_CONJ = np.conj(_FT8_CSYNC)


def _ft8_sync8d(cd0: np.ndarray, i0: int, ctwk: Optional[np.ndarray]) -> float:
    """Sync power for a downsampled FT8 signal (sync8d.f90)."""
    if ctwk is None:
        cc = _FT8_CSYNC_CONJ
    else:
        cc = np.conj(ctwk)[None, :] * _FT8_CSYNC_CONJ
    if _kernels.HAVE_FAST:
        return _kernels.sync8d(cd0, i0, cc, _FT8_NP2)
    sync = 0.0
    for i in range(7):
        for block in (0, 36, 72):
            i1 = i0 + (i + block) * 32
            if 0 <= i1 and i1 + 31 <= _FT8_NP2 - 1:
                z = np.dot(cd0[i1:i1 + 32], cc[i])
                sync += z.real ** 2 + z.imag ** 2
    return sync


# combination tables for the noncoherent multi-symbol metrics
def _combo_tables():
    tables = {}
    for nsym in (1, 2, 3):
        nt = 2 ** (3 * nsym)
        i = np.arange(nt)
        digs = [(i >> (3 * (nsym - 1 - d))) & 7 for d in range(nsym)]
        tables[nsym] = [_FT8_GRAY[d] for d in digs]
    return tables


_FT8_COMBOS = _combo_tables()
_ONE9 = (np.arange(512)[:, None] & (1 << np.arange(9))[None, :]) != 0  # [i, bitpos]
_ONE8 = (np.arange(256)[:, None] & (1 << np.arange(8))[None, :]) != 0


def _ft8_softbits(cs: np.ndarray) -> List[np.ndarray]:
    """Bit metrics for one FT8 signal.  cs: complex spectra (8, 79).

    Returns [llra, llrb, llrc, llrd, llre] (each 174 long), port of the
    metric section of ft8b.f90.
    """
    bmeta = np.zeros(174)
    bmetb = np.zeros(174)
    bmetc = np.zeros(174)
    bmetd = np.zeros(174)
    for nsym in (1, 2, 3):
        nt = 2 ** (3 * nsym)
        combos = _FT8_COMBOS[nsym]
        ibmax = {1: 2, 2: 5, 3: 8}[nsym]
        kvals = np.arange(1, 30, nsym)
        for ihalf in (1, 2):
            ks = kvals + (7 if ihalf == 1 else 43)
            # symbol indices are 1-based in Fortran; cs column = ks-1
            s2 = cs[combos[0]][:, ks - 1]
            for d in range(1, nsym):
                s2 = s2 + cs[combos[d]][:, ks - 1 + d]
            s2 = np.abs(s2)                         # (nt, len(kvals))
            i32 = 1 + (kvals - 1) * 3 + (ihalf - 1) * 87
            for ib in range(ibmax + 1):
                valid = i32 + ib <= 174
                idx = i32[valid] + ib - 1
                mask = _ONE9[:nt, ibmax - ib]
                mx1 = s2[mask].max(axis=0)
                mx0 = s2[~mask].max(axis=0)
                bm = mx1 - mx0
                if nsym == 1:
                    bmeta[idx] = bm[valid]
                    den = np.maximum(mx1, mx0)
                    with np.errstate(invalid="ignore"):
                        bmetd[idx] = np.where(den > 0, bm / den, 0.0)[valid]
                elif nsym == 2:
                    bmetb[idx] = bm[valid]
                else:
                    bmetc[idx] = bm[valid]
    stacked = np.stack([bmeta, bmetb, bmetc])
    bmete = stacked[np.abs(stacked).argmax(axis=0), np.arange(174)]
    return [LLR_SCALE * _normalize_bmet(b)
            for b in (bmeta, bmetb, bmetc, bmetd, bmete)]


def _decode_ft8_events(audio, freq_min, freq_max, syncmin, max_candidates,
                       hashes, depth, mycall, dxcall, npasses):
    """Run the FT8 decoder, yielding (result, is_new) as messages are found.

    ``is_new`` is False when the result is a better copy (higher sync) of an
    already-yielded message.
    """
    hashes = hashes or HashTable()
    dd = _prepare(audio, FT8.NMAX).copy()
    if not np.any(dd):
        return

    max_osd = 2 if depth >= 2 else -1
    ap = _APInfo(mycall, dxcall) if depth >= 3 else None
    if depth < 2:
        npasses = 1

    fs2 = SAMPLE_RATE / FT8.NDOWN  # 200 Hz
    dt2 = 1.0 / fs2
    results = {}
    for _ in range(npasses):
        candidates = _ft8_candidates(dd, freq_min, freq_max, syncmin,
                                     max_candidates)
        if not candidates:
            break
        downsample = _FT8Downsampler(dd)
        new_signals = []
        for f1, xdt, sync_coarse in candidates:
            cd0 = downsample(f1)
            i0 = int(round((xdt + 0.5) * fs2))
            # time search +/- one half symbol
            syncs = [(_ft8_sync8d(cd0, idt, None), idt)
                     for idt in range(i0 - 10, i0 + 11)]
            smax, ibest = max(syncs)
            # frequency search +/- 2.5 Hz
            delfbest = 0.0
            smax = 0.0
            for ifr in range(-5, 6):
                delf = ifr * 0.5
                ctwk = np.exp(1j * 2 * np.pi * delf * dt2 * np.arange(32))
                s = _ft8_sync8d(cd0, ibest, ctwk)
                if s > smax:
                    smax, delfbest = s, delf
            f1 = f1 + delfbest
            cd0 = downsample(f1)
            syncs = [(_ft8_sync8d(cd0, ibest + idt, None), idt)
                     for idt in range(-4, 5)]
            smax, idt = max(syncs)
            ibest += idt
            xdt = (ibest - 1) * dt2

            # extract symbol spectra
            cs = np.zeros((8, FT8.NN), dtype=complex)
            s8 = np.zeros((8, FT8.NN))
            i1 = ibest + np.arange(FT8.NN) * 32
            valid = (i1 >= 0) & (i1 + 31 <= _FT8_NP2 - 1)
            spec = np.fft.fft(cd0[i1[valid, None] + np.arange(32)], axis=1)
            cs[:, valid] = spec[:, :8].T / 1e3
            s8[:, valid] = np.abs(spec[:, :8]).T

            # hard sync quality check
            nsync = 0
            for k in range(7):
                for block in (0, 36, 72):
                    if s8[:, k + block].argmax() == FT8.COSTAS[k]:
                        nsync += 1
            if nsync <= 6:
                continue

            llrs = _ft8_softbits(cs)
            attempts = [(llr, None, 0) for llr in llrs]
            if ap is not None:
                attempts.extend(_ft8_ap_passes(llrs[0], llrs[2], ap))

            for llr, apmask, iaptype in attempts:
                message77, cw, nharderrors, dmin = decode174_91(
                    llr, apmask=apmask, max_osd=max_osd, norder=2,
                    max_iterations=BP_ITERATIONS)
                if message77 is None or nharderrors > 36:
                    continue
                if not cw.any():
                    continue
                i3, n3 = message_i3n3(message77)
                if i3 > 5 or (i3 == 0 and n3 > 6) or (i3 == 0 and n3 == 2):
                    continue
                msg, ok = unpack77(message77, nrx=1, hashes=hashes)
                if not ok:
                    continue
                # SNR estimate from signal/noise tone powers (ft8b.f90)
                itone = ft8_tones_from_bits(message77)
                ksym = np.arange(FT8.NN)
                xsig = (s8[itone, ksym] ** 2).sum()
                xnoi = (s8[(itone + 4) % 7, ksym] ** 2).sum()
                arg = xsig / xnoi - 1.0 if xnoi > 0 else 0.0
                xsnr = 10.0 * np.log10(max(arg, 0.001)) - 27.0
                xsnr = max(xsnr, -25.0)
                result = Decode(message=msg, snr=round(xsnr), dt=xdt - 0.5,
                                freq=f1, sync=sync_coarse, mode="FT8",
                                ap=iaptype)
                prev = results.get(msg)
                if prev is None:
                    new_signals.append((itone, f1, xdt))
                if prev is None or result.sync > prev.sync:
                    results[msg] = result
                    yield result, prev is None
                break

        if not new_signals:
            break
        for itone, f1, tstart in new_signals:
            subtract_ft8(dd, itone, f1, tstart)


def decode_ft8(audio, freq_min: float = 200.0, freq_max: float = 4000.0,
               syncmin: float = 1.3, max_candidates: int = 120,
               hashes: Optional[HashTable] = None, depth: int = 3,
               mycall: str = "", dxcall: str = "",
               npasses: int = 3) -> List[Decode]:
    """Decode all FT8 signals in a 15-second, 12 kHz audio array.

    Parameters
    ----------
    audio : array-like, real audio samples at 12000 S/s (any scale).
    freq_min, freq_max : audio frequency search range in Hz.
    syncmin : minimum coarse sync strength (WSJT-X uses 1.3).
    max_candidates : cap on the number of candidates to try per pass.
    hashes : optional HashTable for <hashed> callsign resolution.
    depth : 1 = BP only, single pass;
            2 = BP+OSD with multi-pass signal subtraction;
            3 = additionally use a-priori (AP) decoding.
    mycall, dxcall : station callsigns enabling the deeper AP types
            (MyCall ..., MyCall DxCall ...); plain "CQ" AP needs neither.
    npasses : maximum subtraction passes at depth >= 2.

    Returns a list of Decode results sorted by frequency.  Decodes found
    with a-priori information carry the AP type in the ``ap`` field.
    """
    results = {}
    for result, _ in _decode_ft8_events(audio, freq_min, freq_max, syncmin,
                                        max_candidates, hashes, depth,
                                        mycall, dxcall, npasses):
        results[result.message] = result
    return sorted(results.values(), key=lambda r: r.freq)


def decode_ft8_stream(audio, freq_min: float = 200.0, freq_max: float = 4000.0,
                      syncmin: float = 1.3, max_candidates: int = 120,
                      hashes: Optional[HashTable] = None, depth: int = 3,
                      mycall: str = "", dxcall: str = "",
                      npasses: int = 3) -> Iterator[Decode]:
    """Like decode_ft8, but yield each message as soon as it decodes.

    Results come in discovery order (strongest candidates first, later
    subtraction passes last) rather than sorted by frequency; each unique
    message is yielded once.  Parameters are those of decode_ft8.
    """
    for result, is_new in _decode_ft8_events(audio, freq_min, freq_max,
                                             syncmin, max_candidates, hashes,
                                             depth, mycall, dxcall, npasses):
        if is_new:
            yield result


# ===========================================================================
# FT4
# ===========================================================================

_FT4_GRAY = np.array(FT4.GRAYMAP)
_FT4_NSS = FT4.NSPS // FT4.NDOWN          # 32 samples/symbol after downsampling
_FT4_NDMAX = FT4.NMAX // FT4.NDOWN        # 4032
_FT4_SYNC_OFFSETS = (0, 33 * _FT4_NSS, 66 * _FT4_NSS, 99 * _FT4_NSS)


def _nuttal_window(n: int) -> np.ndarray:
    i = np.arange(n)
    return (0.3635819
            - 0.4891775 * np.cos(2 * np.pi * i / n)
            + 0.1365995 * np.cos(4 * np.pi * i / n)
            - 0.0106411 * np.cos(6 * np.pi * i / n))


def _ft4_baseline(savg: np.ndarray, nfa: int, nfb: int) -> Optional[np.ndarray]:
    """Percentile-based polynomial baseline fit (ft4_baseline.f90).

    savg is indexed by bin (1-based; index 0 unused).
    """
    nh1 = FT4.NH1
    df = SAMPLE_RATE / FT4.NFFT1
    ia = max(int(round(200.0 / df)), nfa)
    ib = min(nh1, nfb)
    if ib - ia < 50:
        return None
    with np.errstate(divide="ignore"):
        sdb = 10.0 * np.log10(np.maximum(savg[ia:ib + 1], 1e-30))
    npts = ib - ia + 1
    nseg = 10
    nlen = npts // nseg
    i0 = npts // 2
    xs, ys = [], []
    for nseg_i in range(nseg):
        ja = nseg_i * nlen
        jb = ja + nlen
        seg = sdb[ja:jb]
        if seg.size == 0:
            continue
        base = np.percentile(seg, 10)
        sel = seg <= base
        xs.extend((np.arange(ja, jb)[sel] - i0).tolist())
        ys.extend(seg[sel].tolist())
    if len(xs) < 10:
        return None
    coeffs = np.polynomial.polynomial.polyfit(np.array(xs), np.array(ys), 4)
    sbase = np.zeros(nh1 + 1)
    t = np.arange(npts) - i0
    fit = np.polynomial.polynomial.polyval(t, coeffs) + 0.65
    sbase[ia:ib + 1] = 10.0 ** (fit / 10.0)
    return sbase


def _ft4_candidates(dd: np.ndarray, nfa: float, nfb: float, syncmin: float,
                    max_candidates: int) -> List[tuple]:
    """Spectral peak candidate search (getcandidates4.f90). Returns [(f0, strength)]."""
    nfft1, nh1, nstep, nhsym = FT4.NFFT1, FT4.NH1, FT4.NSTEP, FT4.NHSYM
    window = _nuttal_window(nfft1)
    idx = np.arange(nhsym)[:, None] * nstep + np.arange(nfft1)[None, :]
    frames = dd[idx] * window / 300.0
    spec = np.fft.rfft(frames, axis=1)
    s = (spec.real ** 2 + spec.imag ** 2)[:, 1:nh1 + 1]
    savg = np.zeros(nh1 + 1)
    savg[1:] = s.mean(axis=0)

    savsm = np.zeros(nh1 + 1)
    for i in range(8, nh1 - 6):
        savsm[i] = savg[i - 7:i + 8].mean()

    df = SAMPLE_RATE / nfft1
    ia = max(int(nfa / df), int(round(200.0 / df)))
    ib = min(int(nfb / df), int(round(4910.0 / df)))
    sbase = _ft4_baseline(savg, ia, ib)
    if sbase is None or np.any(sbase[ia:ib + 1] <= 0):
        return []
    savsm[ia:ib + 1] /= sbase[ia:ib + 1]
    f_offset = -1.5 * SAMPLE_RATE / FT4.NSPS

    cands = []
    for i in range(ia + 1, ib):
        if savsm[i] >= savsm[i - 1] and savsm[i] >= savsm[i + 1] and savsm[i] >= syncmin:
            den = savsm[i - 1] - 2 * savsm[i] + savsm[i + 1]
            delta = 0.5 * (savsm[i - 1] - savsm[i + 1]) / den if den != 0 else 0.0
            fpeak = (i + delta) * df + f_offset
            if fpeak < 200.0 or fpeak > 4910.0:
                continue
            speak = savsm[i] - 0.25 * (savsm[i - 1] - savsm[i + 1]) * delta
            cands.append((fpeak, speak))
            if len(cands) >= max_candidates:
                break
    cands.sort(key=lambda c: -c[1])
    return cands


class _FT4Downsampler:
    """Mix to baseband at 666.67 S/s (ft4_downsample.f90)."""

    def __init__(self, dd: np.ndarray):
        self.cx = np.fft.rfft(dd)
        nfft2 = _FT4_NDMAX
        df = SAMPLE_RATE / FT4.NMAX
        baud = SAMPLE_RATE / FT4.NSPS
        iwt = int(0.5 * baud / df)
        iwf = int(4 * baud / df)
        window = np.zeros(nfft2)
        window[:iwt] = 0.5 * (1 + np.cos(np.pi * np.arange(iwt - 1, -1, -1) / iwt))
        window[iwt:iwt + iwf] = 1.0
        window[iwt + iwf:2 * iwt + iwf] = 0.5 * (1 + np.cos(np.pi * np.arange(iwt) / iwt))
        iws = int(baud / df)
        self.window = np.roll(window, -iws)
        self.nfft2 = nfft2

    def __call__(self, f0: float) -> np.ndarray:
        df = SAMPLE_RATE / FT4.NMAX
        i0 = int(round(f0 / df))
        nfft2 = self.nfft2
        c1 = np.zeros(nfft2, dtype=complex)
        nmax_half = FT4.NMAX // 2
        if 0 <= i0 <= nmax_half:
            c1[0] = self.cx[i0]
        hi = np.arange(1, nfft2 // 2 + 1)
        sel = i0 + hi <= nmax_half
        c1[hi[sel]] = self.cx[i0 + hi[sel]]
        sel = i0 - hi >= 0
        c1[nfft2 - hi[sel]] = self.cx[i0 - hi[sel]]
        c1 *= self.window / nfft2
        return np.fft.ifft(c1) * nfft2


def _ft4_sync_templates():
    """Stride-2 sync templates (16 samples/symbol) for the 4 Costas blocks."""
    templates = []
    for costas in (FT4.COSTAS_A, FT4.COSTAS_B, FT4.COSTAS_C, FT4.COSTAS_D):
        phases = []
        phi = 0.0
        for tone in costas:
            dphi = 2 * 2 * np.pi * tone / _FT4_NSS
            for _ in range(_FT4_NSS // 2):
                phases.append(phi)
                phi = (phi + dphi) % (2 * np.pi)
        templates.append(np.exp(1j * np.array(phases)))
    return templates


_FT4_TEMPLATES = _ft4_sync_templates()
_FT4_TEMPLATES_ARR = np.stack(_FT4_TEMPLATES).astype(complex)
_FT4_OFFSETS_ARR = np.array(_FT4_SYNC_OFFSETS, dtype=np.int64)
_FT4_DT_EFF = 2.0 / (SAMPLE_RATE / FT4.NDOWN)  # stride-2 sample interval


def _ft4_sync_search(cd2: np.ndarray, idf_range, istart_range) -> tuple:
    """Grid search of sync power over frequency offsets and start samples."""
    istarts = np.fromiter(istart_range, dtype=np.int64)
    idfs = np.fromiter(idf_range, dtype=np.int64)
    if _kernels.HAVE_FAST:
        val, istart, idf = _kernels.ft4_sync_search(
            cd2, istarts, idfs, _FT4_DT_EFF, _FT4_TEMPLATES_ARR, _FT4_OFFSETS_ARR)
        return float(val), int(istart), int(idf)

    n64 = 2 * _FT4_NSS
    taps = np.arange(0, 4 * _FT4_NSS, 2)
    best = (-1.0, 0, 0)
    for idf in idfs.tolist():
        twk = np.exp(1j * 2 * np.pi * idf * _FT4_DT_EFF * np.arange(n64))
        sync = np.zeros(len(istarts))
        for b, boff in enumerate(_FT4_SYNC_OFFSETS):
            i1 = istarts + boff
            valid = (i1 >= 0) & (i1 + 4 * _FT4_NSS - 1 <= _FT4_NDMAX - 1)
            idx = np.where(valid, i1, 0)[:, None] + taps[None, :]
            z = cd2[idx] @ np.conj(_FT4_TEMPLATES[b] * twk)
            sync += np.where(valid, np.abs(z), 0.0)
        imax = int(sync.argmax())
        if sync[imax] > best[0]:
            best = (float(sync[imax]), int(istarts[imax]), idf)
    return best


def _ft4_bitmetrics(cd: np.ndarray):
    """Soft bit metrics for one FT4 signal (get_ft4_bitmetrics.f90).

    cd: complex baseband, NN*NSS samples.  Returns (bitmetrics(206,3), badsync).
    """
    nss, nn = _FT4_NSS, FT4.NN
    spec = np.fft.fft(cd[: nn * nss].reshape(nn, nss), axis=1)
    cs = spec[:, :4].T.copy()
    s4 = np.abs(cs)

    nsync = 0
    for k in range(4):
        for costas, boff in zip((FT4.COSTAS_A, FT4.COSTAS_B, FT4.COSTAS_C, FT4.COSTAS_D),
                                (0, 33, 66, 99)):
            if s4[:, k + boff].argmax() == costas[k]:
                nsync += 1
    if nsync < 8:
        return None, True

    bitmetrics = np.zeros((2 * nn, 3))
    for nseq, nsym in enumerate((1, 2, 4), start=0):
        nt = 2 ** (2 * nsym)
        i = np.arange(nt)
        digs = [(i >> (2 * (nsym - 1 - d))) & 3 for d in range(nsym)]
        combos = [_FT4_GRAY[d] for d in digs]
        ibmax = {1: 1, 2: 3, 4: 7}[nsym]
        ksvals = np.arange(1, nn - nsym + 2, nsym)
        s2 = cs[combos[0]][:, ksvals - 1]
        for d in range(1, nsym):
            s2 = s2 + cs[combos[d]][:, ksvals - 1 + d]
        s2 = np.abs(s2)                             # (nt, len(ksvals))
        ipt = 1 + (ksvals - 1) * 2
        for ib in range(ibmax + 1):
            valid = ipt + ib <= 2 * nn
            mask = _ONE8[:nt, ibmax - ib]
            bm = s2[mask].max(axis=0) - s2[~mask].max(axis=0)
            bitmetrics[ipt[valid] + ib - 1, nseq] = bm[valid]

    bitmetrics[204:206, 1] = bitmetrics[204:206, 0]
    bitmetrics[200:204, 2] = bitmetrics[200:204, 1]
    bitmetrics[204:206, 2] = bitmetrics[204:206, 0]
    for col in range(3):
        bitmetrics[:, col] = _normalize_bmet(bitmetrics[:, col])
    return bitmetrics, False


def _decode_ft4_events(audio, freq_min, freq_max, syncmin, max_candidates,
                       hashes, depth, mycall, dxcall, npasses):
    """Run the FT4 decoder, yielding (result, is_new) as messages are found.

    ``is_new`` is False when the result is a better copy (higher sync) of an
    already-yielded message.
    """
    hashes = hashes or HashTable()
    dd = _prepare(audio, FT4.NMAX).copy()
    if not np.any(dd):
        return

    max_osd = 2 if depth >= 2 else -1
    ap = _APInfo(mycall, dxcall) if depth >= 3 else None
    if depth < 2:
        npasses = 1

    fs2 = SAMPLE_RATE / FT4.NDOWN  # 666.67 Hz
    results = {}
    for _ in range(npasses):
        candidates = _ft4_candidates(dd, freq_min, freq_max, syncmin,
                                     max_candidates)
        if not candidates:
            break
        downsample = _FT4Downsampler(dd)
        new_signals = []
        for f0, strength in candidates:
            cd2 = downsample(f0)
            power = np.mean(np.abs(cd2) ** 2)
            if power > 0:
                cd2 = cd2 / np.sqrt(power)
            # coarse search: +/-12 Hz in 3 Hz steps, start -344..1012 step 4
            smax, ibest, idfbest = _ft4_sync_search(
                cd2, range(-12, 13, 3), range(-344, 1013, 4))
            # refinement
            smax, ibest, idfbest = _ft4_sync_search(
                cd2, range(idfbest - 4, idfbest + 5),
                range(max(-344, ibest - 5), min(1012, ibest + 5) + 1))
            if smax < 1.2:
                continue
            f1 = f0 + idfbest
            if f1 <= 10.0 or f1 >= 4990.0:
                continue
            cb = downsample(f1)
            # normalize over the message duration like ft4_decode.f90
            power = np.sum(np.abs(cb) ** 2) / (_FT4_NSS * FT4.NN)
            if power > 0:
                cb = cb / np.sqrt(power)
            cd = np.zeros(FT4.NN * _FT4_NSS, dtype=complex)
            if ibest >= 0:
                it = min(_FT4_NDMAX - 1, ibest + FT4.NN * _FT4_NSS - 1)
                cd[: it - ibest + 1] = cb[ibest:it + 1]
            else:
                cd[-ibest: FT4.NN * _FT4_NSS] = cb[: FT4.NN * _FT4_NSS + ibest]

            bitmetrics, badsync = _ft4_bitmetrics(cd)
            if badsync:
                continue
            hbits = (bitmetrics[:, 0] >= 0).astype(int)
            ns = (np.sum(hbits[0:8] == np.array([0, 0, 0, 1, 1, 0, 1, 1]))
                  + np.sum(hbits[66:74] == np.array([0, 1, 0, 0, 1, 1, 1, 0]))
                  + np.sum(hbits[132:140] == np.array([1, 1, 1, 0, 0, 1, 0, 0]))
                  + np.sum(hbits[198:206] == np.array([1, 0, 1, 1, 0, 0, 0, 1])))
            if ns < 20:
                continue

            llr_cols = []
            for col in range(3):
                llr = np.zeros(174)
                llr[0:58] = bitmetrics[8:66, col]
                llr[58:116] = bitmetrics[74:132, col]
                llr[116:174] = bitmetrics[140:198, col]
                llr_cols.append(llr * LLR_SCALE)
            attempts = [(llr, None, 0) for llr in llr_cols]
            if ap is not None:
                attempts.extend(_ft4_ap_passes(llr_cols[2], ap))

            for llr, apmask, iaptype in attempts:
                scrambled77, cw, nharderrors, dmin = decode174_91(
                    llr, apmask=apmask, max_osd=max_osd, norder=2,
                    max_iterations=40)
                if scrambled77 is None or nharderrors > 36:
                    continue
                if not scrambled77.any():
                    continue
                message77 = (scrambled77 + FT4.RVEC) % 2
                msg, ok = unpack77(message77, nrx=1, hashes=hashes)
                if not ok:
                    continue
                xsnr = (10 * np.log10(strength - 1.0) - 14.8
                        if strength > 1.0 else -21.0)
                xsnr = max(-21.0, xsnr)
                xdt = ibest / fs2 - 0.5
                result = Decode(message=msg, snr=round(xsnr), dt=xdt, freq=f1,
                                sync=smax, mode="FT4", ap=iaptype)
                prev = results.get(msg)
                if prev is None:
                    itone = ft4_tones_from_bits(message77)
                    new_signals.append((itone, f1, ibest / fs2))
                if prev is None or result.sync > prev.sync:
                    results[msg] = result
                    yield result, prev is None
                break

        if not new_signals:
            break
        for itone, f1, tstart in new_signals:
            subtract_ft4(dd, itone, f1, tstart)


def decode_ft4(audio, freq_min: float = 200.0, freq_max: float = 4000.0,
               syncmin: float = 1.18, max_candidates: int = 100,
               hashes: Optional[HashTable] = None, depth: int = 3,
               mycall: str = "", dxcall: str = "",
               npasses: int = 3) -> List[Decode]:
    """Decode all FT4 signals in a 7.5-second, 12 kHz audio array.

    Accepts audio of up to 6.048 s (72576 samples); shorter arrays are
    zero-padded.  See decode_ft8 for the meaning of depth/mycall/dxcall.
    Returns a list of Decode results sorted by frequency.
    """
    results = {}
    for result, _ in _decode_ft4_events(audio, freq_min, freq_max, syncmin,
                                        max_candidates, hashes, depth,
                                        mycall, dxcall, npasses):
        results[result.message] = result
    return sorted(results.values(), key=lambda r: r.freq)


def decode_ft4_stream(audio, freq_min: float = 200.0, freq_max: float = 4000.0,
                      syncmin: float = 1.18, max_candidates: int = 100,
                      hashes: Optional[HashTable] = None, depth: int = 3,
                      mycall: str = "", dxcall: str = "",
                      npasses: int = 3) -> Iterator[Decode]:
    """Like decode_ft4, but yield each message as soon as it decodes.

    Results come in discovery order (strongest candidates first, later
    subtraction passes last) rather than sorted by frequency; each unique
    message is yielded once.  Parameters are those of decode_ft4.
    """
    for result, is_new in _decode_ft4_events(audio, freq_min, freq_max,
                                             syncmin, max_candidates, hashes,
                                             depth, mycall, dxcall, npasses):
        if is_new:
            yield result
