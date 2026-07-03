"""LDPC(174,91) encoder and decoders.

Ports of WSJT-X lib/ft8/encode174_91.f90, lib/ft8/bpdecode174_91.f90,
lib/ft8/decode174_91.f90 (hybrid BP+OSD) and lib/ft8/osd174_91.f90
(ordered-statistics decoding), using the generator/parity tables from
lib/ft8/ldpc_174_91_c_*.f90.

The OSD reprocessing here differs from the Fortran in one way: instead of
the screened test-pattern search (nextpat91/boxit91), all order-1 and
order-2 error patterns over the 91 most-reliable-basis positions are
evaluated exactly via vectorized linear algebra, which gives coverage at
least as good as WSJT-X's ndeep=2..4 settings.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ._tables import LDPC_GENERATOR_HEX, LDPC_MN, LDPC_NM, LDPC_NRW
from .crc import crc14, crc14_check

N = 174  # codeword bits
K = 91   # message bits (77 + 14-bit CRC)
M = N - K  # parity checks (83)


def _build_generator() -> np.ndarray:
    gen = np.zeros((M, K), dtype=np.uint8)
    for i, row in enumerate(LDPC_GENERATOR_HEX):
        bits = bin(int(row, 16))[2:].zfill(92)[:91]  # 23 hex digits -> 92 bits, use first 91
        gen[i] = np.frombuffer(bits.encode(), dtype=np.uint8) - ord("0")
    return gen


_GEN = _build_generator()

# Parity connectivity, 0-based. NM is padded with -1 where a check has only 6 bits.
_NM = np.array(LDPC_NM, dtype=np.int64) - 1          # (83, 7)
_MN = np.array(LDPC_MN, dtype=np.int64) - 1          # (174, 3)
_NRW = np.array(LDPC_NRW, dtype=np.int64)            # (83,)
_NM_VALID = np.arange(7)[None, :] < _NRW[:, None]    # (83, 7) mask
_NM_SAFE = np.where(_NM_VALID, _NM, 0)

# For each (bit, check-slot) edge, the slot index of that bit within the check's
# NM row, so messages can be gathered/subtracted without inner loops.
_EDGE_SLOT = np.zeros((N, 3), dtype=np.int64)
for _j in range(N):
    for _i in range(3):
        _chk = _MN[_j, _i]
        _slots = np.where(_NM[_chk] == _j)[0]
        assert len(_slots) == 1
        _EDGE_SLOT[_j, _i] = _slots[0]


def encode174_91(message77) -> np.ndarray:
    """Append the 14-bit CRC and LDPC-encode: 77 bits -> 174-bit codeword."""
    msg = np.asarray(message77, dtype=np.uint8)
    if msg.shape != (77,):
        raise ValueError("message77 must be 77 bits")
    crc = crc14(msg.tolist())
    crcbits = np.array([(crc >> (13 - i)) & 1 for i in range(14)], dtype=np.uint8)
    message = np.concatenate([msg, crcbits])
    pchecks = (_GEN @ message) % 2
    return np.concatenate([message, pchecks.astype(np.uint8)])


def check_parity(codeword) -> bool:
    """True if all 83 parity checks are satisfied."""
    cw = np.asarray(codeword, dtype=np.int64)
    synd = np.where(_NM_VALID, cw[_NM_SAFE], 0).sum(axis=1) % 2
    return not synd.any()


def bp_decode(
    llr,
    max_iterations: int = 30,
    apmask=None,
) -> Tuple[Optional[np.ndarray], np.ndarray, int]:
    """Belief-propagation decoder for the (174,91) code.

    Parameters
    ----------
    llr : array of 174 log-likelihood ratios (positive means bit=1,
        matching the WSJT-X convention).
    apmask : optional array of 174 flags; where 1, the LLR is treated as
        a-priori known and not updated.

    Returns
    -------
    (message77, codeword, nharderrors): message77 is None when decoding
    failed (no codeword found, or CRC mismatch).
    """
    llr = np.asarray(llr, dtype=np.float64)
    if apmask is None:
        ap_free = np.ones(N, dtype=bool)
    else:
        ap_free = np.asarray(apmask, dtype=np.int64) != 1

    tov = np.zeros((N, 3))          # check -> bit messages, per bit slot
    toc = np.zeros((M, 7))          # bit -> check messages, per check slot

    # initialize bit-to-check messages with the channel LLRs
    toc[:] = np.where(_NM_VALID, llr[_NM_SAFE], 0.0)

    ncnt = 0
    nclast = 0
    for iteration in range(max_iterations + 1):
        zn = np.where(ap_free, llr + tov.sum(axis=1), llr)

        cw = (zn > 0).astype(np.uint8)
        synd = np.where(_NM_VALID, cw[_NM_SAFE], 0).sum(axis=1) % 2
        ncheck = int(synd.sum())
        if ncheck == 0:
            if crc14_check(cw[:K].tolist()):
                nharderrors = int(np.count_nonzero((2 * cw.astype(int) - 1) * llr < 0))
                return cw[:77].copy(), cw, nharderrors
            # valid codeword but bad CRC: keep iterating like WSJT-X

        if iteration > 0:
            if ncheck - nclast < 0:
                ncnt = 0
            else:
                ncnt += 1
            if ncnt >= 5 and iteration >= 10 and ncheck > 15:
                return None, cw, -1
        nclast = ncheck

        # bit-to-check messages: total belief minus what that check contributed
        contrib = np.zeros((M, 7))
        for s in range(3):
            # bits whose s-th check is i contribute zn[bit]-tov[bit,s]
            checks = _MN[:, s]
            slots = _EDGE_SLOT[:, s]
            contrib[checks, slots] = zn - tov[:, s]
        toc = np.where(_NM_VALID, contrib, 0.0)

        # check-to-bit messages
        tanhtoc = np.tanh(-toc / 2.0)
        tanhtoc = np.where(_NM_VALID, tanhtoc, 1.0)
        prod = tanhtoc.prod(axis=1)                       # (83,)
        with np.errstate(divide="ignore", invalid="ignore"):
            # product excluding each slot
            excl = np.where(np.abs(tanhtoc) > 1e-12, prod[:, None] / tanhtoc, 0.0)
            # recompute exactly where a zero made the quotient unreliable
            bad = np.abs(tanhtoc) <= 1e-12
            if bad.any():
                for i, s in zip(*np.nonzero(bad)):
                    mask = np.ones(7, dtype=bool)
                    mask[s] = False
                    excl[i, s] = tanhtoc[i][mask].prod()
        # gather back to bits: tov[j, s] = 2*atanh(product over check _MN[j,s] excl j)
        tmn = excl[_MN, _EDGE_SLOT]                       # (174, 3)
        tmn = np.clip(-tmn, -0.9999999999, 0.9999999999)
        tov = 2.0 * np.arctanh(tmn)

    return None, (llr > 0).astype(np.uint8), -1


# ---------------------------------------------------------------------------
# Ordered-statistics decoding (osd174_91.f90) and the hybrid decoder
# (decode174_91.f90)
# ---------------------------------------------------------------------------

# Systematic generator for OSD: row i is the codeword of unit message e_i
# (91 information bits, no CRC cascade -- WSJT-X Keff=91 case).
_G_FULL = np.hstack([np.eye(K, dtype=np.uint8), _GEN.T.astype(np.uint8)])


def osd_decode(llr, norder: int = 2, apmask=None):
    """Ordered-statistics decoder for the (174,91) code.

    Reduces the generator to systematic form over the 91 most reliable
    received bits, then tests all error patterns of weight <= norder (0..2)
    on those bits.  Returns (message77, codeword, nhardmin, dmin), with
    message77 None when no codeword with a valid CRC was found.
    """
    rx = np.asarray(llr, dtype=np.float64)
    hdec = (rx >= 0).astype(np.uint8)
    absrx = np.abs(rx)
    if apmask is None:
        apm = np.zeros(N, dtype=np.uint8)
    else:
        apm = np.asarray(apmask, dtype=np.uint8)

    # order columns by decreasing reliability
    order = np.argsort(-absrx, kind="stable")
    genmrb = _G_FULL[:, order].copy()
    indices = order.copy()

    # Gaussian elimination: identity on the first K (most reliable
    # independent) columns, swapping in later columns when necessary.
    for d in range(K):
        pivots = np.nonzero(genmrb[d, d:])[0]
        if pivots.size == 0:
            return None, hdec, -1, 0.0  # degenerate; should not happen
        col = d + pivots[0]
        if col != d:
            genmrb[:, [d, col]] = genmrb[:, [col, d]]
            indices[[d, col]] = indices[[col, d]]
        rows = np.nonzero(genmrb[:, d])[0]
        rows = rows[rows != d]
        if rows.size:
            genmrb[rows] ^= genmrb[d]

    hdec_p = hdec[indices]
    absrx_p = absrx[indices]
    apm_p = apm[indices]

    m0 = hdec_p[:K]
    c0 = (m0 @ genmrb) % 2  # order-0 codeword
    base = (c0 ^ hdec_p).astype(np.float64)

    # dd(S) = sum_n absrx_n * (1 - B_n * prod_{i in S} R_in) / 2  where
    # B = +/-1 form of the order-0 error, R = +/-1 form of generator rows.
    w = absrx_p * (1.0 - 2.0 * base)          # absrx * B
    const = 0.5 * absrx_p.sum()
    R = 1.0 - 2.0 * genmrb.astype(np.float64)  # (K, N) in +/-1
    d0 = const - 0.5 * w.sum()

    best_dd = d0
    best_pattern = ()
    blocked = apm_p[:K] == 1

    if norder >= 1:
        dd1 = const - 0.5 * (R @ w)
        dd1[blocked] = np.inf
        i1 = int(dd1.argmin())
        if dd1[i1] < best_dd:
            best_dd = float(dd1[i1])
            best_pattern = (i1,)

    if norder >= 2:
        Rw = R * w[None, :]
        dd2 = const - 0.5 * (Rw @ R.T)        # (K, K); diagonal = order 0
        dd2[blocked, :] = np.inf
        dd2[:, blocked] = np.inf
        np.fill_diagonal(dd2, np.inf)
        j2 = int(dd2.argmin())
        i2a, i2b = divmod(j2, K)
        if dd2[i2a, i2b] < best_dd:
            best_dd = float(dd2[i2a, i2b])
            best_pattern = (i2a, i2b)

    cw_p = c0.copy()
    for i in best_pattern:
        cw_p ^= genmrb[i]

    cw = np.zeros(N, dtype=np.uint8)
    cw[indices] = cw_p
    nhardmin = int((cw ^ hdec).sum())
    dmin = float(((cw ^ hdec) * absrx).sum())
    if not crc14_check(cw[:K].tolist()):
        return None, cw, nhardmin, dmin
    return cw[:77].copy(), cw, nhardmin, dmin


def decode174_91(
    llr,
    apmask=None,
    max_osd: int = 2,
    norder: int = 2,
    max_iterations: int = 30,
) -> Tuple[Optional[np.ndarray], np.ndarray, int, float]:
    """Hybrid BP + OSD decoder (port of decode174_91.f90).

    max_osd < 0 : belief propagation only
    max_osd = 0 : BP, then one OSD pass on the channel LLRs
    max_osd > 0 : BP, then OSD on the accumulated LLRs of the first
                  max_osd BP iterations

    Returns (message77, codeword, nharderrors, dmin); message77 is None on
    failure.
    """
    llr = np.asarray(llr, dtype=np.float64)
    if apmask is None:
        ap_free = np.ones(N, dtype=bool)
    else:
        ap_free = np.asarray(apmask, dtype=np.int64) != 1

    max_osd = min(max_osd, 3)
    zsave = []
    if max_osd == 0:
        zsave.append(llr.copy())

    tov = np.zeros((N, 3))
    toc = np.where(_NM_VALID, llr[_NM_SAFE], 0.0)
    zsum = np.zeros(N)

    ncnt = 0
    nclast = 0
    for iteration in range(max_iterations + 1):
        zn = np.where(ap_free, llr + tov.sum(axis=1), llr)
        zsum += zn
        if 0 < iteration <= max_osd:
            zsave.append(zsum.copy())

        cw = (zn > 0).astype(np.uint8)
        synd = np.where(_NM_VALID, cw[_NM_SAFE], 0).sum(axis=1) % 2
        if int(synd.sum()) == 0:
            if crc14_check(cw[:K].tolist()):
                hdec = (llr >= 0).astype(np.uint8)
                nharderrors = int((hdec ^ cw).sum())
                dmin = float(((hdec ^ cw) * np.abs(llr)).sum())
                return cw[:77].copy(), cw, nharderrors, dmin

        ncheck = int(synd.sum())
        if iteration > 0:
            if ncheck - nclast < 0:
                ncnt = 0
            else:
                ncnt += 1
            if ncnt >= 5 and iteration >= 10 and ncheck > 15:
                break
        nclast = ncheck

        contrib = np.zeros((M, 7))
        for s in range(3):
            checks = _MN[:, s]
            slots = _EDGE_SLOT[:, s]
            contrib[checks, slots] = zn - tov[:, s]
        toc = np.where(_NM_VALID, contrib, 0.0)

        tanhtoc = np.tanh(-toc / 2.0)
        tanhtoc = np.where(_NM_VALID, tanhtoc, 1.0)
        prod = tanhtoc.prod(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            excl = np.where(np.abs(tanhtoc) > 1e-12, prod[:, None] / tanhtoc, 0.0)
            bad = np.abs(tanhtoc) <= 1e-12
            if bad.any():
                for i, s in zip(*np.nonzero(bad)):
                    mask = np.ones(7, dtype=bool)
                    mask[s] = False
                    excl[i, s] = tanhtoc[i][mask].prod()
        tmn = excl[_MN, _EDGE_SLOT]
        tmn = np.clip(-tmn, -0.9999999999, 0.9999999999)
        tov = 2.0 * np.arctanh(tmn)

    # BP failed; try ordered-statistics decoding on the saved LLR sums
    apmask_arr = None if apmask is None else np.asarray(apmask)
    for zn in zsave:
        message77, cw, nhardmin, _ = osd_decode(zn, norder, apmask_arr)
        if message77 is not None and nhardmin > 0:
            hdec = (llr >= 0).astype(np.uint8)
            dmin = float(((hdec ^ cw) * np.abs(llr)).sum())
            return message77, cw, nhardmin, dmin

    return None, (llr > 0).astype(np.uint8), -1, 0.0
