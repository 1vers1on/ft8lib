"""CRC-14 and LDPC(174,91) tests."""

import numpy as np
import pytest

from ft8lib.crc import crc14, crc14_check
from ft8lib.ldpc import bp_decode, check_parity, encode174_91


def crc14_reference(bits77):
    """Independent CRC implementation: direct port of lib/ft8/get_crc14.f90.

    Divides the 96-bit block (77 msg + 5 pad + 14 zero CRC bits) by the
    polynomial 0x6757 using the Fortran shift-register formulation.
    """
    p = [1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    mc = list(bits77) + [0] * 19  # 96 bits, CRC field zeroed
    r = mc[:15]
    for i in range(0, 96 - 14):
        if i > 0:
            r[14] = mc[i + 14]
        r = [(a + r[0] * b) % 2 for a, b in zip(r, p)]
        r = r[1:] + [r[0]]
    return int("".join(str(b) for b in r[:14]), 2)


def test_crc_against_fortran_reference():
    rng = np.random.default_rng(0)
    for _ in range(200):
        msg = rng.integers(0, 2, 77).tolist()
        assert crc14(msg) == crc14_reference(msg)


def test_crc_check_roundtrip():
    rng = np.random.default_rng(1)
    msg = rng.integers(0, 2, 77).tolist()
    crc = crc14(msg)
    bits91 = msg + [(crc >> (13 - i)) & 1 for i in range(14)]
    assert crc14_check(bits91)
    bits91[40] ^= 1
    assert not crc14_check(bits91)


def test_generator_satisfies_parity_checks():
    # The generator matrix and the parity-check tables are independent data
    # sources in WSJT-X; consistency of the two validates both ports.
    rng = np.random.default_rng(2)
    for _ in range(100):
        cw = encode174_91(rng.integers(0, 2, 77))
        assert check_parity(cw)


def test_bp_decode_noiseless():
    rng = np.random.default_rng(3)
    msg = rng.integers(0, 2, 77)
    cw = encode174_91(msg)
    llr = (2.0 * cw - 1.0) * 4.0
    decoded, _, nharderrors = bp_decode(llr)
    assert decoded is not None
    assert nharderrors == 0
    np.testing.assert_array_equal(decoded, msg)


def test_bp_decode_corrects_errors():
    rng = np.random.default_rng(4)
    msg = rng.integers(0, 2, 77)
    cw = encode174_91(msg).astype(float)
    x = 2.0 * cw - 1.0
    x[::25] *= -1  # flip 7 of 174 bits hard
    decoded, _, _ = bp_decode(4.0 * x)
    assert decoded is not None
    np.testing.assert_array_equal(decoded, msg)


def test_bp_decode_rejects_noise():
    rng = np.random.default_rng(5)
    for _ in range(10):
        decoded, _, _ = bp_decode(rng.normal(0, 3, 174))
        assert decoded is None
