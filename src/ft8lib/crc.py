"""14-bit CRC used by the FT8/FT4 (174,91) code.

Port of WSJT-X lib/crc14.cpp (boost::augmented_crc<14, 0x2757>) as used by
lib/ft8/encode174_91.f90: the CRC is computed over the 77 message bits
followed by 5 zero bits (96-bit augmented block minus the 14 CRC bits).
"""

from __future__ import annotations

from typing import Sequence

CRC_POLYNOMIAL = 0x2757  # truncated polynomial, degree 14


def crc14(bits77: Sequence[int]) -> int:
    """Compute the 14-bit CRC of a 77-bit message (sequence of 0/1)."""
    if len(bits77) != 77:
        raise ValueError(f"expected 77 bits, got {len(bits77)}")
    reg = 0
    for bit in list(bits77) + [0] * 5:  # pad to 82 bits, matching WSJT-X
        reg <<= 1
        if bit:
            reg |= 1
        if reg & (1 << 14):
            reg ^= (1 << 14) | CRC_POLYNOMIAL
    # push through 14 augmentation zeros
    for _ in range(14):
        reg <<= 1
        if reg & (1 << 14):
            reg ^= (1 << 14) | CRC_POLYNOMIAL
    return reg & 0x3FFF


def crc14_check(bits91: Sequence[int]) -> bool:
    """Check a 91-bit message+CRC block (77 message bits + 14 CRC bits)."""
    if len(bits91) != 91:
        raise ValueError(f"expected 91 bits, got {len(bits91)}")
    expected = 0
    for bit in bits91[77:]:
        expected = (expected << 1) | (1 if bit else 0)
    return crc14(bits91[:77]) == expected
