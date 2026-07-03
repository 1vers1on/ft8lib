"""Dispatch to the compiled decoder kernels (_ckernels.c) when available.

The C extension implements the decode hot paths — the LDPC belief-
propagation loop, the OSD Gaussian elimination, and the FT8 fine-sync
correlation — as direct transcriptions of the WSJT-X Fortran.  When the
extension was not built, ldpc.py and decode.py fall back to their pure
numpy implementations (several times slower, same results).
"""

from __future__ import annotations

try:
    from ._ckernels import (  # noqa: F401
        bp_hybrid,
        crc14_check,
        osd,
        osd_ge,
        sync8d,
    )

    HAVE_FAST = True
except ImportError:  # pragma: no cover - extension not built
    HAVE_FAST = False
