"""Dispatch to the compiled decoder kernels (_ckernels.c) when available.

The C extension implements the decode hot paths — the LDPC belief-
propagation loop, the OSD Gaussian elimination, the FT8 fine-sync
correlation, and the WSPR demodulator and Fano sequential decoder — as
direct transcriptions of the WSJT-X Fortran/C.  When the extension was
not built, ldpc.py, decode.py and wspr.py fall back to their pure
numpy/Python implementations (several times slower, same results).
"""

from __future__ import annotations

try:
    from ._ckernels import (  # noqa: F401
        bp_hybrid,
        crc14_check,
        ft4_sync_search,
        osd,
        osd_ge,
        sync8d,
        wspr_fano,
        wspr_ncsd,
        wspr_sync_demod,
    )

    HAVE_FAST = True
except ImportError:  # pragma: no cover - extension not built
    HAVE_FAST = False
