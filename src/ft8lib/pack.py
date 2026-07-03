"""77-bit message packing/unpacking for FT8 and FT4.

Python port of WSJT-X lib/77bit/packjt77.f90.  Supported message types:

  i3.n3   description
  -----   ------------------------------------------------------
  0.0     free text (13 chars)
  0.1     DXpedition mode        K1ABC RR73; W9XYZ <KH1/KH7Z> -11
  0.3/0.4 ARRL Field Day         WA9XYZ KA1ABC R 16A EMA
  0.5     telemetry (18 hex digits)
  0.6     WSPR types 1-3
  1       standard message       WA9XYZ/R KA1ABC/R R FN42
  2       EU VHF ("/P" form)     PA3XYZ/P GM4ABC/P R JO22
  3       ARRL RTTY roundup      TU; W9XYZ K1ABC R 579 MA
  4       nonstandard call       <WA9XYZ> PJ4/KA1ABC RR73
  5       EU VHF contest         <PA3XYZ> <G4ABC/P> R 590003 IO91NP

Messages are exchanged as numpy arrays of 77 bits (dtype uint8).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np

NTOKENS = 2063592
MAX22 = 4194304
MAXGRID4 = 32400

A1 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A2 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A3 = "0123456789"
A4 = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HASH_ALPHABET = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/"
TEXT_ALPHABET = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?"

ARRL_SECTIONS = (
    "AB", "AK", "AL", "AR", "AZ", "BC", "CO", "CT", "DE", "EB",
    "EMA", "ENY", "EPA", "EWA", "GA", "GH", "IA", "ID", "IL", "IN",
    "KS", "KY", "LA", "LAX", "NS", "MB", "MDC", "ME", "MI", "MN",
    "MO", "MS", "MT", "NC", "ND", "NE", "NFL", "NH", "NL", "NLI",
    "NM", "NNJ", "NNY", "TER", "NTX", "NV", "OH", "OK", "ONE", "ONN",
    "ONS", "OR", "ORG", "PAC", "PR", "QC", "RI", "SB", "SC", "SCV",
    "SD", "SDG", "SF", "SFL", "SJV", "SK", "SNJ", "STX", "SV", "TN",
    "UT", "VA", "VI", "VT", "WCF", "WI", "WMA", "WNY", "WPA", "WTX",
    "WV", "WWA", "WY", "DX", "PE", "NB",
)

RTTY_MULTS = (
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "NB", "NS", "QC", "ON", "MB", "SK", "AB", "BC", "NWT", "NF",
    "LB", "NU", "YT", "PEI", "DC", "DR", "FR", "GD", "GR", "OV",
    "ZH", "ZL",
) + tuple(f"X{i:02d}" for i in range(1, 100))


# ---------------------------------------------------------------------------
# Callsign hashing (10/12/22-bit) and the hash table
# ---------------------------------------------------------------------------

def ihashcall(call: str, m: int) -> int:
    """WSJT-X callsign hash: top m bits of (47055833459 * base38(call)) mod 2^64."""
    c0 = call.upper().ljust(11)[:11]
    n8 = 0
    for ch in c0:
        j = HASH_ALPHABET.find(ch)
        if j < 0:
            j = -1  # match Fortran index()-1 for unknown chars
        n8 = 38 * n8 + j
    return ((47055833459 * n8) & 0xFFFFFFFFFFFFFFFF) >> (64 - m)


class HashTable:
    """Stores callsigns seen in <>-hashed form so they can be recovered later.

    Mirrors the calls10/calls12/calls22 tables in packjt77.f90.  Optionally
    knows the local station's callsign and the DX callsign, which are used
    to resolve hashes in received messages just as WSJT-X does.
    """

    def __init__(self, mycall: str = "", dxcall: str = ""):
        self.calls10 = {}
        self.calls12 = {}
        self.calls22 = {}
        self.mycall = ""
        self.dxcall = ""
        self.set_station(mycall, dxcall)

    def set_station(self, mycall: str = "", dxcall: str = "") -> None:
        self.mycall = mycall.strip().upper()
        self.dxcall = dxcall.strip().upper()
        if len(self.mycall) > 2:
            self.save(self.mycall)

    def save(self, call: str) -> Tuple[int, int, int]:
        """Add a callsign to the hash tables; returns (n10, n12, n22)."""
        cw = call.strip().upper()
        if cw.startswith("<"):
            cw = cw[1:]
        cw = cw.rstrip(">").strip()
        if len(cw) < 3 or cw == "...":
            return -1, -1, -1
        n10 = ihashcall(cw, 10)
        n12 = ihashcall(cw, 12)
        n22 = ihashcall(cw, 22)
        self.calls10[n10] = cw
        self.calls12[n12] = cw
        self.calls22[n22] = cw
        return n10, n12, n22

    def lookup10(self, n10: int) -> str:
        call = self.calls10.get(n10)
        return f"<{call}>" if call else "<...>"

    def lookup12(self, n12: int) -> str:
        call = self.calls12.get(n12)
        return f"<{call}>" if call else "<...>"

    def lookup22(self, n22: int) -> str:
        call = self.calls22.get(n22)
        return f"<{call}>" if call else "<...>"


_default_hashes = HashTable()


def default_hash_table() -> HashTable:
    return _default_hashes


# ---------------------------------------------------------------------------
# Callsign validation (ports of chkcall.f90 and the callok function)
# ---------------------------------------------------------------------------

def chkcall(w: str) -> Tuple[str, bool]:
    """Check for a valid standard or compound callsign.

    Returns (basecall, ok).  Port of lib/chkcall.f90.
    """
    w = w.strip().upper()
    bc = w[:6]
    n1 = len(w)
    if n1 > 11 or any(ch in w for ch in ".+-?"):
        return bc, False
    if n1 > 6 and "/" not in w:
        return bc, False
    i0 = w.find("/") + 1  # 1-based like Fortran (0 if absent)
    if i0 and max(i0 - 1, n1 - i0) > 6:
        return bc, False
    if 2 <= i0 <= n1 - 1:
        bc = w[i0:] if i0 - 1 <= n1 - i0 else w[: i0 - 1]
    if len(bc) > 6:
        return bc, False
    if not (bc[:1].isalpha() or bc[1:2].isalpha()):
        return bc, False
    if bc.startswith("Q") and bc[:5] != "QU1RK":
        return bc, False
    if len(bc) >= 2 and bc[1].isdigit():
        i1 = 2
    elif len(bc) >= 3 and bc[2].isdigit():
        i1 = 3
    else:
        return bc, False
    if i1 == len(bc):
        return bc, False
    sfx = bc[i1:]
    if not sfx.isalpha() or not (1 <= len(sfx) <= 3):
        return bc, False
    return bc, True


def _callok(w: str) -> bool:
    """Validity check applied to callsigns decoded by unpack28 (packjt77 callok)."""
    w = w.strip()
    if len(w) < 3 or w.startswith("Q"):
        return False
    i0 = 0
    for i in range(len(w), 0, -1):
        if w[i - 1].isdigit():
            i0 = i
            break
    if i0 not in (2, 3):
        return False
    pfx, sfx = w[: i0 - 1], w[i0:][:3]
    if not any(ch.isalpha() for ch in pfx):
        return False
    if not all(ch.isalpha() for ch in sfx):
        return False
    return True


# ---------------------------------------------------------------------------
# 28-bit callsign encoding
# ---------------------------------------------------------------------------

def pack28(c13: str, hashes: Optional[HashTable] = None) -> int:
    """Pack a token, hashed callsign, or standard callsign into 28 bits."""
    hashes = hashes or _default_hashes
    c13 = c13.strip().upper()
    if c13 == "DE":
        return 0
    if c13 == "QRZ":
        return 1
    if c13 == "CQ":
        return 2
    if c13.startswith("CQ_") and 4 <= len(c13) <= 7:
        tail = c13[3:]
        if tail.isdigit() and len(tail) == 3:
            return 3 + int(tail)
        if tail.isalpha() and 1 <= len(tail) <= 4:
            m = 0
            for ch in tail.rjust(4):
                m = 27 * m + (ord(ch) - ord("A") + 1 if ch.isalpha() else 0)
            return 3 + 1000 + m
    if c13.startswith("<"):
        hashes.save(c13)
        bare = c13[1:].rstrip(">")
        return NTOKENS + ihashcall(bare, 22)

    # Standard callsign check (same digit/letter analysis as pack28 in Fortran)
    n = len(c13)
    iarea = 1
    for i in range(n, 1, -1):
        if c13[i - 1].isdigit():
            iarea = i
            break
    npdig = sum(1 for ch in c13[: iarea - 1] if ch.isdigit())
    nplet = sum(1 for ch in c13[: iarea - 1] if ch.isalpha())
    nslet = sum(1 for ch in c13[iarea:] if ch.isalpha())
    if iarea < 2 or iarea > 3 or nplet == 0 or npdig >= iarea - 1 or nslet > 3:
        # Nonstandard callsign: use its 22-bit hash
        hashes.save(c13)
        return NTOKENS + ihashcall(c13, 22)

    hashes.save(c13)
    callsign = (" " + c13[:5]) if iarea == 2 else c13[:6]
    callsign = callsign.ljust(6)
    i1 = A1.find(callsign[0])
    i2 = A2.find(callsign[1])
    i3 = A3.find(callsign[2])
    i4 = A4.find(callsign[3])
    i5 = A4.find(callsign[4])
    i6 = A4.find(callsign[5])
    n28 = (
        36 * 10 * 27 * 27 * 27 * i1
        + 10 * 27 * 27 * 27 * i2
        + 27 * 27 * 27 * i3
        + 27 * 27 * i4
        + 27 * i5
        + i6
    )
    return (n28 + NTOKENS + MAX22) & ((1 << 28) - 1)


def unpack28(n28: int, hashes: Optional[HashTable] = None) -> Tuple[str, bool]:
    """Unpack a 28-bit field to a callsign or token. Returns (call, success)."""
    hashes = hashes or _default_hashes
    if n28 < NTOKENS:
        if n28 == 0:
            return "DE", True
        if n28 == 1:
            return "QRZ", True
        if n28 == 2:
            return "CQ", True
        if n28 <= 1002:
            return f"CQ_{n28 - 3:03d}", True
        if n28 <= 532443:
            n = n28 - 1003
            chars = []
            for div in (27 ** 3, 27 ** 2, 27, 1):
                j, n = divmod(n, div)
                chars.append(A4[j])
            return "CQ_" + "".join(chars).strip(), True
        return "", False
    n28 -= NTOKENS
    if n28 < MAX22:
        return hashes.lookup22(n28), True

    n = n28 - MAX22
    i1, n = divmod(n, 36 * 10 * 27 * 27 * 27)
    i2, n = divmod(n, 10 * 27 * 27 * 27)
    i3, n = divmod(n, 27 * 27 * 27)
    i4, n = divmod(n, 27 * 27)
    i5, i6 = divmod(n, 27)
    try:
        call = (A1[i1] + A2[i2] + A3[i3] + A4[i4] + A4[i5] + A4[i6]).strip()
    except IndexError:
        return "QU1RK", False
    if not _callok(call) or " " in call:
        return "QU1RK", False
    return call, True


# ---------------------------------------------------------------------------
# Maidenhead grids
# ---------------------------------------------------------------------------

def _grid4_to_n(grid4: str) -> int:
    return (
        (ord(grid4[0]) - ord("A")) * 18 * 10 * 10
        + (ord(grid4[1]) - ord("A")) * 10 * 10
        + (ord(grid4[2]) - ord("0")) * 10
        + (ord(grid4[3]) - ord("0"))
    )


def _n_to_grid4(n: int) -> Optional[str]:
    j1, n = divmod(n, 18 * 10 * 10)
    j2, n = divmod(n, 10 * 10)
    j3, j4 = divmod(n, 10)
    if not (0 <= j1 <= 17 and 0 <= j2 <= 17):
        return None
    return chr(j1 + ord("A")) + chr(j2 + ord("A")) + chr(j3 + ord("0")) + chr(j4 + ord("0"))


def _n_to_grid6(n: int) -> Optional[str]:
    j1, n = divmod(n, 18 * 10 * 10 * 24 * 24)
    j2, n = divmod(n, 10 * 10 * 24 * 24)
    j3, n = divmod(n, 10 * 24 * 24)
    j4, n = divmod(n, 24 * 24)
    j5, j6 = divmod(n, 24)
    if not (0 <= j1 <= 17 and 0 <= j2 <= 17 and 0 <= j5 <= 23 and 0 <= j6 <= 23):
        return None
    return (chr(j1 + ord("A")) + chr(j2 + ord("A")) + chr(j3 + ord("0"))
            + chr(j4 + ord("0")) + chr(j5 + ord("A")) + chr(j6 + ord("A")))


def _n_to_grid46(n: int) -> Optional[str]:
    """4- or 6-char grid with base-25 subsquares (WSPR type 3)."""
    j1, n = divmod(n, 18 * 10 * 10 * 25 * 25)
    j2, n = divmod(n, 10 * 10 * 25 * 25)
    j3, n = divmod(n, 10 * 25 * 25)
    j4, n = divmod(n, 25 * 25)
    j5, j6 = divmod(n, 25)
    if not (0 <= j1 <= 17 and 0 <= j2 <= 17):
        return None
    grid = (chr(j1 + ord("A")) + chr(j2 + ord("A"))
            + chr(j3 + ord("0")) + chr(j4 + ord("0")))
    if not (j5 == 24 and j6 == 24):
        grid += chr(j5 + ord("A")) + chr(j6 + ord("A"))
    return grid


_GRID4_RE = re.compile(r"^[A-R][A-R][0-9][0-9]$")
_GRID6_RE = re.compile(r"^[A-R][A-R][0-9][0-9][A-X][A-X]$")


def _is_grid4(w: str) -> bool:
    return bool(_GRID4_RE.match(w))


def _is_grid6(w: str) -> bool:
    return bool(_GRID6_RE.match(w))


# ---------------------------------------------------------------------------
# Free text (type 0.0)
# ---------------------------------------------------------------------------

def packtext77(text: str) -> str:
    """Pack up to 13 characters of free text into 71 bits (as a '0'/'1' string)."""
    w = text.upper()[:13].rjust(13)
    n = 0
    for ch in w:
        j = TEXT_ALPHABET.find(ch)
        if j < 0:
            j = 0
        n = 42 * n + j
    return format(n, "071b")


def unpacktext77(c71: str) -> str:
    n = int(c71, 2)
    chars = []
    for _ in range(13):
        n, r = divmod(n, 42)
        chars.append(TEXT_ALPHABET[r])
    return "".join(reversed(chars)).strip()


# ---------------------------------------------------------------------------
# split77
# ---------------------------------------------------------------------------

def split77(msg: str) -> Tuple[list, str]:
    """Uppercase, collapse blanks, split into words; fold 'CQ XX' into 'CQ_XX'."""
    msg = " ".join(msg.upper().split())
    words = [w[:13] for w in msg.split()]
    if len(words) >= 3 and words[0] == "CQ":
        _, ok = chkcall(words[2])
        if ok:
            words = [f"CQ_{words[1][:10]}"] + words[2:]
            msg = " ".join(words)
    return words, msg


# ---------------------------------------------------------------------------
# pack77 sub-packers.  Each returns a 77-char '0'/'1' string or None.
# ---------------------------------------------------------------------------

def _pack77_01(words, hashes) -> Optional[str]:
    # DXpedition:  K1ABC RR73; W9XYZ <KH1/KH7Z> -11
    if len(words) != 5 or words[1] != "RR73;":
        return None
    if not (words[3].startswith("<") and ">" in words[3]):
        return None
    try:
        n = int(words[4])
    except ValueError:
        return None
    n5 = min(31, max(0, (n + 30) // 2))
    _, ok1 = chkcall(words[0])
    _, ok2 = chkcall(words[2])
    if not (ok1 and ok2):
        return None
    n28a = pack28(words[0], hashes)
    n28b = pack28(words[2], hashes)
    hashes.save(words[3])
    n10 = ihashcall(words[3][1:].rstrip(">"), 10)
    return f"{n28a:028b}{n28b:028b}{n10:010b}{n5:05b}{1:03b}{0:03b}"


def _pack77_03(words, hashes) -> Optional[str]:
    # ARRL Field Day:  WA9XYZ KA1ABC [R] 16A EMA
    if len(words) < 4 or len(words) > 5:
        return None
    _, ok1 = chkcall(words[0])
    _, ok2 = chkcall(words[1])
    if not (ok1 and ok2):
        return None
    sec = words[-1]
    if sec not in ARRL_SECTIONS:
        return None
    isec = ARRL_SECTIONS.index(sec) + 1
    if len(words) == 5 and words[2] != "R":
        return None
    exch = words[-2]
    if len(exch) < 2 or not exch[:-1].isdigit():
        return None
    ntx = int(exch[:-1])
    if ntx < 1 or ntx > 32:
        return None
    nclass = ord(exch[-1]) - ord("A")
    if nclass < 0 or nclass > 7:
        return None
    n3 = 3
    intx = ntx - 1
    if intx >= 16:
        n3 = 4
        intx = ntx - 17
    n28a = pack28(words[0], hashes)
    n28b = pack28(words[1], hashes)
    ir = 1 if len(words) == 5 else 0
    return f"{n28a:028b}{n28b:028b}{ir:01b}{intx:04b}{nclass:03b}{isec:07b}{n3:03b}{0:03b}"


def _pack77_06(words, hashes, i3_hint, n3_hint) -> Optional[str]:
    # WSPR-style messages (type 0.6)
    nwords = len(words)
    if nwords == 3:
        w1, w2, w3 = words
        if (3 <= len(w1) <= 6 and _is_grid4(w2) and w3.isdigit()
                and 1 <= len(w3) <= 2):
            n28 = pack28(w1, hashes)
            igrid4 = _grid4_to_n(w2)
            idbm = min(60, max(0, int(w3)))
            idbm = round(0.3 * idbm)
            return f"{n28:028b}{igrid4:015b}{idbm:05b}00{0:021b}{6:03b}{0:03b}"
    if nwords == 2:
        w1, w2 = words
        m1 = len(w1)
        if 5 <= m1 <= 10 and len(w2) <= 2 and "/" in w1 and w2.isdigit():
            i1 = w1.find("/") + 1
            if i1 < 2 or i1 == m1:
                return None
            if i1 == m1 - 3 and not w1[-1].isdigit():
                return None
            bcall, ok = chkcall(w1)
            if not ok:
                return None
            if i1 <= 4:
                # prefix
                pfx = w1[: i1 - 1]
                npfx = 0
                for ch in pfx:
                    npfx = 36 * npfx + A2.find(ch)
            else:
                sfx = w1[i1:]
                ns = len(sfx)
                if ns == 1:
                    npfx = A2.find(sfx)
                elif ns == 2:
                    npfx = 36 * A2.find(sfx[0]) + A2.find(sfx[1])
                elif ns == 3:
                    if not sfx[2].isdigit():
                        return None
                    npfx = 360 * A2.find(sfx[0]) + 10 * A2.find(sfx[1]) + A2.find(sfx[2])
                else:
                    return None
                npfx += 46656
            n28 = pack28(bcall, hashes)
            idbm = min(60, max(0, int(w2)))
            idbm = round(0.3 * idbm)
            return f"{n28:028b}{npfx:016b}{idbm:05b}1{0:021b}{6:03b}{0:03b}"
        if (i3_hint == 0 and n3_hint == 6 and 5 <= m1 <= 12 and len(w2) <= 6
                and w1.startswith("<") and ">" in w1):
            g = w2
            if not (_is_grid4(g) or _is_grid6(g)):
                return None
            n28 = pack28(w1, hashes)
            n22 = n28 - NTOKENS
            k = (
                (ord(g[0]) - ord("A")) * 18 * 10 * 10 * 25 * 25
                + (ord(g[1]) - ord("A")) * 10 * 10 * 25 * 25
                + (ord(g[2]) - ord("0")) * 10 * 25 * 25
                + (ord(g[3]) - ord("0")) * 25 * 25
            )
            if len(g) == 4:
                igrid6 = k + 24 * 25 + 24
            else:
                igrid6 = k + (ord(g[4]) - ord("A")) * 25 + (ord(g[5]) - ord("A"))
            return f"{n22:022b}{igrid6:025b}{2:03b}{0:021b}{6:03b}{0:03b}"
    return None


def _pack77_1(words, hashes) -> Optional[str]:
    # Standard message (types 1 and 2)
    nwords = len(words)
    if nwords < 2 or nwords > 4:
        return None
    w1, w2 = words[0], words[1]
    _, ok1 = chkcall(w1)
    _, ok2 = chkcall(w2)
    bcall_1 = chkcall(w1)[0]
    bcall_2 = chkcall(w2)[0]
    if w1 in ("DE", "QRZ") or w1.startswith("CQ_") or w1 == "CQ":
        ok1 = True
    if w1.startswith("<") and w1.find(">") >= 4:
        ok1 = True
    if w2.startswith("<") and w2.find(">") >= 4:
        ok2 = True
    if not (ok1 and ok2):
        return None
    if w1.startswith("<") and "/" in w2:
        return None
    if w2.startswith("<") and "/" in w1:
        return None
    if nwords == 2 and (not ok2 or w2.find("/") >= 1):
        return None

    ir = 0
    irpt = 0
    if nwords >= 3:
        last = words[-1]
        c1, c2 = last[:1], last[:2]
        if not _is_grid4(last[:4]) and c1 not in "+-" \
                and c2 not in ("R+", "R-") and last not in ("RRR", "RR73", "73"):
            return None
        if c1 in "+-":
            try:
                irpt = int(last)
            except ValueError:
                return None
            if -50 <= irpt <= -31:
                irpt += 101
            irpt += 35
        elif c2 in ("R+", "R-"):
            ir = 1
            try:
                irpt = int(last[1:])
            except ValueError:
                return None
            if -50 <= irpt <= -31:
                irpt += 101
            irpt += 35
        elif last == "RRR":
            irpt = 2
        elif last == "RR73":
            irpt = 3
        elif last == "73":
            irpt = 4

    if not (nwords in (2, 3) or (nwords == 4 and words[2] == "R")):
        return None
    i3 = 2 if (w1.endswith("/P") or w2.endswith("/P")) else 1

    c13 = bcall_1
    if w1.startswith("CQ_") or w1.startswith("<"):
        c13 = w1
    elif w1 in ("CQ", "DE", "QRZ"):
        c13 = w1
    n28a = pack28(c13, hashes)
    c13 = bcall_2
    if w2.startswith("<"):
        c13 = w2
    n28b = pack28(c13, hashes)
    ipa = 1 if (w1.endswith("/P") or w1.endswith("/R")) else 0
    ipb = 1 if (w2.endswith("/P") or w2.endswith("/R")) else 0

    if nwords >= 3 and _is_grid4(words[-1][:4]):
        ir = 1 if (nwords == 4 and words[2] == "R") else 0
        igrid4 = _grid4_to_n(words[-1])
    else:
        igrid4 = MAXGRID4 + irpt
    if nwords == 2:
        ir = 0
        igrid4 = MAXGRID4 + 1
    return f"{n28a:028b}{ipa:01b}{n28b:028b}{ipb:01b}{ir:01b}{igrid4:015b}{i3:03b}"


def _pack77_3(words, hashes) -> Optional[str]:
    # ARRL RTTY roundup:  [TU;] W9XYZ K1ABC [R] 579 MA
    nwords = len(words)
    if words[0].startswith("<") and len(words) > 1 and words[1].startswith("<"):
        return None
    if nwords not in (4, 5, 6):
        return None
    itu = 1 if words[0] == "TU;" else 0
    if itu + 4 > nwords:
        return None
    _, ok1 = chkcall(words[itu])
    _, ok2 = chkcall(words[itu + 1])
    if not (ok1 and ok2):
        return None
    crpt = words[-2][:3]
    if "-" in crpt or "+" in crpt:
        return None
    nserial = 0
    if len(crpt) == 3 and crpt[0] == "5" and "2" <= crpt[1] <= "9" and crpt[2] == "9":
        if words[-1].isdigit():
            nserial = int(words[-1])
    imult = -1
    if words[-1] in RTTY_MULTS:
        imult = RTTY_MULTS.index(words[-1]) + 1
    nexch = 0
    if nserial > 0:
        nexch = nserial
    if imult > 0:
        nexch = 8000 + imult
    if imult <= 0 and nserial <= 0:
        return None
    ir = 1 if words[2 + itu] == "R" else 0
    rpt_word = words[2 + itu + ir]
    if not rpt_word.isdigit():
        return None
    irpt = (int(rpt_word) - 509) // 10 - 2
    irpt = min(7, max(0, irpt))
    n28a = pack28(words[itu], hashes)
    n28b = pack28(words[itu + 1], hashes)
    return f"{itu:01b}{n28a:028b}{n28b:028b}{ir:01b}{irpt:03b}{nexch:013b}{3:03b}"


def _pack77_4(words, hashes) -> Optional[str]:
    # Type 4: one nonstandard call and one hashed call
    nwords = len(words)
    if nwords not in (2, 3):
        return None
    w1, w2 = words[0], words[1]
    call_1 = w1[1:-1] if w1.startswith("<") and w1.endswith(">") else w1
    call_2 = w2[1:-1] if w2.startswith("<") and w2.endswith(">") else w2
    bcall_1, ok1 = chkcall(call_1)
    bcall_2, ok2 = chkcall(call_2)
    if call_1 == bcall_1 and call_2 == bcall_2 and ok1 and ok2:
        return None  # two standard calls: not type 4
    if not (w1 == "CQ" or (ok1 and ok2)):
        return None
    if w1 == "CQ" and len(w2) <= 4:
        return None
    icq = 1 if w1 == "CQ" else 0

    if icq:
        n12 = 0
        c11 = call_2[:11].rjust(11)
        hashes.save(w2)
    elif w1.startswith("<"):
        n10, n12, n22 = hashes.save(w1)
        c11 = call_2[:11].rjust(11)
    elif w2.startswith("<"):
        n10, n12, n22 = hashes.save(w2)
        c11 = call_1[:11].rjust(11)
    else:
        return None
    iflip = 1 if (not icq and w2.startswith("<")) else 0
    n58 = 0
    for ch in c11:
        n58 = n58 * 38 + max(0, HASH_ALPHABET.find(ch))
    nrpt = 0
    if nwords == 3:
        if words[2] == "RRR":
            nrpt = 1
        elif words[2] == "RR73":
            nrpt = 2
        elif words[2] == "73":
            nrpt = 3
    if icq:
        iflip = 0
        nrpt = 0
    return f"{n12:012b}{n58:058b}{iflip:01b}{nrpt:02b}{icq:01b}{4:03b}"


def _pack77_5(words, hashes) -> Optional[str]:
    # Type 5: EU VHF contest, two hashed calls + report/serial + grid6
    nwords = len(words)
    if nwords not in (4, 5):
        return None
    if not (words[0].startswith("<") and words[1].startswith("<")):
        return None
    if not words[-2].isdigit():
        return None
    nx = int(words[-2])
    if nx < 520001 or nx > 594095:
        return None
    if not _is_grid6(words[-1][:6]):
        return None
    hashes.save(words[0])
    n12 = ihashcall(words[0][1:].rstrip(">"), 12)
    hashes.save(words[1])
    n22 = ihashcall(words[1][1:].rstrip(">"), 22)
    ir = 1 if (nwords == 5 and words[2] == "R") else 0
    irpt = nx // 10000 - 52
    iserial = min(2047, nx % 10000)
    g = words[-1]
    igrid6 = (
        (ord(g[0]) - ord("A")) * 18 * 10 * 10 * 24 * 24
        + (ord(g[1]) - ord("A")) * 10 * 10 * 24 * 24
        + (ord(g[2]) - ord("0")) * 10 * 24 * 24
        + (ord(g[3]) - ord("0")) * 24 * 24
        + (ord(g[4]) - ord("A")) * 24
        + (ord(g[5]) - ord("A"))
    )
    return f"{n12:012b}{n22:022b}{ir:01b}{irpt:03b}{iserial:011b}{igrid6:025b}{5:03b}"


def _pack_telemetry(msg: str) -> Optional[str]:
    word = msg.split()[0] if msg.split() else ""
    if not word or len(word) > 18:
        return None
    if not all(ch in "0123456789ABCDEFabcdef" for ch in word):
        return None
    h = word.upper().rjust(18, "0")
    ntel = (int(h[0:6], 16), int(h[6:12], 16), int(h[12:18], 16))
    if ntel[0] >= 2 ** 23:
        return None
    return f"{ntel[0]:023b}{ntel[1]:024b}{ntel[2]:024b}{5:03b}{0:03b}"


def pack77(msg: str, hashes: Optional[HashTable] = None,
           i3_hint: int = -1, n3_hint: int = -1) -> np.ndarray:
    """Pack a message into 77 bits (numpy uint8 array). Port of pack77()."""
    hashes = hashes or _default_hashes
    msg = " ".join(msg.upper().split())

    c77 = None
    if i3_hint == 0 and n3_hint == 5:
        c77 = _pack_telemetry(msg)
    else:
        words, msg = split77(msg)
        if not words:
            c77 = None
        elif msg.startswith(("CQ ", "DE ", "QRZ ")) or words[0] in ("CQ", "DE", "QRZ") \
                or words[0].startswith("CQ_"):
            c77 = _std_chain(words, hashes, i3_hint, n3_hint)
        else:
            c77 = _pack77_01(words, hashes)
            if c77 is None:
                c77 = _pack77_03(words, hashes)
            if c77 is None and len(words) < 2:
                c77 = _pack_telemetry(msg)
            if c77 is None:
                c77 = _std_chain(words, hashes, i3_hint, n3_hint)

    if c77 is None:
        # default: free text
        c77 = packtext77(msg[:13]) + f"{0:03b}{0:03b}"
    return np.frombuffer(c77.encode(), dtype=np.uint8) - ord("0")


def _std_chain(words, hashes, i3_hint, n3_hint) -> Optional[str]:
    c77 = _pack77_06(words, hashes, i3_hint, n3_hint)
    if c77 is None:
        c77 = _pack77_1(words, hashes)
    if c77 is None:
        c77 = _pack77_3(words, hashes)
    if c77 is None:
        c77 = _pack77_4(words, hashes)
    if c77 is None:
        c77 = _pack77_5(words, hashes)
    return c77


# ---------------------------------------------------------------------------
# unpack77
# ---------------------------------------------------------------------------

def _bits_to_str(bits77) -> str:
    a = np.asarray(bits77).astype(np.uint8)
    return "".join("1" if b else "0" for b in a)


def unpack77(bits77, nrx: int = 1,
             hashes: Optional[HashTable] = None) -> Tuple[str, bool]:
    """Unpack 77 bits into a message string. Returns (message, success).

    nrx=1 when unpacking a received message, 0 for a to-be-transmitted one
    (affects how hashed callsigns are resolved against mycall/dxcall).
    """
    hashes = hashes or _default_hashes
    c77 = _bits_to_str(bits77)
    if len(c77) != 77:
        return "failed unpack", False
    n3 = int(c77[71:74], 2)
    i3 = int(c77[74:77], 2)
    success = True
    msg = ""

    hashmy10 = hashmy12 = hashmy22 = -2
    hashdx10 = -2
    if len(hashes.mycall) > 2:
        hashmy10 = ihashcall(hashes.mycall, 10)
        hashmy12 = ihashcall(hashes.mycall, 12)
        hashmy22 = ihashcall(hashes.mycall, 22)
    if len(hashes.dxcall) > 2:
        hashdx10 = ihashcall(hashes.dxcall, 10)

    if i3 == 0 and n3 == 0:
        msg = unpacktext77(c77[:71])
        if not msg:
            return msg, False

    elif i3 == 0 and n3 == 1:
        n28a = int(c77[0:28], 2)
        n28b = int(c77[28:56], 2)
        n10 = int(c77[56:66], 2)
        n5 = int(c77[66:71], 2)
        irpt = 2 * n5 - 30
        crpt = f"{irpt:+03d}"
        call_1, ok = unpack28(n28a, hashes)
        if not ok or n28a <= 2:
            success = False
        call_2, ok = unpack28(n28b, hashes)
        if not ok or n28b <= 2:
            success = False
        call_3 = hashes.lookup10(n10)
        if nrx == 1 and hashdx10 == n10:
            call_3 = f"<{hashes.dxcall}>"
        if nrx == 0 and hashmy10 == n10:
            call_3 = f"<{hashes.mycall}>"
        msg = f"{call_1} RR73; {call_2} {call_3} {crpt}"

    elif i3 == 0 and n3 == 2:
        success = False

    elif i3 == 0 and n3 in (3, 4):
        n28a = int(c77[0:28], 2)
        n28b = int(c77[28:56], 2)
        ir = int(c77[56], 2)
        intx = int(c77[57:61], 2)
        nclass = int(c77[61:64], 2)
        isec = int(c77[64:71], 2)
        if isec > len(ARRL_SECTIONS) or isec < 1:
            return "failed unpack", False
        call_1, ok = unpack28(n28a, hashes)
        if not ok or n28a <= 2:
            success = False
        call_2, ok = unpack28(n28b, hashes)
        if not ok or n28b <= 2:
            success = False
        ntx = intx + 1 + (16 if n3 == 4 else 0)
        cntx = f"{ntx}{chr(ord('A') + nclass)}"
        rr = "R " if ir else ""
        msg = f"{call_1} {call_2} {rr}{cntx} {ARRL_SECTIONS[isec - 1]}"

    elif i3 == 0 and n3 == 5:
        ntel = (int(c77[0:23], 2), int(c77[23:47], 2), int(c77[47:71], 2))
        msg = f"{ntel[0]:06X}{ntel[1]:06X}{ntel[2]:06X}".lstrip("0")
        if not msg:
            msg = "0"

    elif i3 == 0 and n3 == 6:
        j48, j49, j50 = int(c77[47]), int(c77[48]), int(c77[49])
        if j50 == 1:
            itype = 2
        elif j49 == 0:
            itype = 1
        elif j48 == 0:
            itype = 3
        else:
            return "failed unpack", False
        if itype == 1:
            n28 = int(c77[0:28], 2)
            igrid4 = int(c77[28:43], 2)
            idbm = round(int(c77[43:48], 2) * 10.0 / 3.0)
            if idbm < 0 or idbm > 60:
                success = False
            call_1, ok = unpack28(n28, hashes)
            if not ok:
                success = False
            grid4 = _n_to_grid4(igrid4)
            if grid4 is None:
                success = False
                grid4 = "AA00"
            msg = f"{call_1} {grid4} {idbm}"
            if success:
                hashes.save(call_1)
        elif itype == 2:
            n28 = int(c77[0:28], 2)
            npfx = int(c77[28:44], 2)
            idbm = round(int(c77[44:49], 2) * 10.0 / 3.0)
            if idbm < 0 or idbm > 60:
                success = False
            call_1, ok = unpack28(n28, hashes)
            if not ok:
                success = False
            if npfx < 46656:
                chars = []
                n = npfx
                for _ in range(3):
                    chars.append(A2[n % 36])
                    n //= 36
                    if n == 0:
                        break
                cpfx = "".join(reversed(chars))
                msg = f"{cpfx}/{call_1} {idbm}"
                hashes.save(f"{cpfx}/{call_1}")
            else:
                n = npfx - 46656
                if n <= 35:
                    cpfx = A2[n]
                elif n <= 1295:
                    cpfx = A2[n // 36] + A2[n % 36]
                elif n <= 12959:
                    cpfx = A2[n // 360] + A2[(n // 10) % 36] + A2[n % 10]
                else:
                    return "failed unpack", False
                msg = f"{call_1}/{cpfx} {idbm}"
                hashes.save(f"{call_1}/{cpfx}")
        else:
            n22 = int(c77[0:22], 2)
            igrid6 = int(c77[22:47], 2)
            call_1, ok = unpack28(n22 + NTOKENS, hashes)
            if not ok:
                success = False
            grid = _n_to_grid46(igrid6)
            if grid is None:
                success = False
                grid = ""
            msg = f"{call_1} {grid}".strip()

    elif i3 == 0 and n3 > 6:
        success = False

    elif i3 in (1, 2):
        n28a = int(c77[0:28], 2)
        ipa = int(c77[28], 2)
        n28b = int(c77[29:57], 2)
        ipb = int(c77[57], 2)
        ir = int(c77[58], 2)
        igrid4 = int(c77[59:74], 2)
        call_1, ok = unpack28(n28a, hashes)
        if nrx == 1 and hashmy22 == n28a - NTOKENS:
            call_1 = f"<{hashes.mycall}>"
            ok = True
        if not ok:
            success = False
        call_2, ok = unpack28(n28b, hashes)
        if not ok:
            success = False
        if call_1.startswith("CQ_"):
            call_1 = "CQ " + call_1[3:]
        if "<" not in call_1 and len(call_1) >= 3 and ipa:
            call_1 += "/R" if i3 == 1 else "/P"
        if "<" not in call_2 and len(call_2) >= 3:
            if ipb:
                call_2 += "/R" if i3 == 1 else "/P"
            hashes.save(call_2)
        if igrid4 <= MAXGRID4:
            grid4 = _n_to_grid4(igrid4)
            if grid4 is None:
                success = False
                grid4 = "AA00"
            msg = f"{call_1} {call_2} R {grid4}" if ir else f"{call_1} {call_2} {grid4}"
            if msg.startswith("CQ ") and ir:
                success = False
        else:
            irpt = igrid4 - MAXGRID4
            if irpt == 1:
                msg = f"{call_1} {call_2}"
            elif irpt == 2:
                msg = f"{call_1} {call_2} RRR"
            elif irpt == 3:
                msg = f"{call_1} {call_2} RR73"
            elif irpt == 4:
                msg = f"{call_1} {call_2} 73"
            else:
                isnr = irpt - 35
                if isnr > 50:
                    isnr -= 101
                crpt = f"{isnr:+03d}"
                msg = f"{call_1} {call_2} R{crpt}" if ir else f"{call_1} {call_2} {crpt}"
            if msg.startswith("CQ ") and irpt >= 2:
                success = False

    elif i3 == 3:
        itu = int(c77[0], 2)
        n28a = int(c77[1:29], 2)
        n28b = int(c77[29:57], 2)
        ir = int(c77[57], 2)
        irpt = int(c77[58:61], 2)
        nexch = int(c77[61:74], 2)
        crpt = f"5{irpt + 2}9"
        call_1, ok = unpack28(n28a, hashes)
        if not ok:
            success = False
        call_2, ok = unpack28(n28b, hashes)
        if not ok:
            success = False
        imult = nexch - 8000 if nexch > 8000 else 0
        nserial = nexch if nexch < 8000 else 0
        prefix = "TU; " if itu else ""
        rr = "R " if ir else ""
        if 1 <= imult <= len(RTTY_MULTS):
            msg = f"{prefix}{call_1} {call_2} {rr}{crpt} {RTTY_MULTS[imult - 1]}"
        elif 1 <= nserial <= 7999:
            msg = f"{prefix}{call_1} {call_2} {rr}{crpt} {nserial:04d}"
        else:
            success = False

    elif i3 == 4:
        n12 = int(c77[0:12], 2)
        n58 = int(c77[12:70], 2)
        iflip = int(c77[70], 2)
        nrpt = int(c77[71:73], 2)
        icq = int(c77[73], 2)
        chars = []
        for _ in range(11):
            n58, r = divmod(n58, 38)
            chars.append(HASH_ALPHABET[r])
        c11 = "".join(reversed(chars)).strip()
        call_3 = hashes.lookup12(n12)
        if iflip == 0:
            call_1 = call_3
            call_2 = c11
            hashes.save(call_2)
            if (nrx == 1 and hashes.dxcall and hashes.mycall
                    and call_2 == hashes.dxcall and n12 == hashmy12):
                call_1 = f"<{hashes.mycall}>"
            if nrx == 1 and "<...>" in call_1 and n12 == hashmy12:
                call_1 = f"<{hashes.mycall}>"
        else:
            call_1 = c11
            call_2 = call_3
            if nrx == 0 and n12 == hashmy12:
                call_2 = f"<{hashes.mycall}>"
        if icq == 0:
            suffix = {0: "", 1: " RRR", 2: " RR73", 3: " 73"}[nrpt]
            msg = f"{call_1} {call_2}{suffix}"
        else:
            msg = f"CQ {call_2}"

    elif i3 == 5:
        n12 = int(c77[0:12], 2)
        n22 = int(c77[12:34], 2)
        ir = int(c77[34], 2)
        irpt = int(c77[35:38], 2)
        iserial = int(c77[38:49], 2)
        igrid6 = int(c77[49:74], 2)
        if igrid6 < 0 or igrid6 > 18662399:
            return "failed unpack", False
        call_1 = hashes.lookup12(n12)
        if n12 == hashmy12:
            call_1 = f"<{hashes.mycall}>"
        call_2 = hashes.lookup22(n22)
        cexch = f"{52 + irpt}{iserial:04d}"
        grid6 = _n_to_grid6(igrid6)
        if grid6 is None:
            return "failed unpack", False
        rr = "R " if ir else ""
        msg = f"{call_1} {call_2} {rr}{cexch} {grid6}"

    else:
        success = False

    if msg.startswith("CQ <"):
        success = False
    return msg.strip(), success


def message_i3n3(bits77) -> Tuple[int, int]:
    """Return (i3, n3) for a 77-bit payload.  n3 is only meaningful (nonzero)
    for i3=0 messages; for other types those bits belong to other fields."""
    c77 = _bits_to_str(bits77)
    i3 = int(c77[74:77], 2)
    n3 = int(c77[71:74], 2) if i3 == 0 else 0
    return i3, n3
