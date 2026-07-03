"""WSPR message source coding: text <-> 50-bit payload.

Ports of the packing side of WSJT-X lib/wsprd/wsprsim_utils.c
(get_wspr_channel_symbols and helpers) and the unpacking side of
lib/wsprd/wsprd_utils.c (unpack50/unpackcall/unpackgrid/unpackpfx/unpk_),
plus the 15-bit callsign hash from lib/wsprd/nhash.c (Bob Jenkins'
lookup3 "hashlittle", seeded with 146).

WSPR message types:

* Type 1: ``K1ABC FN42 33``            call, 4-char grid, power (dBm)
* Type 2: ``PJ4/K1ABC 37``             compound call, power
* Type 3: ``<PJ4/K1ABC> FK52UD 37``    hashed call, 6-char grid, power

Type 3 sends only a 15-bit hash of the callsign, so decoding one requires
having previously decoded the matching type 2 (or type 1) transmission;
a `WsprHashTable` carries that state between decode calls.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_M32 = 0xFFFFFFFF
_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "

# power +nu[power%10] is the nearest legal dBm value (0, 3 or 7 mod 10)
_POWER_NUDGE = (0, -1, 1, 0, -1, 2, 1, 0, -1, 1)


def _rot(x: int, k: int) -> int:
    return ((x << k) | (x >> (32 - k))) & _M32


def nhash(call: str, initval: int = 146) -> int:
    """15-bit hash of a callsign (lookup3 hashlittle, nhash.c)."""
    data = call.encode("ascii")
    length = len(data)
    a = b = c = (0xDEADBEEF + length + initval) & _M32
    k = 0
    while length > 12:
        a = (a + int.from_bytes(data[k:k + 4], "little")) & _M32
        b = (b + int.from_bytes(data[k + 4:k + 8], "little")) & _M32
        c = (c + int.from_bytes(data[k + 8:k + 12], "little")) & _M32
        a = (a - c) & _M32; a ^= _rot(c, 4); c = (c + b) & _M32
        b = (b - a) & _M32; b ^= _rot(a, 6); a = (a + c) & _M32
        c = (c - b) & _M32; c ^= _rot(b, 8); b = (b + a) & _M32
        a = (a - c) & _M32; a ^= _rot(c, 16); c = (c + b) & _M32
        b = (b - a) & _M32; b ^= _rot(a, 19); a = (a + c) & _M32
        c = (c - b) & _M32; c ^= _rot(b, 4); b = (b + a) & _M32
        k += 12
        length -= 12
    if length > 0:
        tail = data[k:k + length] + b"\0" * (12 - length)
        a = (a + int.from_bytes(tail[0:4], "little")) & _M32
        b = (b + int.from_bytes(tail[4:8], "little")) & _M32
        c = (c + int.from_bytes(tail[8:12], "little")) & _M32
        c ^= b; c = (c - _rot(b, 14)) & _M32
        a ^= c; a = (a - _rot(c, 11)) & _M32
        b ^= a; b = (b - _rot(a, 25)) & _M32
        c ^= b; c = (c - _rot(b, 16)) & _M32
        a ^= c; a = (a - _rot(c, 4)) & _M32
        b ^= a; b = (b - _rot(a, 14)) & _M32
        c ^= b; c = (c - _rot(b, 24)) & _M32
    return c & 32767


class WsprHashTable:
    """Callsigns previously seen, indexed by their 15-bit nhash.

    Mirrors the hashtab/loctab files kept by wsprd: type 1 and type 2
    decodes store (callsign, grid), and type 3 decodes look their callsign
    up by hash.  Keep one instance across decode calls to resolve type 3
    messages.
    """

    def __init__(self):
        self.calls = {}
        self.grids = {}

    def save(self, call: str, grid: str = "") -> int:
        call = call.strip().strip("<>").upper()
        if not call:
            return -1
        ihash = nhash(call)
        self.calls[ihash] = call
        if grid:
            self.grids[ihash] = grid
        return ihash

    def lookup(self, ihash: int) -> str:
        call = self.calls.get(ihash)
        return f"<{call}>" if call else "<...>"


def _callsign_code(ch: str) -> int:
    if ch.isdigit():
        return ord(ch) - 48
    if ch == " ":
        return 36
    if "A" <= ch <= "Z":
        return ord(ch) - 55
    raise ValueError(f"bad callsign character {ch!r}")


def _locator_code(ch: str) -> int:
    if ch.isdigit():
        return ord(ch) - 48
    if ch == " ":
        return 36
    if "A" <= ch <= "R":
        return ord(ch) - 65
    raise ValueError(f"bad locator character {ch!r}")


def _pack_call(callsign: str) -> int:
    """28-bit standard-callsign code (pack_call in wsprsim_utils.c)."""
    if len(callsign) > 6:
        raise ValueError(f"callsign too long: {callsign!r}")
    if len(callsign) >= 3 and callsign[2].isdigit():
        call6 = callsign.ljust(6)
    elif len(callsign) >= 2 and callsign[1].isdigit() and len(callsign) <= 5:
        call6 = (" " + callsign).ljust(6)
    else:
        raise ValueError(f"not a standard callsign: {callsign!r}")
    c = [_callsign_code(ch) for ch in call6]
    n = c[0]
    n = n * 36 + c[1]
    n = n * 10 + c[2]
    n = n * 27 + c[3] - 10
    n = n * 27 + c[4] - 10
    n = n * 27 + c[5] - 10
    return n


def _pack_grid4_power(grid4: str, power: int) -> int:
    g = [_locator_code(ch) for ch in grid4]
    m = (179 - 10 * g[0] - g[2]) * 180 + 10 * g[1] + g[3]
    return m * 128 + power + 64


def _pack_prefix(callsign: str) -> Tuple[int, int, int]:
    """Pack a compound call; returns (n, m, nadd) (pack_prefix in C)."""
    i1 = callsign.find("/")
    if i1 < 0:
        raise ValueError("no / in compound callsign")
    if len(callsign) == i1 + 2:
        # single-character suffix
        n = _pack_call(callsign[:i1])
        nc = callsign[i1 + 1]
        if nc.isdigit():
            m = ord(nc) - 48
        elif "A" <= nc <= "Z":
            m = ord(nc) - 65 + 10
        else:
            m = 38
        return n, 60000 - 32768 + m, 1
    if len(callsign) == i1 + 3:
        # two-digit suffix
        n = _pack_call(callsign[:i1])
        m = 10 * (ord(callsign[i1 + 1]) - 48) + (ord(callsign[i1 + 2]) - 48)
        return n, 60000 + 26 + m, 1
    # 1- to 3-character prefix
    pfx, call = callsign[:i1], callsign[i1 + 1:]
    n = _pack_call(call)
    if len(pfx) == 1:
        m = 36 * 37 + 36
    elif len(pfx) == 2:
        m = 36
    else:
        m = 0
    for ch in pfx:
        if ch.isdigit():
            nc = ord(ch) - 48
        elif "A" <= ch <= "Z":
            nc = ord(ch) - 65 + 10
        else:
            nc = 36
        m = 37 * m + nc
    if m > 32768:
        return n, m - 32768, 1
    return n, m, 0


def pack_wspr(message: str) -> np.ndarray:
    """Pack a WSPR message string into the 11-byte data vector.

    The vector holds the 50-bit payload followed by 31 zero tail bits
    (the convolutional encoder flush); port of the packing section of
    get_wspr_channel_symbols in wsprsim_utils.c.
    """
    msg = " ".join(message.upper().split())
    has_slash = "/" in msg
    has_angle = msg.startswith("<")
    parts = msg.replace("<", " ").replace(">", " ").split()

    if has_angle:
        # Type 3: <CALL> GRID6 dBm
        if len(parts) != 3:
            raise ValueError(f"bad type 3 WSPR message: {message!r}")
        callsign, grid, powstr = parts
        power = _clip_power(int(powstr))
        ntype = -(power + 1)
        m = 128 * nhash(callsign) + ntype + 64
        # the 6-char grid is rotated and sent through the callsign packer
        grid6 = (grid[1:] + grid[0]).ljust(6) if len(grid) >= 1 else grid
        n = _pack_call(grid6[:6])
    elif has_slash:
        # Type 2: PREFIX/CALL dBm or CALL/SUFFIX dBm
        if len(parts) != 2:
            raise ValueError(f"bad type 2 WSPR message: {message!r}")
        callsign, powstr = parts
        power = _clip_power(int(powstr))
        n, ng, nadd = _pack_prefix(callsign)
        ntype = power + 1 + nadd
        m = 128 * ng + ntype + 64
    else:
        # Type 1: CALL GRID4 dBm
        if len(parts) != 3 or not 3 <= len(parts[0]) <= 6:
            raise ValueError(f"bad type 1 WSPR message: {message!r}")
        callsign, grid, powstr = parts
        if len(grid) != 4:
            raise ValueError(f"type 1 message needs a 4-character grid: {message!r}")
        n = _pack_call(callsign)
        m = _pack_grid4_power(grid, int(powstr))

    data = np.zeros(11, dtype=np.uint8)
    data[0] = (n >> 20) & 0xFF
    data[1] = (n >> 12) & 0xFF
    data[2] = (n >> 4) & 0xFF
    data[3] = ((n & 0x0F) << 4) | ((m >> 18) & 0x0F)
    data[4] = (m >> 10) & 0xFF
    data[5] = (m >> 2) & 0xFF
    data[6] = (m & 0x03) << 6
    return data


def _clip_power(power: int) -> int:
    power = min(max(power, 0), 60)
    return power + _POWER_NUDGE[power % 10]


def _unpack50(data) -> Tuple[int, int]:
    d = [int(x) & 0xFF for x in data[:7]]
    n1 = (d[0] << 20) | (d[1] << 12) | (d[2] << 4) | (d[3] >> 4)
    n2 = ((d[3] & 0x0F) << 18) | (d[4] << 10) | (d[5] << 2) | (d[6] >> 6)
    return n1, n2


def _unpackcall(ncall: int) -> Optional[str]:
    if ncall >= 262177560:
        return None
    n = ncall
    tmp = [""] * 6
    tmp[5] = _CHARS[n % 27 + 10]; n //= 27
    tmp[4] = _CHARS[n % 27 + 10]; n //= 27
    tmp[3] = _CHARS[n % 27 + 10]; n //= 27
    tmp[2] = _CHARS[n % 10]; n //= 10
    tmp[1] = _CHARS[n % 36]; n //= 36
    if n > 36:
        return None
    tmp[0] = _CHARS[n]
    # strip leading spaces, then truncate at the first remaining space
    call = "".join(tmp).lstrip(" ").ljust(6)
    return call.split(" ")[0]


def _unpackgrid(n2: int) -> Optional[str]:
    ngrid = n2 >> 7
    if ngrid >= 32400:
        return None
    dlat = (ngrid % 180) - 90
    dlong = (ngrid // 180) * 2 - 180 + 2
    if dlong < -180:
        dlong += 360
    if dlong > 180:
        dlong += 360
    nlong = int(60.0 * (180.0 - dlong) / 5.0)
    n1 = nlong // 240
    n2b = (nlong - 240 * n1) // 24
    nlat = int(60.0 * (dlat + 90) / 2.5)
    n3 = nlat // 240
    n4 = (nlat - 240 * n3) // 24
    return _CHARS[10 + n1] + _CHARS[10 + n3] + _CHARS[n2b] + _CHARS[n4]


def _unpackpfx(nprefix: int, call: str) -> Optional[str]:
    if nprefix < 60000:
        # 1- to 3-character prefix
        n = nprefix
        pfx = ""
        for _ in range(3):
            nc = n % 37
            if nc <= 9:
                pfx = chr(nc + 48) + pfx
            elif nc <= 35:
                pfx = chr(nc + 55) + pfx
            else:
                pfx = " " + pfx
            n //= 37
        return pfx.split(" ")[-1] + "/" + call
    # 1- or 2-character suffix
    nc = nprefix - 60000
    if nc <= 9:
        return call + "/" + chr(nc + 48)
    if nc <= 35:
        return call + "/" + chr(nc + 55)
    if nc <= 125:
        return call + "/" + chr((nc - 26) // 10 + 48) + chr((nc - 26) % 10 + 48)
    return None


def unpack_wspr(data, hashes: Optional[WsprHashTable] = None
                ) -> Tuple[Optional[str], Optional[str], bool]:
    """Unpack an 11-byte decoded data vector.

    Returns (message, callsign, ok); port of unpk_ in wsprd_utils.c.  ``ok``
    False corresponds to wsprd's "noprint" flag: the data passed the FEC
    but fails the message sanity checks.  Type 1/2 decodes update
    ``hashes`` so later type 3 messages can resolve the hashed callsign.
    """
    n1, n2 = _unpack50(data)
    callsign = _unpackcall(n1)
    if callsign is None:
        return None, None, False
    grid = _unpackgrid(n2)
    if grid is None:
        return None, None, False
    ntype = (n2 & 127) - 64

    if 0 <= ntype <= 62:
        nu = ntype % 10
        if nu in (0, 3, 7):
            # Type 1: callsign, grid, power
            message = f"{callsign} {grid} {ntype:2d}"
            if hashes is not None:
                hashes.save(callsign, grid)
        else:
            # Type 2: compound callsign, power
            nadd = nu
            if nu > 3:
                nadd = nu - 3
            if nu > 7:
                nadd = nu - 7
            n3 = n2 // 128 + 32768 * (nadd - 1)
            callsign = _unpackpfx(n3, callsign)
            if callsign is None:
                return None, None, False
            ndbm = ntype - nadd
            message = f"{callsign} {ndbm:2d}"
            if ndbm % 10 not in (0, 3, 7):
                return message, callsign, False
            if hashes is not None:
                hashes.save(callsign)
    elif ntype < 0:
        # Type 3: hashed callsign, 6-char grid, power
        ndbm = -(ntype + 1)
        grid6 = (callsign[5] if len(callsign) > 5 else "") + callsign[:5]
        ok = (ndbm % 10 in (0, 3, 7)
              and len(grid6) >= 4
              and grid6[0].isalpha() and grid6[1].isalpha()
              and grid6[2].isdigit() and grid6[3].isdigit())
        ihash = (n2 - ntype - 64) // 128
        callsign = hashes.lookup(ihash) if hashes is not None else "<...>"
        message = f"{callsign} {grid6} {ndbm:2d}"
        if ntype == -64:
            ok = False
        return message, callsign, ok
    else:
        return None, None, False
    return message, callsign, True
