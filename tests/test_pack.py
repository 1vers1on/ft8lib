"""77-bit message codec tests."""

import numpy as np
import pytest

from ft8lib.pack import HashTable, ihashcall, message_i3n3, pack77, unpack77

ROUNDTRIP_MESSAGES = [
    # standard type 1
    "CQ K1ABC FN42",
    "CQ DX PJ2A GG21",
    "CQ TEST K1ABC FN42",
    "CQ 001 K1ABC FN42",
    "K1ABC W9XYZ EN37",
    "K1ABC W9XYZ -11",
    "K1ABC W9XYZ R-09",
    "K1ABC W9XYZ R+15",
    "W9XYZ K1ABC RRR",
    "W9XYZ K1ABC RR73",
    "K1ABC W9XYZ 73",
    "K1ABC W9XYZ",
    "KA1ABC/R W9XYZ/R R FN42",
    # type 2
    "PA3XYZ/P GM4ABC/P R JO22",
    # type 4 (nonstandard calls)
    "CQ PJ4/K1ABC",
    "<PJ4/K1ABC> W9XYZ",
    "PJ4/K1ABC <W9XYZ> RR73",
    # field day
    "WA9XYZ KA1ABC R 16A EMA",
    "WA9XYZ KA1ABC 32A WPA",
    # RTTY roundup
    "TU; W9XYZ K1ABC R 579 MA",
    "W9XYZ K1ABC 559 0013",
    # telemetry
    "123456789ABCDEF012",
    # free text
    "TNX BOB 73 GL",
    "HELLO WORLD",
    # WSPR style (type 0.6)
    "K1ABC FN42 37",
]


@pytest.fixture
def hashes():
    h = HashTable()
    h.save("PJ4/K1ABC")
    h.save("W9XYZ")
    return h


@pytest.mark.parametrize("msg", ROUNDTRIP_MESSAGES)
def test_roundtrip(msg, hashes):
    bits = pack77(msg, hashes=hashes)
    assert bits.shape == (77,)
    out, ok = unpack77(bits, nrx=0, hashes=hashes)
    assert ok
    assert out == msg


def test_dxpedition_roundtrip(hashes):
    hashes.save("KH1/KH7Z")
    bits = pack77("K1ABC RR73; W9XYZ <KH1/KH7Z> -8", hashes=hashes)
    out, ok = unpack77(bits, nrx=0, hashes=hashes)
    assert ok
    assert out == "K1ABC RR73; W9XYZ <KH1/KH7Z> -08"
    assert message_i3n3(bits) == (0, 1)


def test_i3n3_values(hashes):
    assert message_i3n3(pack77("CQ K1ABC FN42")) == (1, 0)
    assert message_i3n3(pack77("PA3XYZ/P GM4ABC/P R JO22")) == (2, 0)
    assert message_i3n3(pack77("TU; W9XYZ K1ABC R 579 MA")) == (3, 0)
    assert message_i3n3(pack77("CQ PJ4/K1ABC")) == (4, 0)
    assert message_i3n3(pack77("HELLO WORLD")) == (0, 0)
    assert message_i3n3(pack77("123456789ABCDEF012")) == (0, 5)
    assert message_i3n3(pack77("WA9XYZ KA1ABC R 16A EMA")) == (0, 3)


def test_hash_resolution_unknown():
    # Without a primed hash table the hashed call cannot be recovered
    h1 = HashTable()
    bits = pack77("<PJ4/K1ABC> W9XYZ", hashes=h1)
    out, ok = unpack77(bits, nrx=1, hashes=HashTable())
    assert ok
    assert out == "<...> W9XYZ"


def test_hash_resolution_learned():
    # After hearing "CQ PJ4/K1ABC" the hash resolves in later messages
    h = HashTable()
    bits1 = pack77("CQ PJ4/K1ABC", hashes=HashTable())
    out1, ok1 = unpack77(bits1, nrx=1, hashes=h)
    assert ok1 and out1 == "CQ PJ4/K1ABC"
    bits2 = pack77("<PJ4/K1ABC> W9XYZ", hashes=HashTable())
    out2, ok2 = unpack77(bits2, nrx=1, hashes=h)
    assert ok2 and out2 == "<PJ4/K1ABC> W9XYZ"


def test_ihashcall_range():
    for call, m in (("K1ABC", 10), ("PJ4/K1ABC", 12), ("VK9XX", 22)):
        n = ihashcall(call, m)
        assert 0 <= n < 2 ** m


def test_free_text_charset():
    bits = pack77("A+B-C./?")
    out, ok = unpack77(bits)
    assert ok
    assert out == "A+B-C./?"


def test_grid_r_prefix():
    bits = pack77("KA1ABC W9XYZ R FN42")
    out, ok = unpack77(bits)
    assert ok
    assert out == "KA1ABC W9XYZ R FN42"


def test_snr_extremes():
    for rpt in ("+50", "-50", "-30", "+00"):
        msg = f"K1ABC W9XYZ {rpt}"
        out, ok = unpack77(pack77(msg))
        assert ok
        assert out == msg
