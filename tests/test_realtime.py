"""Tests for streaming decode generators and the realtime decoder."""

import numpy as np
import pytest

from ft8lib import (
    FT4,
    FT8,
    RealtimeDecoder,
    decode_ft8,
    decode_ft8_stream,
    decode_realtime,
    encode_ft4,
    encode_ft8,
)


def _period_with(msgs_freqs, nmax, encode, rng=None, snr_db=None):
    audio = np.zeros(nmax)
    if snr_db is not None:
        audio = rng.normal(0.0, 1.0, nmax)
    for msg, f0 in msgs_freqs:
        wave = encode(msg, f0=f0)
        if snr_db is not None:
            wave = wave * np.sqrt(10 ** (snr_db / 10) * 2 * 2500 / 6000)
        audio[6000:6000 + len(wave)] += wave
    return audio


def _feed_chunks(rt, audio, chunk=1200):
    """Feed audio in chunks; return {message: seconds_when_reported}."""
    first = {}
    for i in range(0, len(audio), chunk):
        for r in rt.feed(audio[i:i + chunk]):
            assert r.message not in first, "message reported twice in a cycle"
            first[r.message] = (i + chunk) / 12000
    return first


def test_ft8_stream_matches_batch():
    rng = np.random.default_rng(21)
    audio = _period_with([("CQ K1ABC FN42", 1500.0), ("W9XYZ K1ABC -07", 900.0)],
                         FT8.NMAX, encode_ft8, rng=rng, snr_db=-10)
    batch = {r.message for r in decode_ft8(audio)}
    streamed = [r.message for r in decode_ft8_stream(audio)]
    assert set(streamed) == batch
    assert len(streamed) == len(set(streamed))


def test_ft8_realtime_early_decode():
    audio = _period_with([("CQ K1ABC FN42", 1500.0)], FT8.NMAX, encode_ft8)
    rt = RealtimeDecoder("FT8")
    first = _feed_chunks(rt, audio)
    # reported at the first WSJT-X attempt time, well before the period ends
    assert first == {"CQ K1ABC FN42": pytest.approx(11.8, abs=0.15)}
    assert rt.cycle == 1 and rt.t == 0.0  # rolled over into the next cycle


def test_ft8_realtime_two_cycles():
    c0 = _period_with([("CQ K1ABC FN42", 1500.0)], FT8.NMAX, encode_ft8)
    c1 = _period_with([("K1ABC W9XYZ -07", 900.0)], FT8.NMAX, encode_ft8)
    got = list(decode_realtime([c0, c1], mode="FT8"))
    assert [r.message for r in got] == ["CQ K1ABC FN42", "K1ABC W9XYZ -07"]


def test_ft4_realtime_early_decode():
    period = int(FT4.PERIOD * 12000)
    audio = np.zeros(period)
    wave = encode_ft4("K1ABC W9XYZ R-07", f0=1750.0)
    audio[6000:6000 + len(wave)] += wave
    rt = RealtimeDecoder("FT4")
    first = _feed_chunks(rt, audio, chunk=600)
    assert first == {"K1ABC W9XYZ R-07": pytest.approx(5.6, abs=0.1)}
    assert rt.cycle == 1


def test_realtime_flush_mid_cycle():
    audio = _period_with([("CQ K1ABC FN42", 1500.0)], FT8.NMAX, encode_ft8)
    rt = RealtimeDecoder("FT8", attempt_times=(14.7,))
    assert rt.feed(audio[: 13 * 12000]) == []   # no attempt due yet
    flushed = rt.flush()
    assert [r.message for r in flushed] == ["CQ K1ABC FN42"]
    assert rt.flush() == []                     # already reported this cycle


def test_realtime_rejects_bad_mode():
    with pytest.raises(ValueError):
        RealtimeDecoder("JT65")
