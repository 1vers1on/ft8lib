"""WSPR modem tests."""

import numpy as np

from ft8lib import WSPR, decode_wspr, encode_wspr, wspr_tones_from_message


def _embed(wave, nmax, start=12000):
    # transmissions nominally start 1 s into the even 2-minute cycle
    audio = np.zeros(nmax)
    audio[start:start + len(wave)] += wave
    return audio


def test_wspr_tone_structure():
    tones = wspr_tones_from_message("K1ABC FN42 37")
    assert tones.shape == (162,)
    assert tones.min() >= 0 and tones.max() <= 3
    assert tones[:10].tolist() == [3, 3, 0, 0, 2, 0, 0, 0, 1, 0]


def test_wspr_waveform_length():
    wave = encode_wspr("K1ABC FN42 37")
    assert len(wave) == 162 * 8192 == 1327104
    assert np.max(np.abs(wave)) <= 1.0 + 1e-9


def test_wspr_loopback_clean():
    wave = encode_wspr("K1ABC FN42 37", f0=1500.0)
    audio = _embed(wave, WSPR.NMAX)
    results = decode_wspr(audio)
    assert any(r.message == "K1ABC FN42 37" for r in results)
    r = next(r for r in results if r.message == "K1ABC FN42 37")
    assert abs(r.freq - 1500.0) < 1.0
    assert abs(r.dt) < 0.05
    assert abs(r.drift) < 0.5
