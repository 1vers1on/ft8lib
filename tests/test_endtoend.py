"""End-to-end modem tests: encode -> channel -> decode."""

import numpy as np
import pytest

from ft8lib import (
    FT4,
    FT8,
    decode_ft4,
    decode_ft8,
    encode_ft4,
    encode_ft8,
    ft8_tones_from_bits,
    pack77,
)


def _embed(wave, nmax, start=6000, snr_db=None, rng=None):
    """Place a waveform in a receive period, optionally with AWGN at snr_db
    (SNR referenced to 2500 Hz bandwidth, WSJT-X convention)."""
    audio = np.zeros(nmax)
    if snr_db is not None:
        sigma = np.sqrt(1.2 / 10 ** (snr_db / 10))
        audio = rng.normal(0.0, sigma, nmax)
    audio[start:start + len(wave)] += wave
    return audio


def test_ft8_tone_structure():
    tones = ft8_tones_from_bits(pack77("CQ K1ABC FN42"))
    assert tones.shape == (79,)
    costas = list(FT8.COSTAS)
    assert tones[0:7].tolist() == costas
    assert tones[36:43].tolist() == costas
    assert tones[72:79].tolist() == costas
    assert tones.min() >= 0 and tones.max() <= 7


def test_ft8_waveform_length():
    wave = encode_ft8("CQ K1ABC FN42")
    assert len(wave) == FT8.NZ == 151680
    assert np.max(np.abs(wave)) <= 1.0 + 1e-9


def test_ft4_waveform_length():
    wave = encode_ft4("CQ K1ABC FN42")
    assert len(wave) == FT4.NZ2 == 60480


def test_encode_bad_message_raises():
    with pytest.raises(ValueError):
        encode_ft8("")


def test_ft8_loopback_clean():
    wave = encode_ft8("CQ K1ABC FN42", f0=1500.0)
    audio = _embed(wave, FT8.NMAX)
    results = decode_ft8(audio)
    assert any(r.message == "CQ K1ABC FN42" for r in results)
    r = next(r for r in results if r.message == "CQ K1ABC FN42")
    assert abs(r.freq - 1500.0) < 1.0
    assert abs(r.dt) < 0.05


def test_ft4_loopback_clean():
    wave = encode_ft4("W9XYZ K1ABC RR73", f0=1200.0)
    audio = _embed(wave, FT4.NMAX)
    results = decode_ft4(audio)
    assert any(r.message == "W9XYZ K1ABC RR73" for r in results)
    r = next(r for r in results if r.message == "W9XYZ K1ABC RR73")
    assert abs(r.freq - 1200.0) < 2.0


def test_ft8_decode_in_noise():
    rng = np.random.default_rng(11)
    wave = encode_ft8("K1ABC W9XYZ -15", f0=987.0)
    hits = 0
    for _ in range(3):
        audio = _embed(wave, FT8.NMAX, snr_db=-15, rng=rng)
        results = decode_ft8(audio)
        hits += any(r.message == "K1ABC W9XYZ -15" for r in results)
    assert hits == 3


def test_ft4_decode_in_noise():
    rng = np.random.default_rng(12)
    wave = encode_ft4("K1ABC W9XYZ R-07", f0=1750.0)
    hits = 0
    for _ in range(3):
        audio = _embed(wave, FT4.NMAX, snr_db=-12, rng=rng)
        results = decode_ft4(audio)
        hits += any(r.message == "K1ABC W9XYZ R-07" for r in results)
    assert hits == 3


def test_ft8_snr_estimate_tracks_truth():
    rng = np.random.default_rng(13)
    wave = encode_ft8("CQ K1ABC FN42", f0=1500.0)
    audio = _embed(wave, FT8.NMAX, snr_db=-10, rng=rng)
    results = decode_ft8(audio)
    r = next(r for r in results if r.message == "CQ K1ABC FN42")
    assert -14 <= r.snr <= -6


def test_ft8_multiple_signals():
    rng = np.random.default_rng(14)
    signals = [
        ("CQ K1ABC FN42", 400.0, -8),
        ("W9XYZ K1ABC -07", 1200.0, -12),
        ("JA1XYZ VK2ABC RR73", 2200.0, -15),
    ]
    sigma = 1.0
    audio = rng.normal(0, sigma, FT8.NMAX)
    for msg, f0, snr in signals:
        amp = sigma * np.sqrt(10 ** (snr / 10) * 2 * 2500 / 6000)
        wave = amp * encode_ft8(msg, f0=f0)
        audio[6000:6000 + len(wave)] += wave
    results = decode_ft8(audio)
    decoded = {r.message for r in results}
    assert {msg for msg, _, _ in signals} <= decoded


def test_ft8_time_offset():
    wave = encode_ft8("CQ K1ABC FN42", f0=1000.0)
    audio = _embed(wave, FT8.NMAX, start=12000)  # DT = +0.5 s
    results = decode_ft8(audio)
    r = next(r for r in results if r.message == "CQ K1ABC FN42")
    assert 0.4 < r.dt < 0.6


def test_no_false_decodes_in_pure_noise():
    rng = np.random.default_rng(15)
    assert decode_ft8(rng.normal(0, 1, FT8.NMAX)) == []
    assert decode_ft4(rng.normal(0, 1, FT4.NMAX)) == []
