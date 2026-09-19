from __future__ import annotations

import numpy as np

from eeg_decode.entropy import permutation_entropy, spectral_entropy


def test_spectral_entropy_range_tone_vs_noise():
    fs = 256.0
    t = np.arange(512) / fs
    tone = np.sin(2 * np.pi * 10.0 * t)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(512)
    h_tone = spectral_entropy(tone, fs)
    h_noise = spectral_entropy(noise, fs)
    assert 0.0 <= h_tone <= 1.0
    assert 0.0 <= h_noise <= 1.0
    assert h_tone < h_noise


def test_spectral_entropy_bad_input():
    assert np.isnan(spectral_entropy(np.zeros(4), 256))
    x = np.ones(128)
    x[3] = np.nan
    assert np.isnan(spectral_entropy(x, 256))


def test_permutation_entropy_range():
    rng = np.random.default_rng(1)
    h = permutation_entropy(rng.standard_normal(400), order=3)
    assert 0.0 <= h <= 1.0
    t = np.linspace(0, 4, 400)
    h2 = permutation_entropy(np.sin(2 * np.pi * 3 * t), order=3)
    assert h2 < h
