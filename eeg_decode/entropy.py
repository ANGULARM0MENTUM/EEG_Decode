"""Entropy metrics used as the online 'decode' output.

Spectral entropy
    Shannon entropy of the normalized power spectral density in [fmin, fmax].
    Divided by log2(K) so the value is in [0, 1]:
      0   -> energy concentrated in one frequency bin (highly regular)
      1   -> flat spectrum in the band (noise-like)

Permutation entropy
    Bandt-Pompe ordinal pattern entropy, normalized to [0, 1].
    order=3, delay=1 by default.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def spectral_entropy(
    x: np.ndarray,
    fs: float,
    fmin: float = 1.0,
    fmax: float = 40.0,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 8 or not np.isfinite(x).all() or fs <= 0:
        return float("nan")
    x = x - np.mean(x)
    window = np.hanning(n)
    spec = np.fft.rfft(x * window)
    psd = np.real(spec * np.conj(spec))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return float("nan")
    band = psd[mask]
    total = band.sum()
    if total <= 0:
        return float("nan")
    p = band / total
    p = p[p > 0]
    k = p.size
    if k <= 1:
        return 0.0
    return float(-np.sum(p * np.log2(p)) / np.log2(k))


def permutation_entropy(
    x: np.ndarray,
    order: int = 3,
    delay: int = 1,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size - (order - 1) * delay
    if n < 8 or order < 2:
        return float("nan")
    counts: Counter[tuple[int, ...]] = Counter()
    for i in range(n):
        window = x[i : i + order * delay : delay]
        ranks = tuple(np.argsort(window, kind="stable"))
        counts[ranks] += 1
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()], dtype=np.float64)
    n_patterns = float(math.factorial(order))
    ent = -np.sum(probs * np.log2(probs))
    return float(ent / np.log2(n_patterns)) if n_patterns > 1 else 0.0


def channel_entropy(
    data: np.ndarray,
    fs: float,
    method: str = "spectral",
    **kwargs,
) -> np.ndarray:
    """data shape (n_channels, n_samples) -> (n_channels,) entropy."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    out = np.empty(data.shape[0], dtype=np.float64)
    if method == "permutation":
        order = int(kwargs.get("perm_order", 3))
        delay = int(kwargs.get("perm_delay", 1))
        for i, row in enumerate(data):
            out[i] = permutation_entropy(row, order=order, delay=delay)
    else:
        fmin = float(kwargs.get("fmin", 1.0))
        fmax = float(kwargs.get("fmax", 40.0))
        for i, row in enumerate(data):
            out[i] = spectral_entropy(row, fs, fmin=fmin, fmax=fmax)
    return out
