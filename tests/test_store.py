from __future__ import annotations

import numpy as np

from eeg_decode.store import new_session


def test_ring_buffer_window_timestamps(cfg):
    store = new_session(256.0, 2, "upload", cfg)
    fs = 256.0
    for i in range(20):
        n = 64
        t0 = i * n / fs
        t = t0 + np.arange(n) / fs
        x = np.ones((2, n)) * i
        store.append_samples(t, x, x)
    win, t_end = store.window_at(256, 0)
    assert win is not None
    assert win.shape == (2, 256)
    assert t_end == pytest_approx_last(store)
    older, t2 = store.window_at(256, 64)
    assert older is not None
    assert t2 == t_end - 64 / fs


def pytest_approx_last(store):
    t, _ = store.snapshot_display()
    return float(t[-1])
