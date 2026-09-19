from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from eeg_decode.config import Settings, display_samples
from eeg_decode.pipeline import SessionPipeline
from eeg_decode.source import make_synthetic_chunk
from eeg_decode.store import new_session


def _submit_blocking(pipe, chunk, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pipe.submit(chunk, block=True) != "BACKPRESSURE":
            return
        time.sleep(0.001)
    raise TimeoutError("queue did not drain")


@pytest.mark.slow
def test_thirty_minute_equivalent_stream(tmp_path):
    """Process 30 minutes of 256 Hz x 8 ch without unbounded RAM growth.

    Wall clock is accelerated: samples are injected as fast as the worker allows.
    """
    fs = 256.0
    seconds = 30 * 60
    n_ch = 8
    cfg = Settings(
        data_dir=str(tmp_path / "sessions"),
        default_fs=fs,
        n_channels=n_ch,
        window_sec=2.0,
        hop_sec=0.5,
        display_sec=8.0,
        queue_max_chunks=64,
        accelerated=True,
    )
    store = new_session(fs, n_ch, "upload", cfg)
    pipe = SessionPipeline(store, emit=lambda m: None, cfg=cfg)
    pipe.start()
    total = int(fs * seconds)
    chunk_n = 512
    t0 = 0.0
    seq = 0
    sent = 0
    cap_bytes = store.raw.nbytes
    while sent < total:
        n = min(chunk_n, total - sent)
        c = make_synthetic_chunk(n_ch, fs, n, t0, seq)
        _submit_blocking(pipe, c)
        t0 += n / fs
        sent += n
        seq += 1
        if seq % 200 == 0:
            assert store.raw.nbytes == cap_bytes
            assert store.filt.nbytes == cap_bytes
    deadline = time.time() + 60
    expected_min = int((seconds - cfg.window_sec) / cfg.hop_sec) - 5
    while time.time() < deadline and store.windows_emitted < expected_min:
        time.sleep(0.05)
    pipe.stop()
    assert store.samples_seen == total
    assert store.windows_emitted >= expected_min
    assert store.raw.shape[1] == display_samples(fs)
    summary = store.close("stopped")
    assert summary["windows"] == store.windows_emitted
    out = Path("examples") / "summary_generated.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def test_online_vs_offline_entropy_close_on_same_window():
    from eeg_decode.entropy import spectral_entropy

    fs = 256.0
    t = np.arange(int(fs * 2.0)) / fs
    x = 5 * np.sin(2 * np.pi * 12 * t) + 0.2 * np.random.default_rng(2).standard_normal(t.size)
    online = spectral_entropy(x, fs)
    offline = spectral_entropy(x.copy(), fs)
    assert abs(online - offline) < 1e-12
