from __future__ import annotations

import time

import numpy as np
import pytest

from eeg_decode.pipeline import SessionPipeline
from eeg_decode.source import make_synthetic_chunk
from eeg_decode.store import new_session


def _wait_windows(store, n: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if store.windows_emitted >= n:
            return
        time.sleep(0.02)
    raise AssertionError(f"only {store.windows_emitted} windows, expected >= {n}")


def _submit_blocking(pipe, chunk, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = pipe.submit(chunk)
        if st != "BACKPRESSURE":
            return st
        time.sleep(0.002)
    return "BACKPRESSURE"


def test_stream_windows_status_and_alignment(cfg):
    store = new_session(cfg.default_fs, cfg.n_channels, "upload", cfg)
    frames: list[dict] = []
    pipe = SessionPipeline(store, emit=frames.append, cfg=cfg)
    pipe.start()
    fs = cfg.default_fs
    chunk_n = 64
    t0 = 0.0
    for seq in range(20):
        c = make_synthetic_chunk(cfg.n_channels, fs, chunk_n, t0, seq)
        assert _submit_blocking(pipe, c) == "OK"
        t0 += chunk_n / fs
    # 20*64=1280 samples, window=256, hop=64 -> windows after filling 256
    _wait_windows(store, 10)
    pipe.stop()

    waves = [m for m in frames if m.get("type") == "waveform"]
    ents = [m for m in frames if m.get("type") == "entropy"]
    stats = [m for m in frames if m.get("type") == "status"]
    assert waves, "waveform frames missing"
    assert ents, "entropy frames missing"
    assert stats, "status frames missing"
    assert all("t" in w and "samples" in w for w in waves)
    ts = [e["t_end"] for e in ents]
    assert ts == sorted(ts)
    hops = np.diff(ts)
    assert np.median(hops) == pytest_approx_hop(cfg.hop_sec)
    for e in ents:
        assert e["t_end"] - e["t_start"] == pytest.approx(cfg.window_sec)
        assert len(e["per_channel"]) == cfg.n_channels
        assert e["e2e_ms"] is not None
        for v in e["per_channel"]:
            if v is not None:
                assert 0.0 <= v <= 1.0
    assert any(s["throughput_sps"] >= 0 for s in stats)
    assert store.last_state in {"OK", "QUALITY_BAD", "BACKPRESSURE"}


def pytest_approx_hop(hop: float):
    return pytest.approx(hop, rel=0.15, abs=0.08)


def test_backpressure_drop_oldest(cfg):
    cfg.queue_max_chunks = 2
    store = new_session(cfg.default_fs, cfg.n_channels, "upload", cfg)
    pipe = SessionPipeline(store, emit=lambda m: None, cfg=cfg)
    # do not start worker yet so queue fills
    fs = cfg.default_fs
    statuses = []
    t0 = 0.0
    for seq in range(12):
        c = make_synthetic_chunk(cfg.n_channels, fs, 32, t0, seq)
        statuses.append(pipe.submit(c))
        t0 += 32 / fs
    assert "BACKPRESSURE" in statuses
    assert store.dropped_chunks >= 1
    pipe.start()
    time.sleep(0.4)
    pipe.stop()
    assert store.samples_seen > 0
    assert store.last_state in {"OK", "BACKPRESSURE", "STREAM_GAP", "QUALITY_BAD"}


def test_gap_then_resume(cfg):
    store = new_session(cfg.default_fs, cfg.n_channels, "upload", cfg)
    pipe = SessionPipeline(store, emit=lambda m: None, cfg=cfg)
    pipe.start()
    fs = cfg.default_fs
    t0 = 0.0
    for seq in range(8):
        c = make_synthetic_chunk(cfg.n_channels, fs, 64, t0, seq)
        _submit_blocking(pipe, c)
        t0 += 64 / fs
    time.sleep(cfg.gap_sec + 0.25)
    deadline = time.time() + 2
    while time.time() < deadline and store.last_state != "STREAM_GAP":
        time.sleep(0.02)
    assert store.last_state == "STREAM_GAP"
    for seq in range(8, 16):
        c = make_synthetic_chunk(cfg.n_channels, fs, 64, t0, seq)
        _submit_blocking(pipe, c)
        t0 += 64 / fs
    _wait_windows(store, 8)
    deadline = time.time() + 2
    while time.time() < deadline and store.last_state == "STREAM_GAP":
        time.sleep(0.02)
    assert store.last_state != "STREAM_GAP"
    pipe.stop()


def test_anomaly_quality_status(cfg):
    store = new_session(cfg.default_fs, cfg.n_channels, "upload", cfg)
    pipe = SessionPipeline(store, emit=lambda m: None, cfg=cfg)
    pipe.start()
    fs = cfg.default_fs
    t0 = 0.0
    for seq in range(6):
        c = make_synthetic_chunk(cfg.n_channels, fs, 128, t0, seq, anomaly="sat")
        _submit_blocking(pipe, c)
        t0 += 128 / fs
    deadline = time.time() + 5
    while time.time() < deadline and store.last_state != "QUALITY_BAD":
        time.sleep(0.02)
    pipe.stop()
    assert store.last_state == "QUALITY_BAD" or store.windows_emitted == 0
    if store.windows_emitted:
        # saturated channel 0 should be null in last persisted sense: stats still run
        assert store.entropy_stats is not None


def test_mid_session_and_final_summary(cfg):
    store = new_session(cfg.default_fs, cfg.n_channels, "upload", cfg)
    pipe = SessionPipeline(store, emit=lambda m: None, cfg=cfg)
    pipe.start()
    fs = cfg.default_fs
    t0 = 0.0
    for seq in range(12):
        c = make_synthetic_chunk(cfg.n_channels, fs, 128, t0, seq)
        _submit_blocking(pipe, c)
        t0 += 128 / fs
    _wait_windows(store, 8)
    mid = store.summary()
    assert mid["windows"] >= 8
    assert "channel_entropy" in mid
    assert "global_entropy" in mid
    assert "latency_ms" in mid
    more = 0
    for seq in range(12, 20):
        c = make_synthetic_chunk(cfg.n_channels, fs, 128, t0, seq)
        _submit_blocking(pipe, c)
        t0 += 128 / fs
        more += 1
    _wait_windows(store, mid["windows"] + 1)
    pipe.stop()
    final = store.close("stopped")
    assert final["windows"] >= mid["windows"]
    assert (store.root / "summary.json").exists()
    assert (store.root / "windows.jsonl").stat().st_size > 0
    ge = final["global_entropy"]
    assert ge["count"] >= 1
    assert ge["mean"] is not None
    assert "intervals" in final
