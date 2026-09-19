"""Bounded session store: RAM stays O(display window), disk holds window rows.

A 30+ minute session must not grow in-memory caches without bound.
Raw samples live in a circular buffer sized to display_sec.
Window entropy rows are append-only JSONL on disk.
Running stats are O(n_channels) via Welford.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from eeg_decode.config import Settings, display_samples, settings as default_settings


class Welford:
    def __init__(self, n: int):
        self.n = np.zeros(n, dtype=np.int64)
        self.mean = np.zeros(n, dtype=np.float64)
        self.m2 = np.zeros(n, dtype=np.float64)
        self.min = np.full(n, np.inf)
        self.max = np.full(n, -np.inf)

    def update(self, values: np.ndarray, valid: np.ndarray | None = None) -> None:
        values = np.asarray(values, dtype=np.float64)
        if valid is None:
            valid = np.isfinite(values)
        else:
            valid = valid & np.isfinite(values)
        for i, ok in enumerate(valid):
            if not ok:
                continue
            x = values[i]
            self.n[i] += 1
            delta = x - self.mean[i]
            self.mean[i] += delta / self.n[i]
            delta2 = x - self.mean[i]
            self.m2[i] += delta * delta2
            self.min[i] = min(self.min[i], x)
            self.max[i] = max(self.max[i], x)

    def snapshot(self) -> dict[str, list]:
        var = np.divide(self.m2, np.maximum(self.n - 1, 1), dtype=np.float64)
        var[self.n < 2] = 0.0
        mn = np.where(np.isfinite(self.min), self.min, np.nan)
        mx = np.where(np.isfinite(self.max), self.max, np.nan)
        return {
            "count": self.n.tolist(),
            "mean": self.mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "min": mn.tolist(),
            "max": mx.tolist(),
            "range": (mx - mn).tolist(),
        }


@dataclass
class SessionMeta:
    session_id: str
    created_ts: float
    fs: float
    n_channels: int
    channel_names: list[str]
    window_sec: float
    hop_sec: float
    entropy_method: str
    source: str
    status: str = "running"
    ended_ts: float | None = None


@dataclass
class SessionStore:
    meta: SessionMeta
    cfg: Settings
    root: Path
    lock: threading.Lock = field(default_factory=threading.Lock)
    write_idx: int = 0
    samples_seen: int = 0
    windows_emitted: int = 0
    dropped_chunks: int = 0
    last_ingest_ts: float = 0.0
    last_sample_t: float | None = None
    e2e_ms: Welford = field(default_factory=lambda: Welford(1))
    entropy_stats: Welford | None = None
    global_stats: Welford = field(default_factory=lambda: Welford(1))
    latency_stats: Welford = field(default_factory=lambda: Welford(1))
    throughput_sps: float = 0.0
    last_state: str = "INIT"
    recent_entropy: list[list[float]] = field(default_factory=list)
    recent_global: list[float] = field(default_factory=list)
    recent_t: list[float] = field(default_factory=list)
    interval_labels: list[dict] = field(default_factory=list)
    _trend_buf: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        n = self.meta.n_channels
        cap = display_samples(self.meta.fs)
        self.raw = np.zeros((n, cap), dtype=np.float32)
        self.filt = np.zeros((n, cap), dtype=np.float32)
        self.times = np.zeros(cap, dtype=np.float64)
        self.cap = cap
        self.entropy_stats = Welford(n)
        (self.root / "meta.json").write_text(json.dumps(self.meta.__dict__, indent=2), encoding="utf-8")
        (self.root / "windows.jsonl").touch()
        (self.root / "events.jsonl").touch()

    def append_samples(self, t: np.ndarray, raw: np.ndarray, filt: np.ndarray) -> None:
        n_new = raw.shape[1]
        cap = self.cap
        with self.lock:
            self.samples_seen += n_new
            if n_new >= cap:
                self.raw[:] = raw[:, -cap:]
                self.filt[:] = filt[:, -cap:]
                self.times[:] = t[-cap:]
                self.write_idx = cap
                return
            space = cap - self.write_idx
            if n_new <= space:
                sl = slice(self.write_idx, self.write_idx + n_new)
                self.raw[:, sl] = raw
                self.filt[:, sl] = filt
                self.times[sl] = t
                self.write_idx += n_new
            else:
                first = space
                self.raw[:, self.write_idx :] = raw[:, :first]
                self.filt[:, self.write_idx :] = filt[:, :first]
                self.times[self.write_idx :] = t[:first]
                rest = n_new - first
                self.raw[:, :rest] = raw[:, first:]
                self.filt[:, :rest] = filt[:, first:]
                self.times[:rest] = t[first:]
                self.write_idx = rest

    def snapshot_display(self, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        n = n or self.cap
        with self.lock:
            filled = min(self.samples_seen, self.cap)
            if filled <= 0:
                return self.times[:0], self.raw[:, :0]
            if self.samples_seen < self.cap:
                t = self.times[:filled].copy()
                x = self.raw[:, :filled].copy()
            else:
                idx = (self.write_idx + np.arange(self.cap)) % self.cap
                t = self.times[idx].copy()
                x = self.raw[:, idx].copy()
        if n < x.shape[1]:
            step = max(1, x.shape[1] // n)
            return t[::step], x[:, ::step]
        return t, x

    def last_filtered_window(self, win: int) -> np.ndarray | None:
        x, _ = self.window_at(win, 0)
        return x

    def window_at(self, win: int, end_offset: int = 0) -> tuple[np.ndarray | None, float | None]:
        """Filtered window whose last sample is `end_offset` samples before the newest."""
        with self.lock:
            if win <= 0 or end_offset < 0:
                return None, None
            if self.samples_seen < win + end_offset:
                return None, None
            if self.samples_seen < self.cap:
                stop = self.write_idx - end_offset
                start = stop - win
                if start < 0 or stop <= 0:
                    return None, None
                return self.filt[:, start:stop].copy(), float(self.times[stop - 1])
            idx = (self.write_idx - end_offset - win + np.arange(win)) % self.cap
            last_i = (self.write_idx - 1 - end_offset) % self.cap
            return self.filt[:, idx].copy(), float(self.times[last_i])

    def log_event(self, kind: str, **payload: Any) -> None:
        rec = {"ts": time.time(), "kind": kind, **payload}
        with self.lock:
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def record_window(self, row: dict) -> None:
        per = np.asarray(row["per_channel"], dtype=np.float64)
        valid = np.isfinite(per)
        g = row.get("global")
        with self.lock:
            self.windows_emitted += 1
            self.entropy_stats.update(per, valid)
            if g is not None and np.isfinite(g):
                self.global_stats.update(np.array([g]))
            if row.get("e2e_ms") is not None:
                self.latency_stats.update(np.array([row["e2e_ms"]]))
            self.recent_entropy.append(per.tolist())
            self.recent_global.append(float(g) if g is not None else float("nan"))
            self.recent_t.append(float(row["t_end"]))
            if len(self.recent_entropy) > 240:
                self.recent_entropy = self.recent_entropy[-240:]
                self.recent_global = self.recent_global[-240:]
                self.recent_t = self.recent_t[-240:]
            self._update_intervals(float(g) if g is not None else float("nan"), row)
            with (self.root / "windows.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=_json_default) + "\n")

    def _update_intervals(self, g: float, row: dict) -> None:
        if not np.isfinite(g):
            label = "ANOMALOUS"
        else:
            self._trend_buf.append(g)
            if len(self._trend_buf) > 24:
                self._trend_buf = self._trend_buf[-24:]
            if len(self._trend_buf) >= 8:
                arr = np.asarray(self._trend_buf, dtype=np.float64)
                recent = arr[-4:].mean()
                older = arr[:-4].mean()
                spread = arr.std() + 1e-6
                if row.get("quality_bad"):
                    label = "ANOMALOUS"
                elif abs(recent - older) > 1.5 * spread:
                    label = "CHANGING"
                else:
                    label = "STABLE"
            else:
                label = "STABLE" if not row.get("quality_bad") else "ANOMALOUS"
        t = float(row["t_end"])
        if self.interval_labels and self.interval_labels[-1]["label"] == label:
            self.interval_labels[-1]["t_end"] = t
            self.interval_labels[-1]["windows"] += 1
        else:
            self.interval_labels.append({"label": label, "t_start": t, "t_end": t, "windows": 1})

    def summary(self) -> dict:
        with self.lock:
            ch_stats = self.entropy_stats.snapshot() if self.entropy_stats else {}
            g_stats = self.global_stats.snapshot()
            lat = self.latency_stats.snapshot()
            trend = _trend_from_series(self.recent_global)
            meta = dict(self.meta.__dict__)
            return {
                "session": meta,
                "samples_seen": self.samples_seen,
                "windows": self.windows_emitted,
                "dropped_chunks": self.dropped_chunks,
                "throughput_sps": self.throughput_sps,
                "channel_entropy": ch_stats,
                "global_entropy": {
                    "mean": g_stats["mean"][0] if g_stats["mean"] else None,
                    "min": g_stats["min"][0] if g_stats["min"] else None,
                    "max": g_stats["max"][0] if g_stats["max"] else None,
                    "range": g_stats["range"][0] if g_stats["range"] else None,
                    "std": g_stats["std"][0] if g_stats["std"] else None,
                    "count": g_stats["count"][0] if g_stats["count"] else 0,
                },
                "latency_ms": {
                    "mean": lat["mean"][0] if lat["mean"] else None,
                    "max": lat["max"][0] if lat["max"] else None,
                    "min": lat["min"][0] if lat["min"] else None,
                },
                "trend": trend,
                "intervals": list(self.interval_labels),
                "last_state": self.last_state,
            }

    def write_summary(self) -> dict:
        s = self.summary()
        (self.root / "summary.json").write_text(json.dumps(s, indent=2, default=_json_default), encoding="utf-8")
        return s

    def close(self, status: str = "stopped") -> dict:
        self.meta.status = status
        self.meta.ended_ts = time.time()
        (self.root / "meta.json").write_text(json.dumps(self.meta.__dict__, indent=2), encoding="utf-8")
        return self.write_summary()


def _trend_from_series(xs: list[float]) -> str:
    arr = np.asarray([x for x in xs if np.isfinite(x)], dtype=np.float64)
    if arr.size < 6:
        return "insufficient"
    half = arr.size // 2
    d = arr[half:].mean() - arr[:half].mean()
    if abs(d) < 0.02:
        return "stable"
    return "increasing" if d > 0 else "decreasing"


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def new_session(
    fs: float,
    n_channels: int,
    source: str,
    cfg: Settings | None = None,
    session_id: str | None = None,
    channel_names: list[str] | None = None,
) -> SessionStore:
    cfg = cfg or default_settings
    sid = session_id or uuid.uuid4().hex[:12]
    root = Path(cfg.data_dir) / sid
    names = channel_names or [f"EEG{i + 1}" for i in range(n_channels)]
    meta = SessionMeta(
        session_id=sid,
        created_ts=time.time(),
        fs=fs,
        n_channels=n_channels,
        channel_names=names,
        window_sec=cfg.window_sec,
        hop_sec=cfg.hop_sec,
        entropy_method=cfg.entropy_method,
        source=source,
    )
    return SessionStore(meta=meta, cfg=cfg, root=root)
