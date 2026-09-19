"""Streaming pipeline: ingest -> bounded queue -> window -> entropy -> emit.

Backpressure strategy (drop-oldest):
  The ingest queue is bounded (queue_max_chunks). When the producer is faster
  than the worker, the oldest unread chunk is discarded, a BACKPRESSURE status
  is emitted, and the session stays alive. This keeps RAM and latency bounded
  instead of letting a queue grow without limit.

Time alignment:
  Each window is tagged with t_end = last sample timestamp in that window.
  Entropy is computed on the filtered samples that occupy [t_end - window, t_end].
  The UI plots entropy at t_end against the raw waveform using the same clock.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from eeg_decode.config import Settings, hop_samples, settings as default_settings, window_samples
from eeg_decode.entropy import channel_entropy
from eeg_decode.preprocess import OnlinePreprocessor
from eeg_decode.source import StreamChunk
from eeg_decode.store import SessionStore

log = logging.getLogger(__name__)

EmitFn = Callable[[dict], None]


class SessionPipeline:
    def __init__(self, store: SessionStore, emit: EmitFn, cfg: Settings | None = None):
        self.store = store
        self.emit = emit
        self.cfg = cfg or default_settings
        fs = store.meta.fs
        n = store.meta.n_channels
        self.pre = OnlinePreprocessor(n, fs, self.cfg)
        self.q: queue.Queue[Optional[StreamChunk]] = queue.Queue(maxsize=self.cfg.queue_max_chunks)
        self.win = window_samples(fs, store.meta.window_sec)
        self.hop = hop_samples(fs, store.meta.hop_sec)
        self._next_emit = self.win
        self._smooth = np.full(n, np.nan)
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._bytes_window: list[tuple[float, int]] = []
        self._t0_wall = time.time()

    def start(self) -> None:
        self._worker = threading.Thread(target=self._run, name=f"eeg-{self.store.meta.session_id}", daemon=True)
        self._worker.start()
        self.store.last_state = "OK"
        self.store.log_event("pipeline_start")

    def stop(self) -> None:
        self._stop.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        if self._worker:
            self._worker.join(timeout=2.0)

    def submit(self, chunk: StreamChunk, block: bool = False) -> str:
        """Push a chunk. Returns OK, BACKPRESSURE, or STOPPED.

        BACKPRESSURE means this chunk was accepted after dropping the oldest
        queued chunk. Do not resubmit the same chunk.
        """
        if self._stop.is_set():
            return "STOPPED"
        if block:
            try:
                self.q.put(chunk, timeout=10.0)
                return "OK"
            except queue.Full:
                with self.store.lock:
                    self.store.dropped_chunks += 1
                self.store.last_state = "BACKPRESSURE"
                return "BACKPRESSURE"
        try:
            self.q.put_nowait(chunk)
            return "OK"
        except queue.Full:
            dropped = 0
            try:
                self.q.get_nowait()
                dropped = 1
                self.q.put_nowait(chunk)
            except queue.Full:
                dropped = 1
            except queue.Empty:
                try:
                    self.q.put_nowait(chunk)
                except queue.Full:
                    dropped = 1
            with self.store.lock:
                self.store.dropped_chunks += dropped
            self.store.last_state = "BACKPRESSURE"
            self.store.log_event("backpressure", dropped=dropped, qsize=self.q.qsize())
            return "BACKPRESSURE"

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                now = time.time()
                if self.store.last_ingest_ts and (now - self.store.last_ingest_ts) > self.cfg.gap_sec:
                    if self.store.last_state != "STREAM_GAP":
                        self.store.last_state = "STREAM_GAP"
                        self.store.log_event("stream_gap", idle_sec=now - self.store.last_ingest_ts)
                        self._emit_status()
                continue
            if item is None:
                break
            try:
                self._handle(item)
            except Exception as exc:
                log.exception("window compute failed")
                self.store.last_state = "COMPUTE_ERROR"
                self.store.log_event("compute_error", error=str(exc))
                self._emit_status()

    def _handle(self, chunk: StreamChunk) -> None:
        t_recv = chunk.recv_ts
        data = chunk.data
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[0] != self.store.meta.n_channels:
            n = min(data.shape[0], self.store.meta.n_channels)
            padded = np.zeros((self.store.meta.n_channels, data.shape[1]))
            padded[:n] = data[:n]
            data = padded
        t = chunk.t
        if t.size != data.shape[1]:
            t = (self.store.last_sample_t or 0.0) + np.arange(1, data.shape[1] + 1) / self.store.meta.fs

        if self.store.last_sample_t is not None and t.size:
            dt = t[0] - self.store.last_sample_t
            if dt > self.cfg.gap_sec:
                self.store.last_state = "STREAM_GAP"
                self.store.log_event("timestamp_gap", dt=float(dt))
            elif dt < -1e-3:
                self.store.log_event("reorder", dt=float(dt))
                order = np.argsort(t)
                t = t[order]
                data = data[:, order]

        qf = self.pre.quality(data)
        t0p = time.perf_counter()
        filt = self.pre.process(data)
        preprocess_ms = (time.perf_counter() - t0p) * 1000.0
        self.store.append_samples(t, data, filt)
        self.store.last_ingest_ts = time.time()
        if t.size:
            self.store.last_sample_t = float(t[-1])

        n_new = data.shape[1]
        self._note_throughput(n_new)

        quality_bad = qf.any_bad
        if quality_bad:
            self.store.last_state = "QUALITY_BAD"
        elif self.store.last_state in {"STREAM_GAP", "COMPUTE_ERROR", "INIT", "BACKPRESSURE", "QUALITY_BAD"}:
            self.store.last_state = "OK"

        while self.store.samples_seen >= self._next_emit:
            end_offset = self.store.samples_seen - self._next_emit
            win, t_end = self.store.window_at(self.win, end_offset)
            self._next_emit += self.hop
            if win is None or t_end is None:
                continue
            t0e = time.perf_counter()
            per = channel_entropy(
                win,
                self.store.meta.fs,
                method=self.cfg.entropy_method,
                fmin=self.cfg.spectral_fmin,
                fmax=self.cfg.spectral_fmax,
                perm_order=self.cfg.perm_order,
                perm_delay=self.cfg.perm_delay,
            )
            entropy_ms = (time.perf_counter() - t0e) * 1000.0
            if quality_bad:
                per = np.where(qf.saturated | qf.flatline | qf.nan_inf, np.nan, per)
            alpha = self.cfg.smooth_alpha
            if not np.any(np.isfinite(self._smooth)):
                self._smooth = per.copy()
            else:
                for i, v in enumerate(per):
                    if not np.isfinite(v):
                        continue
                    if not np.isfinite(self._smooth[i]):
                        self._smooth[i] = v
                    else:
                        self._smooth[i] = alpha * v + (1 - alpha) * self._smooth[i]
            g_raw = float(np.nanmean(per)) if np.any(np.isfinite(per)) else float("nan")
            g_s = float(np.nanmean(self._smooth)) if np.any(np.isfinite(self._smooth)) else float("nan")
            e2e_ms = (time.time() - t_recv) * 1000.0
            row = {
                "t_end": float(t_end),
                "t_start": float(t_end) - self.store.meta.window_sec,
                "per_channel": [None if not np.isfinite(v) else float(v) for v in per],
                "smoothed": [None if not np.isfinite(v) else float(v) for v in self._smooth],
                "global": None if not np.isfinite(g_raw) else g_raw,
                "global_smoothed": None if not np.isfinite(g_s) else g_s,
                "preprocess_ms": preprocess_ms,
                "entropy_ms": entropy_ms,
                "e2e_ms": e2e_ms,
                "quality": qf.as_dict(),
                "quality_bad": quality_bad,
                "state": self.store.last_state,
            }
            self.store.record_window(row)
            self.emit({"type": "entropy", **row, "session_id": self.store.meta.session_id})

        t_disp, x_disp = self.store.snapshot_display(n=256)
        self.emit(
            {
                "type": "waveform",
                "session_id": self.store.meta.session_id,
                "t": t_disp.tolist(),
                "samples": x_disp.tolist(),
                "fs": self.store.meta.fs,
                "channels": self.store.meta.channel_names,
            }
        )
        self._emit_status()

    def _note_throughput(self, n_new: int) -> None:
        now = time.time()
        self._bytes_window.append((now, n_new))
        self._bytes_window = [(t, n) for t, n in self._bytes_window if now - t < 2.0]
        span = max(now - self._bytes_window[0][0], 1e-6)
        self.store.throughput_sps = sum(n for _, n in self._bytes_window) / span

    def _emit_status(self) -> None:
        self.emit(
            {
                "type": "status",
                "session_id": self.store.meta.session_id,
                "state": self.store.last_state,
                "qsize": self.q.qsize(),
                "qmax": self.cfg.queue_max_chunks,
                "dropped_chunks": self.store.dropped_chunks,
                "samples_seen": self.store.samples_seen,
                "windows": self.store.windows_emitted,
                "throughput_sps": self.store.throughput_sps,
                "uptime_sec": time.time() - self._t0_wall,
            }
        )
