"""BrainFlow Synthetic Board producer and a numpy fallback.

The live path uses BrainFlow BoardShim with SYNTHETIC_BOARD so sample rate,
channel count, and timestamps match a device-like stream. Tests and hosts
without the native library can use the fallback generator, which emits the
same (timestamps, data) contract.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

import numpy as np

from eeg_decode.config import Settings, settings as default_settings

log = logging.getLogger(__name__)


class StreamChunk:
    __slots__ = ("t", "data", "recv_ts", "seq")

    def __init__(self, t: np.ndarray, data: np.ndarray, seq: int):
        self.t = np.asarray(t, dtype=np.float64)
        self.data = np.asarray(data, dtype=np.float64)
        self.recv_ts = time.time()
        self.seq = seq


def make_synthetic_chunk(
    n_channels: int,
    fs: float,
    n_samples: int,
    t0: float,
    seq: int,
    anomaly: str | None = None,
) -> StreamChunk:
    t = t0 + np.arange(n_samples, dtype=np.float64) / fs
    rng = np.random.default_rng(seq + 17)
    data = np.zeros((n_channels, n_samples), dtype=np.float64)
    for ch in range(n_channels):
        f_alpha = 9.0 + 0.3 * ch
        f_beta = 18.0 + 0.2 * ch
        data[ch] = (
            8.0 * np.sin(2 * np.pi * f_alpha * t)
            + 3.0 * np.sin(2 * np.pi * f_beta * t)
            + 1.2 * rng.standard_normal(n_samples)
        )
    if anomaly == "sat":
        data[0] = 250.0
    elif anomaly == "flat":
        data[1] = 0.0
    elif anomaly == "nan":
        data[2, : max(1, n_samples // 4)] = np.nan
    return StreamChunk(t=t, data=data, seq=seq)


class FallbackProducer:
    def __init__(self, n_channels: int, fs: float, chunk_sec: float, accelerated: bool = False):
        self.n_channels = n_channels
        self.fs = fs
        self.chunk_n = max(1, int(round(fs * chunk_sec)))
        self.accelerated = accelerated
        self._t0 = 0.0
        self._seq = 0
        self.channel_names = [f"EEG{i + 1}" for i in range(n_channels)]

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def read(self) -> StreamChunk | None:
        chunk = make_synthetic_chunk(
            self.n_channels, self.fs, self.chunk_n, self._t0, self._seq
        )
        self._t0 += self.chunk_n / self.fs
        self._seq += 1
        if not self.accelerated:
            time.sleep(self.chunk_n / self.fs)
        return chunk

    def iter(self) -> Iterator[StreamChunk]:
        while True:
            yield self.read()  # type: ignore[misc]


class BrainFlowProducer:
    def __init__(self, cfg: Settings | None = None, accelerated: bool | None = None):
        self.cfg = cfg or default_settings
        self.accelerated = self.cfg.accelerated if accelerated is None else accelerated
        self._board = None
        self.fs = float(self.cfg.default_fs)
        self.n_channels = int(self.cfg.n_channels)
        self.eeg_idx: list[int] = []
        self.ts_idx: int | None = None
        self.channel_names: list[str] = []
        self._seq = 0
        self._chunk_n = max(1, int(round(self.fs * self.cfg.producer_chunk_sec)))

    def start(self) -> None:
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()
        board_id = int(self.cfg.brainflow_board_id)
        if board_id == -1:
            board_id = int(BoardIds.SYNTHETIC_BOARD)
        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream(450000)
        self._board = board
        self.fs = float(BoardShim.get_sampling_rate(board_id))
        self.eeg_idx = list(BoardShim.get_eeg_channels(board_id))
        self.n_channels = len(self.eeg_idx)
        try:
            names = BoardShim.get_eeg_names(board_id)
            self.channel_names = list(names)[: self.n_channels]
        except Exception:
            self.channel_names = [f"EEG{i + 1}" for i in range(self.n_channels)]
        try:
            self.ts_idx = int(BoardShim.get_timestamp_channel(board_id))
        except Exception:
            self.ts_idx = None
        self._chunk_n = max(1, int(round(self.fs * self.cfg.producer_chunk_sec)))
        log.info("BrainFlow synthetic board fs=%s channels=%s", self.fs, self.n_channels)

    def stop(self) -> None:
        if self._board is None:
            return
        try:
            self._board.stop_stream()
        finally:
            try:
                self._board.release_session()
            except Exception:
                pass
            self._board = None

    def read(self) -> StreamChunk | None:
        if self._board is None:
            return None
        data = self._board.get_board_data()
        if data is None or data.size == 0 or data.shape[1] == 0:
            if not self.accelerated:
                time.sleep(self.cfg.producer_chunk_sec)
            return None
        eeg = data[self.eeg_idx, :]
        n = eeg.shape[1]
        if self.ts_idx is not None:
            t = data[self.ts_idx, :]
        else:
            now = time.time()
            t = now - (n - 1) / self.fs + np.arange(n) / self.fs
        chunk = StreamChunk(t=t, data=eeg, seq=self._seq)
        self._seq += 1
        if not self.accelerated:
            time.sleep(self.cfg.producer_chunk_sec)
        return chunk


def create_producer(source: str, cfg: Settings | None = None, accelerated: bool = False):
    cfg = cfg or default_settings
    if source == "brainflow":
        try:
            prod = BrainFlowProducer(cfg, accelerated=accelerated)
            prod.start()
            return prod
        except Exception as exc:
            log.warning("BrainFlow unavailable (%s); using numpy fallback producer", exc)
    return FallbackProducer(
        n_channels=cfg.n_channels,
        fs=cfg.default_fs,
        chunk_sec=cfg.producer_chunk_sec,
        accelerated=accelerated,
    )
