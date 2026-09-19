"""Stateful online EEG preprocessing.

Filters run sample-continuously (SOS / IIR zi) so window edges do not reset.
This is the right model for a live stream; offline batch filtering of each
window independently would leak transients every hop.

Used libraries:
  scipy.signal  – Butterworth bandpass + IIR notch design / sosfilt
  numpy         – vector math, quality metrics
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

from eeg_decode.config import Settings, settings as default_settings


@dataclass
class QualityFlags:
    saturated: np.ndarray
    flatline: np.ndarray
    emg_like: np.ndarray
    nan_inf: np.ndarray

    @property
    def any_bad(self) -> bool:
        return bool(
            np.any(self.saturated)
            | np.any(self.flatline)
            | np.any(self.emg_like)
            | np.any(self.nan_inf)
        )

    def as_dict(self) -> dict:
        return {
            "saturated": self.saturated.tolist(),
            "flatline": self.flatline.tolist(),
            "emg_like": self.emg_like.tolist(),
            "nan_inf": self.nan_inf.tolist(),
        }


class OnlinePreprocessor:
    def __init__(self, n_channels: int, fs: float, cfg: Settings | None = None):
        self.cfg = cfg or default_settings
        self.n_channels = n_channels
        self.fs = float(fs)
        nyq = 0.5 * self.fs
        lo, hi = self.cfg.bandpass_hz
        lo = max(0.01, lo / nyq)
        hi = min(0.99, hi / nyq)
        if lo >= hi:
            lo, hi = 0.02, 0.4
        self._bp_sos = butter(self.cfg.filter_order, [lo, hi], btype="band", output="sos")
        w0 = self.cfg.notch_hz / nyq
        if 0 < w0 < 1:
            b, a = iirnotch(self.cfg.notch_hz, self.cfg.notch_q, fs=self.fs)
            self._notch_sos = tf2sos(b, a)
        else:
            self._notch_sos = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        self._bp_zi = np.zeros((self._bp_sos.shape[0], 2, n_channels))
        self._notch_zi = np.zeros((self._notch_sos.shape[0], 2, n_channels))

    def reset(self) -> None:
        self._bp_zi.fill(0.0)
        self._notch_zi.fill(0.0)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """chunk (n_channels, n_samples) -> filtered same shape."""
        x = np.asarray(chunk, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.empty_like(x)
        for ch in range(x.shape[0]):
            y[ch], self._notch_zi[:, :, ch] = sosfilt(self._notch_sos, x[ch], zi=self._notch_zi[:, :, ch])
            y[ch], self._bp_zi[:, :, ch] = sosfilt(self._bp_sos, y[ch], zi=self._bp_zi[:, :, ch])
        return y

    def quality(self, raw_chunk: np.ndarray) -> QualityFlags:
        x = np.asarray(raw_chunk, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        nan_inf = ~np.isfinite(x).all(axis=1)
        peak = np.nanmax(np.abs(x), axis=1)
        std = np.nanstd(x, axis=1)
        saturated = peak >= self.cfg.sat_abs_uv
        flatline = std <= self.cfg.flatline_std_uv
        # crude EMG proxy: high-frequency energy share via first difference
        d = np.diff(x, axis=1)
        hf = np.mean(d * d, axis=1)
        tot = np.mean(x * x, axis=1) + 1e-12
        emg_like = (hf / tot) >= self.cfg.emg_highband_ratio
        return QualityFlags(saturated=saturated, flatline=flatline, emg_like=emg_like, nan_inf=nan_inf)
