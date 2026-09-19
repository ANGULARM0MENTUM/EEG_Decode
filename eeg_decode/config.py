from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Override with env vars prefixed EEG_."""

    model_config = SettingsConfigDict(env_prefix="EEG_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: str = "data/sessions"

    # Streaming / windowing
    default_fs: float = 256.0
    n_channels: int = 8
    window_sec: float = 2.0
    hop_sec: float = 0.25
    display_sec: float = 8.0

    # Preprocessing (online IIR, stateful)
    bandpass_hz: tuple[float, float] = (1.0, 40.0)
    notch_hz: float = 50.0
    notch_q: float = 30.0
    filter_order: int = 4

    # Entropy
    entropy_method: str = "spectral"  # spectral | permutation
    spectral_fmin: float = 1.0
    spectral_fmax: float = 40.0
    perm_order: int = 3
    perm_delay: int = 1
    smooth_alpha: float = 0.35

    # Quality / artifacts
    sat_abs_uv: float = 200.0
    flatline_std_uv: float = 0.15
    emg_highband_ratio: float = 0.55

    # Backpressure: bounded chunk queue. Drop-oldest when full.
    queue_max_chunks: int = 48
    gap_sec: float = 0.75
    ingest_timeout_sec: float = 3.0

    # Producer
    producer_chunk_sec: float = 0.05
    brainflow_board_id: int = -1  # SYNTHETIC_BOARD
    accelerated: bool = False

    log_level: str = "INFO"


settings = Settings()


def window_samples(fs: float, window_sec: float | None = None) -> int:
    return max(8, int(round(fs * (window_sec or settings.window_sec))))


def hop_samples(fs: float, hop_sec: float | None = None) -> int:
    return max(1, int(round(fs * (hop_sec or settings.hop_sec))))


def display_samples(fs: float) -> int:
    return max(window_samples(fs), int(round(fs * settings.display_sec)))
