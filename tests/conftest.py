from __future__ import annotations

import pytest

from eeg_decode.config import Settings


@pytest.fixture
def cfg(tmp_path) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "sessions"),
        default_fs=256.0,
        n_channels=8,
        window_sec=1.0,
        hop_sec=0.25,
        display_sec=4.0,
        queue_max_chunks=32,
        gap_sec=0.4,
        producer_chunk_sec=0.05,
        accelerated=True,
    )
