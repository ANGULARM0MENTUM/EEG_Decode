"""Optional wall-clock long session. Default tests use accelerated injection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eeg_decode.config import Settings
from eeg_decode.pipeline import SessionPipeline
from eeg_decode.source import create_producer
from eeg_decode.store import new_session


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--accelerated", action="store_true")
    args = p.parse_args()
    cfg = Settings(accelerated=args.accelerated)
    prod = create_producer("brainflow", cfg, accelerated=args.accelerated)
    store = new_session(prod.fs, prod.n_channels, "brainflow", cfg, channel_names=prod.channel_names)
    pipe = SessionPipeline(store, emit=lambda m: None, cfg=cfg)
    pipe.start()
    t_end = time.time() + args.seconds
    try:
        while time.time() < t_end:
            chunk = prod.read()
            if chunk is not None:
                pipe.submit(chunk)
    finally:
        pipe.stop()
        try:
            prod.stop()
        except Exception:
            pass
        summary = store.close("stopped")
        out = Path("data/sessions") / store.meta.session_id / "summary.json"
        print(json.dumps({"path": str(out), "windows": summary["windows"], "samples": summary["samples_seen"]}, indent=2))


if __name__ == "__main__":
    main()
