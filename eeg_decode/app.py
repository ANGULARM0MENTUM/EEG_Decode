"""FastAPI app: session lifecycle, WebSocket fan-out, BrainFlow producer thread."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from eeg_decode.config import Settings, settings
from eeg_decode.pipeline import SessionPipeline
from eeg_decode.source import StreamChunk, create_producer
from eeg_decode.store import SessionStore, new_session

log = logging.getLogger("eeg_decode")
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


class Hub:
    def __init__(self) -> None:
        self.sessions: dict[str, SessionStore] = {}
        self.pipelines: dict[str, SessionPipeline] = {}
        self.producers: dict[str, Any] = {}
        self.producer_threads: dict[str, threading.Thread] = {}
        self.ws: dict[str, set[WebSocket]] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self._stop_flags: dict[str, threading.Event] = {}

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def emit(self, session_id: str, message: dict) -> None:
        loop = self.loop
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(session_id, message), loop)
        except RuntimeError:
            pass

    async def _broadcast(self, session_id: str, message: dict) -> None:
        peers = list(self.ws.get(session_id, set()))
        dead: list[WebSocket] = []
        payload = json.dumps(message, default=str)
        for ws in peers:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws.get(session_id, set()).discard(ws)

    def create(
        self,
        source: str,
        accelerated: bool = False,
        cfg: Settings | None = None,
        fs: float | None = None,
        n_channels: int | None = None,
    ) -> SessionStore:
        cfg = cfg or settings
        if source == "brainflow":
            prod = create_producer("brainflow", cfg, accelerated=accelerated)
            store = new_session(prod.fs, prod.n_channels, source="brainflow", cfg=cfg, channel_names=prod.channel_names)
        else:
            store = new_session(
                fs or cfg.default_fs,
                n_channels or cfg.n_channels,
                source=source,
                cfg=cfg,
            )
            prod = None
        pipe = SessionPipeline(store, emit=lambda m: self.emit(store.meta.session_id, m), cfg=cfg)
        pipe.start()
        sid = store.meta.session_id
        self.sessions[sid] = store
        self.pipelines[sid] = pipe
        self.ws.setdefault(sid, set())
        self._stop_flags[sid] = threading.Event()
        if prod is not None:
            self.producers[sid] = prod
            th = threading.Thread(target=self._produce, args=(sid, prod), daemon=True, name=f"src-{sid}")
            self.producer_threads[sid] = th
            th.start()
        return store

    def _produce(self, sid: str, prod: Any) -> None:
        flag = self._stop_flags[sid]
        pipe = self.pipelines[sid]
        while not flag.is_set():
            try:
                chunk = prod.read()
            except Exception as exc:
                log.warning("producer read failed: %s", exc)
                continue
            if chunk is None:
                continue
            pipe.submit(chunk)
        try:
            prod.stop()
        except Exception:
            pass

    def stop(self, sid: str) -> dict:
        store = self.sessions.get(sid)
        if store is None:
            raise KeyError(sid)
        flag = self._stop_flags.get(sid)
        if flag:
            flag.set()
        pipe = self.pipelines.get(sid)
        if pipe:
            pipe.stop()
        summary = store.close("stopped")
        return summary

    def shutdown(self) -> None:
        for sid in list(self.sessions):
            try:
                self.stop(sid)
            except Exception:
                pass


hub = Hub()


class StartBody(BaseModel):
    source: str = Field("brainflow", description="brainflow | upload")
    accelerated: bool = False


class IngestBody(BaseModel):
    t: list[float]
    data: list[list[float]]
    seq: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    hub.bind_loop(asyncio.get_running_loop())
    yield
    hub.shutdown()


app = FastAPI(title="EEG Decode", version="1.0.0", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "sessions": len(hub.sessions)}


@app.post("/api/sessions")
def start_session(body: StartBody) -> dict:
    source = body.source if body.source in {"brainflow", "upload"} else "brainflow"
    store = hub.create(source=source, accelerated=body.accelerated)
    return {"session_id": store.meta.session_id, "meta": store.meta.__dict__}


@app.get("/api/sessions")
def list_sessions() -> dict:
    return {"sessions": [s.meta.__dict__ for s in hub.sessions.values()]}


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> dict:
    store = hub.sessions.get(sid)
    if not store:
        raise HTTPException(404, "unknown session")
    return {"meta": store.meta.__dict__, "status": store.last_state, "windows": store.windows_emitted}


@app.post("/api/sessions/{sid}/stop")
def stop_session(sid: str) -> dict:
    if sid not in hub.sessions:
        raise HTTPException(404, "unknown session")
    return hub.stop(sid)


@app.get("/api/sessions/{sid}/summary")
def get_summary(sid: str) -> dict:
    store = hub.sessions.get(sid)
    if not store:
        path = Path(settings.data_dir) / sid / "summary.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        raise HTTPException(404, "unknown session")
    return store.summary()


@app.post("/api/sessions/{sid}/ingest")
def ingest(sid: str, body: IngestBody) -> dict:
    pipe = hub.pipelines.get(sid)
    if not pipe:
        raise HTTPException(404, "unknown session")
    data = StreamChunk(t=np.asarray(body.t), data=np.asarray(body.data), seq=body.seq)
    status = pipe.submit(data)
    return {"status": status}


@app.websocket("/ws/sessions/{sid}")
async def ws_session(ws: WebSocket, sid: str):
    await ws.accept()
    if sid not in hub.sessions:
        await ws.send_text(json.dumps({"type": "error", "error": "unknown session"}))
        await ws.close()
        return
    hub.ws.setdefault(sid, set()).add(ws)
    try:
        await ws.send_text(json.dumps({"type": "hello", "session_id": sid, "meta": hub.sessions[sid].meta.__dict__}))
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "ingest":
                pipe = hub.pipelines.get(sid)
                if pipe:
                    import numpy as np

                    chunk = StreamChunk(
                        t=np.asarray(msg["t"], dtype=float),
                        data=np.asarray(msg["data"], dtype=float),
                        seq=int(msg.get("seq", 0)),
                    )
                    pipe.submit(chunk)
            elif msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        hub.ws.get(sid, set()).discard(ws)


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")
