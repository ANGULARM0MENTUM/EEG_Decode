from __future__ import annotations

from fastapi.testclient import TestClient

from eeg_decode.app import app, hub
from eeg_decode.config import settings


def test_health_and_upload_session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "sessions"))
    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        r = client.post("/api/sessions", json={"source": "upload", "accelerated": True})
        assert r.status_code == 200
        sid = r.json()["session_id"]
        r = client.get(f"/api/sessions/{sid}/summary")
        assert r.status_code == 200
        body = r.json()
        assert body["windows"] >= 0
        r = client.post(f"/api/sessions/{sid}/stop")
        assert r.status_code == 200
        assert r.json()["session"]["status"] == "stopped"
    hub.sessions.clear()
