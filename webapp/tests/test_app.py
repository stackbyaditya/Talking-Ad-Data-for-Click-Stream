"""Run from the repo root:  python -m pytest webapp/tests -q"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "webapp"))
sys.path.insert(0, str(REPO / "Model4" / "deploy"))

MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO / "Model4" / "deploy" / "artifacts" / "clean"))
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "pipeline.joblib").exists(), reason="pickle not found")

import verify_features as V  # noqa: E402  (scenario simulators)
from app import create_app  # noqa: E402

TOKEN = "test-token"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("COLLECT_TOKEN", TOKEN)
    app = create_app(model_dir=MODEL_DIR, store_dir=tmp_path)
    app.config["TESTING"] = True
    return app.test_client(), tmp_path


def payload(events, start, end, **extra):
    return {"sessionStart": start, "sessionEnd": end, "events": events, **extra}


def human_session():
    ev, s, e = V.scenario_human_like(np.random.default_rng(1), 60, 0.3, 5)
    return payload(ev, s, e)


def test_health_and_model(client):
    c, _ = client
    assert c.get("/health").get_json()["status"] == "ok"
    info = c.get("/api/model").get_json()
    assert info["classes"] == ["human", "moderate_bot", "advanced_bot"] and len(info["features"]) == 19
    assert info["model"]["gate"] == {"seconds": 5.0, "mouse_events": 20, "clicks": 3}
    assert c.get("/").status_code == 200


def test_predict_rejects_bad_input(client):
    c, _ = client
    assert c.post("/api/predict", data="nope", content_type="application/json").status_code == 400
    assert c.post("/api/predict", json={"events": []}).status_code == 400
    assert c.post("/api/predict", json={"sessionStart": 1, "sessionEnd": 1e15, "events": []}).status_code == 400


def test_predict_waits_for_evidence(client):
    c, _ = client
    r = c.post("/api/predict", json=payload([], 1_790_000_000_000, 1_790_000_001_000)).get_json()
    assert r["ready"] is False and r["reasons"]


def test_predict_returns_probabilities_and_explanation(client):
    c, _ = client
    r = c.post("/api/predict", json=human_session()).get_json()
    assert r["ready"] is True
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-3
    assert r["verdict"] in ("human", "bot") and len(r["explanation"]) == 5
    assert set(r["features"]) == set(__import__("features").BASE_FEATURES)


def test_hostile_events_are_dropped(client):
    c, _ = client
    junk = [{"type": "mousemove", "ts": "x", "x": 1, "y": 1}, {"type": "evil", "ts": 1},
            {"type": "click", "ts": 5, "x": float("inf"), "y": 1}, "str", None]
    r = c.post("/api/predict", json=payload(junk, 1_790_000_000_000, 1_790_000_030_000)).get_json()
    assert r["ready"] is False


def test_save_rules(client):
    c, tmp = client
    s = human_session()
    assert c.post("/api/save", json={**s, "label": "human"}).status_code == 400            # no consent
    assert c.post("/api/save", json={**s, "label": "bot_linear"}).status_code == 403       # no token
    assert c.post("/api/save", json={**s, "label": "weird", "consent": True}).status_code == 400
    ok = c.post("/api/save", json={**s, "label": "human", "consent": True})
    assert ok.status_code == 200 and ok.get_json()["saved"] is True
    bot = c.post("/api/save", json={**s, "label": "bot_linear"}, headers={"X-Collect-Token": TOKEN})
    assert bot.status_code == 200
    files = list((tmp / "sessions").glob("*.json"))
    assert len(files) == 2
    rec = json.loads(files[0].read_text())
    assert "ip" not in json.dumps(rec).lower() and "user_agent" not in rec


def test_payload_too_large(client):
    c, _ = client
    assert c.post("/api/predict", data="x" * (5 * 1024 * 1024), content_type="application/json").status_code == 413


def test_save_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("SAVE_ENABLED", "0")
    monkeypatch.setenv("COLLECT_TOKEN", TOKEN)
    app = create_app(model_dir=MODEL_DIR, store_dir=tmp_path)
    c = app.test_client()
    s = human_session()
    assert c.post("/api/save", json={**s, "label": "human", "consent": True}).status_code == 403
    assert b'id="consent"' not in c.get("/").data
    assert c.post("/api/predict", json=s).get_json()["ready"] is True
