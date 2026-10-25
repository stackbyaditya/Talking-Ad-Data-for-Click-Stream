"""Flask app: real-time human/bot prediction from a live browsing session.

Run locally (from the repo root):
    pip install -r webapp/requirements.txt
    python webapp/app.py            # http://127.0.0.1:5000

Environment variables
    MODEL_DIR      folder with pipeline.joblib + feature_schema.json  (default Model4/deploy/artifacts/clean)
    STORE_DIR      where consented / scripted sessions are saved      (default data/collected)
    SAVE_ENABLED   "0" disables /api/save and hides the consent UI (use for a public deployment) (default 1)
    COLLECT_TOKEN  secret required to save bot-labelled sessions (unset => bot saves are rejected)
    RATE_LIMIT     max requests per minute per client IP              (default 120)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from flask import Flask, jsonify, render_template, request

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Model4" / "deploy"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
from storage import SessionStore  # noqa: E402

CLASS_NAMES = ["human", "moderate_bot", "advanced_bot"]
HUMAN_LABEL = "human"
BOT_LABELS = {"bot_naive", "bot_linear", "bot_curved", "bot_clicker", "bot_other"}
MAX_SESSION_MS = 24 * 3600 * 1000
UNCERTAIN_BELOW = 0.70

CAVEATS = [
    "Model v1 was trained on simulated bots and mostly synthetic humans.",
    "Known blind spot: bots that move the mouse far and click slowly can look human.",
    "Treat the verdict as a demo of the pipeline, not as a production fraud decision.",
]

log = logging.getLogger("webapp")


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            if len(self.hits) > 5000:  # crude memory bound
                self.hits = defaultdict(deque, {key: q})
            return True


def create_app(model_dir: str | Path | None = None, store_dir: str | Path | None = None) -> Flask:
    # Static files live in <repo>/public/static: Vercel serves public/** from its CDN,
    # and Flask serves the same files when running locally or in Docker.
    app = Flask(__name__, static_folder=str(REPO_ROOT / "public" / "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

    model_dir = Path(model_dir or os.environ.get("MODEL_DIR") or REPO_ROOT / "Model4" / "deploy" / "artifacts" / "clean")
    pipe = joblib.load(model_dir / "pipeline.joblib")
    schema = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
    if schema["column_order"] != F.MODEL_COLUMNS:  # fail fast on training/serving skew
        raise RuntimeError("feature_schema.json column order differs from features.MODEL_COLUMNS")
    card_path = model_dir / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
    model_info = {
        "variant": card.get("variant", "unknown"),
        "trained_utc": card.get("trained_utc"),
        "cv5_accuracy": card.get("metrics", {}).get("cv5_accuracy_mean"),
        "test_accuracy": card.get("metrics", {}).get("accuracy"),
        "n_features": len(F.MODEL_COLUMNS),
        "versions": card.get("versions", {}),
        "gate": {"seconds": F.MIN_SECONDS, "mouse_events": F.MIN_MOUSE_EVENTS, "clicks": F.MIN_CLICKS_WITHOUT_MOUSE},
    }

    token = os.environ.get("COLLECT_TOKEN", "")
    save_enabled = os.environ.get("SAVE_ENABLED", "1") != "0"
    model_info["save_enabled"] = save_enabled
    # Only touch the disk when saving is on: serverless filesystems (Vercel) are read-only.
    store = SessionStore(store_dir or os.environ.get("STORE_DIR") or REPO_ROOT / "data" / "collected") if save_enabled else None
    limiter = RateLimiter(int(os.environ.get("RATE_LIMIT", "120")))

    # ------------------------------------------------------------------ helpers
    def parse_session(body):
        if not isinstance(body, dict):
            return None, "JSON object expected"
        start = body.get("sessionStart")
        if not F._finite(start):
            return None, "sessionStart (epoch ms) required"
        events = F.sanitize_events(body.get("events"))
        end = body.get("sessionEnd")
        last = max((e["ts"] for e in events), default=float(start))
        end = max(float(end), last) if F._finite(end) else last
        if end - float(start) > MAX_SESSION_MS or end < float(start):
            return None, "implausible session duration"
        return F.session_from_events(events, float(start), end), None

    def explain(X, cls_idx: int, top: int = 5):
        """Per-session TreeSHAP contributions (log-odds) toward the predicted class."""
        Xt = pipe.named_steps["prep"].transform(X)
        contribs = pipe.named_steps["clf"].get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
        c = np.asarray(contribs)[0][cls_idx][:-1]
        order = np.argsort(-np.abs(c))[:top]
        return [
            {"feature": F.MODEL_COLUMNS[i], "value": round(float(X.iloc[0, i]), 4), "contribution": round(float(c[i]), 3)}
            for i in order
        ]

    def client_key() -> str:
        """Rate-limit key. One trusted proxy hop (Vercel, an ALB) appends the real client address
        as the LAST X-Forwarded-For entry; earlier entries are client-supplied and ignored."""
        fwd = request.headers.get("X-Forwarded-For", "").split(",")[-1].strip()
        return fwd or request.remote_addr or "?"

    def too_many():
        return jsonify(error="rate limit exceeded"), 429

    # ------------------------------------------------------------------ routes
    @app.get("/")
    def index():
        return render_template("index.html", model=model_info, caveats=CAVEATS)

    @app.get("/health")
    def health():
        return jsonify(status="ok", model=model_info["variant"], sessions_saved=store.count() if store else 0)

    @app.get("/api/model")
    def model():
        return jsonify(model=model_info, classes=CLASS_NAMES, features=F.MODEL_COLUMNS, caveats=CAVEATS)

    @app.post("/api/predict")
    def predict():
        if not limiter.allow(client_key()):
            return too_many()
        session, err = parse_session(request.get_json(silent=True))
        if err:
            return jsonify(error=err), 400
        ready, reasons, stats = F.evidence_status(session)
        resp = {"ready": ready, "reasons": reasons, "stats": stats, "caveats": CAVEATS}
        if not ready:
            return jsonify(resp)
        t0 = time.perf_counter()
        X = F.build_model_input(session)
        proba = pipe.predict_proba(X)[0]
        cls = int(np.argmax(proba))
        resp.update(
            probabilities={n: round(float(p), 4) for n, p in zip(CLASS_NAMES, proba)},
            predicted=CLASS_NAMES[cls],
            verdict="human" if cls == 0 else "bot",
            confidence=round(float(proba[cls]), 4),
            uncertain=bool(proba[cls] < UNCERTAIN_BELOW),
            features={c: round(float(X.iloc[0][c]), 4) for c in F.BASE_FEATURES},
            explanation=explain(X, cls),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return jsonify(resp)

    @app.post("/api/save")
    def save():
        """Store a labelled session for retraining. Humans must consent; bot labels need COLLECT_TOKEN."""
        if not save_enabled:
            return jsonify(error="saving is disabled on this deployment"), 403
        if not limiter.allow(client_key()):
            return too_many()
        body = request.get_json(silent=True)
        session, err = parse_session(body)
        if err:
            return jsonify(error=err), 400
        label = body.get("label")
        if label == HUMAN_LABEL:
            if body.get("consent") is not True:
                return jsonify(error="consent required to save a human session"), 400
        elif label in BOT_LABELS:
            if not token or request.headers.get("X-Collect-Token") != token:
                return jsonify(error="bot sessions require a valid collect token"), 403
        else:
            return jsonify(error="unknown label"), 400
        ready, reasons, _ = F.evidence_status(session)
        if not ready:
            return jsonify(error="session too short to save", reasons=reasons), 400

        sid = uuid.uuid4().hex
        record = {
            "id": sid,
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "session": {k: session[k] for k in ("temporal", "behavior", "events")},
            "base_features": F.compute_base_features(session),
            "client": {"viewport": body.get("viewport") if isinstance(body.get("viewport"), dict) else None},
        }  # no IP address and no user-agent are stored
        store.save(record)
        return jsonify(saved=True, id=sid, total=store.count())

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="payload too large"), 413

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")))
