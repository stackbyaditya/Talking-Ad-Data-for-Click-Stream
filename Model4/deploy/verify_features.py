"""Verify serving-time features against training, and probe how the clean model treats
realistic live-session scenarios.

Run from the repo root (read-only; writes nothing):
    python Model4/deploy/verify_features.py

Checks
  1. add_derived_features here == the one used in train_and_pickle.py
  2. MODEL_COLUMNS == clean/feature_schema.json column_order
  3. PARITY: features.py recomputes the 12 base features from the raw collector JSON and must
     match the training CSV for the real human sessions
  4. RECONSTRUCTION: rebuilding temporal/behavior aggregates from raw events (what the live app
     does) vs the collector's own aggregates
  5. SCENARIOS: hand-built human-like / bot-like sessions -> model probabilities
     (my own simulations, indicative only, NOT ground truth)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as F  # noqa: E402
import train_and_pickle as T  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)


def load_raw_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("records", "data", "items", "sessions"):
                if isinstance(obj.get(key), list):
                    return obj[key]
            return [obj]
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def latest_per_session(records: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for rec in records:
        payload = rec.get("payload", rec)
        sid = str(payload.get("sessionId"))
        sess = {
            "sessionId": sid,
            "temporal": payload.get("temporal", {}),
            "behavior": payload.get("behavior", {}),
            "events": payload.get("events", []),
            "_received": float(rec.get("receivedAt", 0) or 0),
        }
        if sid not in best or sess["_received"] >= best[sid]["_received"]:
            best[sid] = sess
    return best


# ----------------------------------------------------------------------------- scenarios
BASE_TS = 1_790_000_000_000.0


def _segment(rng, t0_ms, p0, p1, dur_s, hz=75, curved=True, jitter=1.2):
    n = max(int(dur_s * hz), 3)
    u = np.linspace(0, 1, n)
    s = u * u * (3 - 2 * u) if curved else u
    c = ((p0[0] + p1[0]) / 2 + (rng.normal(0, 70) if curved else 0),
         (p0[1] + p1[1]) / 2 + (rng.normal(0, 70) if curved else 0))
    x = (1 - s) ** 2 * p0[0] + 2 * (1 - s) * s * c[0] + s ** 2 * p1[0]
    y = (1 - s) ** 2 * p0[1] + 2 * (1 - s) * s * c[1] + s ** 2 * p1[1]
    if curved:
        x = x + rng.normal(0, jitter, n)
        y = y + rng.normal(0, jitter, n)
    dts = rng.uniform(0.7, 1.3, n) if curved else np.ones(n)
    ts = t0_ms + np.cumsum(dts / dts.sum()) * dur_s * 1000.0
    ev = [{"type": "mousemove", "ts": float(t), "x": int(round(a)), "y": int(round(b))} for t, a, b in zip(ts, x, y)]
    return ev, (int(round(x[-1])), int(round(y[-1]))), float(ts[-1])


def scenario_human_like(rng, duration_s, active_frac, n_clicks):
    events, pos, t = [], (int(rng.integers(100, 900)), int(rng.integers(100, 600))), BASE_TS
    move_total = duration_s * active_frac
    seg_len = move_total / n_clicks
    idle = (duration_s - move_total) / n_clicks
    for _ in range(n_clicks):
        t += idle * 1000.0 * rng.lognormal(0, 0.4)
        target = (int(rng.integers(50, 1200)), int(rng.integers(50, 700)))
        ev, pos, t = _segment(rng, t, pos, target, seg_len * rng.lognormal(0, 0.3), curved=True)
        events += ev
        t += rng.uniform(80, 250)
        events.append({"type": "click", "ts": t, "x": pos[0], "y": pos[1]})
    return events, BASE_TS, max(BASE_TS + duration_s * 1000.0, t)


def scenario_linear_bot(rng, duration_s=30, step_s=2.0):
    events, pos, t = [], (100, 100), BASE_TS
    while t - BASE_TS < duration_s * 1000.0:
        target = (int(rng.integers(50, 1200)), int(rng.integers(50, 700)))
        ev, pos, t = _segment(rng, t, pos, target, 0.25, hz=100, curved=False)
        events += ev
        events.append({"type": "click", "ts": t + 5, "x": pos[0], "y": pos[1]})
        t += step_s * 1000.0
    return events, BASE_TS, t


def scenario_clicker_bot(rng, n=40, gap_s=0.5):
    events = [{"type": "click", "ts": BASE_TS + i * gap_s * 1000.0 + rng.normal(0, 8), "x": 500, "y": 300} for i in range(n)]
    return events, BASE_TS, BASE_TS + n * gap_s * 1000.0


def scenario_advanced_bot(rng, duration_s=30, n_clicks=8, gap_s=3.0):
    events, pos, t = [], (200, 200), BASE_TS
    for _ in range(n_clicks):
        target = (int(rng.integers(50, 1200)), int(rng.integers(50, 700)))
        ev, pos, t_end = _segment(rng, t, pos, target, 0.8, curved=True)
        events += ev
        events.append({"type": "click", "ts": t_end + 5, "x": pos[0], "y": pos[1]})
        t = t_end + gap_s * 1000.0 + rng.normal(0, 40)
    return events, BASE_TS, max(BASE_TS + duration_s * 1000.0, t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/clickstream_20260318_235610.json")
    ap.add_argument("--csv", default=T.DEFAULT_DATASET)
    ap.add_argument("--artifacts", default="Model4/deploy/artifacts/clean")
    args = ap.parse_args()
    ok_all = True

    # 1 -------------------------------------------------------------------------------
    print("\n=== 1. derived-feature formulas identical to training script ===")
    rng = np.random.default_rng(0)
    probe = pd.DataFrame({c: np.abs(rng.normal(10, 5, 50)) for c in F.BASE_FEATURES})
    a, b = F.add_derived_features(probe), T.add_derived_features(probe)
    same = all(np.allclose(a[c], b[c]) for c in F.DERIVED_FEATURES)
    print("PASS" if same else "FAIL"); ok_all &= same

    # 2 -------------------------------------------------------------------------------
    print("\n=== 2. MODEL_COLUMNS == feature_schema.json column_order ===")
    schema = json.loads((Path(args.artifacts) / "feature_schema.json").read_text(encoding="utf-8"))
    same = schema["column_order"] == F.MODEL_COLUMNS
    print("PASS" if same else f"FAIL\nschema: {schema['column_order']}\nfeatures.py: {F.MODEL_COLUMNS}"); ok_all &= same

    # 3 & 4 ---------------------------------------------------------------------------
    raw_path, csv_path = Path(args.raw), Path(args.csv)
    if raw_path.exists() and csv_path.exists():
        sessions = latest_per_session(load_raw_records(raw_path))
        df = pd.read_csv(csv_path).drop_duplicates()
        df = df[df["session_id"].isin(sessions.keys())]
        print(f"\n=== 3. PARITY vs training CSV: {len(sessions)} raw sessions, {len(df)} matched in CSV ===")
        if df.empty:
            print("No session_id matched. Sample raw ids:", list(sessions)[:3],
                  "| sample csv ids:", pd.read_csv(csv_path)["session_id"].head(3).tolist())
        rows = []
        for _, r in df.iterrows():
            got = F.compute_base_features(sessions[r["session_id"]])
            bad = {k: (round(got[k], 6), round(float(r[k]), 6)) for k in F.BASE_FEATURES
                   if not np.isclose(got[k], float(r[k]), rtol=1e-6, atol=1e-6)}
            rows.append((r["session_id"], bad))
        n_bad = sum(1 for _, bad in rows if bad)
        for sid, bad in rows:
            if bad:
                print(f"  MISMATCH {sid}: {bad}   (feature: (features.py, csv))")
        print(f"{len(rows) - n_bad}/{len(rows)} sessions match on all 12 base features")
        ok_all &= (n_bad == 0 and len(rows) > 0)

        print("\n=== 4. RECONSTRUCTION from raw events (live-app path) vs collector aggregates ===")
        out = []
        for sid, sess in sessions.items():
            ev = F.sanitize_events(sess["events"])
            t = sess["temporal"]
            if not ev or "sessionStart" not in t:
                continue
            rebuilt = F.session_from_events(ev, t["sessionStart"], t.get("sessionEnd"))
            f_new, f_old = F.compute_base_features(rebuilt), F.compute_base_features(sess)
            rel = {k: abs(f_new[k] - f_old[k]) / (abs(f_old[k]) + 1e-9) for k in F.BASE_FEATURES}
            out.append({"session": sid[:8], **{k: round(v, 3) for k, v in rel.items() if v > 0.02}})
        print(pd.DataFrame(out).fillna("").to_string(index=False) if out else "no sessions with events")
        print("(columns show relative difference > 2% only; blank = within 2%)")
    else:
        print(f"\n=== 3/4 skipped: missing {raw_path if not raw_path.exists() else csv_path} ===")

    # 5 -------------------------------------------------------------------------------
    pipe = joblib.load(Path(args.artifacts) / "pipeline.joblib")
    print("\n=== 5. SCENARIO PROBE (simulated by me; indicative only) ===")
    rng = np.random.default_rng(7)
    scen = {
        "human_like_45s_5clicks": scenario_human_like(rng, 45, 0.35, 5),
        "human_like_120s_8clicks": scenario_human_like(rng, 120, 0.25, 8),
        "human_like_20min_idle_6clicks": scenario_human_like(rng, 1200, 0.03, 6),
        "linear_bot_30s": scenario_linear_bot(rng),
        "clicker_bot_no_mouse_20s": scenario_clicker_bot(rng),
        "advanced_bot_curved_metronomic": scenario_advanced_bot(rng),
    }
    rows = []
    for name, (events, start, end) in scen.items():
        sess = F.session_from_events(F.sanitize_events(events), start, end)
        ready, reasons, stats = F.evidence_status(sess)
        base = F.compute_base_features(sess)
        p = pipe.predict_proba(F.build_model_input(sess))[0]
        rows.append({
            "scenario": name, "ready": ready, "secs": round(stats["seconds"], 1),
            "moves": stats["mouse_events"], "clicks": stats["clicks"],
            "path_px": round(base["mouse_path_length"]), "cpm": round(base["clicks_per_minute"], 1),
            "P_human": round(float(p[0]), 3), "P_mod": round(float(p[1]), 3), "P_adv": round(float(p[2]), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nOverall checks 1-3:", "PASS" if ok_all else "FAIL (see above)")


if __name__ == "__main__":
    main()
