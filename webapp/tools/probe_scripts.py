"""Probe how the REAL pickled model scores variants of the demo bot scripts, without a browser.

Why: the demo scripts in public/static/bots.js should behave like the simulated bot classes the model was trained
on. This tool simulates the events each script variant would produce (a mousemove every ~16 ms like
requestAnimationFrame, clicks on a timer), scores them with your model and prints the class odds, so a variant
can be chosen from evidence instead of guessing.

Run from the repo root, in the environment that has the full xgboost:
    python webapp/tools/probe_scripts.py
    python webapp/tools/probe_scripts.py --runs 20 --csv probe_results.csv

Send the printed table (or the CSV) back and the best variant is ported into bots.js.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Model4" / "deploy"))
import features as F  # noqa: E402

CLASSES = ["human", "moderate_bot", "advanced_bot"]
T0 = 1_790_000_000_000.0
FRAME_MS = 16.7
# the demo page: three ads about 270 px apart; clicks land inside a tile, 20% of its size around the centre
TILES = [(372, 810), (659, 810), (945, 810)]


def tile_point(rng):
    cx, cy = TILES[int(rng.integers(len(TILES)))]
    return cx + rng.uniform(-0.2, 0.2) * 260, cy + rng.uniform(-0.2, 0.2) * 180


# ------------------------------------------------------------------ script variants
def spammer(rng, gap_ms=470, wander_px=0.0, wander_hz=0.0, seconds=14.0):
    """Stays on one tile and clicks every gap_ms. wander_px > 0 adds a slow drift of the cursor around the tile."""
    x0, y0 = tile_point(rng)
    ev = [{"type": "mousemove", "ts": T0, "x": int(x0), "y": int(y0)}]
    t = T0
    next_click = T0 + gap_ms
    phase = rng.uniform(0, 6.28)
    while t - T0 < seconds * 1000:
        t += FRAME_MS
        if wander_px > 0 and wander_hz > 0:
            a = 2 * np.pi * wander_hz * (t - T0) / 1000 + phase
            ev.append({"type": "mousemove", "ts": t, "x": int(x0 + wander_px * np.cos(a)), "y": int(y0 + wander_px * np.sin(a * 0.7))})
        if t >= next_click:
            last = ev[-1]
            ev.append({"type": "click", "ts": t, "x": last["x"], "y": last["y"]})
            next_click += gap_ms
    return ev, T0, t


def patient(rng, move_ms=1400, pause_ms=2400, curve=0.25, jitter_px=0.0, seconds=24.0, click_jitter_ms=0.0):
    """Smooth eased curve to a random tile, click, wait pause_ms, repeat. jitter_px adds per-frame noise."""
    cur = tile_point(rng)
    ev = [{"type": "mousemove", "ts": T0, "x": int(cur[0]), "y": int(cur[1])}]
    t = T0
    while t - T0 < seconds * 1000:
        to = tile_point(rng)
        dx, dy = to[0] - cur[0], to[1] - cur[1]
        dist = np.hypot(dx, dy) or 1.0
        cx = (cur[0] + to[0]) / 2 - dy / dist * dist * curve
        cy = (cur[1] + to[1]) / 2 + dx / dist * dist * curve
        n = max(int(move_ms / FRAME_MS), 3)
        for i in range(1, n + 1):
            u = i / n
            s = u * u * (3 - 2 * u)
            q = 1 - s
            x = q * q * cur[0] + 2 * q * s * cx + s * s * to[0]
            y = q * q * cur[1] + 2 * q * s * cy + s * s * to[1]
            if jitter_px:
                x += rng.normal(0, jitter_px)
                y += rng.normal(0, jitter_px)
            t += FRAME_MS
            ev.append({"type": "mousemove", "ts": t, "x": int(round(x)), "y": int(round(y))})
        ev.append({"type": "click", "ts": t, "x": int(round(to[0])), "y": int(round(to[1]))})
        cur = to
        t += pause_ms + (rng.normal(0, click_jitter_ms) if click_jitter_ms else 0.0)
    return ev, T0, t


VARIANTS = {
    # target: moderate bot
    "spammer: as shipped (still, 470 ms)": ("spammer", dict()),
    "spammer: still, 350 ms": ("spammer", dict(gap_ms=350)),
    "spammer: still, 700 ms": ("spammer", dict(gap_ms=700)),
    "spammer: drift 6 px, 470 ms": ("spammer", dict(wander_px=6, wander_hz=0.4)),
    "spammer: drift 20 px, 470 ms": ("spammer", dict(wander_px=20, wander_hz=0.4)),
    "spammer: drift 60 px, 470 ms": ("spammer", dict(wander_px=60, wander_hz=0.3)),
    "spammer: drift 60 px, 700 ms": ("spammer", dict(wander_px=60, wander_hz=0.3, gap_ms=700)),
    "spammer: drift 120 px, 700 ms, 20 s": ("spammer", dict(wander_px=120, wander_hz=0.25, gap_ms=700, seconds=20.0)),
    # target: advanced bot
    "patient: as shipped (smooth, 2.4 s pause)": ("patient", dict()),
    "patient: jitter 1 px": ("patient", dict(jitter_px=1.0)),
    "patient: jitter 2 px": ("patient", dict(jitter_px=2.0)),
    "patient: jitter 2 px, pause 1.2 s": ("patient", dict(jitter_px=2.0, pause_ms=1200)),
    "patient: jitter 2 px, pause 3 s +-300 ms": ("patient", dict(jitter_px=2.0, pause_ms=3000, click_jitter_ms=300)),
    "patient: jitter 3 px, move 800 ms, pause 1.5 s": ("patient", dict(jitter_px=3.0, move_ms=800, pause_ms=1500)),
    "patient: jitter 3 px, move 800 ms, pause 3 s +-400 ms": ("patient", dict(jitter_px=3.0, move_ms=800, pause_ms=3000, click_jitter_ms=400)),
    "patient: jitter 1 px, move 2200 ms, pause 1 s": ("patient", dict(jitter_px=1.0, move_ms=2200, pause_ms=1000)),
}
RUNNERS = {"spammer": spammer, "patient": patient}


def score(pipe, events, start, end):
    session = F.session_from_events(events, start, end)
    ready, reasons, _ = F.evidence_status(session)
    if not ready:
        return None
    X = F.build_model_input(session)
    return pipe.predict_proba(X)[0], X.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "Model4" / "deploy" / "artifacts" / "clean"))
    ap.add_argument("--runs", type=int, default=15, help="random repeats per variant")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    pipe = joblib.load(Path(args.model_dir) / "pipeline.joblib")
    rng = np.random.default_rng(args.seed)
    rows = []
    for name, (kind, kw) in VARIANTS.items():
        probs, wins = [], []
        for _ in range(args.runs):
            ev, s, e = RUNNERS[kind](rng, **kw)
            out = score(pipe, ev, s, e)
            if out is None:
                continue
            p, _x = out
            probs.append(p)
            wins.append(int(np.argmax(p)))
        if not probs:
            rows.append({"variant": name, "note": "evidence gate not passed"})
            continue
        m = np.mean(probs, axis=0)
        share = np.bincount(wins, minlength=3) / len(wins)
        rows.append({
            "variant": name,
            "human": round(float(m[0]), 3), "moderate": round(float(m[1]), 3), "advanced": round(float(m[2]), 3),
            "wins_human": round(float(share[0]), 2), "wins_moderate": round(float(share[1]), 2), "wins_advanced": round(float(share[2]), 2),
        })
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(df.to_string(index=False))
    print("\nwins_* = share of runs where that class had the highest probability. Targets: spammer -> moderate, patient -> advanced.")
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"saved {args.csv}")


if __name__ == "__main__":
    main()
