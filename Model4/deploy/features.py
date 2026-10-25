"""Serving-time feature extraction for the Model 4 'clean' pipeline.

The base-feature logic is a line-for-line port of
  scripts/convert_clickstream_json_to_training_csv.py  (compute_entropy, wrap_angle,
  compute_event_motion_features, and the feature assembly in build_rows)
so a live session produces the same numbers the model was trained on.

Entry points
  sanitize_events(raw_events)              -> clean list of events (untrusted browser input)
  session_from_events(events, start, end)  -> collector-style dict (temporal / behavior / events)
  evidence_status(session)                 -> (ready, reasons, stats): is there enough data to predict?
  build_model_input(session)               -> 1-row DataFrame in MODEL_COLUMNS order
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

INTERVAL_BINS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0, 3600.0]

BASE_FEATURES = [
    "mouse_speed_mean", "mouse_speed_std", "mouse_path_length", "direction_change_count",
    "movement_std", "coordinate_entropy", "session_duration_sec", "request_interval_mean",
    "request_interval_std", "clicks_per_minute", "burstiness", "click_interval_entropy",
]
DERIVED_FEATURES = [
    "movement_acceleration", "mouse_direction_entropy", "click_burst_score",
    "session_idle_ratio", "trajectory_smoothness", "interaction_variability",
    "behavioral_complexity",
]
MODEL_COLUMNS = BASE_FEATURES + DERIVED_FEATURES  # must equal clean/feature_schema.json column_order

ALLOWED_EVENT_TYPES = {"mousemove", "click", "scroll", "hover_start", "hover_end", "ad_click", "visibility"}

# Minimum evidence before the app shows a verdict (training sessions mostly had far more data).
MIN_SECONDS = 5.0
MIN_MOUSE_EVENTS = 20
MIN_CLICKS_WITHOUT_MOUSE = 3


# ----------------------------------------------------------------------------- port of converter
def compute_entropy(intervals: Sequence[float]) -> float:
    """Discretized entropy of inter-click intervals (seconds)."""
    if not len(intervals):
        return 0.0
    counts, _ = np.histogram(np.asarray(intervals, dtype=float), bins=INTERVAL_BINS)
    if counts.sum() == 0:
        return 0.0
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def compute_event_motion_features(
    events: Sequence[Mapping[str, object]],
    path_length: float,
    active_time_sec: float,
    movement_count: int,
    scroll_count: int,
    click_count: int,
) -> Dict[str, float]:
    points: List[Tuple[float, float, float]] = []
    for event in events:
        ts = event.get("ts")
        x = event.get("x")
        y = event.get("y")
        if ts is None or x is None or y is None:
            continue
        try:
            points.append((float(ts), float(x), float(y)))
        except (TypeError, ValueError):
            continue

    points = sorted(set(points), key=lambda item: item[0])
    step_distances: List[float] = []
    step_speeds: List[float] = []
    step_angles: List[float] = []
    for (ts1, x1, y1), (ts2, x2, y2) in zip(points, points[1:]):
        dt = max((ts2 - ts1) / 1000.0, 0.0)
        if dt <= 0.0:
            continue
        dx = x2 - x1
        dy = y2 - y1
        distance = math.hypot(dx, dy)
        step_distances.append(distance)
        step_speeds.append(distance / dt)
        if distance > 0.0:
            step_angles.append(math.atan2(dy, dx))

    direction_changes = 0
    for angle_1, angle_2 in zip(step_angles, step_angles[1:]):
        if abs(wrap_angle(angle_2 - angle_1)) >= (math.pi / 4.0):
            direction_changes += 1

    if step_speeds:
        mouse_speed_mean = float(np.mean(step_speeds))
        mouse_speed_std = float(np.std(step_speeds))
    else:
        mouse_speed_mean = float(path_length / max(active_time_sec, 1.0))
        variability_factor = 0.2 + 0.6 * min(scroll_count + click_count, movement_count + 1) / max(movement_count + 1, 1)
        mouse_speed_std = float(mouse_speed_mean * variability_factor)

    if step_distances:
        movement_std = float(np.std(step_distances))
    else:
        movement_std = float(path_length / max(movement_count, 1))

    if len(points) >= 2:
        cells = Counter((int(x // 100), int(y // 100)) for _, x, y in points)
        total = sum(cells.values())
        p = np.array([c / total for c in cells.values()], dtype=float)
        coordinate_entropy = float(-(p * np.log2(p)).sum())
    else:
        coordinate_entropy = float(min(5.0, math.log2(max(movement_count + scroll_count + 1, 2)) / 1.5))

    if direction_changes == 0 and movement_count > 1:
        direction_changes = int(max(round((movement_count / 12.0) + (scroll_count * 0.5) + max(click_count - 1, 0) * 0.75), 0))

    return {
        "mouse_speed_mean": max(mouse_speed_mean, 0.0),
        "mouse_speed_std": max(mouse_speed_std, 0.0),
        "direction_change_count": int(max(direction_changes, 0)),
        "movement_std": max(movement_std, 0.0),
        "coordinate_entropy": max(coordinate_entropy, 0.0),
    }


def compute_base_features(session: Mapping[str, Any]) -> Dict[str, float]:
    """The 12 base features from a collector-style session dict (temporal / behavior / events)."""
    temporal = session.get("temporal", {}) or {}
    behavior = session.get("behavior", {}) or {}
    events = session.get("events", [])

    session_duration_sec = float(temporal.get("sessionDuration", 0.0) or 0.0) / 1000.0
    active_time_sec = float(temporal.get("activeTime", 0.0) or 0.0) / 1000.0
    path_length = float(behavior.get("mousePathLength", 0.0) or 0.0)
    click_count = int(temporal.get("clickFrequency", 0) or 0)
    intervals_sec = [
        float(interval) / 1000.0
        for interval in list(temporal.get("clickIntervals", []) or [])
        if interval is not None
    ]
    request_interval_mean = (
        float(np.mean(intervals_sec))
        if intervals_sec
        else float(temporal.get("avgClickInterval", 0.0) or 0.0) / 1000.0
    )
    request_interval_std = float(np.std(intervals_sec)) if intervals_sec else 0.0
    clicks_per_minute = float(
        temporal.get("clicksPerMinute", 0.0)
        or (click_count / max(session_duration_sec / 60.0, 1e-6))
    )
    burstiness = float(request_interval_std / max(request_interval_mean, 1e-6))
    motion = compute_event_motion_features(
        events=events if isinstance(events, list) else [],
        path_length=path_length,
        active_time_sec=max(active_time_sec, session_duration_sec),
        movement_count=int(behavior.get("mouseMovementCount", 0) or 0),
        scroll_count=int(behavior.get("scrollCount", 0) or 0),
        click_count=click_count,
    )
    return {
        "mouse_speed_mean": float(motion["mouse_speed_mean"]),
        "mouse_speed_std": float(motion["mouse_speed_std"]),
        "mouse_path_length": float(path_length),
        "direction_change_count": int(motion["direction_change_count"]),
        "movement_std": float(motion["movement_std"]),
        "coordinate_entropy": float(motion["coordinate_entropy"]),
        "session_duration_sec": float(session_duration_sec),
        "request_interval_mean": float(request_interval_mean),
        "request_interval_std": float(request_interval_std),
        "clicks_per_minute": float(clicks_per_minute),
        "burstiness": float(burstiness),
        "click_interval_entropy": float(compute_entropy(intervals_sec)),
    }


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same formulas as Model4/deploy/train_and_pickle.py (verified equal by verify_features.py)
    and preprocessing/behavioral_feature_engineering.py, but UNSCALED."""
    out = df.copy()
    out["movement_acceleration"] = out["mouse_speed_std"] / (out["mouse_speed_mean"] + 1e-5)
    out["mouse_direction_entropy"] = out["direction_change_count"] / (out["mouse_path_length"] + 1.0)
    out["click_burst_score"] = out["clicks_per_minute"] / (out["request_interval_mean"] + 1.0)
    out["session_idle_ratio"] = out["request_interval_std"] / (out["session_duration_sec"] + 1.0)
    out["trajectory_smoothness"] = out["mouse_path_length"] / (out["direction_change_count"] + 1.0)
    out["interaction_variability"] = (
        out["mouse_speed_std"] + out["request_interval_std"] + out["click_interval_entropy"]
    ) / 3.0
    out["behavioral_complexity"] = (
        out["movement_std"] + out["coordinate_entropy"] + out["interaction_variability"]
    )
    return out


# ----------------------------------------------------------------------------- live sessions
def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def sanitize_events(raw_events: Any, max_events: int = 20000) -> List[Dict[str, Any]]:
    """Validate untrusted browser input: known types, finite numbers, bounded size."""
    if not isinstance(raw_events, list):
        return []
    clean: List[Dict[str, Any]] = []
    for e in raw_events[:max_events]:
        if not isinstance(e, dict) or e.get("type") not in ALLOWED_EVENT_TYPES or not _finite(e.get("ts")):
            continue
        item: Dict[str, Any] = {"type": e["type"], "ts": float(e["ts"])}
        if e["type"] in ("mousemove", "click"):
            if not (_finite(e.get("x")) and _finite(e.get("y"))):
                continue
            if abs(e["x"]) > 100000 or abs(e["y"]) > 100000:
                continue
            item["x"], item["y"] = float(e["x"]), float(e["y"])
        clean.append(item)
    clean.sort(key=lambda ev: ev["ts"])
    return clean


def session_from_events(
    events: Sequence[Mapping[str, Any]],
    session_start_ms: float,
    session_end_ms: float | None = None,
) -> Dict[str, Any]:
    """Rebuild the collector's temporal/behavior aggregates from raw events, server-side,
    so client-computed aggregates never have to be trusted."""
    events = list(events)
    last_ts = max((float(e["ts"]) for e in events), default=session_start_ms)
    end = float(session_end_ms) if session_end_ms is not None else last_ts
    duration_ms = max(end - float(session_start_ms), 0.0)

    moves = [e for e in events if e.get("type") == "mousemove"]
    clicks = sorted(float(e["ts"]) for e in events if e.get("type") == "click")
    scrolls = sum(1 for e in events if e.get("type") == "scroll")

    path = 0.0
    for a, b in zip(moves, moves[1:]):
        path += math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"]))

    intervals = [b - a for a, b in zip(clicks, clicks[1:])]
    clicks_per_minute = (len(clicks) / (duration_ms / 60000.0)) if duration_ms > 0 else 0.0
    return {
        "temporal": {
            "sessionStart": float(session_start_ms),
            "sessionEnd": end,
            "sessionDuration": duration_ms,
            "clickFrequency": len(clicks),
            "clicksPerMinute": clicks_per_minute,
            "clickIntervals": intervals,
            "avgClickInterval": float(np.mean(intervals)) if intervals else 0.0,
            "activeTime": duration_ms,
        },
        "behavior": {
            "mouseMovementCount": len(moves),
            "mousePathLength": path,
            "scrollCount": scrolls,
        },
        "events": events,
    }


def evidence_status(session: Mapping[str, Any]) -> Tuple[bool, List[str], Dict[str, float]]:
    """Is there enough activity for a meaningful verdict? Returns (ready, reasons, stats)."""
    temporal = session.get("temporal", {}) or {}
    behavior = session.get("behavior", {}) or {}
    seconds = float(temporal.get("sessionDuration", 0.0) or 0.0) / 1000.0
    mouse = int(behavior.get("mouseMovementCount", 0) or 0)
    clicks = int(temporal.get("clickFrequency", 0) or 0)
    reasons: List[str] = []
    if seconds < MIN_SECONDS:
        reasons.append(f"need at least {MIN_SECONDS:.0f}s of activity (have {seconds:.1f}s)")
    if mouse < MIN_MOUSE_EVENTS and clicks < MIN_CLICKS_WITHOUT_MOUSE:
        reasons.append(
            f"need {MIN_MOUSE_EVENTS}+ mouse movements or {MIN_CLICKS_WITHOUT_MOUSE}+ clicks "
            f"(have {mouse} movements, {clicks} clicks)"
        )
    return (not reasons), reasons, {"seconds": seconds, "mouse_events": mouse, "clicks": clicks}


def build_model_input(session: Mapping[str, Any]) -> pd.DataFrame:
    """One-row DataFrame, columns in MODEL_COLUMNS order, ready for pipeline.predict_proba."""
    base = compute_base_features(session)
    df = add_derived_features(pd.DataFrame([base]))
    return df[MODEL_COLUMNS].astype(float)
