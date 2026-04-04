"""Convert deployed website clickstream JSON into a synthetic-like training CSV.

Example:
    python scripts/convert_clickstream_json_to_training_csv.py ^
        --input data/clickstream_20260318_235610.json ^
        --output data/processed/clickstream_20260318_235610_advanced.csv
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


LOGGER = logging.getLogger("clickstream_json_to_training_csv")
REFERENCE_REALISTIC_DATASET = Path("data/processed/final_training_dataset_realistic.csv")
REFERENCE_ADVANCED_DATASET = Path("data/processed/final_training_dataset_advanced.csv")
INTERVAL_BINS = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0, 3600.0], dtype=float)
ANOMALY_FEATURES = [
    "clicks_per_minute",
    "request_interval_mean",
    "request_interval_std",
    "movement_std",
    "coordinate_entropy",
]
RAW_ADVANCED_FEATURES = [
    "movement_acceleration",
    "mouse_direction_entropy",
    "click_burst_score",
    "session_idle_ratio",
    "trajectory_smoothness",
    "interaction_variability",
    "behavioral_complexity",
]
OUTPUT_COLUMNS = [
    "session_id",
    "mouse_speed_mean",
    "mouse_speed_std",
    "mouse_path_length",
    "direction_change_count",
    "movement_std",
    "coordinate_entropy",
    "session_duration_sec",
    "request_interval_mean",
    "request_interval_std",
    "clicks_per_minute",
    "requests_per_minute",
    "success_rate",
    "browser",
    "operating_system",
    "device_type",
    "user_agent",
    "ip_address",
    "country",
    "region",
    "is_proxy",
    "bot_likelihood_score",
    "anomaly_score",
    "label",
    "label_name",
    "session_click_count",
    "burstiness",
    "click_interval_entropy",
    "app",
    "channel",
    "device",
    "os",
    "source_click_time",
    "source_attributed_time",
    "movement_acceleration",
    "mouse_direction_entropy",
    "click_burst_score",
    "session_idle_ratio",
    "trajectory_smoothness",
    "interaction_variability",
    "behavioral_complexity",
]
DEVICE_CODE_MAP = {"desktop": 1, "mobile": 2, "tablet": 3}
OS_CODE_MAP = {
    "unknown": 0,
    "Windows": 1,
    "MacOS": 2,
    "Linux": 3,
    "Android": 4,
    "iOS": 5,
}


def configure_logging() -> None:
    """Configure INFO-level logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the raw clickstream JSON export.")
    parser.add_argument("--output", required=True, help="Path to the output CSV.")
    parser.add_argument(
        "--reference-realistic",
        default=str(REFERENCE_REALISTIC_DATASET),
        help="Reference realistic synthetic dataset used for anomaly scoring and scaling.",
    )
    parser.add_argument(
        "--reference-advanced",
        default=str(REFERENCE_ADVANCED_DATASET),
        help="Reference advanced synthetic dataset used for output schema validation.",
    )
    return parser.parse_args()


def load_json_records(path: Path) -> List[dict]:
    """Load the raw JSON records."""
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected the clickstream export to be a JSON list.")
    records = [row for row in payload if isinstance(row, dict)]
    if not records:
        raise ValueError("The input JSON contains no object records.")
    return records


def normalize_summary_record(record: Mapping[str, object]) -> dict | None:
    """Normalize nested website records into a flat session-summary shape."""
    payload = record.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}

    session_id = record.get("sessionId")
    if not isinstance(session_id, str):
        session_id = payload.get("sessionId")

    temporal = record.get("temporal")
    if not isinstance(temporal, Mapping):
        temporal = payload.get("temporal")

    behavior = record.get("behavior")
    if not isinstance(behavior, Mapping):
        behavior = payload.get("behavior")

    traffic = record.get("traffic")
    if not isinstance(traffic, Mapping):
        traffic = payload.get("traffic")

    events = record.get("events")
    if not isinstance(events, list):
        events = payload.get("events")

    if not (
        isinstance(session_id, str)
        and isinstance(temporal, Mapping)
        and isinstance(behavior, Mapping)
        and isinstance(traffic, Mapping)
    ):
        return None

    device = record.get("device")
    device = device if isinstance(device, Mapping) else {}

    return {
        "sessionId": session_id,
        "temporal": dict(temporal),
        "behavior": dict(behavior),
        "traffic": dict(traffic),
        "events": list(events) if isinstance(events, list) else [],
        "device": dict(device),
        "ipAddress": record.get("ipAddress", "unknown"),
        "geoLocation": record.get("geoLocation", "unknown"),
        "createdAt": record.get("createdAt", record.get("receivedAt")),
        "botType": payload.get("botType", record.get("botType")),
        "rageClicks": payload.get("rageClicks", record.get("rageClicks")),
        "engagementScore": payload.get("engagementScore", record.get("engagementScore")),
    }


def extract_latest_session_rows(records: Sequence[dict]) -> List[dict]:
    """Keep the latest summary row per session."""
    session_rows = []
    for row in records:
        normalized = normalize_summary_record(row)
        if normalized is not None:
            session_rows.append(normalized)
    if not session_rows:
        raise ValueError("No session summary rows were found in the JSON export.")

    latest_by_session: Dict[str, dict] = {}
    for row in session_rows:
        session_id = str(row["sessionId"])
        created_at = pd.to_datetime(row.get("createdAt"), utc=True, errors="coerce")
        session_end = pd.to_datetime(
            row.get("temporal", {}).get("sessionEnd"),
            unit="ms",
            utc=True,
            errors="coerce",
        )
        sort_key = created_at if not pd.isna(created_at) else session_end
        if session_id not in latest_by_session:
            latest_by_session[session_id] = row | {"_sort_key": sort_key}
            continue
        previous_key = latest_by_session[session_id]["_sort_key"]
        if pd.isna(previous_key) or (not pd.isna(sort_key) and sort_key >= previous_key):
            latest_by_session[session_id] = row | {"_sort_key": sort_key}

    latest_rows = [dict(row) for row in latest_by_session.values()]
    for row in latest_rows:
        row.pop("_sort_key", None)
    latest_rows.sort(
        key=lambda row: pd.to_datetime(
            row.get("temporal", {}).get("sessionStart"),
            unit="ms",
            utc=True,
            errors="coerce",
        )
    )
    LOGGER.info("Selected %s latest session rows from %s summary records.", len(latest_rows), len(session_rows))
    return latest_rows


def extract_domain(value: str | None) -> str:
    """Extract a lowercase hostname from a URL-like value."""
    if not value:
        return ""
    parsed = urlparse(value)
    domain = parsed.netloc or parsed.path
    return domain.lower().strip()


def stable_code(value: str, modulo: int, offset: int = 0) -> int:
    """Create a deterministic integer code from a string."""
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) % modulo) + offset


def parse_browser(user_agent: str) -> str:
    """Infer a browser family from the user agent."""
    ua = (user_agent or "").lower()
    if "edg/" in ua:
        return "Edge"
    if "firefox" in ua:
        return "Firefox"
    if "electron" in ua:
        return "Electron"
    if "headlesschrome" in ua:
        return "HeadlessChrome"
    if "chrome" in ua or "chromium" in ua:
        return "Chrome"
    if "safari" in ua:
        return "Safari"
    if "curl" in ua:
        return "curl"
    if "python-requests" in ua:
        return "python-requests"
    return "unknown"


def parse_operating_system(user_agent: str) -> str:
    """Infer an operating system family from the user agent."""
    ua = (user_agent or "").lower()
    if "windows" in ua:
        return "Windows"
    if "iphone" in ua or "ipad" in ua or "ios" in ua:
        return "iOS"
    if "android" in ua:
        return "Android"
    if "mac os" in ua or "macintosh" in ua:
        return "MacOS"
    if "linux" in ua:
        return "Linux"
    return "unknown"


def parse_device_type(user_agent: str) -> str:
    """Infer a device type from the user agent."""
    ua = (user_agent or "").lower()
    if "ipad" in ua or "tablet" in ua:
        return "tablet"
    if "iphone" in ua or "android" in ua or "mobile" in ua:
        return "mobile"
    return "desktop"


def parse_iso_timestamp(timestamp_ms: float | int | None) -> str:
    """Convert a millisecond epoch timestamp into an ISO string."""
    ts = pd.to_datetime(timestamp_ms, unit="ms", utc=True, errors="coerce")
    if pd.isna(ts):
        return "unknown"
    return ts.isoformat()


def is_proxy_ip(value: str) -> int:
    """Treat local/private/reserved IPs as proxied or masked addresses."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return 0
    return int(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def compute_entropy(intervals: Sequence[float]) -> float:
    """Compute discretized entropy for inter-click intervals."""
    if not intervals:
        return 0.0
    counts, _ = np.histogram(np.asarray(intervals, dtype=float), bins=INTERVAL_BINS)
    if counts.sum() == 0:
        return 0.0
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def wrap_angle(angle: float) -> float:
    """Wrap an angle difference into [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def compute_event_motion_features(
    events: Sequence[Mapping[str, object]],
    path_length: float,
    active_time_sec: float,
    movement_count: int,
    scroll_count: int,
    click_count: int,
) -> Dict[str, float]:
    """Compute motion features from raw event coordinates with sensible fallbacks."""
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
        cells = [(int(x // 100), int(y // 100)) for _, x, y in points]
        counts = pd.Series(cells).value_counts(normalize=True)
        coordinate_entropy = float(-(counts * np.log2(counts)).sum())
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


def compute_success_rate(
    temporal: Mapping[str, object],
    behavior: Mapping[str, object],
    traffic: Mapping[str, object],
    bot_type: str | None,
    rage_clicks: int | float | None,
    engagement_score: int | float | None,
) -> float:
    """Build a bounded engagement-based proxy for success_rate."""
    duration_ms = float(temporal.get("sessionDuration", 0.0) or 0.0)
    active_ratio = float(np.clip(float(temporal.get("activeTimeRatio", 0.0) or 0.0), 0.0, 1.0))
    hover_ratio = float(
        np.clip(float(behavior.get("hoverTime", 0.0) or 0.0) / max(duration_ms, 1.0), 0.0, 1.0)
    )
    scroll_signal = float(np.clip(np.log1p(float(behavior.get("scrollDepth", 0.0) or 0.0)) / 7.0, 0.0, 1.0))
    movement_signal = float(
        np.clip(np.log1p(float(behavior.get("mouseMovementCount", 0.0) or 0.0)) / 6.0, 0.0, 1.0)
    )
    ctr_signal = float(np.clip(float(traffic.get("clickThroughRate", 0.0) or 0.0) / 10.0, 0.0, 1.0))
    page_signal = float(np.clip(float(traffic.get("pagesVisited", 0.0) or 0.0) / 3.0, 0.0, 1.0))
    impression_signal = float(np.clip(float(traffic.get("adImpressions", 0.0) or 0.0) / 6.0, 0.0, 1.0))
    engagement_signal = float(
        np.clip(np.log1p(float(engagement_score or 0.0)) / 15.0, 0.0, 1.0)
    )
    success = (
        0.28 * active_ratio
        + 0.12 * hover_ratio
        + 0.10 * scroll_signal
        + 0.10 * movement_signal
        + 0.12 * ctr_signal
        + 0.08 * page_signal
        + 0.05 * impression_signal
        + 0.15 * engagement_signal
    )
    success -= min(float(rage_clicks or 0.0), 6.0) * 0.04
    if bot_type == "human":
        success += 0.12
    elif bot_type == "bot":
        success -= 0.18
    return float(np.clip(success, 0.0, 1.0))


def add_raw_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the unscaled advanced behavioral features."""
    result = df.copy()
    result["movement_acceleration"] = result["mouse_speed_std"] / (result["mouse_speed_mean"] + 1e-5)
    result["mouse_direction_entropy"] = result["direction_change_count"] / (result["mouse_path_length"] + 1.0)
    result["click_burst_score"] = result["clicks_per_minute"] / (result["request_interval_mean"] + 1.0)
    result["session_idle_ratio"] = result["request_interval_std"] / (result["session_duration_sec"] + 1.0)
    result["trajectory_smoothness"] = result["mouse_path_length"] / (result["direction_change_count"] + 1.0)
    result["interaction_variability"] = (
        result["mouse_speed_std"] + result["request_interval_std"] + result["click_interval_entropy"]
    ) / 3.0
    result["behavioral_complexity"] = (
        result["movement_std"] + result["coordinate_entropy"] + result["interaction_variability"]
    )
    return result


def normalize_against_reference(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Min-max normalize values against a reference series."""
    ref_min = float(reference.min())
    ref_max = float(reference.max())
    denom = max(ref_max - ref_min, 1e-6)
    return ((values - ref_min) / denom).clip(0.0, 1.0)


def build_rows(latest_rows: Sequence[dict]) -> pd.DataFrame:
    """Convert latest session summary rows into the target tabular schema."""
    rows: List[Dict[str, object]] = []
    for row in latest_rows:
        temporal = row.get("temporal", {})
        behavior = row.get("behavior", {})
        traffic = row.get("traffic", {})
        events = row.get("events", [])
        device = row.get("device", {})
        device = device if isinstance(device, Mapping) else {}
        user_agent = str(device.get("userAgent") or row.get("userAgent") or "")
        browser = str(device.get("browser") or "").strip() or parse_browser(user_agent)
        operating_system = str(device.get("os") or "").strip() or parse_operating_system(user_agent)
        device_type = str(device.get("deviceType") or "").strip() or parse_device_type(user_agent)
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
        requests_per_minute = clicks_per_minute
        burstiness = float(request_interval_std / max(request_interval_mean, 1e-6))
        motion_features = compute_event_motion_features(
            events=events if isinstance(events, list) else [],
            path_length=path_length,
            active_time_sec=max(active_time_sec, session_duration_sec),
            movement_count=int(behavior.get("mouseMovementCount", 0) or 0),
            scroll_count=int(behavior.get("scrollCount", 0) or 0),
            click_count=click_count,
        )
        success_rate = compute_success_rate(
            temporal=temporal,
            behavior=behavior,
            traffic=traffic,
            bot_type=row.get("botType"),
            rage_clicks=row.get("rageClicks"),
            engagement_score=row.get("engagementScore"),
        )
        referrer_domain = extract_domain(str(traffic.get("referrer", "") or ""))
        landing_domain = extract_domain(str(traffic.get("landingPage", "") or ""))
        os_code = OS_CODE_MAP.get(operating_system, stable_code(operating_system or "unknown", modulo=500, offset=10))
        label = -1
        label_name = "unknown"
        if row.get("botType") == "human":
            label = 0
            label_name = "human"
        elif row.get("botType") == "bot":
            label = 2
            label_name = "advanced_bot"

        rows.append(
            {
                "session_id": str(row.get("sessionId")),
                "mouse_speed_mean": float(motion_features["mouse_speed_mean"]),
                "mouse_speed_std": float(motion_features["mouse_speed_std"]),
                "mouse_path_length": float(path_length),
                "direction_change_count": int(motion_features["direction_change_count"]),
                "movement_std": float(motion_features["movement_std"]),
                "coordinate_entropy": float(motion_features["coordinate_entropy"]),
                "session_duration_sec": float(session_duration_sec),
                "request_interval_mean": float(request_interval_mean),
                "request_interval_std": float(request_interval_std),
                "clicks_per_minute": float(clicks_per_minute),
                "requests_per_minute": float(requests_per_minute),
                "success_rate": float(success_rate),
                "browser": browser,
                "operating_system": operating_system,
                "device_type": device_type,
                "user_agent": user_agent or "unknown",
                "ip_address": str(row.get("ipAddress", "unknown") or "unknown"),
                "country": "unknown",
                "region": "unknown",
                "is_proxy": int(is_proxy_ip(str(row.get("ipAddress", "")))),
                "label": label,
                "label_name": label_name,
                "session_click_count": click_count,
                "burstiness": float(burstiness),
                "click_interval_entropy": float(compute_entropy(intervals_sec)),
                "app": int(stable_code(landing_domain or "unknown_app", modulo=1000, offset=1)),
                "channel": int(stable_code(referrer_domain or "direct", modulo=1000, offset=0)),
                "device": int(DEVICE_CODE_MAP.get(device_type, 0)),
                "os": int(os_code),
                "source_click_time": parse_iso_timestamp(temporal.get("sessionStart")),
                "source_attributed_time": "not_attributed",
            }
        )

    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("No output rows were created from the clickstream JSON.")
    return output


def attach_reference_scores(output_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Add anomaly and bot-likelihood scores using the synthetic reference dataset."""
    enriched = output_df.copy()
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(reference_df[ANOMALY_FEATURES])
    enriched["anomaly_score"] = -iso.decision_function(enriched[ANOMALY_FEATURES])

    rpm_norm = normalize_against_reference(enriched["requests_per_minute"], reference_df["requests_per_minute"])
    burst_norm = normalize_against_reference(enriched["burstiness"], reference_df["burstiness"])
    anomaly_norm = normalize_against_reference(enriched["anomaly_score"], reference_df["anomaly_score"])
    enriched["bot_likelihood_score"] = (
        0.35 * rpm_norm
        + 0.25 * burst_norm
        + 0.25 * anomaly_norm
        + 0.15 * (1.0 - enriched["success_rate"])
    ).clip(0.0, 1.0)
    return enriched


def scale_advanced_features(output_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Scale advanced behavioral features using the synthetic reference fit."""
    reference_raw = add_raw_advanced_features(reference_df)
    scaler = RobustScaler()
    scaler.fit(reference_raw[RAW_ADVANCED_FEATURES])

    transformed = add_raw_advanced_features(output_df)
    transformed[RAW_ADVANCED_FEATURES] = scaler.transform(transformed[RAW_ADVANCED_FEATURES])
    return transformed


def validate_output_columns(output_df: pd.DataFrame, reference_advanced_df: pd.DataFrame) -> None:
    """Ensure the generated CSV matches the advanced synthetic schema."""
    reference_columns = list(reference_advanced_df.columns)
    if reference_columns != OUTPUT_COLUMNS:
        raise ValueError("The hard-coded output column order no longer matches the reference advanced dataset.")
    if list(output_df.columns) != reference_columns:
        raise ValueError("Generated output columns do not match the reference advanced dataset schema.")
    if output_df.isna().any().any():
        raise ValueError("Generated output contains missing values.")


def main() -> None:
    """Run the JSON-to-CSV conversion."""
    configure_logging()
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    reference_realistic_path = Path(args.reference_realistic)
    reference_advanced_path = Path(args.reference_advanced)

    LOGGER.info("Loading clickstream JSON from %s", input_path)
    records = load_json_records(input_path)
    latest_rows = extract_latest_session_rows(records)

    LOGGER.info("Loading reference datasets.")
    reference_realistic_df = pd.read_csv(reference_realistic_path)
    reference_advanced_df = pd.read_csv(reference_advanced_path, nrows=1)

    output_df = build_rows(latest_rows)
    output_df = attach_reference_scores(output_df, reference_realistic_df)
    output_df = scale_advanced_features(output_df, reference_realistic_df)
    output_df = output_df[OUTPUT_COLUMNS].copy()
    validate_output_columns(output_df, reference_advanced_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    LOGGER.info("Saved %s session rows to %s", len(output_df), output_path)


if __name__ == "__main__":
    main()
