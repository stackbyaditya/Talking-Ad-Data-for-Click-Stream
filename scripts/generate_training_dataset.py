"""Generate a balanced synthetic session-level dataset for bot detection.

Example:
    python scripts/generate_training_dataset.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


LOGGER = logging.getLogger("generate_training_dataset")
RANDOM_STATE = 42
TARGET_COUNTS = {0: 5000, 1: 2000, 2: 2000}
LABEL_NAMES = {0: "human", 1: "moderate_bot", 2: "advanced_bot"}
BROWSERS = np.array(["Chrome", "Firefox", "Safari", "Edge"])
DEVICE_TYPES = np.array(["mobile", "desktop", "tablet"])
COUNTRIES = np.array(["US", "IN", "GB", "DE", "CA", "SG"])
REGIONS = np.array(["California", "Karnataka", "England", "Bavaria", "Ontario", "Central"])
HUMAN_UA = np.array(
    [
        "Mozilla/5.0 Chrome/122 Mobile",
        "Mozilla/5.0 Safari/17 iPhone",
        "Mozilla/5.0 Firefox/123 Windows",
        "Mozilla/5.0 Edge/122 MacOS",
    ]
)
BOT_UA = np.array(
    [
        "python-requests/2.31",
        "Go-http-client/1.1",
        "curl/8.5.0",
        "HeadlessChrome/122.0",
    ]
)
REFERENCE_STAT_COLUMNS = [
    "mouse_speed_mean",
    "requests_per_minute",
    "click_interval_entropy",
    "session_duration_sec",
    "success_rate",
]


def configure_logging() -> None:
    """Configure INFO logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def resolve_input_path() -> Path:
    """Resolve the raw training sample path with a repo-local fallback."""
    primary = Path("data/raw/train_sample.csv")
    fallback = Path("datasets/train_sample.csv")
    if primary.exists():
        return primary
    if fallback.exists():
        LOGGER.warning("Input %s missing; falling back to %s", primary, fallback)
        return fallback
    raise FileNotFoundError("Expected data/raw/train_sample.csv or datasets/train_sample.csv.")


def load_source_data(input_path: Path) -> pd.DataFrame:
    """Load the Kaggle source rows without modifying the raw file."""
    LOGGER.info("Loading source data from %s", input_path)
    df = pd.read_csv(input_path, parse_dates=["click_time", "attributed_time"])
    required = ["ip", "app", "device", "os", "channel", "click_time", "is_attributed"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")
    return df


def load_reference_stats() -> Dict[str, Dict[str, float]]:
    """Load summary statistics from the previously generated dataset."""
    reference_path = Path("data/processed/final_training_dataset.csv")
    defaults = {
        "mouse_speed_mean": {"mean": 100.0, "std": 60.0, "min": 5.0, "max": 275.0},
        "requests_per_minute": {"mean": 55.0, "std": 56.0, "min": 1.0, "max": 200.0},
        "click_interval_entropy": {"mean": 1.75, "std": 0.9, "min": 0.0, "max": 3.0},
        "session_duration_sec": {"mean": 44.0, "std": 54.0, "min": 0.15, "max": 302.0},
        "success_rate": {"mean": 0.35, "std": 0.24, "min": 0.0, "max": 1.0},
    }
    if not reference_path.exists():
        LOGGER.warning("Reference dataset %s missing; using built-in baseline stats.", reference_path)
        return defaults

    reference_df = pd.read_csv(reference_path)
    stats = {
        col: {
            "mean": float(reference_df[col].mean()),
            "std": float(reference_df[col].std()),
            "min": float(reference_df[col].min()),
            "max": float(reference_df[col].max()),
        }
        for col in REFERENCE_STAT_COLUMNS
        if col in reference_df.columns
    }
    for col, fallback in defaults.items():
        stats.setdefault(col, fallback)
    LOGGER.info("Loaded reference stats from %s", reference_path)
    return stats


def sample_base_rows(df: pd.DataFrame, target_counts: Dict[int, int], rng: np.random.Generator) -> pd.DataFrame:
    """Sample base Kaggle rows with replacement to support the target session counts."""
    total_sessions = int(sum(target_counts.values()))
    sampled = df.sample(n=total_sessions, replace=len(df) < total_sessions, random_state=RANDOM_STATE).reset_index(drop=True)
    labels: List[int] = []
    for label, count in target_counts.items():
        labels.extend([label] * count)
    sampled["label"] = labels
    sampled["label_name"] = sampled["label"].map(LABEL_NAMES)
    return sampled.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def make_intervals(label: int, session_length: int, rng: np.random.Generator) -> np.ndarray:
    """Create realistic per-click intervals for a session."""
    base_seconds = np.array([0.5, 1, 2, 5, 10, 30, 60], dtype=float)
    if session_length <= 1:
        return np.array([], dtype=float)

    if label == 0:
        expo = rng.exponential(scale=8, size=session_length - 1)
        pauses = rng.choice(base_seconds, size=session_length - 1, p=np.array([0.12, 0.14, 0.18, 0.2, 0.16, 0.12, 0.08]))
        intervals = expo + pauses * rng.uniform(0.2, 1.0, size=session_length - 1)
        pause_mask = rng.random(session_length - 1) < 0.15
        intervals[pause_mask] += rng.uniform(10, 45, size=pause_mask.sum())
    elif label == 1:
        intervals = rng.exponential(scale=4.0, size=session_length - 1)
        intervals += np.abs(rng.normal(loc=0.9, scale=0.5, size=session_length - 1))
        intervals += rng.choice([0.0, 0.5, 1.0, 2.5], size=session_length - 1, p=[0.5, 0.2, 0.2, 0.1])
    else:
        intervals = rng.exponential(scale=2.0, size=session_length - 1)
        intervals += np.abs(rng.normal(loc=0.65, scale=0.35, size=session_length - 1))
        intervals += rng.choice([0.0, 0.25, 0.75, 2.0], size=session_length - 1, p=[0.58, 0.2, 0.14, 0.08])

    return np.clip(intervals, 0.1, 120.0)


def build_session_times(start_time: pd.Timestamp, intervals: np.ndarray) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return the session start and end time from intervals."""
    start_time = pd.Timestamp(start_time)
    duration = float(intervals.sum()) if len(intervals) else 0.0
    end_time = start_time + pd.to_timedelta(duration, unit="s")
    return start_time, end_time


def simulate_behavioral_features(
    label: int,
    session_length: int,
    session_duration_sec: float,
    reference_stats: Dict[str, Dict[str, float]],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Simulate behavioural features that mimic mouse and movement patterns."""
    ref_mouse = reference_stats["mouse_speed_mean"]
    if label == 0:
        mouse_speed_mean = rng.normal(120, 50) + rng.normal(0, 10)
        mouse_speed_std = rng.normal(30, 15)
        direction_change_count = int(rng.poisson(6))
        movement_std = rng.normal(50, 20) + rng.normal(0, 5)
        coordinate_entropy = rng.normal(3.2, 1.2)
    elif label == 1:
        mouse_speed_mean = rng.normal(90, 40)
        mouse_speed_std = rng.normal(20, 12)
        direction_change_count = int(rng.poisson(4))
        movement_std = rng.normal(35, 15) + rng.normal(0, 4)
        coordinate_entropy = rng.normal(2.2, 1.0)
    else:
        mouse_speed_mean = rng.normal(70, 35)
        mouse_speed_std = rng.normal(15, 10)
        direction_change_count = int(rng.poisson(3))
        movement_std = rng.normal(25, 12) + rng.normal(0, 4)
        coordinate_entropy = rng.normal(1.8, 0.8)

    mouse_speed_mean = np.clip(mouse_speed_mean, 5, min(300, ref_mouse["max"] + 30))
    mouse_speed_std = np.clip(mouse_speed_std, 1, 120)
    movement_std = np.clip(movement_std, 1, 150)
    coordinate_entropy = np.clip(coordinate_entropy, 0.1, 5.0)
    path_multiplier = rng.uniform(0.6, 1.4)
    entropy_multiplier = rng.uniform(0.75, 1.25)
    movement_multiplier = rng.uniform(0.7, 1.3)
    duration_factor = max(session_duration_sec, 5.0)
    mouse_path_length = mouse_speed_mean * (duration_factor / 8.0) * path_multiplier + rng.normal(0, 120)
    coordinate_entropy = coordinate_entropy * entropy_multiplier
    movement_std = movement_std * movement_multiplier

    return {
        "mouse_speed_mean": float(np.clip(mouse_speed_mean, 5, 300)),
        "mouse_speed_std": float(np.clip(mouse_speed_std, 0.1, 120)),
        "mouse_path_length": float(np.clip(mouse_path_length, 1.0, 5000.0)),
        "direction_change_count": int(max(direction_change_count, 0)),
        "movement_std": float(np.clip(movement_std, 0.1, 180.0)),
        "coordinate_entropy": float(np.clip(coordinate_entropy, 0.1, 5.0)),
    }


def simulate_network_features(base_row: pd.Series, label: int, rng: np.random.Generator) -> Dict[str, object]:
    """Generate realistic network and device metadata when Kaggle fields are missing."""
    browser = str(rng.choice(BROWSERS, p=[0.5, 0.17, 0.18, 0.15]))
    device_type = str(rng.choice(DEVICE_TYPES, p=[0.62, 0.28, 0.10]))
    operating_system = f"os_{base_row['os']}"
    user_agent = str(rng.choice(HUMAN_UA if label == 0 else BOT_UA))
    ip_address = str(base_row["ip"])
    country = str(rng.choice(COUNTRIES, p=[0.2, 0.35, 0.12, 0.1, 0.13, 0.1]))
    region = str(rng.choice(REGIONS))
    proxy_prob = 0.08 if label == 0 else 0.25 if label == 1 else 0.4
    is_proxy = int(rng.random() < proxy_prob)
    return {
        "browser": browser,
        "operating_system": operating_system,
        "device_type": device_type,
        "user_agent": user_agent,
        "ip_address": ip_address,
        "country": country,
        "region": region,
        "is_proxy": is_proxy,
    }


def generate_sessions(
    base_rows: pd.DataFrame,
    reference_stats: Dict[str, Dict[str, float]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create one synthetic multi-click session from each sampled source row."""
    LOGGER.info("Generating %s synthetic sessions.", len(base_rows))
    rows: List[Dict[str, object]] = []
    ref_duration = reference_stats["session_duration_sec"]
    ref_entropy = reference_stats["click_interval_entropy"]

    for session_idx, base_row in base_rows.iterrows():
        label = int(base_row["label"])
        session_length = int(rng.integers(2, 9))
        intervals = make_intervals(label, session_length, rng)
        interval_mean = float(intervals.mean()) if len(intervals) else 0.0
        interval_std = float(intervals.std()) if len(intervals) else 0.0
        raw_entropy = float(pd.Series(np.round(intervals, 2)).value_counts(normalize=True).pipe(lambda s: -(s * np.log2(s)).sum())) if len(intervals) else 0.0

        if label == 0:
            requests_per_minute = rng.gamma(shape=2, scale=15)
        elif label == 1:
            requests_per_minute = rng.gamma(shape=3, scale=20)
        else:
            requests_per_minute = rng.gamma(shape=5, scale=18)

        target_duration_sec = max((session_length / max(requests_per_minute, 0.1)) * 60.0, 0.5)
        duration_sec = 0.55 * max(float(intervals.sum()), 0.0) + 0.45 * target_duration_sec
        duration_sec *= rng.uniform(0.8, 1.25)

        if label == 0 and rng.random() < rng.uniform(0.10, 0.15):
            requests_per_minute *= rng.uniform(2.0, 5.0)
            raw_entropy -= rng.uniform(0.2, 0.5)

        if label in {1, 2} and rng.random() < 0.15:
            idle_factor = rng.uniform(1.5, 3.0)
            duration_sec *= idle_factor
            interval_mean *= idle_factor

        if label == 0:
            interval_mean += rng.normal(0, 0.3)
        else:
            requests_per_minute += rng.normal(0, 8)
            raw_entropy += rng.normal(0, 0.15)

        duration_sec = float(np.clip(duration_sec, ref_duration["min"], max(ref_duration["max"] * 1.4, 420.0)))
        interval_mean = float(np.clip(interval_mean, 0.05, 180.0))
        interval_std = float(np.clip(interval_std * rng.uniform(0.75, 1.35), 0.0, 120.0))
        requests_per_minute = float(np.clip(requests_per_minute, 0.5, 500.0))
        clicks_per_minute = float(np.clip(requests_per_minute * rng.uniform(0.9, 1.1), 0.5, 500.0))
        click_interval_entropy = float(np.clip(raw_entropy + rng.normal(ref_entropy["mean"] * 0.05, 0.2), 0.0, 5.0))
        burstiness = float(np.clip(interval_std / max(interval_mean, 1e-6), 0.0, 10.0))
        start_time, end_time = build_session_times(base_row["click_time"], np.array([duration_sec], dtype=float))

        if label == 0:
            success_rate = float(rng.beta(4, 2))
        elif label == 1:
            success_rate = float(rng.beta(2, 3))
        else:
            success_rate = float(rng.beta(1, 4))
        success_rate = float(np.clip(0.7 * success_rate + 0.3 * float(base_row.get("is_attributed", 0)), 0.0, 1.0))

        behavioural = simulate_behavioral_features(label, session_length, duration_sec, reference_stats, rng)
        network = simulate_network_features(base_row, label, rng)

        rows.append(
            {
                "session_id": f"session_{session_idx:05d}",
                "session_click_count": int(session_length),
                "session_duration_sec": float(duration_sec),
                "request_interval_mean": float(interval_mean),
                "request_interval_std": float(interval_std),
                "clicks_per_minute": float(clicks_per_minute),
                "requests_per_minute": float(requests_per_minute),
                "success_rate": float(success_rate),
                "burstiness": float(burstiness),
                "click_interval_entropy": float(max(click_interval_entropy, 0.0)),
                "source_click_time": str(pd.Timestamp(base_row["click_time"]).isoformat()),
                "source_attributed_time": "not_attributed"
                if pd.isna(base_row.get("attributed_time"))
                else str(pd.Timestamp(base_row["attributed_time"]).isoformat()),
                "app": int(base_row["app"]),
                "channel": int(base_row["channel"]),
                "device": int(base_row["device"]),
                "os": int(base_row["os"]),
                "label": label,
                "label_name": LABEL_NAMES[label],
            }
            | behavioural
            | network
        )

    return pd.DataFrame(rows)


def add_anomaly_and_bot_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add IsolationForest anomaly scores and the final bot likelihood score."""
    df = df.copy()
    model_features = [
        "clicks_per_minute",
        "request_interval_mean",
        "request_interval_std",
        "movement_std",
        "coordinate_entropy",
    ]
    iso = IsolationForest(contamination=0.1, random_state=RANDOM_STATE)
    iso.fit(df[model_features])
    df["anomaly_score"] = -iso.decision_function(df[model_features])

    rpm_norm = (df["requests_per_minute"] - df["requests_per_minute"].min()) / max(df["requests_per_minute"].max() - df["requests_per_minute"].min(), 1e-6)
    burst_norm = (df["burstiness"] - df["burstiness"].min()) / max(df["burstiness"].max() - df["burstiness"].min(), 1e-6)
    anomaly_norm = (df["anomaly_score"] - df["anomaly_score"].min()) / max(df["anomaly_score"].max() - df["anomaly_score"].min(), 1e-6)
    df["bot_likelihood_score"] = (
        0.35 * rpm_norm
        + 0.25 * burst_norm
        + 0.25 * anomaly_norm
        + 0.15 * (1.0 - df["success_rate"])
    )
    df["bot_likelihood_score"] = df["bot_likelihood_score"].clip(0.0, 1.0)
    return df


def enforce_no_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the final dataset has no missing values."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(float(df[col].median()) if not df[col].dropna().empty else 0.0)
    for col in object_cols:
        df[col] = df[col].fillna("unknown")
    if df.isna().any().any():
        raise ValueError("Final dataset still contains missing values.")
    return df


def save_plot_class_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Save class distribution bar chart."""
    counts = df["label_name"].value_counts().reindex(["human", "moderate_bot", "advanced_bot"])
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", color=["#4C956C", "#F4A259", "#BC4B51"], ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Session Count")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close(fig)


def save_plot_behavioral_overlap(df: pd.DataFrame, output_dir: Path) -> None:
    """Save a scatter plot to inspect feature overlap by class."""
    colors = {"human": "#4C956C", "moderate_bot": "#F4A259", "advanced_bot": "#BC4B51"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for label_name, subset in df.groupby("label_name"):
        sampled = subset.sample(n=min(1200, len(subset)), random_state=RANDOM_STATE)
        ax.scatter(
            sampled["mouse_speed_mean"],
            sampled["requests_per_minute"],
            s=12,
            alpha=0.35,
            c=colors[label_name],
            label=label_name,
        )
    ax.set_title("Behavioural Overlap")
    ax.set_xlabel("mouse_speed_mean")
    ax.set_ylabel("requests_per_minute")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "behavioral_overlap.png", dpi=160)
    plt.close(fig)


def save_plot_feature_correlation(df: pd.DataFrame, output_dir: Path) -> None:
    """Save a correlation heatmap for key numeric features."""
    feature_cols = [
        "mouse_speed_mean",
        "mouse_speed_std",
        "movement_std",
        "coordinate_entropy",
        "session_duration_sec",
        "request_interval_mean",
        "request_interval_std",
        "clicks_per_minute",
        "requests_per_minute",
        "success_rate",
        "bot_likelihood_score",
        "anomaly_score",
    ]
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels(feature_cols)
    ax.set_title("Feature Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_correlation_heatmap.png", dpi=160)
    plt.close(fig)


def build_report(df: pd.DataFrame) -> Dict[str, object]:
    """Build the dataset generation report JSON payload."""
    feature_cols = [
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
        "bot_likelihood_score",
        "anomaly_score",
    ]
    feature_statistics = {
        col: {
            "min": float(df[col].min()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "max": float(df[col].max()),
        }
        for col in feature_cols
    }
    threshold = float(df["anomaly_score"].quantile(0.95))
    outliers = df[df["anomaly_score"] >= threshold]
    return {
        "session_count": int(len(df)),
        "class_distribution": {LABEL_NAMES[k]: int(v) for k, v in df["label"].value_counts().sort_index().to_dict().items()},
        "feature_statistics": feature_statistics,
        "outlier_summary": {
            "anomaly_score_95th_percentile": threshold,
            "outlier_session_count": int(len(outliers)),
            "top_outlier_sessions": outliers.nlargest(10, "anomaly_score")[["session_id", "label_name", "anomaly_score"]].to_dict(orient="records"),
        },
    }


def save_outputs(df: pd.DataFrame, report: Dict[str, object]) -> None:
    """Save the final dataset, report, and analysis plots."""
    processed_dir = Path("data/processed")
    reports_dir = Path("reports")
    analysis_dir = Path("Model1/analysis/dataset_generation/realistic")
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = processed_dir / "final_training_dataset_realistic.csv"
    report_path = reports_dir / "dataset_generation_report_realistic.json"

    LOGGER.info("Saving dataset to %s", dataset_path)
    df.to_csv(dataset_path, index=False)
    LOGGER.info("Saving report to %s", report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    LOGGER.info("Saving analysis plots to %s", analysis_dir)
    save_plot_class_distribution(df, analysis_dir)
    save_plot_behavioral_overlap(df, analysis_dir)
    save_plot_feature_correlation(df, analysis_dir)


def validate_output(df: pd.DataFrame) -> None:
    """Validate class balance, nulls, and key constraints."""
    expected = pd.Series(TARGET_COUNTS).sort_index()
    actual = df["label"].value_counts().sort_index()
    if not actual.equals(expected):
        raise ValueError(f"Class balance mismatch. Expected {expected.to_dict()}, got {actual.to_dict()}.")
    if df.isna().any().any():
        raise ValueError("Final dataset contains missing values.")
    if (df["clicks_per_minute"] > 500).any() or (df["requests_per_minute"] > 500).any():
        raise ValueError("Requests per minute cap violated.")


def main() -> None:
    """Generate the final synthetic training dataset."""
    configure_logging()
    rng = np.random.default_rng(RANDOM_STATE)
    input_path = resolve_input_path()
    source_df = load_source_data(input_path)
    reference_stats = load_reference_stats()
    base_rows = sample_base_rows(source_df, TARGET_COUNTS, rng)
    session_df = generate_sessions(base_rows, reference_stats, rng)
    session_df = add_anomaly_and_bot_scores(session_df)
    session_df = enforce_no_missing(session_df)

    ordered_cols = [
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
    ]
    session_df = session_df[ordered_cols]
    validate_output(session_df)
    report = build_report(session_df)
    save_outputs(session_df, report)
    LOGGER.info("Synthetic dataset generation complete. Sessions: %s", len(session_df))


if __name__ == "__main__":
    main()
