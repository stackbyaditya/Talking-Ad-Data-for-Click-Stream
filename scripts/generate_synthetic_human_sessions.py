"""Generate a larger human-only synthetic clickstream session dataset.

The generator is designed for small session-level behavioral datasets where we
need to preserve human-like variability and avoid bot-like regularity. It uses
an empirical Gaussian-copula style sampler over core human features, then
recomputes derived fields under explicit behavioral constraints.

Example:
    python scripts/generate_synthetic_human_sessions.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import uuid
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm, rankdata, skew, wasserstein_distance
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest


LOGGER = logging.getLogger("synthetic_human_sessions")
DEFAULT_INPUT_CANDIDATES = [
    Path("clickstream_advanced.csv"),
    Path("data/processed/clickstream_20260318_235610_advanced.csv"),
]
DEFAULT_OUTPUT = Path("synthetic_human_sessions.csv")
DEFAULT_COMBINED_OUTPUT = Path("reports/human_sessions_with_synthetic_flag.csv")
DEFAULT_SUMMARY_OUTPUT = Path("reports/synthetic_human_validation_summary.json")
DEFAULT_PLOT_DIR = Path("Model1/analysis/human_synthetic")
RANDOM_STATE = 42
BASE_FEATURES = [
    "session_duration_sec",
    "session_click_count",
    "request_interval_mean",
    "request_interval_std",
    "mouse_speed_mean",
    "mouse_speed_std",
    "mouse_path_length",
    "direction_change_count",
    "movement_std",
    "coordinate_entropy",
    "click_interval_entropy",
    "success_rate",
]
PLOT_FEATURES = [
    "session_duration_sec",
    "session_click_count",
    "request_interval_mean",
    "request_interval_std",
    "clicks_per_minute",
    "mouse_speed_mean",
    "mouse_path_length",
    "direction_change_count",
    "movement_std",
    "coordinate_entropy",
    "click_interval_entropy",
    "success_rate",
]
BOT_LABEL_NAMES = {"bot", "advanced_bot", "moderate_bot"}


def configure_logging() -> None:
    """Configure INFO logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=None,
        help="Optional input CSV. Falls back to clickstream_advanced.csv and then the processed clickstream CSV.",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=25,
        help="Expansion factor for synthetic rows relative to the human-safe source rows.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Main synthetic CSV output path.",
    )
    parser.add_argument(
        "--combined-output",
        default=str(DEFAULT_COMBINED_OUTPUT),
        help="Combined original+synthetic CSV with synthetic_flag.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARY_OUTPUT),
        help="Validation summary JSON output path.",
    )
    parser.add_argument(
        "--plot-dir",
        default=str(DEFAULT_PLOT_DIR),
        help="Directory for validation plots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed.",
    )
    return parser.parse_args()


def resolve_input_path(explicit_path: str | None) -> Path:
    """Resolve the input dataset path."""
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Input dataset not found: {path}")
        return path

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find clickstream_advanced.csv or data/processed/clickstream_20260318_235610_advanced.csv."
    )


def load_human_dataset(input_path: Path) -> pd.DataFrame:
    """Load the source dataset and keep only human-safe rows."""
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input dataset is empty.")

    if "label_name" in df.columns:
        keep_mask = ~df["label_name"].astype(str).str.lower().isin(BOT_LABEL_NAMES)
        filtered = df.loc[keep_mask].copy()
    elif "label" in df.columns:
        filtered = df.loc[df["label"].astype(float) <= 0].copy()
    else:
        filtered = df.copy()

    filtered = filtered.drop_duplicates().reset_index(drop=True)
    if len(filtered) < 5:
        raise ValueError(
            f"Human-safe dataset is too small after filtering ({len(filtered)} rows). Need at least 5 rows."
        )

    if "source_click_time" in filtered.columns:
        filtered["source_click_time"] = pd.to_datetime(filtered["source_click_time"], utc=True, errors="coerce")

    LOGGER.info(
        "Loaded %s rows from %s; %s rows kept as human-safe source sessions.",
        len(df),
        input_path,
        len(filtered),
    )
    return filtered


def get_bounds(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Collect per-feature robust bounds from the source dataset."""
    bounds: Dict[str, Dict[str, float]] = {}
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        q_low = float(series.quantile(0.05))
        q_high = float(series.quantile(0.95))
        iqr = float(series.quantile(0.75) - series.quantile(0.25))
        bounds[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "q05": q_low,
            "q95": q_high,
            "iqr": iqr,
        }
    return bounds


def empirical_to_normal_matrix(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Map each numeric feature to Gaussian scores via empirical ranks."""
    z = pd.DataFrame(index=df.index)
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        ranks = rankdata(values, method="average")
        uniform = ranks / (len(values) + 1.0)
        z[column] = norm.ppf(uniform)
    return z


def inverse_empirical_transform(z_values: np.ndarray, reference: pd.Series) -> np.ndarray:
    """Map Gaussian scores back into the empirical feature distribution."""
    uniform = norm.cdf(z_values)
    reference_values = np.sort(pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float))
    quantiles = np.linspace(0.0, 1.0, len(reference_values))
    if len(reference_values) == 1:
        return np.full_like(z_values, reference_values[0], dtype=float)
    return np.interp(uniform, quantiles, reference_values)


def fit_numeric_sampler(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, object]:
    """Fit the empirical copula sampler for core human features."""
    z_df = empirical_to_normal_matrix(df, columns)
    cov = LedoitWolf().fit(z_df[columns]).covariance_
    cov += np.eye(len(columns)) * 1e-6
    return {
        "columns": list(columns),
        "z_df": z_df,
        "cov": cov,
        "means": np.zeros(len(columns), dtype=float),
    }


def sample_core_numeric(
    source_df: pd.DataFrame,
    sampler: Dict[str, object],
    n_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample synthetic core features by blending copula draws with donor rows."""
    columns = sampler["columns"]
    donor_indices = rng.integers(0, len(source_df), size=n_samples)
    donor_z = sampler["z_df"].iloc[donor_indices][columns].to_numpy(dtype=float)
    copula_z = rng.multivariate_normal(mean=sampler["means"], cov=sampler["cov"], size=n_samples)
    noise = rng.multivariate_normal(
        mean=np.zeros(len(columns), dtype=float),
        cov=sampler["cov"] * 0.07,
        size=n_samples,
    )
    blend = rng.uniform(0.55, 0.85, size=(n_samples, 1))
    sampled_z = blend * donor_z + (1.0 - blend) * copula_z + noise

    synthetic = pd.DataFrame(index=range(n_samples))
    for idx, column in enumerate(columns):
        synthetic[column] = inverse_empirical_transform(sampled_z[:, idx], source_df[column])
    synthetic["__donor_index"] = donor_indices
    return synthetic


def clip_positive(value: float, minimum: float = 0.0) -> float:
    """Clip to a minimum positive threshold."""
    return float(max(value, minimum))


def rebuild_session_logic(
    raw_df: pd.DataFrame,
    source_df: pd.DataFrame,
    bounds: Dict[str, Dict[str, float]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Enforce human-like logical relationships and recompute derived base fields."""
    rebuilt_rows: List[Dict[str, float]] = []
    donor_clicks = source_df["session_click_count"].to_numpy(dtype=float)
    duration_multiplier_source = (
        source_df["session_duration_sec"]
        / (
            (
                source_df["request_interval_mean"]
                * np.maximum(source_df["session_click_count"] - 1, 1)
            ).replace(0, np.nan)
        )
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if duration_multiplier_source.empty:
        duration_multiplier_source = pd.Series([1.0, 1.25, 1.5])
    duration_multiplier_source = duration_multiplier_source.clip(
        lower=max(float(duration_multiplier_source.quantile(0.10)), 0.9),
        upper=min(float(duration_multiplier_source.quantile(0.80)), 8.0),
    )
    path_ratio_source = (
        source_df["mouse_path_length"]
        / ((source_df["mouse_speed_mean"] + 1e-6) * (source_df["session_duration_sec"] + 1e-6))
    ).replace([np.inf, -np.inf], np.nan).dropna()
    path_ratio_source = path_ratio_source.clip(
        lower=max(float(path_ratio_source.quantile(0.10)), 0.002),
        upper=min(float(path_ratio_source.quantile(0.90)), 5.0),
    )
    movement_ratio_source = (
        source_df["movement_std"] / (source_df["mouse_path_length"] + 1e-6)
    ).replace([np.inf, -np.inf], np.nan).dropna()
    movement_ratio_source = movement_ratio_source.clip(
        lower=max(float(movement_ratio_source.quantile(0.10)), 0.0002),
        upper=min(float(movement_ratio_source.quantile(0.90)), 0.12),
    )
    path_per_direction_source = (
        source_df["mouse_path_length"] / (source_df["direction_change_count"] + 1.0)
    ).replace([np.inf, -np.inf], np.nan).dropna()
    path_per_direction_source = path_per_direction_source.clip(
        lower=max(float(path_per_direction_source.quantile(0.10)), 10.0),
        upper=min(float(path_per_direction_source.quantile(0.90)), 5000.0),
    )

    for _, row in raw_df.iterrows():
        clicks = int(np.clip(round(row["session_click_count"]), 1, max(int(np.ceil(bounds["session_click_count"]["max"])), 2)))
        donor_click = int(max(round(float(rng.choice(donor_clicks))), 1))
        if rng.random() < 0.3:
            clicks = int(np.clip(round((0.65 * clicks) + (0.35 * donor_click)), 1, bounds["session_click_count"]["max"]))

        mean_interval = clip_positive(row["request_interval_mean"], 0.05)
        std_interval = clip_positive(row["request_interval_std"], 0.0)
        duration = clip_positive(row["session_duration_sec"], 1.0)
        click_entropy = clip_positive(row["click_interval_entropy"], 0.0)
        success_rate = float(np.clip(row["success_rate"], 0.0, 1.0))

        if clicks == 1:
            mean_interval = 0.0
            std_interval = 0.0
            click_entropy = 0.0
            duration = max(duration, rng.uniform(15.0, 0.8 * bounds["session_duration_sec"]["q95"] + 15.0))
        else:
            base_interval = duration / max(clicks - 1, 1)
            sampled_duration_multiplier = float(rng.choice(duration_multiplier_source.to_numpy(dtype=float)))
            mean_interval = 0.65 * mean_interval + 0.35 * base_interval
            mean_interval = clip_positive(mean_interval, 0.1)
            if std_interval < (0.05 * mean_interval):
                std_interval = mean_interval * rng.uniform(0.15, 0.65)
            if rng.random() < 0.25:
                std_interval *= rng.uniform(1.1, 1.7)
            active_window = mean_interval * (clicks - 1)
            target_duration = active_window * sampled_duration_multiplier
            duration = max(
                0.55 * duration + 0.45 * target_duration,
                active_window * 0.9,
                clicks * rng.uniform(2.5, 10.0),
            )
            entropy_floor = min(math.log2(clicks), 0.65)
            entropy_ceiling = min(math.log2(clicks + 1), bounds["click_interval_entropy"]["max"] + 0.2)
            click_entropy = float(np.clip(click_entropy + rng.normal(0.0, 0.08), entropy_floor, entropy_ceiling))

        clicks_per_minute = float(clicks / max(duration / 60.0, 1e-6))
        human_cpm_cap = max(bounds["session_click_count"]["q95"], 1.0) * 4.0
        clicks_per_minute = float(np.clip(clicks_per_minute, 0.05, min(bounds["session_duration_sec"]["max"], human_cpm_cap, 120.0)))
        requests_per_minute = clicks_per_minute * rng.uniform(0.97, 1.03)

        mouse_speed_mean = clip_positive(row["mouse_speed_mean"], 0.05)
        mouse_speed_std = clip_positive(row["mouse_speed_std"], 0.01)
        movement_std = clip_positive(row["movement_std"], 0.01)
        coordinate_entropy = float(np.clip(row["coordinate_entropy"], 0.0, max(bounds["coordinate_entropy"]["max"], 5.0)))
        path_length = clip_positive(row["mouse_path_length"], 1.0)
        direction_changes = int(max(round(row["direction_change_count"]), 0))

        sampled_path_ratio = float(rng.choice(path_ratio_source.to_numpy(dtype=float)))
        target_path = mouse_speed_mean * duration * sampled_path_ratio
        path_length = float(np.clip(
            0.70 * path_length + 0.30 * target_path,
            max(bounds["mouse_path_length"]["min"] * 0.85, 1.0),
            bounds["mouse_path_length"]["max"] * 1.10,
        ))
        sampled_path_per_direction = float(rng.choice(path_per_direction_source.to_numpy(dtype=float)))
        target_directions = max(int(round(path_length / sampled_path_per_direction)) - 1, 0)
        direction_changes = int(round(0.55 * direction_changes + 0.45 * target_directions))
        direction_changes = max(direction_changes, int(round((clicks - 1) * rng.uniform(0.08, 0.55))))
        direction_changes = int(np.clip(direction_changes, 0, max(int(bounds["direction_change_count"]["max"] * 1.05), clicks * 6, 4)))

        if mouse_speed_std < 0.05 * mouse_speed_mean:
            mouse_speed_std = mouse_speed_mean * rng.uniform(0.12, 0.55)
        sampled_movement_ratio = float(rng.choice(movement_ratio_source.to_numpy(dtype=float)))
        target_movement_std = path_length * sampled_movement_ratio
        movement_std = float(np.clip(
            0.65 * movement_std + 0.35 * target_movement_std,
            max(bounds["movement_std"]["min"] * 0.90, 0.01),
            bounds["movement_std"]["max"] * 1.15,
        ))
        coordinate_entropy = float(np.clip(coordinate_entropy + rng.normal(0.0, 0.15), 0.0, 5.0))

        if rng.random() < 0.2 and clicks > 3:
            duration *= rng.uniform(1.05, 1.30)
            std_interval *= rng.uniform(1.05, 1.35)

        success_rate = float(np.clip(success_rate + rng.normal(0.0, 0.04), 0.0, 1.0))

        rebuilt_rows.append(
            {
                "session_duration_sec": duration,
                "session_click_count": clicks,
                "request_interval_mean": mean_interval,
                "request_interval_std": std_interval,
                "mouse_speed_mean": mouse_speed_mean,
                "mouse_speed_std": mouse_speed_std,
                "mouse_path_length": path_length,
                "direction_change_count": direction_changes,
                "movement_std": movement_std,
                "coordinate_entropy": coordinate_entropy,
                "click_interval_entropy": click_entropy,
                "success_rate": success_rate,
                "clicks_per_minute": clicks_per_minute,
                "requests_per_minute": float(np.clip(requests_per_minute, 0.05, 120.0)),
                "burstiness": float(std_interval / max(mean_interval, 1e-6)) if clicks > 1 else 0.0,
            }
        )

    rebuilt = pd.DataFrame(rebuilt_rows)
    return rebuilt


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute advanced derived features from the base fields."""
    output = df.copy()
    output["movement_acceleration"] = output["mouse_speed_std"] / (output["mouse_speed_mean"] + 1e-5)
    output["mouse_direction_entropy"] = output["direction_change_count"] / (output["mouse_path_length"] + 1.0)
    output["click_burst_score"] = output["clicks_per_minute"] / (output["request_interval_mean"] + 1.0)
    output["session_idle_ratio"] = output["request_interval_std"] / (output["session_duration_sec"] + 1.0)
    output["trajectory_smoothness"] = output["mouse_path_length"] / (output["direction_change_count"] + 1.0)
    output["interaction_variability"] = (
        output["mouse_speed_std"] + output["request_interval_std"] + output["click_interval_entropy"]
    ) / 3.0
    output["behavioral_complexity"] = (
        output["movement_std"] + output["coordinate_entropy"] + output["interaction_variability"]
    )
    return output


def add_human_scoring(df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """Generate anomaly and bot-likelihood scores anchored to the human-only source."""
    output = df.copy()
    feature_columns = [
        "clicks_per_minute",
        "request_interval_mean",
        "request_interval_std",
        "movement_std",
        "coordinate_entropy",
        "success_rate",
    ]
    iso = IsolationForest(contamination=min(0.15, max(0.05, 2.0 / len(source_df))), random_state=RANDOM_STATE)
    iso.fit(source_df[feature_columns])
    output["anomaly_score"] = -iso.decision_function(output[feature_columns])

    def normalize(values: pd.Series, reference: pd.Series) -> pd.Series:
        ref_min = float(reference.min())
        ref_max = float(reference.max())
        return ((values - ref_min) / max(ref_max - ref_min, 1e-6)).clip(0.0, 1.0)

    rpm_norm = normalize(output["requests_per_minute"], source_df["requests_per_minute"])
    burst_norm = normalize(output["burstiness"], source_df["burstiness"])
    anomaly_norm = normalize(output["anomaly_score"], source_df["anomaly_score"])
    success_inverse = 1.0 - output["success_rate"]
    output["bot_likelihood_score"] = (
        0.20 * rpm_norm + 0.15 * burst_norm + 0.15 * anomaly_norm + 0.50 * success_inverse
    ).clip(0.0, 0.45)
    return output


def jitter_timestamp(base_time: pd.Timestamp, rng: np.random.Generator) -> pd.Timestamp:
    """Generate a human-like timestamp preserving coarse temporal habits."""
    if pd.isna(base_time):
        base_time = pd.Timestamp("2026-03-01T12:00:00Z")
    day_offset = int(rng.integers(0, 90))
    hour_jitter = int(rng.integers(-2, 3))
    minute_jitter = int(rng.integers(-25, 26))
    second_jitter = int(rng.integers(-40, 41))
    return base_time + pd.Timedelta(days=day_offset, hours=hour_jitter, minutes=minute_jitter, seconds=second_jitter)


def attach_metadata(
    synthetic_base_df: pd.DataFrame,
    source_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Populate categorical fields and timestamps from human donor sessions."""
    donor_indices = synthetic_base_df["__donor_index"].to_numpy(dtype=int)
    donor_df = source_df.iloc[donor_indices].reset_index(drop=True)
    output = pd.DataFrame(index=range(len(synthetic_base_df)))

    output["session_id"] = [f"synthetic_human_{uuid.uuid4().hex[:16]}" for _ in range(len(synthetic_base_df))]
    for column in source_df.columns:
        if column in output.columns or column in BASE_FEATURES:
            continue
        if column in {
            "clicks_per_minute",
            "requests_per_minute",
            "burstiness",
            "movement_acceleration",
            "mouse_direction_entropy",
            "click_burst_score",
            "session_idle_ratio",
            "trajectory_smoothness",
            "interaction_variability",
            "behavioral_complexity",
            "bot_likelihood_score",
            "anomaly_score",
        }:
            continue
        output[column] = donor_df[column].to_numpy()

    if "source_click_time" in donor_df.columns:
        synthetic_times = [jitter_timestamp(ts, rng).isoformat() for ts in donor_df["source_click_time"]]
        output["source_click_time"] = synthetic_times
    if "source_attributed_time" in donor_df.columns:
        output["source_attributed_time"] = donor_df["source_attributed_time"].fillna("not_attributed").to_numpy()

    if "label" in source_df.columns:
        output["label"] = 0
    if "label_name" in source_df.columns:
        output["label_name"] = "human"

    output["synthetic_flag"] = 1
    return output


def deduplicate_numeric_rows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add tiny jitter to remove repeated synthetic sessions."""
    output = df.copy()
    numeric_columns = output.select_dtypes(include=[np.number]).columns.tolist()
    if "synthetic_flag" in numeric_columns:
        numeric_columns.remove("synthetic_flag")
    duplicate_mask = output.duplicated(subset=numeric_columns, keep=False)
    if not duplicate_mask.any():
        return output

    dup_indices = output.index[duplicate_mask].tolist()
    for idx in dup_indices:
        for column in [
            "session_duration_sec",
            "request_interval_mean",
            "request_interval_std",
            "mouse_speed_mean",
            "mouse_speed_std",
            "mouse_path_length",
            "movement_std",
            "coordinate_entropy",
            "click_interval_entropy",
            "success_rate",
        ]:
            if column in output.columns:
                output.at[idx, column] = float(output.at[idx, column]) * rng.uniform(0.985, 1.015)
    return output


def generate_synthetic_dataset(source_df: pd.DataFrame, multiplier: int, seed: int) -> pd.DataFrame:
    """Generate the final synthetic human-only dataset."""
    rng = np.random.default_rng(seed)
    target_rows = max(len(source_df) * multiplier, len(source_df) * 10)
    bounds = get_bounds(source_df, BASE_FEATURES)
    sampler = fit_numeric_sampler(source_df, BASE_FEATURES)
    raw_numeric = sample_core_numeric(source_df, sampler, target_rows, rng)
    rebuilt = rebuild_session_logic(raw_numeric, source_df, bounds, rng)
    rebuilt = add_advanced_features(rebuilt)
    rebuilt = add_human_scoring(rebuilt, source_df)
    metadata = attach_metadata(raw_numeric, source_df, rng)

    synthetic_df = pd.concat([metadata, rebuilt], axis=1)
    synthetic_df = deduplicate_numeric_rows(synthetic_df, rng)

    ordered_columns = list(source_df.columns)
    synthetic_df = synthetic_df.reindex(columns=ordered_columns + ["synthetic_flag"])
    synthetic_df["synthetic_flag"] = synthetic_df["synthetic_flag"].fillna(1).astype(int)
    if "label" in synthetic_df.columns:
        synthetic_df["label"] = 0
    if "label_name" in synthetic_df.columns:
        synthetic_df["label_name"] = "human"
    return synthetic_df


def build_stat_comparison(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Build numeric summary comparisons."""
    comparison: Dict[str, Dict[str, float]] = {}
    for column in columns:
        orig = pd.to_numeric(original_df[column], errors="coerce")
        synth = pd.to_numeric(synthetic_df[column], errors="coerce")
        comparison[column] = {
            "original_mean": float(orig.mean()),
            "synthetic_mean": float(synth.mean()),
            "original_std": float(orig.std(ddof=1)),
            "synthetic_std": float(synth.std(ddof=1)),
            "original_skew": float(skew(orig, bias=False)),
            "synthetic_skew": float(skew(synth, bias=False)),
            "wasserstein_distance": float(wasserstein_distance(orig, synth)),
        }
    return comparison


def save_distribution_plots(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    plot_dir: Path,
) -> None:
    """Save distribution comparison plots for key features."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    axes_flat = axes.ravel()
    for ax, column in zip(axes_flat, PLOT_FEATURES):
        sns.histplot(original_df[column], ax=ax, stat="density", color="#4C78A8", alpha=0.45, bins=14, label="original")
        sns.histplot(synthetic_df[column], ax=ax, stat="density", color="#F58518", alpha=0.35, bins=18, label="synthetic")
        ax.set_title(column)
        ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "feature_distribution_comparison.png", dpi=160)
    plt.close(fig)


def save_correlation_plot(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    plot_dir: Path,
) -> float:
    """Save side-by-side correlation heatmaps and return mean absolute difference."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    corr_features = BASE_FEATURES + ["clicks_per_minute", "burstiness"]
    original_corr = original_df[corr_features].corr()
    synthetic_corr = synthetic_df[corr_features].corr()
    corr_diff = float((original_corr - synthetic_corr).abs().mean().mean())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(original_corr, cmap="coolwarm", center=0.0, ax=axes[0])
    axes[0].set_title("Original Human Correlations")
    sns.heatmap(synthetic_corr, cmap="coolwarm", center=0.0, ax=axes[1])
    axes[1].set_title("Synthetic Human Correlations")
    fig.tight_layout()
    fig.savefig(plot_dir / "correlation_comparison.png", dpi=160)
    plt.close(fig)
    return corr_diff


def build_validation_summary(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    input_path: Path,
    multiplier: int,
    corr_diff: float,
) -> Dict[str, object]:
    """Assemble a validation summary for the generated dataset."""
    stat_comparison = build_stat_comparison(original_df, synthetic_df, PLOT_FEATURES)
    duplicate_rate = float(synthetic_df.duplicated().mean())
    synthetic_numeric = synthetic_df.select_dtypes(include=[np.number]).drop(columns=["synthetic_flag"], errors="ignore")
    zero_variance_columns = [col for col in synthetic_numeric.columns if float(synthetic_numeric[col].std()) == 0.0]
    high_freq_cap = float(synthetic_df["clicks_per_minute"].quantile(0.99))

    return {
        "input_path": str(input_path),
        "source_row_count": int(len(original_df)),
        "synthetic_row_count": int(len(synthetic_df)),
        "expansion_factor": float(len(synthetic_df) / max(len(original_df), 1)),
        "requested_multiplier": int(multiplier),
        "human_only_assumption": "Rows with explicit bot labels were excluded; unlabeled rows were treated as human-safe website sessions.",
        "duplicate_rate": duplicate_rate,
        "mean_absolute_correlation_difference": corr_diff,
        "high_frequency_guardrail_clicks_per_minute_p99": high_freq_cap,
        "zero_variance_columns": zero_variance_columns,
        "stat_comparison": stat_comparison,
    }


def save_outputs(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    summary: Dict[str, object],
    output_path: Path,
    combined_output_path: Path,
    summary_output_path: Path,
) -> None:
    """Save CSV and JSON outputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    synthetic_df.drop(columns=["synthetic_flag"], errors="ignore").to_csv(output_path, index=False)

    original_with_flag = original_df.copy()
    original_with_flag["synthetic_flag"] = 0
    combined = pd.concat([original_with_flag, synthetic_df], ignore_index=True)
    combined.to_csv(combined_output_path, index=False)

    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    """Entry point for synthetic human session generation."""
    configure_logging()
    args = parse_args()
    input_path = resolve_input_path(args.input)
    source_df = load_human_dataset(input_path)
    synthetic_df = generate_synthetic_dataset(source_df, args.multiplier, args.seed)

    plot_dir = Path(args.plot_dir)
    save_distribution_plots(source_df, synthetic_df, plot_dir)
    corr_diff = save_correlation_plot(source_df, synthetic_df, plot_dir)
    summary = build_validation_summary(
        original_df=source_df,
        synthetic_df=synthetic_df,
        input_path=input_path,
        multiplier=args.multiplier,
        corr_diff=corr_diff,
    )
    save_outputs(
        original_df=source_df,
        synthetic_df=synthetic_df,
        summary=summary,
        output_path=Path(args.output),
        combined_output_path=Path(args.combined_output),
        summary_output_path=Path(args.summary_output),
    )

    LOGGER.info(
        "Synthetic human dataset saved to %s with %s rows (source rows: %s).",
        args.output,
        len(synthetic_df),
        len(source_df),
    )


if __name__ == "__main__":
    main()
