"""Generate advanced session-level behavioral features for modelling."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.preprocessing import RobustScaler


LOGGER = logging.getLogger("behavioral_feature_engineering")
SOURCE_DATASET = Path("data/processed/final_training_dataset_realistic.csv")
OUTPUT_DATASET = Path("data/processed/final_training_dataset_advanced.csv")
SUMMARY_PATH = Path("reports/behavioral_feature_summary.json")
NEW_FEATURES = [
    "movement_acceleration",
    "mouse_direction_entropy",
    "click_burst_score",
    "session_idle_ratio",
    "trajectory_smoothness",
    "interaction_variability",
    "behavioral_complexity",
]


def configure_logging() -> None:
    """Configure INFO logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_dataset(dataset_path: Path = SOURCE_DATASET) -> pd.DataFrame:
    """Load the realistic training dataset and validate integrity."""
    LOGGER.info("Loading dataset from %s", dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if df.isna().any().any():
        raise ValueError("Source dataset contains missing values.")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        LOGGER.info("Removing %s duplicate rows before feature engineering.", duplicate_count)
        df = df.drop_duplicates().reset_index(drop=True)

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values after deduplication.")
    return df


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute advanced behavioral features from existing mouse and temporal signals."""
    df = df.copy()
    df["movement_acceleration"] = df["mouse_speed_std"] / (df["mouse_speed_mean"] + 1e-5)
    df["mouse_direction_entropy"] = df["direction_change_count"] / (df["mouse_path_length"] + 1.0)
    df["click_burst_score"] = df["clicks_per_minute"] / (df["request_interval_mean"] + 1.0)
    df["session_idle_ratio"] = df["request_interval_std"] / (df["session_duration_sec"] + 1.0)
    df["trajectory_smoothness"] = df["mouse_path_length"] / (df["direction_change_count"] + 1.0)
    df["interaction_variability"] = (
        df["mouse_speed_std"] + df["request_interval_std"] + df["click_interval_entropy"]
    ) / 3.0
    df["behavioral_complexity"] = (
        df["movement_std"] + df["coordinate_entropy"] + df["interaction_variability"]
    )
    return df


def scale_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply RobustScaler to the newly engineered behavioral features."""
    df = df.copy()
    scaler = RobustScaler()
    df[NEW_FEATURES] = scaler.fit_transform(df[NEW_FEATURES])
    return df


def build_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Create summary statistics for the engineered features."""
    feature_stats = {
        feature: {
            "min": float(df[feature].min()),
            "mean": float(df[feature].mean()),
            "median": float(df[feature].median()),
            "max": float(df[feature].max()),
            "std": float(df[feature].std()),
        }
        for feature in NEW_FEATURES
    }
    class_means = (
        df.groupby("label_name")[NEW_FEATURES]
        .mean()
        .round(6)
        .to_dict(orient="index")
    )
    return {
        "input_dataset": str(SOURCE_DATASET),
        "output_dataset": str(OUTPUT_DATASET),
        "row_count": int(len(df)),
        "new_features": NEW_FEATURES,
        "feature_statistics": feature_stats,
        "class_means": class_means,
    }


def save_outputs(df: pd.DataFrame, summary: Dict[str, object]) -> None:
    """Persist the advanced dataset and the behavioral feature summary."""
    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving advanced dataset to %s", OUTPUT_DATASET)
    df.to_csv(OUTPUT_DATASET, index=False)
    LOGGER.info("Saving behavioral feature summary to %s", SUMMARY_PATH)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    """Entry point for upgrading the realistic dataset with new behavioral features."""
    configure_logging()
    df = load_dataset()
    df = add_behavioral_features(df)
    df = scale_new_features(df)
    summary = build_summary(df)
    save_outputs(df, summary)
    LOGGER.info("Behavioral feature engineering complete.")


if __name__ == "__main__":
    main()
