"""Generate fixed-length temporal interaction sequences for CNN-LSTM modelling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("session_sequence_generator")
DATASET_PATH = Path("data/processed/final_training_dataset_advanced.csv")
OUTPUT_PATH = Path("model_outputs/lstm_training_data_v3.npz")
PLOT_PATH = Path("analysis/plots/example_session_sequence.png")
SEQUENCE_LENGTH = 25
SEQUENCE_FEATURES = [
    "mouse_speed_mean",
    "mouse_speed_std",
    "movement_std",
    "coordinate_entropy",
    "clicks_per_minute",
    "requests_per_minute",
    "request_interval_mean",
    "request_interval_std",
    "click_burst_score",
    "behavioral_complexity",
    "interaction_variability",
    "session_idle_ratio",
    "mouse_speed_delta",
    "click_event",
    "pause_event",
]
FEATURE_COUNT = len(SEQUENCE_FEATURES)


def configure_logging() -> None:
    """Configure INFO logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load and validate the advanced session-level dataset."""
    LOGGER.info("Loading advanced dataset from %s", dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if df.isna().any().any():
        raise ValueError("Advanced dataset contains missing values.")
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Advanced dataset contains {duplicate_count} duplicate rows.")
    return df


def simulate_session_sequence(row: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Generate a 25-step temporal interaction sequence from one session summary row."""
    mouse_speed = rng.normal(
        loc=float(row["mouse_speed_mean"]),
        scale=max(float(row["mouse_speed_std"]), 1e-3),
        size=SEQUENCE_LENGTH,
    )
    mouse_speed = np.clip(mouse_speed, 0.0, None)

    request_intervals = rng.normal(
        loc=float(row["request_interval_mean"]),
        scale=max(float(row["request_interval_std"]), 1e-3),
        size=SEQUENCE_LENGTH,
    )
    request_intervals = np.clip(request_intervals, 0.0, None)

    click_probability = float(np.clip(float(row["clicks_per_minute"]) / 60.0, 0.0, 1.0))
    click_events = rng.binomial(1, click_probability, size=SEQUENCE_LENGTH).astype(float)

    percentile_75 = float(np.percentile(request_intervals, 75))
    pause_events = (request_intervals > percentile_75).astype(float)
    mouse_speed_delta = np.diff(mouse_speed, prepend=mouse_speed[0])

    movement_std_seq = np.clip(
        rng.normal(float(row["movement_std"]), max(float(row["movement_std"]) * 0.1, 1e-3), size=SEQUENCE_LENGTH),
        0.0,
        None,
    )
    coordinate_entropy_seq = np.clip(
        rng.normal(float(row["coordinate_entropy"]), 0.15, size=SEQUENCE_LENGTH),
        0.0,
        None,
    )
    clicks_per_minute_seq = np.clip(
        rng.normal(float(row["clicks_per_minute"]), max(float(row["clicks_per_minute"]) * 0.08, 1e-3), size=SEQUENCE_LENGTH),
        0.0,
        None,
    )
    requests_per_minute_seq = np.clip(
        rng.normal(float(row["requests_per_minute"]), max(float(row["requests_per_minute"]) * 0.08, 1e-3), size=SEQUENCE_LENGTH),
        0.0,
        None,
    )
    click_burst_score_seq = np.clip(
        rng.normal(float(row["click_burst_score"]), max(abs(float(row["click_burst_score"])) * 0.12, 0.05), size=SEQUENCE_LENGTH),
        -10.0,
        10.0,
    )
    behavioral_complexity_seq = rng.normal(
        float(row["behavioral_complexity"]),
        max(abs(float(row["behavioral_complexity"])) * 0.08, 0.05),
        size=SEQUENCE_LENGTH,
    )
    interaction_variability_seq = rng.normal(
        float(row["interaction_variability"]),
        max(abs(float(row["interaction_variability"])) * 0.08, 0.05),
        size=SEQUENCE_LENGTH,
    )
    session_idle_ratio_seq = np.clip(
        rng.normal(float(row["session_idle_ratio"]), 0.05, size=SEQUENCE_LENGTH),
        -10.0,
        10.0,
    )

    sequence = np.column_stack(
        [
            mouse_speed,
            np.full(SEQUENCE_LENGTH, float(row["mouse_speed_std"]), dtype=float),
            movement_std_seq,
            coordinate_entropy_seq,
            clicks_per_minute_seq,
            requests_per_minute_seq,
            request_intervals,
            np.full(SEQUENCE_LENGTH, float(row["request_interval_std"]), dtype=float),
            click_burst_score_seq,
            behavioral_complexity_seq,
            interaction_variability_seq,
            session_idle_ratio_seq,
            mouse_speed_delta,
            click_events,
            pause_events,
        ]
    )
    return sequence.astype(np.float32)


def build_sequence_dataset(df: pd.DataFrame, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate the full fixed-length sequence tensor and labels."""
    rng = np.random.default_rng(random_state)
    X_sequences = np.zeros((len(df), SEQUENCE_LENGTH, FEATURE_COUNT), dtype=np.float32)
    y_labels = df["label"].to_numpy(dtype=np.int64)

    for idx, (_, row) in enumerate(df.iterrows()):
        X_sequences[idx] = simulate_session_sequence(row, rng)

    if np.isnan(X_sequences).any():
        raise ValueError("Generated sequence tensor contains NaN values.")

    constant_mask = np.all(np.isclose(X_sequences, X_sequences[:, :1, :]), axis=(1, 2))
    if constant_mask.any():
        raise ValueError(f"Found {int(constant_mask.sum())} constant sequences.")

    label_distribution = df["label"].value_counts().sort_index().to_dict()
    LOGGER.info("Label distribution in sequence dataset: %s", label_distribution)
    return X_sequences, y_labels, SEQUENCE_FEATURES


def save_sequence_dataset(
    X_sequences: np.ndarray,
    y_labels: np.ndarray,
    feature_names: List[str],
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Save the sequence tensor, labels, and feature names."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_sequences=X_sequences,
        y_labels=y_labels,
        feature_names=np.asarray(feature_names, dtype=object),
    )


def plot_example_session(
    X_sequences: np.ndarray,
    output_path: Path = PLOT_PATH,
    session_index: int = 0,
) -> None:
    """Plot one example session's mouse speed, click events, and request intervals."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = X_sequences[session_index]
    timesteps = np.arange(1, session.shape[0] + 1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(timesteps, session[:, 0], color="#4C78A8")
    axes[0].set_ylabel("mouse_speed")
    axes[0].set_title("Example Session Sequence")

    axes[1].step(timesteps, session[:, 13], where="mid", color="#F58518")
    axes[1].set_ylabel("click_event")

    axes[2].plot(timesteps, session[:, 6], color="#54A24B")
    axes[2].set_ylabel("request_interval")
    axes[2].set_xlabel("timestep")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Generate and save the v3 temporal sequence dataset."""
    configure_logging()
    df = load_dataset()
    X_sequences, y_labels, feature_names = build_sequence_dataset(df)
    save_sequence_dataset(X_sequences, y_labels, feature_names)
    plot_example_session(X_sequences)
    print("Sequence dataset shape:", X_sequences.shape)


if __name__ == "__main__":
    main()
