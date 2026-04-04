"""Generate fixed-length temporal interaction sequences for CNN-LSTM modelling.

v4 improvements over v3:
- SEQUENCE_LENGTH increased from 25 → 50 for richer temporal context.
- Class-specific temporal arc injection (human warm-up/active/cooldown,
  moderate-bot flat+burst, advanced-bot perfectly-flat).
- AR(1) autocorrelation on mouse_speed and request_intervals so consecutive
  timesteps are correlated instead of i.i.d.
- Wider class-separating noise budgets on click_burst_score and
  session_idle_ratio.
- Feature-interaction channel: burst_x_complexity added as channel 15.
- Output saved to lstm_training_data_v4.npz to keep v3 intact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("session_sequence_generator")
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "processed" / "final_training_dataset_advanced.csv"
OUTPUT_PATH = MODEL_ROOT / "outputs" / "lstm_training_data_v4.npz"
PLOT_PATH = MODEL_ROOT / "analysis" / "plots" / "example_session_sequence_v4.png"

SEQUENCE_LENGTH = 50          # ↑ from 25 — more temporal context per session
AR1_ALPHA = 0.55              # autocorrelation coefficient for AR(1) smoothing

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
    "burst_x_complexity",      # NEW: interaction feature
]
FEATURE_COUNT = len(SEQUENCE_FEATURES)


# ---------------------------------------------------------------------------
# Temporal arc helpers
# ---------------------------------------------------------------------------

def _human_arc(length: int) -> np.ndarray:
    """Warm-up → active → cool-down envelope scaled to [0.7, 1.3]."""
    t = np.linspace(0, np.pi, length)
    return 0.7 + 0.6 * np.sin(t)


def _moderate_bot_arc(length: int, rng: np.random.Generator) -> np.ndarray:
    """Flat baseline with 1-2 random burst windows."""
    arc = np.ones(length)
    n_bursts = rng.integers(1, 3)
    for _ in range(n_bursts):
        start = rng.integers(0, length - 5)
        width = rng.integers(3, 8)
        arc[start: start + width] += rng.uniform(0.4, 0.9)
    return arc


def _advanced_bot_arc(length: int) -> np.ndarray:
    """Near-perfectly flat — the hallmark of scripted traffic."""
    return np.ones(length)


def _get_arc(label: int, length: int, rng: np.random.Generator) -> np.ndarray:
    if label == 0:
        return _human_arc(length)
    elif label == 1:
        return _moderate_bot_arc(length, rng)
    else:
        return _advanced_bot_arc(length)


# ---------------------------------------------------------------------------
# AR(1) signal generator
# ---------------------------------------------------------------------------

def _ar1_signal(
    mean: float,
    std: float,
    length: int,
    rng: np.random.Generator,
    alpha: float = AR1_ALPHA,
    clip_low: float = 0.0,
    clip_high: float = np.inf,
) -> np.ndarray:
    """Generate an AR(1) correlated signal: x_t = alpha*x_{t-1} + (1-alpha)*mean + eps."""
    signal = np.empty(length, dtype=np.float64)
    signal[0] = np.clip(rng.normal(mean, max(std, 1e-3)), clip_low, clip_high)
    noise_std = max(std, 1e-3) * np.sqrt(1 - alpha ** 2)
    for t in range(1, length):
        signal[t] = alpha * signal[t - 1] + (1 - alpha) * mean + rng.normal(0.0, noise_std)
        signal[t] = np.clip(signal[t], clip_low, clip_high)
    return signal.astype(np.float32)


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    LOGGER.info("Loading advanced dataset from %s", dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    df = pd.read_csv(dataset_path)
    if df.isna().any().any():
        raise ValueError("Advanced dataset contains missing values.")
    if df.duplicated().any():
        raise ValueError("Advanced dataset contains duplicate rows.")
    return df


# ---------------------------------------------------------------------------
# Core sequence simulator (v4)
# ---------------------------------------------------------------------------

def simulate_session_sequence(row: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Generate a SEQUENCE_LENGTH-step correlated temporal sequence.

    Key changes vs v3
    -----------------
    1. AR(1) autocorrelation on mouse_speed and request_intervals.
    2. Class-specific temporal arc modulates mouse_speed amplitude.
    3. Wider noise budgets on click_burst_score / session_idle_ratio to
       increase class separation in the feature space seen by the DL models.
    4. New interaction feature: burst_x_complexity.
    """
    label = int(row["label"])
    arc = _get_arc(label, SEQUENCE_LENGTH, rng)

    # ── mouse speed (AR1 + arc) ──────────────────────────────────────────
    mouse_speed_base = _ar1_signal(
        mean=float(row["mouse_speed_mean"]),
        std=float(row["mouse_speed_std"]),
        length=SEQUENCE_LENGTH,
        rng=rng,
        clip_low=0.0,
    )
    mouse_speed = mouse_speed_base * arc

    # ── request intervals (AR1) ───────────────────────────────────────────
    request_intervals = _ar1_signal(
        mean=float(row["request_interval_mean"]),
        std=float(row["request_interval_std"]),
        length=SEQUENCE_LENGTH,
        rng=rng,
        clip_low=0.0,
    )

    # ── binary events ────────────────────────────────────────────────────
    click_probability = float(np.clip(float(row["clicks_per_minute"]) / 60.0, 0.0, 1.0))
    click_events = rng.binomial(1, click_probability, size=SEQUENCE_LENGTH).astype(np.float32)

    percentile_75 = float(np.percentile(request_intervals, 75))
    pause_events = (request_intervals > percentile_75).astype(np.float32)

    # ── delta ─────────────────────────────────────────────────────────────
    mouse_speed_delta = np.diff(mouse_speed, prepend=mouse_speed[0]).astype(np.float32)

    # ── movement_std (AR1, class-scaled noise) ────────────────────────────
    movement_noise = {0: 0.25, 1: 0.10, 2: 0.04}[label]
    movement_std_seq = _ar1_signal(
        mean=float(row["movement_std"]),
        std=max(float(row["movement_std"]) * movement_noise, 1e-3),
        length=SEQUENCE_LENGTH,
        rng=rng,
        clip_low=0.0,
    )

    # ── coordinate_entropy (AR1) ──────────────────────────────────────────
    coordinate_entropy_seq = _ar1_signal(
        mean=float(row["coordinate_entropy"]),
        std=0.20,
        length=SEQUENCE_LENGTH,
        rng=rng,
        clip_low=0.0,
    )

    # ── click/request rate (AR1) ──────────────────────────────────────────
    clicks_per_minute_seq = _ar1_signal(
        mean=float(row["clicks_per_minute"]),
        std=max(float(row["clicks_per_minute"]) * 0.10, 1e-3),
        length=SEQUENCE_LENGTH,
        rng=rng,
        clip_low=0.0,
    )
    requests_per_minute_seq = _ar1_signal(
        mean=float(row["requests_per_minute"]),
        std=max(float(row["requests_per_minute"]) * 0.10, 1e-3),
        length=SEQUENCE_LENGTH,
        rng=rng,
        clip_low=0.0,
    )

    # ── click_burst_score — WIDER noise budget per class ─────────────────
    burst_noise = {0: 0.30, 1: 0.18, 2: 0.08}[label]
    click_burst_score_seq = _ar1_signal(
        mean=float(row["click_burst_score"]),
        std=max(abs(float(row["click_burst_score"])) * burst_noise, 0.08),
        length=SEQUENCE_LENGTH,
        rng=rng,
        clip_low=-10.0,
        clip_high=10.0,
    )

    # ── behavioral_complexity (AR1) ───────────────────────────────────────
    behavioral_complexity_seq = _ar1_signal(
        mean=float(row["behavioral_complexity"]),
        std=max(abs(float(row["behavioral_complexity"])) * 0.10, 0.05),
        length=SEQUENCE_LENGTH,
        rng=rng,
    )

    # ── interaction_variability (AR1) ─────────────────────────────────────
    interaction_variability_seq = _ar1_signal(
        mean=float(row["interaction_variability"]),
        std=max(abs(float(row["interaction_variability"])) * 0.10, 0.05),
        length=SEQUENCE_LENGTH,
        rng=rng,
    )

    # ── session_idle_ratio — WIDER noise budget per class ────────────────
    idle_noise = {0: 0.20, 1: 0.10, 2: 0.04}[label]
    session_idle_ratio_seq = _ar1_signal(
        mean=float(row["session_idle_ratio"]),
        std=max(abs(float(row["session_idle_ratio"])) * idle_noise, 0.06),
        length=SEQUENCE_LENGTH,
        rng=rng,
        clip_low=-10.0,
        clip_high=10.0,
    )

    # ── NEW interaction feature ──────────────────────────────────────────
    burst_x_complexity = (click_burst_score_seq * behavioral_complexity_seq).astype(np.float32)

    sequence = np.column_stack([
        mouse_speed,
        np.full(SEQUENCE_LENGTH, float(row["mouse_speed_std"]), dtype=np.float32),
        movement_std_seq,
        coordinate_entropy_seq,
        clicks_per_minute_seq,
        requests_per_minute_seq,
        request_intervals,
        np.full(SEQUENCE_LENGTH, float(row["request_interval_std"]), dtype=np.float32),
        click_burst_score_seq,
        behavioral_complexity_seq,
        interaction_variability_seq,
        session_idle_ratio_seq,
        mouse_speed_delta,
        click_events,
        pause_events,
        burst_x_complexity,
    ])
    return sequence.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_sequence_dataset(
    df: pd.DataFrame, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
    LOGGER.info("Label distribution: %s", label_distribution)
    LOGGER.info("Sequence tensor shape: %s", X_sequences.shape)
    return X_sequences, y_labels, SEQUENCE_FEATURES


def save_sequence_dataset(
    X_sequences: np.ndarray,
    y_labels: np.ndarray,
    feature_names: List[str],
    output_path: Path = OUTPUT_PATH,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_sequences=X_sequences,
        y_labels=y_labels,
        feature_names=np.asarray(feature_names, dtype=object),
    )
    LOGGER.info("Saved sequence dataset to %s", output_path)


def plot_example_session(
    X_sequences: np.ndarray,
    output_path: Path = PLOT_PATH,
    session_index: int = 0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = X_sequences[session_index]
    timesteps = np.arange(1, session.shape[0] + 1)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(timesteps, session[:, 0], color="#4C78A8")
    axes[0].set_ylabel("mouse_speed")
    axes[0].set_title("Example Session Sequence (v4 — AR1 + arc)")

    axes[1].step(timesteps, session[:, 13], where="mid", color="#F58518")
    axes[1].set_ylabel("click_event")

    axes[2].plot(timesteps, session[:, 6], color="#54A24B")
    axes[2].set_ylabel("request_interval")

    axes[3].plot(timesteps, session[:, 15], color="#B279A2")
    axes[3].set_ylabel("burst_x_complexity")
    axes[3].set_xlabel("timestep")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    LOGGER.info("Saved example plot to %s", output_path)


def main() -> None:
    configure_logging()
    df = load_dataset()
    X_sequences, y_labels, feature_names = build_sequence_dataset(df)
    save_sequence_dataset(X_sequences, y_labels, feature_names)
    plot_example_session(X_sequences)
    print("Sequence dataset shape:", X_sequences.shape)


if __name__ == "__main__":
    main()
