"""Visualize advanced behavioral features for human and bot sessions."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LOGGER = logging.getLogger("behavioral_feature_plots")
DATASET_PATH = Path("data/processed/final_training_dataset_advanced.csv")
PLOT_DIR = Path("analysis/plots")
PLOT_FEATURES = [
    "movement_acceleration",
    "mouse_direction_entropy",
    "click_burst_score",
    "session_idle_ratio",
    "trajectory_smoothness",
    "behavioral_complexity",
]


def configure_logging() -> None:
    """Configure INFO logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Create per-feature class distribution plots."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    palette = {"human": "#4C956C", "moderate_bot": "#F4A259", "advanced_bot": "#BC4B51"}

    for feature in PLOT_FEATURES:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(
            data=df,
            x=feature,
            hue="label_name",
            common_norm=False,
            fill=True,
            alpha=0.25,
            linewidth=1.5,
            palette=palette,
            ax=ax,
        )
        ax.set_title(f"{feature} Distribution")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{feature}_distribution.png", dpi=160)
        plt.close(fig)


def main() -> None:
    """Entry point for advanced behavioral feature visualizations."""
    configure_logging()
    LOGGER.info("Loading advanced dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    plot_feature_distributions(df)
    LOGGER.info("Behavioral feature plots saved to %s", PLOT_DIR)


if __name__ == "__main__":
    main()
