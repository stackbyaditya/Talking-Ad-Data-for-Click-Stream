"""Generate the Model5 sequence dataset from the balanced real-human source dataset."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model5.models.model5_config import (  # noqa: E402
    DATASET_PATH,
    PLOTS_DIR,
    RANDOM_STATE,
    SEQUENCE_DATASET_PATH,
    SEQUENCE_METADATA_PATH,
)
from Model5.models.model5_utils import ensure_model5_directories, load_and_validate_dataset, save_json  # noqa: E402
from preprocessing.session_sequence_generator import (  # noqa: E402
    build_sequence_dataset,
    plot_example_session,
    save_sequence_dataset,
)


LOGGER = logging.getLogger("Model5SequencePrep")


def prepare_sequence_dataset() -> Dict[str, object]:
    """Create and save the Model5 temporal sequence dataset."""
    ensure_model5_directories()
    df, dataset_report = load_and_validate_dataset(DATASET_PATH)
    LOGGER.info("Generating Model5 sequence dataset from balanced source data.")
    X_sequences, y_labels, feature_names = build_sequence_dataset(df, random_state=RANDOM_STATE)
    save_sequence_dataset(X_sequences, y_labels, feature_names, output_path=SEQUENCE_DATASET_PATH)
    plot_example_session(X_sequences, output_path=PLOTS_DIR / "example_session_sequence.png")

    metadata = {
        "source_dataset_path": str(DATASET_PATH),
        "sequence_dataset_path": str(SEQUENCE_DATASET_PATH),
        "sequence_shape": list(X_sequences.shape),
        "sequence_dtype": str(X_sequences.dtype),
        "label_shape": list(y_labels.shape),
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "class_distribution": dataset_report["class_distribution"],
        "contains_nan": bool(np.isnan(X_sequences).any()),
    }
    save_json(metadata, SEQUENCE_METADATA_PATH)
    return metadata


def main() -> None:
    """Entry point for preparing the Model5 sequence dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    prepare_sequence_dataset()


if __name__ == "__main__":
    main()
