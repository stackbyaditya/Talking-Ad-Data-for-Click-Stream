"""Build a balanced advanced training dataset using real human website sessions.

This pipeline:
1. extracts session-level human rows from the nested website clickstream JSON,
2. converts them into the same 41-column advanced schema used by prior models,
3. expands the human class from the real sessions,
4. combines the human pool with balanced moderate and advanced bot rows from
   the previous advanced dataset.

Example:
    python scripts/build_balanced_dataset_from_real_humans.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convert_clickstream_json_to_training_csv import (  # noqa: E402
    OUTPUT_COLUMNS,
    attach_reference_scores,
    build_rows,
    extract_latest_session_rows,
    load_json_records,
    scale_advanced_features,
    validate_output_columns,
)
from scripts.generate_synthetic_human_sessions import generate_synthetic_dataset  # noqa: E402


LOGGER = logging.getLogger("build_balanced_dataset_from_real_humans")
DEFAULT_JSON_PATH = REPO_ROOT / "data" / "clickstream_20260318_235610.json"
DEFAULT_EXTRACTED_HUMAN_PATH = REPO_ROOT / "data" / "processed" / "real_human_clickstream_advanced.csv"
DEFAULT_EXPANDED_HUMAN_PATH = REPO_ROOT / "data" / "processed" / "real_human_clickstream_expanded_advanced.csv"
DEFAULT_FINAL_DATASET_PATH = REPO_ROOT / "data" / "processed" / "final_training_dataset_real_human_balanced_advanced.csv"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "reports" / "real_human_balanced_dataset_summary.json"
REFERENCE_REALISTIC_DATASET = REPO_ROOT / "data" / "processed" / "final_training_dataset_realistic.csv"
REFERENCE_ADVANCED_DATASET = REPO_ROOT / "data" / "processed" / "final_training_dataset_advanced.csv"
DEFAULT_TARGET_PER_CLASS = 2000


def configure_logging() -> None:
    """Configure INFO logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", default=str(DEFAULT_JSON_PATH), help="Path to the website clickstream JSON export.")
    parser.add_argument(
        "--extracted-human-output",
        default=str(DEFAULT_EXTRACTED_HUMAN_PATH),
        help="Path for the extracted real-human advanced CSV.",
    )
    parser.add_argument(
        "--expanded-human-output",
        default=str(DEFAULT_EXPANDED_HUMAN_PATH),
        help="Path for the expanded human-only advanced CSV.",
    )
    parser.add_argument(
        "--final-output",
        default=str(DEFAULT_FINAL_DATASET_PATH),
        help="Path for the final balanced advanced training dataset.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--reference-realistic",
        default=str(REFERENCE_REALISTIC_DATASET),
        help="Reference realistic dataset used for anomaly scoring and advanced-feature scaling.",
    )
    parser.add_argument(
        "--reference-advanced",
        default=str(REFERENCE_ADVANCED_DATASET),
        help="Reference advanced dataset used for bot sampling and schema validation.",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=DEFAULT_TARGET_PER_CLASS,
        help="Balanced row target per class in the final dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def build_real_human_dataframe(
    input_json_path: Path,
    reference_realistic_path: Path,
    reference_advanced_path: Path,
) -> pd.DataFrame:
    """Extract and convert real human sessions into the advanced training schema."""
    LOGGER.info("Loading website clickstream JSON from %s", input_json_path)
    records = load_json_records(input_json_path)
    latest_rows = extract_latest_session_rows(records)

    LOGGER.info("Converting %s unique website sessions into advanced-schema rows.", len(latest_rows))
    reference_realistic_df = pd.read_csv(reference_realistic_path)
    reference_advanced_df = pd.read_csv(reference_advanced_path, nrows=1)

    human_df = build_rows(latest_rows)
    human_df["label"] = 0
    human_df["label_name"] = "human"
    human_df = attach_reference_scores(human_df, reference_realistic_df)
    human_df = scale_advanced_features(human_df, reference_realistic_df)
    human_df = human_df[OUTPUT_COLUMNS].copy()
    validate_output_columns(human_df, reference_advanced_df)

    if human_df.empty:
        raise ValueError("No real human sessions were extracted from the JSON export.")
    return human_df


def build_balanced_human_pool(
    real_human_df: pd.DataFrame,
    target_per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the final balanced human pool, keeping all real sessions."""
    real_human_df = real_human_df.drop_duplicates().reset_index(drop=True)
    if len(real_human_df) > target_per_class:
        LOGGER.info(
            "Real human rows (%s) exceed target_per_class (%s); sampling down to the target.",
            len(real_human_df),
            target_per_class,
        )
        balanced_humans = real_human_df.sample(n=target_per_class, random_state=seed).reset_index(drop=True)
        return balanced_humans, real_human_df.copy()

    synthetic_needed = target_per_class - len(real_human_df)
    if synthetic_needed <= 0:
        return real_human_df.copy(), real_human_df.copy()

    generation_source_df = real_human_df.copy()
    if "source_click_time" in generation_source_df.columns:
        generation_source_df["source_click_time"] = pd.to_datetime(
            generation_source_df["source_click_time"],
            utc=True,
            errors="coerce",
        )

    multiplier = max(int(math.ceil((synthetic_needed * 1.25) / max(len(real_human_df), 1))), 10)
    LOGGER.info(
        "Generating expanded human pool from %s real rows; need %s extra rows, using multiplier=%s.",
        len(real_human_df),
        synthetic_needed,
        multiplier,
    )
    expanded_human_df = generate_synthetic_dataset(generation_source_df, multiplier=multiplier, seed=seed)
    expanded_human_df = expanded_human_df.drop(columns=["synthetic_flag"], errors="ignore")
    expanded_human_df = expanded_human_df[OUTPUT_COLUMNS].drop_duplicates().reset_index(drop=True)

    sampled_synthetic = expanded_human_df.sample(n=synthetic_needed, random_state=seed).reset_index(drop=True)
    balanced_humans = pd.concat([real_human_df, sampled_synthetic], ignore_index=True)
    balanced_humans = balanced_humans.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced_humans, expanded_human_df


def sample_bot_classes(reference_advanced_df: pd.DataFrame, target_per_class: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample balanced moderate and advanced bot rows from the previous dataset."""
    moderate_df = reference_advanced_df.loc[reference_advanced_df["label_name"] == "moderate_bot"].copy()
    advanced_df = reference_advanced_df.loc[reference_advanced_df["label_name"] == "advanced_bot"].copy()
    if moderate_df.empty or advanced_df.empty:
        raise ValueError("Reference advanced dataset does not contain both moderate_bot and advanced_bot rows.")

    moderate_sample = moderate_df.sample(n=target_per_class, replace=len(moderate_df) < target_per_class, random_state=seed)
    advanced_sample = advanced_df.sample(n=target_per_class, replace=len(advanced_df) < target_per_class, random_state=seed)
    return moderate_sample.reset_index(drop=True), advanced_sample.reset_index(drop=True)


def build_summary(
    real_human_df: pd.DataFrame,
    expanded_human_df: pd.DataFrame,
    final_df: pd.DataFrame,
    target_per_class: int,
) -> dict:
    """Build a compact JSON summary for the generated dataset."""
    return {
        "real_human_rows_extracted": int(len(real_human_df)),
        "expanded_human_pool_rows": int(len(expanded_human_df)),
        "target_per_class": int(target_per_class),
        "final_row_count": int(len(final_df)),
        "final_class_distribution": {
            key: int(value)
            for key, value in final_df["label_name"].value_counts().sort_index().to_dict().items()
        },
        "output_columns": list(final_df.columns),
    }


def main() -> None:
    """Run the end-to-end real-human balanced dataset build."""
    configure_logging()
    args = parse_args()

    input_json_path = Path(args.input_json)
    extracted_human_output = Path(args.extracted_human_output)
    expanded_human_output = Path(args.expanded_human_output)
    final_output = Path(args.final_output)
    summary_output = Path(args.summary_output)
    reference_realistic_path = Path(args.reference_realistic)
    reference_advanced_path = Path(args.reference_advanced)
    target_per_class = int(args.target_per_class)

    real_human_df = build_real_human_dataframe(
        input_json_path=input_json_path,
        reference_realistic_path=reference_realistic_path,
        reference_advanced_path=reference_advanced_path,
    )
    reference_advanced_df = pd.read_csv(reference_advanced_path)

    balanced_human_df, expanded_human_df = build_balanced_human_pool(
        real_human_df=real_human_df,
        target_per_class=target_per_class,
        seed=args.seed,
    )
    moderate_bot_df, advanced_bot_df = sample_bot_classes(reference_advanced_df, target_per_class=target_per_class, seed=args.seed)

    final_df = pd.concat(
        [balanced_human_df, moderate_bot_df, advanced_bot_df],
        ignore_index=True,
    )
    final_df = final_df[OUTPUT_COLUMNS].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    extracted_human_output.parent.mkdir(parents=True, exist_ok=True)
    expanded_human_output.parent.mkdir(parents=True, exist_ok=True)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    real_human_df.to_csv(extracted_human_output, index=False)
    expanded_human_df.to_csv(expanded_human_output, index=False)
    final_df.to_csv(final_output, index=False)

    summary = build_summary(
        real_human_df=real_human_df,
        expanded_human_df=expanded_human_df,
        final_df=final_df,
        target_per_class=target_per_class,
    )
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    LOGGER.info("Saved extracted real-human CSV to %s", extracted_human_output)
    LOGGER.info("Saved expanded human CSV to %s", expanded_human_output)
    LOGGER.info("Saved final balanced advanced dataset to %s", final_output)
    LOGGER.info("Final class distribution: %s", summary["final_class_distribution"])


if __name__ == "__main__":
    main()
