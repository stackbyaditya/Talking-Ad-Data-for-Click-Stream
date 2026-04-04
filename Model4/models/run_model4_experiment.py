"""End-to-end orchestration for the Model4 balanced-dataset experiment."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model4.models.deep_learning.train_dl_models import train_deep_learning_models  # noqa: E402
from Model4.models.model4_config import (  # noqa: E402
    BOOSTING_REPORTS_JSON,
    BOOSTING_SUMMARY_JSON,
    COMBINED_REPORTS_JSON,
    COMBINED_SUMMARY_CSV,
    COMBINED_SUMMARY_JSON,
    DATASET_VALIDATION_REPORT_PATH,
    DL_REPORTS_JSON,
    DL_SUMMARY_JSON,
    MODEL4_REPORT_PATH,
    PLOTS_DIR,
)
from Model4.models.model4_utils import (  # noqa: E402
    configure_logging,
    ensure_model4_directories,
    generate_report_markdown,
    load_and_validate_dataset,
    plot_classwise_metric_heatmap,
    plot_metric_comparison,
    save_json,
)
from Model4.models.prepare_sequence_dataset import prepare_sequence_dataset  # noqa: E402
from Model4.models.train_boosting_models import train_boosting_models  # noqa: E402


LOGGER = logging.getLogger("RunModel4Experiment")


def load_json_if_exists(path: Path) -> dict:
    """Load JSON from disk when present, otherwise return an empty dict."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Run the complete Model4 workflow or selected subsets."""
    parser = argparse.ArgumentParser(description="Run the Model4 balanced clickstream experiment.")
    parser.add_argument("--skip-boosting", action="store_true", help="Skip tabular boosting model training.")
    parser.add_argument("--skip-sequence", action="store_true", help="Skip sequence dataset generation.")
    parser.add_argument("--skip-dl", action="store_true", help="Skip deep-learning model training.")
    parser.add_argument("--dl-epochs", type=int, default=30, help="Epoch count for deep-learning training.")
    parser.add_argument("--dl-batch-size", type=int, default=64, help="Batch size for deep-learning training.")
    args = parser.parse_args()

    configure_logging()
    ensure_model4_directories()

    _, dataset_report = load_and_validate_dataset()
    save_json(dataset_report, DATASET_VALIDATION_REPORT_PATH)

    if not args.skip_boosting:
        boosting_summary = train_boosting_models()
    else:
        boosting_summary = load_json_if_exists(BOOSTING_SUMMARY_JSON)

    if not args.skip_sequence:
        prepare_sequence_dataset()

    if not args.skip_dl:
        dl_summary = train_deep_learning_models(epochs=args.dl_epochs, batch_size=args.dl_batch_size)
    else:
        dl_summary = load_json_if_exists(DL_SUMMARY_JSON)

    combined_summary = {**boosting_summary, **dl_summary}
    if combined_summary:
        save_json(combined_summary, COMBINED_SUMMARY_JSON)
        (
            pd.DataFrame.from_dict(combined_summary, orient="index")
            .reset_index()
            .rename(columns={"index": "model"})
            .sort_values(by="accuracy", ascending=False)
            .to_csv(COMBINED_SUMMARY_CSV, index=False)
        )
        plot_metric_comparison(combined_summary, PLOTS_DIR / "model_comparison_combined.png", "Model4 Combined Model Comparison")

    combined_reports = {**load_json_if_exists(BOOSTING_REPORTS_JSON), **load_json_if_exists(DL_REPORTS_JSON)}
    if combined_reports:
        save_json(combined_reports, COMBINED_REPORTS_JSON)
        plot_classwise_metric_heatmap(
            combined_reports,
            PLOTS_DIR / "classwise_f1_combined.png",
            metric_name="f1-score",
            title="Model4 Combined Class-wise F1 Scores",
        )

    report_markdown = generate_report_markdown(dataset_report, boosting_summary, dl_summary)
    MODEL4_REPORT_PATH.write_text(report_markdown, encoding="utf-8")
    LOGGER.info("Model4 report written to %s", MODEL4_REPORT_PATH)


if __name__ == "__main__":
    main()
