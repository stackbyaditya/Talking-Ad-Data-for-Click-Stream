"""End-to-end orchestration for the Model5 adversarially robust experiment."""

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

from Model5.adversarial.run_robustness_pipeline import run_robustness_pipeline  # noqa: E402
from Model5.models.deep_learning.train_dl_models import train_deep_learning_models  # noqa: E402
from Model5.models.model5_config import (  # noqa: E402
    BASELINE_REPORT_PATH,
    BOOSTING_REPORTS_JSON,
    BOOSTING_SUMMARY_JSON,
    COMBINED_REPORTS_JSON,
    COMBINED_SUMMARY_CSV,
    COMBINED_SUMMARY_JSON,
    DATASET_VALIDATION_REPORT_PATH,
    DL_REPORTS_JSON,
    DL_SUMMARY_JSON,
    PLOTS_DIR,
)
from Model5.models.model5_utils import (  # noqa: E402
    configure_logging,
    ensure_model5_directories,
    generate_baseline_report_markdown,
    load_and_validate_dataset,
    load_json_if_exists,
    plot_classwise_metric_heatmap,
    plot_metric_comparison,
    save_json,
)
from Model5.models.prepare_sequence_dataset import prepare_sequence_dataset  # noqa: E402
from Model5.models.train_boosting_models import train_boosting_models  # noqa: E402


LOGGER = logging.getLogger("RunModel5Experiment")


def main() -> None:
    """Run the complete Model5 workflow or selected subsets."""
    parser = argparse.ArgumentParser(description="Run the Model5 adversarially robust clickstream experiment.")
    parser.add_argument("--skip-boosting", action="store_true", help="Skip tabular boosting model training.")
    parser.add_argument("--skip-sequence", action="store_true", help="Skip sequence dataset generation.")
    parser.add_argument("--skip-dl", action="store_true", help="Skip deep-learning model training.")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip adversarial robustness evaluation.")
    parser.add_argument("--dl-epochs", type=int, default=30, help="Epoch count for deep-learning training.")
    parser.add_argument("--dl-batch-size", type=int, default=64, help="Batch size for deep-learning training.")
    args = parser.parse_args()

    configure_logging()
    ensure_model5_directories()

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
        plot_metric_comparison(combined_summary, PLOTS_DIR / "model_comparison_combined.png", "Model5 Combined Model Comparison")

    combined_reports = {**load_json_if_exists(BOOSTING_REPORTS_JSON), **load_json_if_exists(DL_REPORTS_JSON)}
    if combined_reports:
        save_json(combined_reports, COMBINED_REPORTS_JSON)
        plot_classwise_metric_heatmap(
            combined_reports,
            PLOTS_DIR / "classwise_f1_combined.png",
            metric_name="f1-score",
            title="Model5 Combined Class-wise F1 Scores",
        )

    baseline_report = generate_baseline_report_markdown(dataset_report, boosting_summary, dl_summary)
    BASELINE_REPORT_PATH.write_text(baseline_report, encoding="utf-8")
    LOGGER.info("Model5 baseline report written to %s", BASELINE_REPORT_PATH)

    if not args.skip_robustness:
        run_robustness_pipeline()


if __name__ == "__main__":
    main()
