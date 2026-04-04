"""Train and evaluate Model4 boosting models on the balanced dataset."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model4.models.model4_config import (  # noqa: E402
    BOOSTING_MODELS_DIR,
    BOOSTING_REPORTS_JSON,
    BOOSTING_SUMMARY_CSV,
    BOOSTING_SUMMARY_JSON,
    DATASET_PATH,
    FEATURE_NAMES_PATH,
    PLOTS_DIR,
    PREPROCESSING_PIPELINE_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from Model4.models.model4_utils import (  # noqa: E402
    ensure_model4_directories,
    evaluate_classifier,
    load_and_validate_dataset,
    plot_behavioral_feature_distributions,
    plot_class_distribution,
    plot_classwise_metric_heatmap,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metric_comparison,
    plot_roc_curves,
    save_json,
    save_summary_tables,
    split_summary_and_reports,
)
from preprocessing.preprocess_dataset import prepare_train_test_data  # noqa: E402


LOGGER = logging.getLogger("Model4Boosting")


def train_boosting_models() -> Dict[str, Dict[str, float]]:
    """Train Model4 tabular models and persist artifacts."""
    ensure_model4_directories()
    df, _ = load_and_validate_dataset(DATASET_PATH)
    plot_class_distribution(df, PLOTS_DIR / "dataset_class_distribution.png")
    plot_behavioral_feature_distributions(df, PLOTS_DIR)

    LOGGER.info("Preparing tabular train/test split for Model4.")
    bundle = prepare_train_test_data(
        dataset_path=DATASET_PATH,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    X_train_processed = bundle["X_train_processed"]
    X_test_processed = bundle["X_test_processed"]
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()
    X_train_processed = pd.DataFrame(X_train_processed, columns=bundle["feature_names"])
    X_test_processed = pd.DataFrame(X_test_processed, columns=bundle["feature_names"])

    joblib.dump(bundle["pipeline"], PREPROCESSING_PIPELINE_PATH)
    save_json({"feature_names": bundle["feature_names"]}, FEATURE_NAMES_PATH)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=3,
            n_jobs=1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            objective="multiclass",
            num_class=3,
            n_jobs=1,
            verbosity=-1,
        ),
    }

    model_paths = {
        "RandomForest": BOOSTING_MODELS_DIR / "random_forest_model4.pkl",
        "XGBoost": BOOSTING_MODELS_DIR / "xgboost_model4.pkl",
        "LightGBM": BOOSTING_MODELS_DIR / "lightgbm_model4.pkl",
    }
    confusion_paths = {
        "RandomForest": PLOTS_DIR / "confusion_matrix_rf.png",
        "XGBoost": PLOTS_DIR / "confusion_matrix_xgb.png",
        "LightGBM": PLOTS_DIR / "confusion_matrix_lgbm.png",
    }
    importance_paths = {
        "RandomForest": PLOTS_DIR / "feature_importance_rf.png",
        "XGBoost": PLOTS_DIR / "feature_importance_xgb.png",
        "LightGBM": PLOTS_DIR / "feature_importance_lgbm.png",
    }

    summary: Dict[str, Dict[str, object]] = {}
    roc_inputs: Dict[str, object] = {}

    for name, model in models.items():
        LOGGER.info("Training %s", name)
        model.fit(X_train_processed, bundle["y_train"])
        joblib.dump(model, model_paths[name])

        metrics, y_pred, y_proba = evaluate_classifier(name, model, X_test_processed, bundle["y_test"])
        summary[name] = metrics
        roc_inputs[name] = y_proba

        plot_confusion_matrix(bundle["y_test"], y_pred, f"{name} Confusion Matrix", confusion_paths[name])
        plot_feature_importance(
            model,
            bundle["feature_names"],
            f"{name} Feature Importance",
            importance_paths[name],
        )

    plot_roc_curves(roc_inputs, bundle["y_test"], PLOTS_DIR / "roc_curves_boosting_models.png", "Model4 Boosting ROC Curves")
    metrics_only, report_only = split_summary_and_reports(summary)
    save_summary_tables(summary, BOOSTING_SUMMARY_JSON, BOOSTING_SUMMARY_CSV)
    save_json(report_only, BOOSTING_REPORTS_JSON)
    plot_metric_comparison(metrics_only, PLOTS_DIR / "model_comparison_boosting.png", "Model4 Boosting Model Comparison")
    plot_classwise_metric_heatmap(
        report_only,
        PLOTS_DIR / "classwise_f1_boosting.png",
        metric_name="f1-score",
        title="Boosting Class-wise F1 Scores",
    )
    return metrics_only


def main() -> None:
    """Entry point for Model4 boosting model training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    train_boosting_models()


if __name__ == "__main__":
    main()
