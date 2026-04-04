"""Train tree-based models on the realistic synthetic bot-detection dataset."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

import joblib
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model1.models.model_utils import (  # noqa: E402
    ensure_output_dir,
    evaluate_model,
    export_lstm_training_data,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
    save_model,
    save_performance_summary,
)
from preprocessing.preprocess_dataset import prepare_train_test_data  # noqa: E402
from preprocessing.behavioral_feature_engineering import main as build_advanced_dataset  # noqa: E402


LOGGER = logging.getLogger("train_boosting_models")
OUTPUT_DIR = MODEL_ROOT / "outputs"
COMMON_SEQUENCE_DATA_PATH = REPO_ROOT / "model_outputs" / "lstm_training_data_v3.npz"


def configure_logging() -> None:
    """Configure INFO-level logging for model training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def train_models() -> Dict[str, Dict[str, float]]:
    """Run preprocessing, train models, evaluate them, and save all outputs."""
    ensure_output_dir(OUTPUT_DIR)

    LOGGER.info("Refreshing advanced behavioral dataset.")
    build_advanced_dataset()
    LOGGER.info("Loading dataset and preprocessing features.")
    bundle = prepare_train_test_data()
    X_train_processed = bundle["X_train_processed"]
    X_test_processed = bundle["X_test_processed"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]
    feature_names = bundle["feature_names"]

    joblib.dump(bundle["pipeline"], OUTPUT_DIR / "preprocessing_pipeline.pkl")

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            random_state=42,
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=3,
            n_jobs=1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            objective="multiclass",
            num_class=3,
            n_jobs=1,
        ),
    }

    model_paths = {
        "RandomForest": OUTPUT_DIR / "random_forest_model.pkl",
        "XGBoost": OUTPUT_DIR / "xgboost_model.pkl",
        "LightGBM": OUTPUT_DIR / "lightgbm_model.pkl",
    }
    confusion_paths = {
        "RandomForest": OUTPUT_DIR / "confusion_matrix_rf.png",
        "XGBoost": OUTPUT_DIR / "confusion_matrix_xgb.png",
        "LightGBM": OUTPUT_DIR / "confusion_matrix_lgbm.png",
    }

    summary: Dict[str, Dict[str, float]] = {}
    roc_inputs = []

    for name, model in models.items():
        LOGGER.info("Training %s", name)
        model.fit(X_train_processed, y_train)
        save_model(model, model_paths[name])

        metrics, y_proba = evaluate_model(name, model, X_test_processed, y_test)
        summary[name] = metrics
        y_pred = model.predict(X_test_processed)
        plot_confusion_matrix(y_test, y_pred, f"{name} Confusion Matrix", confusion_paths[name])
        roc_inputs.append((name, y_proba, y_pred))

        if name == "RandomForest":
            plot_feature_importance(
                model,
                feature_names,
                "Random Forest Feature Importance",
                OUTPUT_DIR / "feature_importance_rf.png",
            )
        if name == "XGBoost":
            plot_feature_importance(
                model,
                feature_names,
                "XGBoost Feature Importance",
                OUTPUT_DIR / "feature_importance_xgb.png",
            )

    LOGGER.info("Evaluation complete.")
    plot_roc_curves(roc_inputs, y_test, OUTPUT_DIR / "roc_curves_boosting.png")
    save_performance_summary(summary, OUTPUT_DIR / "model_performance_summary.json")
    export_lstm_training_data(bundle["df"], COMMON_SEQUENCE_DATA_PATH)
    LOGGER.info("Saving outputs complete.")
    return summary


def main() -> None:
    """Entry point for training all tree-based models."""
    configure_logging()
    train_models()


if __name__ == "__main__":
    main()
