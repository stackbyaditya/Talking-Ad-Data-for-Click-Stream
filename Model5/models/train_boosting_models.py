"""Train and evaluate Model5 boosting models on the balanced real-human dataset."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model5.models.model5_config import (  # noqa: E402
    BOOSTING_MODELS_DIR,
    BOOSTING_REPORTS_JSON,
    BOOSTING_SUMMARY_CSV,
    BOOSTING_SUMMARY_JSON,
    DATASET_PATH,
    EXCLUDED_TABULAR_FEATURES,
    FEATURE_NAMES_PATH,
    PLOTS_DIR,
    PREPROCESSING_PIPELINE_PATH,
    RANDOM_STATE,
    TABULAR_FEATURE_METADATA_PATH,
    TABULAR_SPLIT_ARTIFACTS_PATH,
    TEST_SIZE,
)
from Model5.models.model5_utils import (  # noqa: E402
    ensure_model5_directories,
    evaluate_classifier,
    load_and_validate_dataset,
    plot_behavioral_feature_distributions,
    plot_class_distribution,
    plot_classwise_metric_heatmap,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metric_comparison,
    plot_roc_curves,
    save_joblib,
    save_json,
    save_summary_tables,
    split_summary_and_reports,
)
from preprocessing.preprocess_dataset import (  # noqa: E402
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    DROP_COLUMNS,
    build_preprocessor,
    load_dataset,
)


LOGGER = logging.getLogger("Model5Boosting")


def _prepare_model5_train_test_data(
    dataset_path: Path,
    test_size: float,
    random_state: int,
) -> Dict[str, object]:
    """Prepare the Model5 tabular split while excluding leakage-prone geo features."""
    df = load_dataset(dataset_path)
    drop_cols = [column for column in DROP_COLUMNS if column in df.columns]
    X = df.drop(columns=["label"] + drop_cols)
    y = df["label"].copy()

    categorical_features = [
        column
        for column in BASE_CATEGORICAL_FEATURES
        if column in X.columns and column not in EXCLUDED_TABULAR_FEATURES
    ]
    numeric_features = [
        column
        for column in BASE_NUMERIC_FEATURES
        if column in X.columns and column not in EXCLUDED_TABULAR_FEATURES
    ]
    selected_features = numeric_features + categorical_features
    X = X[selected_features].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    pipeline = Pipeline(steps=[("preprocessor", build_preprocessor(numeric_features, categorical_features))])
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "pipeline": pipeline,
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "excluded_features": EXCLUDED_TABULAR_FEATURES,
    }


def _build_feature_metadata(bundle: Dict[str, object], X_train_processed: np.ndarray) -> Dict[str, object]:
    """Save raw and processed range metadata for domain-constrained attacks."""
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    scaler = preprocessor.named_transformers_["num"]
    numeric_features = bundle["numeric_features"]
    feature_names = bundle["feature_names"]
    X_train = bundle["X_train"].reset_index(drop=True)

    metadata = {
        "feature_names": feature_names,
        "numeric_features": numeric_features,
        "categorical_features": bundle["categorical_features"],
        "excluded_features": bundle.get("excluded_features", []),
        "processed_numeric_feature_names": feature_names[: len(numeric_features)],
        "processed_categorical_feature_names": feature_names[len(numeric_features) :],
        "numeric_feature_metadata": {},
    }

    for index, raw_name in enumerate(numeric_features):
        processed_name = feature_names[index]
        transformed_values = X_train_processed[:, index]
        metadata["numeric_feature_metadata"][processed_name] = {
            "raw_name": raw_name,
            "processed_name": processed_name,
            "raw_min": float(X_train[raw_name].min()),
            "raw_max": float(X_train[raw_name].max()),
            "raw_mean": float(X_train[raw_name].mean()),
            "raw_std": float(X_train[raw_name].std()),
            "processed_min": float(np.min(transformed_values)),
            "processed_max": float(np.max(transformed_values)),
            "processed_mean": float(np.mean(transformed_values)),
            "processed_std": float(np.std(transformed_values)),
            "scaler_center": float(scaler.center_[index]),
            "scaler_scale": float(scaler.scale_[index]),
        }
    return metadata


def train_boosting_models() -> Dict[str, Dict[str, float]]:
    """Train Model5 tabular models and persist artifacts."""
    ensure_model5_directories()
    df, _ = load_and_validate_dataset(DATASET_PATH)
    plot_class_distribution(df, PLOTS_DIR / "dataset_class_distribution.png")
    plot_behavioral_feature_distributions(df, PLOTS_DIR)

    LOGGER.info("Preparing tabular train/test split for Model5.")
    bundle = _prepare_model5_train_test_data(
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
    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

    X_train_df = pd.DataFrame(X_train_processed, columns=bundle["feature_names"])
    X_test_df = pd.DataFrame(X_test_processed, columns=bundle["feature_names"])

    save_joblib(bundle["pipeline"], PREPROCESSING_PIPELINE_PATH)
    save_json({"feature_names": bundle["feature_names"]}, FEATURE_NAMES_PATH)
    save_json(_build_feature_metadata(bundle, X_train_processed), TABULAR_FEATURE_METADATA_PATH)
    save_joblib(
        {
            "X_train_raw": bundle["X_train"].reset_index(drop=True),
            "X_test_raw": bundle["X_test"].reset_index(drop=True),
            "X_train_processed": X_train_processed,
            "X_test_processed": X_test_processed,
            "y_train": bundle["y_train"].to_numpy(dtype=np.int64),
            "y_test": bundle["y_test"].to_numpy(dtype=np.int64),
            "feature_names": bundle["feature_names"],
            "numeric_features": bundle["numeric_features"],
            "categorical_features": bundle["categorical_features"],
        },
        TABULAR_SPLIT_ARTIFACTS_PATH,
    )

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
        "RandomForest": BOOSTING_MODELS_DIR / "random_forest_model5.pkl",
        "XGBoost": BOOSTING_MODELS_DIR / "xgboost_model5.pkl",
        "LightGBM": BOOSTING_MODELS_DIR / "lightgbm_model5.pkl",
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
        model.fit(X_train_df, bundle["y_train"])
        joblib.dump(model, model_paths[name])

        metrics, y_pred, y_proba = evaluate_classifier(name, model, X_test_df, bundle["y_test"])
        summary[name] = metrics
        roc_inputs[name] = y_proba

        plot_confusion_matrix(bundle["y_test"], y_pred, f"{name} Confusion Matrix", confusion_paths[name])
        plot_feature_importance(
            model,
            bundle["feature_names"],
            f"{name} Feature Importance",
            importance_paths[name],
        )

    plot_roc_curves(roc_inputs, bundle["y_test"], PLOTS_DIR / "roc_curves_boosting_models.png", "Model5 Boosting ROC Curves")
    metrics_only, report_only = split_summary_and_reports(summary)
    save_summary_tables(summary, BOOSTING_SUMMARY_JSON, BOOSTING_SUMMARY_CSV)
    save_json(report_only, BOOSTING_REPORTS_JSON)
    plot_metric_comparison(metrics_only, PLOTS_DIR / "model_comparison_boosting.png", "Model5 Boosting Model Comparison")
    plot_classwise_metric_heatmap(
        report_only,
        PLOTS_DIR / "classwise_f1_boosting.png",
        metric_name="f1-score",
        title="Model5 Boosting Class-wise F1 Scores",
    )
    return metrics_only


def main() -> None:
    """Entry point for Model5 boosting model training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    train_boosting_models()


if __name__ == "__main__":
    main()
