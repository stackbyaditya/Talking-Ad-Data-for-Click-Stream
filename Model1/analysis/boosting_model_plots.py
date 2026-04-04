"""Generate publication-quality evaluation plots for boosting models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = MODEL_ROOT / "analysis" / "plots"
MODEL_PATHS = {
    "RandomForest": MODEL_ROOT / "outputs" / "random_forest_model.pkl",
    "XGBoost": MODEL_ROOT / "outputs" / "xgboost_model.pkl",
    "LightGBM": MODEL_ROOT / "outputs" / "lightgbm_model.pkl",
}
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "processed" / "final_training_dataset_advanced.csv"
DEFAULT_PIPELINE_PATH = MODEL_ROOT / "outputs" / "preprocessing_pipeline.pkl"
DROP_COLS = [
    "session_id",
    "ip_address",
    "user_agent",
    "label_name",
    "source_click_time",
    "source_attributed_time",
]
METRIC_ORDER = ["accuracy", "precision", "recall", "f1 score", "roc_auc"]
CLASS_PALETTE = ["#0F4C5C", "#E36414", "#6A994E"]
MODEL_PALETTE = {
    "RandomForest": "#1D3557",
    "XGBoost": "#D62828",
    "LightGBM": "#2A9D8F",
}


def configure_plot_style() -> None:
    """Apply a consistent, paper-friendly plotting style."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "font.size": 12,
        }
    )


def resolve_existing_path(*candidates: Path) -> Path:
    """Return the first existing path from a list of candidates."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find any expected file. Checked: {checked}")


def load_models() -> Dict[str, object]:
    """Load persisted boosting models from disk."""
    return {
        name: joblib.load(path)
        for name, path in MODEL_PATHS.items()
    }


def prepare_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str], pd.Series]:
    """Load, preprocess, and split the advanced training dataset."""
    dataset_path = resolve_existing_path(
        DEFAULT_DATASET_PATH,
        REPO_ROOT / "data" / "processed" / "final_training_dataset_advanced.csv",
    )
    pipeline_path = resolve_existing_path(
        DEFAULT_PIPELINE_PATH,
        MODEL_ROOT / "outputs" / "preprocessing_pipeline.pkl",
    )

    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["label"] + DROP_COLS)
    y = df["label"]

    preprocessing_pipeline = joblib.load(pipeline_path)
    feature_names = list(preprocessing_pipeline.get_feature_names_out())
    X_processed = preprocessing_pipeline.transform(X)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = pd.DataFrame(X_processed, columns=feature_names, index=df.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), feature_names, y


def evaluate_models(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute evaluation metrics and collect predictions."""
    metrics_by_model: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, np.ndarray] = {}
    probabilities: Dict[str, np.ndarray] = {}

    X_test_array = X_test.to_numpy()

    for name, model in models.items():
        model_input = X_test if name == "LightGBM" else X_test_array
        y_pred = model.predict(model_input)
        y_proba = model.predict_proba(model_input)

        predictions[name] = y_pred
        probabilities[name] = y_proba
        metrics_by_model[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1 score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")),
        }

    return metrics_by_model, predictions, probabilities


def plot_roc_curves(
    model_probabilities: Dict[str, np.ndarray],
    y_test: np.ndarray,
) -> None:
    """Generate a combined ROC curve plot for the boosting models."""
    fig, ax = plt.subplots(figsize=(9, 7))
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)

    for name, y_proba in model_probabilities.items():
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
        auc_value = auc(fpr, tpr)

        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_value, name=name)
        display.plot(
            ax=ax,
            name=f"{name} (AUC={auc_value:.3f})",
            curve_kwargs={"color": MODEL_PALETTE[name]},
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)
    ax.set_title("ROC Curves for Boosting Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curves_boosting_models.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
) -> None:
    """Generate one confusion matrix per model."""
    file_names = {
        "RandomForest": "confusion_matrix_rf.png",
        "XGBoost": "confusion_matrix_xgb.png",
        "LightGBM": "confusion_matrix_lgbm.png",
    }

    for name, y_pred in predictions.items():
        labels = np.unique(np.concatenate([y_test, y_pred]))
        matrix = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / file_names[name], dpi=300, bbox_inches="tight")
        plt.close(fig)


def extract_feature_importance(model, model_name: str) -> np.ndarray:
    """Return feature importance values for a tree-based estimator."""
    if model_name == "LightGBM":
        if hasattr(model, "booster_"):
            return model.booster_.feature_importance(importance_type="gain")
        if hasattr(model, "feature_importance"):
            return model.feature_importance(importance_type="gain")
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    raise AttributeError(f"Model '{model_name}' does not expose feature importances.")


def plot_feature_importance(
    models: Dict[str, object],
    feature_names: List[str],
) -> None:
    """Generate top-20 feature importance plots for all boosting models."""
    file_names = {
        "RandomForest": "feature_importance_rf.png",
        "XGBoost": "feature_importance_xgb.png",
        "LightGBM": "feature_importance_lgbm.png",
    }

    for name, model in models.items():
        importances = np.asarray(extract_feature_importance(model, name), dtype=float)
        order = np.argsort(importances)[::-1][:20]
        top_features = [feature_names[index] for index in order]
        top_values = importances[order]

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(
            x=top_values,
            y=top_features,
            orient="h",
            color=MODEL_PALETTE[name],
            ax=ax,
        )
        ax.set_title(f"Top 20 Feature Importances - {name}")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / file_names[name], dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_model_comparison(metrics_by_model: Dict[str, Dict[str, float]]) -> None:
    """Generate a grouped bar chart comparing boosting-model performance."""
    rows = []
    for model_name, metrics in metrics_by_model.items():
        for metric_name in METRIC_ORDER:
            rows.append(
                {
                    "Model": model_name,
                    "Metric": metric_name,
                    "Score": metrics[metric_name],
                }
            )

    comparison_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=comparison_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=MODEL_PALETTE,
        ax=ax,
    )
    ax.set_title("Boosting Model Performance Comparison")
    ax.set_xlabel("Evaluation Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.legend(title="Model", loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_comparison_boosting.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(y: pd.Series) -> None:
    """Generate a dataset class distribution plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x=y, palette=CLASS_PALETTE, ax=ax, hue=y, dodge=False, legend=False)
    ax.set_title("Dataset Class Distribution")
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "dataset_class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all requested boosting-model visualizations."""
    configure_plot_style()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    models = load_models()
    _, X_test, _, y_test, feature_names, y = prepare_dataset()
    metrics_by_model, predictions, probabilities = evaluate_models(models, X_test, y_test)

    plot_roc_curves(probabilities, y_test)
    plot_confusion_matrices(predictions, y_test)
    plot_feature_importance(models, feature_names)
    plot_model_comparison(metrics_by_model)
    plot_class_distribution(y)

    print("Boosting model visualizations generated successfully")


if __name__ == "__main__":
    main()
