"""Shared utilities for training, evaluation, plotting, and sequence export."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


LOGGER = logging.getLogger(__name__)
CLASS_NAMES = ["human", "moderate_bot", "advanced_bot"]
DEFAULT_SEQUENCE_DATA_PATH = Path("model_outputs/lstm_training_data_v3.npz")


def ensure_output_dir(output_dir: Path) -> None:
    """Create the output directory if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)


def save_model(model, output_path: Path) -> None:
    """Persist a trained model using joblib."""
    joblib.dump(model, output_path)


def evaluate_model(name: str, model, X_test, y_test) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute metrics and return predicted probabilities."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")),
        "classification_report": classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
    }
    LOGGER.info("%s evaluation complete. Accuracy=%.4f F1=%.4f ROC_AUC=%.4f", name, metrics["accuracy"], metrics["f1_score"], metrics["roc_auc"])
    return metrics, y_proba


def plot_confusion_matrix(y_true, y_pred, title: str, output_path: Path) -> None:
    """Save a seaborn confusion matrix plot."""
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_roc_curves(
    roc_inputs: Iterable[Tuple[str, np.ndarray, np.ndarray]],
    y_true,
    output_path: Path,
) -> None:
    """Save combined micro-average ROC curves for all models."""
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, y_proba, _ in roc_inputs:
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        auc_value = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(
    model,
    feature_names: List[str],
    title: str,
    output_path: Path,
    top_n: int = 25,
) -> None:
    """Save a feature importance plot for tree-based models."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    order = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[idx] for idx in order]
    top_importances = importances[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top_importances, y=top_features, orient="h", ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_performance_summary(summary: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Write the model performance summary JSON."""
    serializable = {}
    for model_name, metrics in summary.items():
        serializable[model_name] = {
            key: value
            for key, value in metrics.items()
            if key != "classification_report"
        }
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def export_lstm_training_data(df, output_path: Path) -> None:
    """Export temporal sequences for future LSTM work using the v3 generator."""
    from preprocessing.session_sequence_generator import build_sequence_dataset

    X_sequences, y_labels, feature_names = build_sequence_dataset(df)
    np.savez_compressed(
        output_path,
        X_sequences=X_sequences,
        y_labels=y_labels,
        feature_names=np.asarray(feature_names, dtype=object),
    )


def load_lstm_training_data(input_path: Path = DEFAULT_SEQUENCE_DATA_PATH) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the v3 LSTM-ready temporal sequence dataset."""
    payload = np.load(input_path, allow_pickle=True)
    feature_names = payload["feature_names"].tolist()
    return payload["X_sequences"], payload["y_labels"], feature_names
