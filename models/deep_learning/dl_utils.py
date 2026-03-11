"""Utility functions for deep learning sequence training and evaluation."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


LOGGER = logging.getLogger(__name__)
CLASS_NAMES = ["human", "moderate_bot", "advanced_bot"]
SEQUENCE_DATA_PATH = Path("model_outputs/lstm_training_data_v3.npz")


def set_global_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and TensorFlow random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_sequence_dataset(input_path: Path = SEQUENCE_DATA_PATH) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the generated temporal sequence dataset."""
    payload = np.load(input_path, allow_pickle=True)
    X_sequences = payload["X_sequences"].astype(np.float32)
    y_labels = payload["y_labels"].astype(np.int64)
    feature_names = payload["feature_names"].tolist()
    LOGGER.info("Loaded sequence dataset: X=%s y=%s", X_sequences.shape, y_labels.shape)
    return X_sequences, y_labels, feature_names


def split_and_scale_sequences(
    X_sequences: np.ndarray,
    y_labels: np.ndarray,
    scaler_path: Path,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Split sequences into train/validation/test sets and scale them using StandardScaler."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_sequences,
        y_labels,
        test_size=0.15,
        stratify=y_labels,
        random_state=random_state,
    )
    validation_fraction = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction,
        stratify=y_train_val,
        random_state=random_state,
    )

    scaler = StandardScaler()
    train_shape = X_train.shape
    val_shape = X_val.shape
    test_shape = X_test.shape

    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, train_shape[-1])).reshape(train_shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, val_shape[-1])).reshape(val_shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, test_shape[-1])).reshape(test_shape)
    joblib.dump(scaler, scaler_path)

    bundle = {
        "X_train": X_train_scaled.astype(np.float32),
        "X_val": X_val_scaled.astype(np.float32),
        "X_test": X_test_scaled.astype(np.float32),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_onehot": to_categorical(y_train, num_classes=3),
        "y_val_onehot": to_categorical(y_val, num_classes=3),
        "y_test_onehot": to_categorical(y_test, num_classes=3),
    }
    LOGGER.info(
        "Sequence split complete. train=%s val=%s test=%s",
        bundle["X_train"].shape,
        bundle["X_val"].shape,
        bundle["X_test"].shape,
    )
    return bundle


def build_callbacks() -> List[tf.keras.callbacks.Callback]:
    """Create common training callbacks."""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]


def evaluate_dl_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate a deep learning model on the test set."""
    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, y_pred, y_proba


def plot_training_curves(histories: Dict[str, Dict[str, List[float]]], output_dir: Path) -> None:
    """Save combined training loss and accuracy curve plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_loss, ax_loss = plt.subplots(figsize=(9, 6))
    for model_name, history in histories.items():
        ax_loss.plot(history["loss"], label=f"{model_name} train")
        ax_loss.plot(history["val_loss"], linestyle="--", label=f"{model_name} val")
    ax_loss.set_title("Training Loss Curves")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=8)
    fig_loss.tight_layout()
    fig_loss.savefig(output_dir / "training_loss_curves.png", dpi=160)
    plt.close(fig_loss)

    fig_acc, ax_acc = plt.subplots(figsize=(9, 6))
    for model_name, history in histories.items():
        ax_acc.plot(history["accuracy"], label=f"{model_name} train")
        ax_acc.plot(history["val_accuracy"], linestyle="--", label=f"{model_name} val")
    ax_acc.set_title("Training Accuracy Curves")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(fontsize=8)
    fig_acc.tight_layout()
    fig_acc.savefig(output_dir / "training_accuracy_curves.png", dpi=160)
    plt.close(fig_acc)


def plot_roc_curves_dl(roc_inputs: Dict[str, np.ndarray], y_test: np.ndarray, output_path: Path) -> None:
    """Save ROC curves for all deep learning models on one figure."""
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, y_proba in roc_inputs.items():
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
        auc_value = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curves - Deep Learning Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrices_dl(
    confusion_inputs: Dict[str, np.ndarray],
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    """Save confusion matrices for all deep learning models in one figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, (model_name, y_pred) in enumerate(confusion_inputs.items()):
        matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=axes[idx],
        )
        axes[idx].set_title(model_name)
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
    for idx in range(len(confusion_inputs), len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_model_comparison(
    boosting_summary: Dict[str, Dict[str, float]],
    dl_summary: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Create a grouped bar chart comparing boosting and deep learning models."""
    combined = {**boosting_summary, **dl_summary}
    model_names = list(combined.keys())
    metrics = ["accuracy", "f1_score", "roc_auc"]
    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, metric in enumerate(metrics):
        values = [combined[name][metric] for name in model_names]
        ax.bar(x + (idx - 1) * width, values, width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_summary(summary: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Save the deep learning model summary to JSON."""
    serializable = {}
    for model_name, metrics in summary.items():
        serializable[model_name] = {
            key: value
            for key, value in metrics.items()
            if key != "classification_report"
        }
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def print_performance_summary(
    boosting_summary: Dict[str, Dict[str, float]],
    dl_summary: Dict[str, Dict[str, float]],
) -> None:
    """Print the final combined performance summary."""
    combined = {**boosting_summary, **dl_summary}
    best_model = max(combined.items(), key=lambda item: item[1]["accuracy"])[0]

    print("==============================")
    print("MODEL PERFORMANCE SUMMARY")
    print("==============================")
    for model_name, metrics in combined.items():
        print(f"{model_name} Accuracy: {metrics['accuracy']:.4f}")
    print()
    print(f"Best Model: {best_model}")
