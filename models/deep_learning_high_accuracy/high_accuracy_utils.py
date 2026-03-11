"""Utilities for the isolated high-accuracy CNN-BiLSTM experiment."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, Tuple

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
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


LOGGER = logging.getLogger(__name__)
CLASS_NAMES = ["human", "moderate_bot", "advanced_bot"]


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_sequence_dataset(input_path: Path) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the pre-generated behavioural sequence dataset."""
    payload = np.load(input_path, allow_pickle=True)
    X_sequences = payload["X_sequences"].astype(np.float32)
    y_labels = payload["y_labels"].astype(np.int64)
    feature_names = payload["feature_names"].tolist()
    LOGGER.info("Loaded sequence dataset: X=%s y=%s", X_sequences.shape, y_labels.shape)
    return X_sequences, y_labels, feature_names


def scale_sequences(
    X_sequences: np.ndarray,
    scaler_path: Path,
) -> Tuple[np.ndarray, StandardScaler]:
    """Apply StandardScaler across the flattened timestep dimension."""
    sample_count, timestep_count, feature_count = X_sequences.shape
    scaler = StandardScaler()
    flattened = X_sequences.reshape(sample_count * timestep_count, feature_count)
    scaled = scaler.fit_transform(flattened).reshape(sample_count, timestep_count, feature_count)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    LOGGER.info("Saved high-accuracy sequence scaler to %s", scaler_path)
    return scaled.astype(np.float32), scaler


def split_dataset(
    X_sequences: np.ndarray,
    y_labels: np.ndarray,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Create stratified train/validation/test splits."""
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
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_onehot": to_categorical(y_train, num_classes=3),
        "y_val_onehot": to_categorical(y_val, num_classes=3),
        "y_test_onehot": to_categorical(y_test, num_classes=3),
    }


def compute_training_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Compute sklearn-balanced class weights from the training split."""
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    LOGGER.info("Computed class weights: %s", class_weights)
    return class_weights


def build_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Create the longer-running callback set for the experiment."""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=4,
            verbose=1,
            min_lr=1e-5,
        ),
    ]


def evaluate_split(
    model,
    X_split: np.ndarray,
    y_split: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Compute evaluation metrics for a validation or test split."""
    y_proba = model.predict(X_split, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_split, y_pred)),
        "precision": float(precision_score(y_split, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_split, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_split, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_split, y_proba, multi_class="ovr", average="weighted")),
        "classification_report": classification_report(
            y_split,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, y_pred, y_proba


def plot_training_history(history: Dict[str, list[float]], plots_dir: Path) -> None:
    """Save training accuracy and loss curves."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    ax_acc.plot(history["accuracy"], label="train_accuracy")
    ax_acc.plot(history["val_accuracy"], label="val_accuracy", linestyle="--")
    ax_acc.set_title("High Accuracy Training Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    fig_acc.tight_layout()
    fig_acc.savefig(plots_dir / "high_accuracy_training_accuracy.png", dpi=160)
    plt.close(fig_acc)

    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(history["loss"], label="train_loss")
    ax_loss.plot(history["val_loss"], label="val_loss", linestyle="--")
    ax_loss.set_title("High Accuracy Training Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    fig_loss.tight_layout()
    fig_loss.savefig(plots_dir / "high_accuracy_training_loss.png", dpi=160)
    plt.close(fig_loss)


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save the experiment confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title("High Accuracy CNN-BiLSTM Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
) -> None:
    """Save the experiment ROC curve plot."""
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    auc_value = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"HighAccuracy-CNN-BiLSTM (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("High Accuracy ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_results(results: Dict[str, object], output_path: Path) -> None:
    """Persist the metrics summary JSON."""
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def print_summary(validation_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> None:
    """Print the requested console summary."""
    print("=============================")
    print("HIGH ACCURACY MODEL RESULTS")
    print("=============================")
    print(f"Validation Accuracy: {validation_metrics['accuracy'] * 100:.2f}%")
    print(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
    print()
    print("Best performing model: HighAccuracy CNN-BiLSTM")
