"""Utility functions for deep learning sequence training and evaluation.

v2 improvements over v1:
- focal_loss() replaces categorical cross-entropy to focus on hard examples.
- compute_class_weights() corrects for the 5000/2000/2000 imbalance.
- evaluate_dl_model() also accepts an optional ensemble model.
- load_sequence_dataset() defaults to the new v4 .npz file.
- All other helpers (plotting, summary saving, …) are unchanged.
"""

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
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


LOGGER = logging.getLogger(__name__)
CLASS_NAMES = ["human", "moderate_bot", "advanced_bot"]
# ↓ updated to v4 — change back to v3 if you still want the old sequences
MODEL_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_DATA_PATH = MODEL_ROOT / "outputs" / "lstm_training_data_v4.npz"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Focal loss for multi-class classification.

    Reduces the loss contribution from easy-to-classify examples so the
    model focuses on hard/misclassified ones — critical when bot patterns
    overlap with human sessions.

    Parameters
    ----------
    gamma : focusing parameter (2.0 is the standard choice).
    alpha : base weighting factor (fine-tune if needed).
    """
    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -y_true * tf.math.log(y_pred)                         # cross-entropy per class
        weight = alpha * y_true * tf.pow(1.0 - y_pred, gamma)      # focal weight
        focal = weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))

    loss_fn.__name__ = f"focal_loss_g{gamma}"
    return loss_fn


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(y_labels: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights to counter the 5k/2k/2k imbalance."""
    classes = np.unique(y_labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_labels)
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    LOGGER.info("Class weights: %s", weight_dict)
    return weight_dict


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_sequence_dataset(
    input_path: Path = SEQUENCE_DATA_PATH,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    payload = np.load(input_path, allow_pickle=True)
    X_sequences = payload["X_sequences"].astype(np.float32)
    y_labels = payload["y_labels"].astype(np.int64)
    feature_names = payload["feature_names"].tolist()
    LOGGER.info("Loaded sequence dataset: X=%s y=%s", X_sequences.shape, y_labels.shape)
    return X_sequences, y_labels, feature_names


# ---------------------------------------------------------------------------
# Train / val / test split + scaling
# ---------------------------------------------------------------------------

def split_and_scale_sequences(
    X_sequences: np.ndarray,
    y_labels: np.ndarray,
    scaler_path: Path,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_sequences, y_labels,
        test_size=0.15, stratify=y_labels, random_state=random_state,
    )
    validation_fraction = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=validation_fraction, stratify=y_train_val, random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_s   = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_s  = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    joblib.dump(scaler, scaler_path)

    bundle = {
        "X_train": X_train_s.astype(np.float32),
        "X_val":   X_val_s.astype(np.float32),
        "X_test":  X_test_s.astype(np.float32),
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "y_train_onehot": to_categorical(y_train, num_classes=3),
        "y_val_onehot":   to_categorical(y_val,   num_classes=3),
        "y_test_onehot":  to_categorical(y_test,  num_classes=3),
    }
    LOGGER.info(
        "Split → train=%s  val=%s  test=%s",
        bundle["X_train"].shape, bundle["X_val"].shape, bundle["X_test"].shape,
    )
    return bundle


# ---------------------------------------------------------------------------
# Training callbacks
# ---------------------------------------------------------------------------

def build_callbacks() -> List[tf.keras.callbacks.Callback]:
    return [
        EarlyStopping(
            monitor="val_loss", patience=7,      # ↑ from 5 — give models more time
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,  # ↑ patience from 2
            min_lr=1e-6, verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dl_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score":  float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")),
        "classification_report": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0,
        ),
    }
    return metrics, y_pred, y_proba


# ---------------------------------------------------------------------------
# Plotting helpers  (unchanged from v1)
# ---------------------------------------------------------------------------

def plot_training_curves(histories: Dict[str, Dict[str, List[float]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    for model_name, history in histories.items():
        ax_loss.plot(history["loss"], label=f"{model_name} train")
        ax_loss.plot(history["val_loss"], linestyle="--", label=f"{model_name} val")
    ax_loss.set_title("Training Loss Curves")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=7)
    fig_loss.tight_layout()
    fig_loss.savefig(output_dir / "training_loss_curves.png", dpi=160)
    plt.close(fig_loss)

    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    for model_name, history in histories.items():
        ax_acc.plot(history["accuracy"], label=f"{model_name} train")
        ax_acc.plot(history["val_accuracy"], linestyle="--", label=f"{model_name} val")
    ax_acc.set_title("Training Accuracy Curves")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(fontsize=7)
    fig_acc.tight_layout()
    fig_acc.savefig(output_dir / "training_accuracy_curves.png", dpi=160)
    plt.close(fig_acc)


def plot_roc_curves_dl(
    roc_inputs: Dict[str, np.ndarray], y_test: np.ndarray, output_path: Path
) -> None:
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, y_proba in roc_inputs.items():
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
        auc_value = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curves — Deep Learning Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrices_dl(
    confusion_inputs: Dict[str, np.ndarray],
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    n = len(confusion_inputs)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()
    for idx, (model_name, y_pred) in enumerate(confusion_inputs.items()):
        matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[idx],
        )
        axes[idx].set_title(model_name)
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
    for idx in range(n, len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_model_comparison(
    boosting_summary: Dict[str, Dict[str, float]],
    dl_summary: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    combined = {**boosting_summary, **dl_summary}
    model_names = list(combined.keys())
    metrics = ["accuracy", "f1_score", "roc_auc"]
    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
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


# ---------------------------------------------------------------------------
# Summary persistence
# ---------------------------------------------------------------------------

def save_summary(summary: Dict[str, Dict[str, float]], output_path: Path) -> None:
    serializable = {
        name: {k: v for k, v in metrics.items() if k != "classification_report"}
        for name, metrics in summary.items()
    }
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def print_performance_summary(
    boosting_summary: Dict[str, Dict[str, float]],
    dl_summary: Dict[str, Dict[str, float]],
) -> None:
    combined = {**boosting_summary, **dl_summary}
    best_model = max(combined.items(), key=lambda item: item[1]["accuracy"])[0]

    print("=" * 40)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 40)
    for model_name, metrics in combined.items():
        print(
            f"{model_name:<28} Acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1_score']:.4f}  AUC={metrics['roc_auc']:.4f}"
        )
    print()
    print(f"Best Model: {best_model}")
