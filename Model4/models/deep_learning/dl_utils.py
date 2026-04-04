"""Utilities for Model4 deep-learning training and sequence evaluation."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from Model4.models.model4_config import CLASS_NAMES, RANDOM_STATE


LOGGER = logging.getLogger("Model4DLUtils")


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Set seeds for reproducible Model4 deep-learning training."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_sequence_dataset(input_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the Model4 sequence dataset from disk."""
    payload = np.load(input_path, allow_pickle=True)
    X_sequences = payload["X_sequences"].astype(np.float32)
    y_labels = payload["y_labels"].astype(np.int64)
    feature_names = payload["feature_names"].tolist()
    LOGGER.info("Loaded Model4 sequence dataset: X=%s y=%s", X_sequences.shape, y_labels.shape)
    return X_sequences, y_labels, feature_names


def split_and_scale_sequences(
    X_sequences: np.ndarray,
    y_labels: np.ndarray,
    scaler_path: Path,
    random_state: int = RANDOM_STATE,
) -> Dict[str, np.ndarray]:
    """Create stratified train/validation/test splits and save a StandardScaler."""
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
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    joblib.dump(scaler, scaler_path)

    return {
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


def build_callbacks() -> List[tf.keras.callbacks.Callback]:
    """Create common callbacks for Model4 deep-learning training."""
    return [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5, verbose=1),
    ]


def evaluate_dl_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    """Evaluate a deep-learning classifier on the Model4 test split."""
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
    """Save combined accuracy and loss curves for Model4 deep-learning models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    for model_name, history in histories.items():
        ax_loss.plot(history["loss"], label=f"{model_name} train")
        ax_loss.plot(history["val_loss"], linestyle="--", label=f"{model_name} val")
    ax_loss.set_title("Model4 Deep Learning Training Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=8)
    fig_loss.tight_layout()
    fig_loss.savefig(output_dir / "training_loss_curves.png", dpi=180)
    plt.close(fig_loss)

    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    for model_name, history in histories.items():
        ax_acc.plot(history["accuracy"], label=f"{model_name} train")
        ax_acc.plot(history["val_accuracy"], linestyle="--", label=f"{model_name} val")
    ax_acc.set_title("Model4 Deep Learning Training Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(fontsize=8)
    fig_acc.tight_layout()
    fig_acc.savefig(output_dir / "training_accuracy_curves.png", dpi=180)
    plt.close(fig_acc)


def save_histories(histories: Dict[str, Dict[str, List[float]]], output_path: Path) -> None:
    """Persist Keras training histories as JSON."""
    serializable = {
        model_name: {
            metric_name: [float(value) for value in values]
            for metric_name, values in history.items()
        }
        for model_name, history in histories.items()
    }
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
