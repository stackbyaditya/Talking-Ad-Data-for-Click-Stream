"""Unified training and evaluation pipeline for improved deep learning models."""

from __future__ import annotations

import importlib.util
import json
import logging
import random
import sys
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
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


LOGGER = logging.getLogger("run_improved_dl_models")
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
DATASET_PATH = REPO_ROOT / "model_outputs" / "lstm_training_data_v3.npz"
SCALER_PATH = MODEL_ROOT / "outputs" / "sequence_scaler_improved.pkl"
SAVED_MODELS_DIR = MODEL_ROOT / "outputs" / "saved_models"
PLOTS_DIR = MODEL_ROOT / "analysis" / "plots"
SUMMARY_PATH = MODEL_ROOT / "outputs" / "improved_model_performance.json"
CLASS_NAMES = ["human", "moderate_bot", "advanced_bot"]


def configure_logging() -> None:
    """Configure INFO logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_module(module_path: Path, module_name: str):
    """Load a Python module from a file path without modifying repo-wide imports."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_model_builders() -> Tuple[Dict[str, callable], Dict[str, object]]:
    """Load architecture and utility modules and discover trainable model builders."""
    architecture_module = load_module(MODULE_DIR / "dl_model_architectures.py", "dl_model_architectures_improved")
    utilities_module = load_module(MODULE_DIR / "dl_utils.py", "dl_utils_improved")

    if hasattr(architecture_module, "get_model_builders"):
        builders = architecture_module.get_model_builders()
    else:
        builders = {
            name.replace("build_", "").replace("_model", "").replace("_", "-"): getattr(architecture_module, name)
            for name in dir(architecture_module)
            if name.startswith("build_") and name.endswith("_model")
        }
    if not builders:
        raise ValueError("No model builders found in models/deep_learning_improved/dl_model_architectures.py")
    return builders, {"architecture_module": architecture_module, "utilities_module": utilities_module}


def load_sequence_dataset(input_path: Path = DATASET_PATH) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the existing v3 sequence dataset."""
    payload = np.load(input_path, allow_pickle=True)
    X_sequences = payload["X_sequences"].astype(np.float32)
    y_labels = payload["y_labels"].astype(np.int64)
    feature_names = payload["feature_names"].tolist()
    LOGGER.info("Loaded sequence dataset: X=%s y=%s", X_sequences.shape, y_labels.shape)
    return X_sequences, y_labels, feature_names


def scale_sequences(X_sequences: np.ndarray, scaler_path: Path) -> np.ndarray:
    """Scale features across all samples and timesteps using StandardScaler."""
    scaler = StandardScaler()
    samples, timesteps, features = X_sequences.shape
    flattened = X_sequences.reshape(samples * timesteps, features)
    scaled = scaler.fit_transform(flattened).reshape(samples, timesteps, features).astype(np.float32)
    joblib.dump(scaler, scaler_path)
    LOGGER.info("Saved scaler to %s", scaler_path)
    return scaled


def split_dataset(X_sequences: np.ndarray, y_labels: np.ndarray) -> Dict[str, np.ndarray]:
    """Create stratified 70/15/15 train, validation, and test splits."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_sequences,
        y_labels,
        test_size=0.15,
        stratify=y_labels,
        random_state=42,
    )
    validation_fraction = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction,
        stratify=y_train_val,
        random_state=42,
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


def build_callbacks() -> List[tf.keras.callbacks.Callback]:
    """Create common training callbacks for improved models."""
    return [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, verbose=1, min_lr=1e-5),
    ]


def evaluate_model(model, X_eval: np.ndarray, y_eval: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate a model and compute classification metrics."""
    y_proba = model.predict(X_eval, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "precision": float(precision_score(y_eval, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_eval, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_eval, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_eval, y_proba, multi_class="ovr", average="weighted")),
        "classification_report": classification_report(
            y_eval,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, y_pred, y_proba


def sanitize_model_name(model_name: str) -> str:
    """Convert model name into a safe file stem."""
    return model_name.lower().replace(" ", "_").replace("-", "_")


def plot_training_curves(histories: Dict[str, Dict[str, List[float]]], plots_dir: Path) -> None:
    """Save combined training accuracy and loss curves."""
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    for model_name, history in histories.items():
        ax_acc.plot(history["accuracy"], label=f"{model_name} train")
        ax_acc.plot(history["val_accuracy"], linestyle="--", label=f"{model_name} val")
    ax_acc.set_title("Improved Model Training Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(fontsize=8)
    fig_acc.tight_layout()
    fig_acc.savefig(plots_dir / "training_accuracy_curves.png", dpi=160)
    plt.close(fig_acc)

    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    for model_name, history in histories.items():
        ax_loss.plot(history["loss"], label=f"{model_name} train")
        ax_loss.plot(history["val_loss"], linestyle="--", label=f"{model_name} val")
    ax_loss.set_title("Improved Model Training Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=8)
    fig_loss.tight_layout()
    fig_loss.savefig(plots_dir / "training_loss_curves.png", dpi=160)
    plt.close(fig_loss)


def plot_confusion_matrices(confusion_inputs: Dict[str, np.ndarray], y_true: np.ndarray, output_path: Path) -> None:
    """Save confusion matrices for all improved models in a single figure."""
    cols = 3
    rows = int(np.ceil(len(confusion_inputs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_1d(axes).flatten()
    for idx, (model_name, y_pred) in enumerate(confusion_inputs.items()):
        matrix = confusion_matrix(y_true, y_pred)
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


def plot_roc_curves(roc_inputs: Dict[str, np.ndarray], y_true: np.ndarray, output_path: Path) -> None:
    """Save ROC curves for all improved models."""
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, y_proba in roc_inputs.items():
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        auc_value = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curves - Improved Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_model_comparison(summary: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Save grouped comparison chart for improved models."""
    model_names = list(summary.keys())
    metrics = ["accuracy", "f1_score", "roc_auc"]
    x = np.arange(len(model_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, metric in enumerate(metrics):
        values = [summary[name][metric] for name in model_names]
        ax.bar(x + (idx - 1) * width, values, width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Improved Deep Learning Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_summary(summary: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Persist metrics summary without nested classification reports."""
    serializable = {}
    for model_name, metrics in summary.items():
        serializable[model_name] = {
            key: value for key, value in metrics.items() if key != "classification_report"
        }
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def print_summary(summary: Dict[str, Dict[str, float]]) -> None:
    """Print the required console summary table."""
    best_model = max(summary.items(), key=lambda item: item[1]["accuracy"])[0]
    print("=================================")
    print("IMPROVED DEEP LEARNING RESULTS")
    print("=================================")
    print("Model                Accuracy    F1 Score    ROC-AUC")
    print("----------------------------------------------------")
    for model_name, metrics in summary.items():
        print(f"{model_name:<20} {metrics['accuracy']:.4f}      {metrics['f1_score']:.4f}      {metrics['roc_auc']:.4f}")
    print()
    print(f"Best Model: {best_model}")


def main() -> None:
    """Run the unified improved deep-learning training pipeline."""
    configure_logging()
    set_global_seed(42)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    builders, metadata = discover_model_builders()
    LOGGER.info("Discovered improved model builders: %s", list(builders.keys()))

    X_sequences, y_labels, feature_names = load_sequence_dataset(DATASET_PATH)
    X_sequences = scale_sequences(X_sequences, SCALER_PATH)
    split_bundle = split_dataset(X_sequences, y_labels)

    input_shape = (split_bundle["X_train"].shape[1], split_bundle["X_train"].shape[2])
    LOGGER.info("Using input shape %s with %s features.", input_shape, len(feature_names))

    histories: Dict[str, Dict[str, List[float]]] = {}
    summary: Dict[str, Dict[str, float]] = {}
    roc_inputs: Dict[str, np.ndarray] = {}
    confusion_inputs: Dict[str, np.ndarray] = {}

    for model_name, builder in builders.items():
        improved_name = f"{model_name.replace('-', '_')}_improved"
        LOGGER.info("Training %s", improved_name)
        model = builder(input_shape)
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss=CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        history = model.fit(
            split_bundle["X_train"],
            split_bundle["y_train_onehot"],
            validation_data=(split_bundle["X_val"], split_bundle["y_val_onehot"]),
            batch_size=64,
            epochs=50,
            callbacks=build_callbacks(),
            verbose=0,
        )

        model_path = SAVED_MODELS_DIR / f"{sanitize_model_name(improved_name)}.h5"
        model.save(model_path, include_optimizer=True)

        metrics, y_pred, y_proba = evaluate_model(
            model,
            split_bundle["X_test"],
            split_bundle["y_test"],
        )
        histories[improved_name] = history.history
        summary[improved_name] = metrics
        roc_inputs[improved_name] = y_proba
        confusion_inputs[improved_name] = y_pred
        LOGGER.info(
            "%s complete. Accuracy=%.4f F1=%.4f ROC_AUC=%.4f",
            improved_name,
            metrics["accuracy"],
            metrics["f1_score"],
            metrics["roc_auc"],
        )

    plot_training_curves(histories, PLOTS_DIR)
    plot_confusion_matrices(confusion_inputs, split_bundle["y_test"], PLOTS_DIR / "confusion_matrix_improved_models.png")
    plot_roc_curves(roc_inputs, split_bundle["y_test"], PLOTS_DIR / "roc_curves_improved_models.png")
    plot_model_comparison(summary, PLOTS_DIR / "model_comparison_improved.png")
    save_summary(summary, SUMMARY_PATH)
    print_summary(summary)


if __name__ == "__main__":
    main()
