"""Train and evaluate deep learning models on temporal behavioural sequences."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict

from tensorflow.keras.optimizers import Adam


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.deep_learning.dl_model_architectures import get_model_builders  # noqa: E402
from models.deep_learning.dl_utils import (  # noqa: E402
    build_callbacks,
    evaluate_dl_model,
    load_sequence_dataset,
    plot_confusion_matrices_dl,
    plot_model_comparison,
    plot_roc_curves_dl,
    plot_training_curves,
    print_performance_summary,
    save_summary,
    set_global_seed,
    split_and_scale_sequences,
)


LOGGER = logging.getLogger("train_dl_models")
OUTPUT_DIR = REPO_ROOT / "model_outputs"
PLOTS_DIR = REPO_ROOT / "analysis" / "plots"
SEQUENCE_SCALER_PATH = OUTPUT_DIR / "sequence_scaler.pkl"
BOOSTING_SUMMARY_PATH = OUTPUT_DIR / "model_performance_summary.json"


def configure_logging() -> None:
    """Configure INFO-level logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_boosting_summary(summary_path: Path = BOOSTING_SUMMARY_PATH) -> Dict[str, Dict[str, float]]:
    """Load the existing boosting-model summary."""
    return json.loads(summary_path.read_text(encoding="utf-8"))


def train_deep_learning_models() -> Dict[str, Dict[str, float]]:
    """Train all deep learning models and save artifacts."""
    set_global_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading sequence dataset.")
    X_sequences, y_labels, _ = load_sequence_dataset()
    split_bundle = split_and_scale_sequences(X_sequences, y_labels, SEQUENCE_SCALER_PATH, random_state=42)

    X_train = split_bundle["X_train"]
    X_val = split_bundle["X_val"]
    X_test = split_bundle["X_test"]
    y_train = split_bundle["y_train"]
    y_val = split_bundle["y_val"]
    y_test = split_bundle["y_test"]
    y_train_onehot = split_bundle["y_train_onehot"]
    y_val_onehot = split_bundle["y_val_onehot"]

    input_shape = (X_train.shape[1], X_train.shape[2])
    model_builders = get_model_builders()
    model_filenames = {
        "CNN": "cnn_model.h5",
        "LSTM": "lstm_model.h5",
        "CNN-LSTM": "cnn_lstm_model.h5",
        "CNN-BiLSTM": "cnn_bilstm_model.h5",
        "CNN-Attention-LSTM": "cnn_attention_lstm_model.h5",
    }

    histories: Dict[str, Dict[str, list[float]]] = {}
    summary: Dict[str, Dict[str, float]] = {}
    roc_inputs: Dict[str, object] = {}
    confusion_inputs: Dict[str, object] = {}

    for model_name, builder in model_builders.items():
        LOGGER.info("Training %s", model_name)
        model = builder(input_shape)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(
            X_train,
            y_train_onehot,
            validation_data=(X_val, y_val_onehot),
            batch_size=64,
            epochs=30,
            callbacks=build_callbacks(),
            verbose=0,
        )
        model.save(OUTPUT_DIR / model_filenames[model_name], include_optimizer=True)

        metrics, y_pred, y_proba = evaluate_dl_model(model, X_test, y_test)
        histories[model_name] = history.history
        summary[model_name] = metrics
        roc_inputs[model_name] = y_proba
        confusion_inputs[model_name] = y_pred
        LOGGER.info(
            "%s evaluation complete. Accuracy=%.4f F1=%.4f ROC_AUC=%.4f",
            model_name,
            metrics["accuracy"],
            metrics["f1_score"],
            metrics["roc_auc"],
        )

    plot_training_curves(histories, PLOTS_DIR)
    plot_roc_curves_dl(roc_inputs, y_test, PLOTS_DIR / "roc_curves_dl_models.png")
    plot_confusion_matrices_dl(confusion_inputs, y_test, PLOTS_DIR / "confusion_matrix_dl_models.png")

    boosting_summary = load_boosting_summary()
    plot_model_comparison(boosting_summary, summary, PLOTS_DIR / "model_comparison_bar_chart.png")
    save_summary(summary, OUTPUT_DIR / "dl_model_performance_summary.json")
    print_performance_summary(boosting_summary, summary)
    return summary


def main() -> None:
    """Entry point for deep learning model training."""
    configure_logging()
    train_deep_learning_models()


if __name__ == "__main__":
    main()
