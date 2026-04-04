"""Train and evaluate deep learning models on temporal behavioural sequences.

v2 improvements over v1:
- Uses focal_loss instead of categorical cross-entropy.
- Passes class_weight dict to model.fit() to correct for label imbalance.
- Trains the new Transformer model alongside the existing five.
- After all base models are trained, builds + evaluates a soft-voting Ensemble.
- Epochs raised from 30 → 50; callbacks have patience 7 so early-stopping
  still fires if the model has genuinely converged.
- Output summary saved to dl_model_performance_summary_v2.json.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict

from tensorflow.keras.optimizers import Adam


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model2.models.deep_learning_improved.dl_model_architectures import (  # noqa: E402
    build_ensemble_model,
    get_model_builders,
)
from Model2.models.deep_learning_improved.dl_utils import (  # noqa: E402
    build_callbacks,
    compute_class_weights,
    evaluate_dl_model,
    focal_loss,
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
OUTPUT_DIR = MODEL_ROOT / "outputs"
PLOTS_DIR = MODEL_ROOT / "analysis" / "plots"
SEQUENCE_SCALER_PATH = OUTPUT_DIR / "sequence_scaler_v2.pkl"
BOOSTING_SUMMARY_PATH = REPO_ROOT / "Model1" / "outputs" / "model_performance_summary.json"

# Model filename map — Ensemble is assembled at runtime, not loaded from disk
MODEL_FILENAMES = {
    "CNN":                 "cnn_model_v2.h5",
    "LSTM":                "lstm_model_v2.h5",
    "CNN-LSTM":            "cnn_lstm_model_v2.h5",
    "CNN-BiLSTM":          "cnn_bilstm_model_v2.h5",
    "CNN-Attention-LSTM":  "cnn_attention_lstm_model_v2.h5",
    "Transformer":         "transformer_model_v2.h5",
}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_boosting_summary(summary_path: Path = BOOSTING_SUMMARY_PATH) -> Dict[str, Dict[str, float]]:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def train_deep_learning_models() -> Dict[str, Dict[str, float]]:
    set_global_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────
    LOGGER.info("Loading sequence dataset (v4).")
    X_sequences, y_labels, _ = load_sequence_dataset()
    split_bundle = split_and_scale_sequences(X_sequences, y_labels, SEQUENCE_SCALER_PATH, random_state=42)

    X_train      = split_bundle["X_train"]
    X_val        = split_bundle["X_val"]
    X_test       = split_bundle["X_test"]
    y_train      = split_bundle["y_train"]
    y_val        = split_bundle["y_val"]
    y_test       = split_bundle["y_test"]
    y_train_oh   = split_bundle["y_train_onehot"]
    y_val_oh     = split_bundle["y_val_onehot"]

    input_shape = (X_train.shape[1], X_train.shape[2])
    LOGGER.info("Input shape: %s", input_shape)

    # ── class weights (corrects 5k human / 2k bot imbalance) ─────────────
    class_weights = compute_class_weights(y_train)

    # ── focal loss instance ───────────────────────────────────────────────
    loss_fn = focal_loss(gamma=2.0, alpha=0.25)

    model_builders = get_model_builders()
    histories:        Dict[str, Dict] = {}
    summary:          Dict[str, Dict] = {}
    roc_inputs:       Dict[str, object] = {}
    confusion_inputs: Dict[str, object] = {}
    trained_models    = []   # kept in memory for ensemble assembly

    # ── train each model ──────────────────────────────────────────────────
    for model_name, builder in model_builders.items():
        LOGGER.info("Training %s …", model_name)
        model = builder(input_shape)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss_fn,
            metrics=["accuracy"],
        )
        history = model.fit(
            X_train, y_train_oh,
            validation_data=(X_val, y_val_oh),
            batch_size=64,
            epochs=50,                  # ↑ from 30; early-stopping fires when needed
            callbacks=build_callbacks(),
            class_weight=class_weights,
            verbose=0,
        )
        model.save(OUTPUT_DIR / MODEL_FILENAMES[model_name], include_optimizer=True)
        trained_models.append(model)

        metrics, y_pred, y_proba = evaluate_dl_model(model, X_test, y_test)
        histories[model_name]        = history.history
        summary[model_name]          = metrics
        roc_inputs[model_name]       = y_proba
        confusion_inputs[model_name] = y_pred

        LOGGER.info(
            "%s — Acc=%.4f  F1=%.4f  AUC=%.4f",
            model_name, metrics["accuracy"], metrics["f1_score"], metrics["roc_auc"],
        )

    # ── soft-voting Ensemble ──────────────────────────────────────────────
    LOGGER.info("Building soft-voting Ensemble from %d base models …", len(trained_models))
    ensemble = build_ensemble_model(trained_models, input_shape)
    # Ensemble needs no training — it averages already-trained softmax outputs.
    ensemble_metrics, ens_pred, ens_proba = evaluate_dl_model(ensemble, X_test, y_test)
    summary["Ensemble"]          = ensemble_metrics
    roc_inputs["Ensemble"]       = ens_proba
    confusion_inputs["Ensemble"] = ens_pred
    LOGGER.info(
        "Ensemble — Acc=%.4f  F1=%.4f  AUC=%.4f",
        ensemble_metrics["accuracy"], ensemble_metrics["f1_score"], ensemble_metrics["roc_auc"],
    )

    # ── plots ─────────────────────────────────────────────────────────────
    plot_training_curves(histories, PLOTS_DIR)
    plot_roc_curves_dl(roc_inputs, y_test, PLOTS_DIR / "roc_curves_dl_models_v2.png")
    plot_confusion_matrices_dl(confusion_inputs, y_test, PLOTS_DIR / "confusion_matrix_dl_models_v2.png")

    boosting_summary = load_boosting_summary()
    plot_model_comparison(boosting_summary, summary, PLOTS_DIR / "model_comparison_bar_chart_v2.png")

    # ── save & print ─────────────────────────────────────────────────────
    save_summary(summary, OUTPUT_DIR / "dl_model_performance_summary_v2.json")
    print_performance_summary(boosting_summary, summary)
    return summary


def main() -> None:
    configure_logging()
    train_deep_learning_models()


if __name__ == "__main__":
    main()
