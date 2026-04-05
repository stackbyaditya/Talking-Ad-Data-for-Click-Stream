"""Train and evaluate Model5 deep-learning models on generated sequence data."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict

from tensorflow.keras.optimizers import Adam


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model5.models.deep_learning.dl_model_architectures import get_model_builders  # noqa: E402
from Model5.models.deep_learning.dl_utils import (  # noqa: E402
    build_callbacks,
    evaluate_dl_model,
    load_sequence_dataset,
    plot_training_curves,
    save_histories,
    set_global_seed,
    split_and_scale_sequences,
)
from Model5.models.model5_config import (  # noqa: E402
    BOOSTING_SUMMARY_JSON,
    DL_HISTORIES_JSON,
    DL_MODELS_DIR,
    DL_REPORTS_JSON,
    DL_SUMMARY_CSV,
    DL_SUMMARY_JSON,
    PLOTS_DIR,
    SEQUENCE_DATASET_PATH,
    SEQUENCE_SCALER_PATH,
)
from Model5.models.model5_utils import (  # noqa: E402
    ensure_model5_directories,
    plot_classwise_metric_heatmap,
    plot_metric_comparison,
    plot_multi_confusion_matrices,
    plot_roc_curves,
    save_json,
    save_summary_tables,
    split_summary_and_reports,
)
from Model5.models.prepare_sequence_dataset import prepare_sequence_dataset  # noqa: E402


LOGGER = logging.getLogger("Model5DeepLearning")


def load_boosting_summary() -> Dict[str, Dict[str, float]]:
    """Load existing Model5 boosting metrics if available."""
    if not BOOSTING_SUMMARY_JSON.exists():
        return {}
    return json.loads(BOOSTING_SUMMARY_JSON.read_text(encoding="utf-8"))


def train_deep_learning_models(epochs: int = 30, batch_size: int = 64) -> Dict[str, Dict[str, float]]:
    """Train the baseline deep-learning family for Model5."""
    ensure_model5_directories()
    set_global_seed()

    if not SEQUENCE_DATASET_PATH.exists():
        LOGGER.info("Sequence dataset missing. Generating Model5 sequence artifact first.")
        prepare_sequence_dataset()

    X_sequences, y_labels, _ = load_sequence_dataset(SEQUENCE_DATASET_PATH)
    split_bundle = split_and_scale_sequences(X_sequences, y_labels, SEQUENCE_SCALER_PATH)
    input_shape = (split_bundle["X_train"].shape[1], split_bundle["X_train"].shape[2])

    model_builders = get_model_builders()
    model_files = {
        "CNN": "cnn_model5.h5",
        "LSTM": "lstm_model5.h5",
        "CNN-LSTM": "cnn_lstm_model5.h5",
        "CNN-BiLSTM": "cnn_bilstm_model5.h5",
        "CNN-Attention-LSTM": "cnn_attention_lstm_model5.h5",
        "Transformer": "transformer_model5.h5",
    }

    histories: Dict[str, Dict[str, list[float]]] = {}
    summary: Dict[str, Dict[str, object]] = {}
    roc_inputs: Dict[str, object] = {}
    confusion_inputs: Dict[str, object] = {}

    for model_name, builder in model_builders.items():
        LOGGER.info("Training %s", model_name)
        model = builder(input_shape)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(
            split_bundle["X_train"],
            split_bundle["y_train_onehot"],
            validation_data=(split_bundle["X_val"], split_bundle["y_val_onehot"]),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=build_callbacks(),
            verbose=0,
        )
        model.save(DL_MODELS_DIR / model_files[model_name], include_optimizer=True)

        metrics, y_pred, y_proba = evaluate_dl_model(model, split_bundle["X_test"], split_bundle["y_test"])
        histories[model_name] = history.history
        summary[model_name] = metrics
        roc_inputs[model_name] = y_proba
        confusion_inputs[model_name] = y_pred
        LOGGER.info(
            "%s metrics | accuracy=%.4f f1=%.4f roc_auc=%.4f",
            model_name,
            metrics["accuracy"],
            metrics["f1_score"],
            metrics["roc_auc"],
        )

    plot_training_curves(histories, PLOTS_DIR)
    plot_roc_curves(roc_inputs, split_bundle["y_test"], PLOTS_DIR / "roc_curves_dl_models.png", "Model5 Deep Learning ROC Curves")
    plot_multi_confusion_matrices(
        confusion_inputs,
        split_bundle["y_test"],
        PLOTS_DIR / "confusion_matrix_dl_models.png",
        "Deep Learning",
    )
    metrics_only, report_only = split_summary_and_reports(summary)
    save_summary_tables(summary, DL_SUMMARY_JSON, DL_SUMMARY_CSV)
    save_json(report_only, DL_REPORTS_JSON)
    save_histories(histories, DL_HISTORIES_JSON)
    plot_metric_comparison(metrics_only, PLOTS_DIR / "model_comparison_dl.png", "Model5 Deep Learning Model Comparison")
    plot_classwise_metric_heatmap(
        report_only,
        PLOTS_DIR / "classwise_f1_dl.png",
        metric_name="f1-score",
        title="Model5 Deep Learning Class-wise F1 Scores",
    )

    boosting_summary = load_boosting_summary()
    if boosting_summary:
        combined = {**boosting_summary, **metrics_only}
        plot_metric_comparison(combined, PLOTS_DIR / "model_comparison_combined.png", "Model5 Boosting vs Deep Learning")
    return metrics_only


def main() -> None:
    """Entry point for Model5 deep-learning training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    train_deep_learning_models()


if __name__ == "__main__":
    main()
