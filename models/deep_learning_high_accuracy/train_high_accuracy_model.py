"""Train an isolated high-capacity CNN-BiLSTM sequence classifier."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.deep_learning_high_accuracy.high_accuracy_architecture import (  # noqa: E402
    build_high_accuracy_cnn_bilstm,
)
from models.deep_learning_high_accuracy.high_accuracy_utils import (  # noqa: E402
    build_callbacks,
    compute_training_class_weights,
    evaluate_split,
    load_sequence_dataset,
    plot_confusion,
    plot_roc,
    plot_training_history,
    print_summary,
    save_results,
    scale_sequences,
    set_global_seed,
    split_dataset,
)


LOGGER = logging.getLogger("train_high_accuracy_model")
SEQUENCE_DATA_PATH = REPO_ROOT / "model_outputs" / "lstm_training_data_v3.npz"
SCALER_PATH = REPO_ROOT / "model_outputs" / "high_accuracy_sequence_scaler.pkl"
MODEL_PATH = REPO_ROOT / "model_outputs" / "high_accuracy_cnn_bilstm_model.h5"
RESULTS_PATH = REPO_ROOT / "model_outputs" / "high_accuracy_model_results.json"
PLOTS_DIR = REPO_ROOT / "analysis" / "plots"


def configure_logging() -> None:
    """Configure INFO logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main() -> None:
    """Run the isolated high-accuracy CNN-BiLSTM experiment."""
    configure_logging()
    set_global_seed(42)

    LOGGER.info("Loading sequence dataset.")
    X_sequences, y_labels, feature_names = load_sequence_dataset(SEQUENCE_DATA_PATH)
    LOGGER.info("Normalizing sequences with StandardScaler.")
    X_scaled, _ = scale_sequences(X_sequences, SCALER_PATH)

    LOGGER.info("Creating train/validation/test splits.")
    split_bundle = split_dataset(X_scaled, y_labels, random_state=42)
    class_weights = compute_training_class_weights(split_bundle["y_train"])

    LOGGER.info("Building high-accuracy CNN-BiLSTM model.")
    model = build_high_accuracy_cnn_bilstm(
        input_shape=(X_scaled.shape[1], X_scaled.shape[2]),
        num_classes=3,
    )
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    LOGGER.info("Training high-accuracy CNN-BiLSTM.")
    history = model.fit(
        split_bundle["X_train"],
        split_bundle["y_train_onehot"],
        validation_data=(split_bundle["X_val"], split_bundle["y_val_onehot"]),
        epochs=60,
        batch_size=64,
        callbacks=build_callbacks(),
        class_weight=class_weights,
        verbose=0,
    )

    LOGGER.info("Evaluating validation split.")
    validation_metrics, _, _ = evaluate_split(
        model,
        split_bundle["X_val"],
        split_bundle["y_val"],
    )
    LOGGER.info("Evaluating test split.")
    test_metrics, test_predictions, test_probabilities = evaluate_split(
        model,
        split_bundle["X_test"],
        split_bundle["y_test"],
    )

    LOGGER.info("Saving high-accuracy model to %s", MODEL_PATH)
    model.save(MODEL_PATH, include_optimizer=True)

    plot_training_history(history.history, PLOTS_DIR)
    plot_confusion(
        split_bundle["y_test"],
        test_predictions,
        PLOTS_DIR / "high_accuracy_confusion_matrix.png",
    )
    plot_roc(
        split_bundle["y_test"],
        test_probabilities,
        PLOTS_DIR / "high_accuracy_roc_curve.png",
    )

    results = {
        "model": "HighAccuracy-CNN-BiLSTM",
        "feature_names": feature_names,
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_precision": validation_metrics["precision"],
        "validation_recall": validation_metrics["recall"],
        "validation_f1_score": validation_metrics["f1_score"],
        "validation_roc_auc": validation_metrics["roc_auc"],
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1_score": test_metrics["f1_score"],
        "roc_auc": test_metrics["roc_auc"],
        "class_weights": class_weights,
    }
    save_results(results, RESULTS_PATH)
    print_summary(validation_metrics, test_metrics)


if __name__ == "__main__":
    main()
