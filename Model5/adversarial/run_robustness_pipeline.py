"""End-to-end adversarial robustness pipeline runner for Model5."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Callable, Dict

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model5.adversarial.attacks.domain_constraints import SequenceDomainConstraints, TabularDomainConstraints  # noqa: E402
from Model5.adversarial.attacks.fgsm import fgsm_attack, fgsm_sequence_attack  # noqa: E402
from Model5.adversarial.attacks.pgd import pgd_attack, pgd_sequence_attack  # noqa: E402
from Model5.adversarial.defense.adversarial_training import (  # noqa: E402
    augment_training_data,
    train_hardened_sequence_model,
    train_hardened_tabular_model,
)
from Model5.adversarial.defense.feature_squeezing import FeatureSqueezer  # noqa: E402
from Model5.adversarial.evaluation.robustness_metrics import attack_success_rate, epsilon_accuracy_curve  # noqa: E402
from Model5.adversarial.evaluation.robustness_report import (  # noqa: E402
    plot_clean_vs_attacked_vs_hardened,
    plot_defense_recovery,
    plot_epsilon_curve,
    save_report,
    write_markdown_report,
)
from Model5.adversarial.surrogate_model import SurrogateModel  # noqa: E402
from Model5.adversarial.threat_model import DEFAULT_THREAT_MODEL  # noqa: E402
from Model5.models.deep_learning.dl_model_architectures import get_model_builders  # noqa: E402
from Model5.models.deep_learning.dl_utils import evaluate_dl_model  # noqa: E402
from Model5.models.model5_config import (  # noqa: E402
    ADVERSARIAL_DIR,
    ADVERSARIAL_EXAMPLES_DIR,
    DL_MODELS_DIR,
    DL_SUMMARY_JSON,
    EPSILON_GRID,
    HARDENED_MODELS_DIR,
    ROBUSTNESS_MARKDOWN_PATH,
    ROBUSTNESS_SUMMARY_JSON,
    SEQUENCE_DATASET_PATH,
    SEQUENCE_METADATA_PATH,
    SEQUENCE_SPLIT_ARTIFACTS_PATH,
    SURROGATE_MODEL_PATH,
    TABULAR_SPLIT_ARTIFACTS_PATH,
)
from Model5.models.model5_utils import build_metric_dict, ensure_model5_directories, load_json_if_exists, save_json  # noqa: E402


LOGGER = logging.getLogger("Model5Robustness")
TABULAR_MODEL_FILES = {
    "RandomForest": "random_forest_model5.pkl",
    "XGBoost": "xgboost_model5.pkl",
    "LightGBM": "lightgbm_model5.pkl",
}
SEQUENCE_MODEL_FILES = {
    "CNN": "cnn_model5.h5",
    "LSTM": "lstm_model5.h5",
    "CNN-LSTM": "cnn_lstm_model5.h5",
    "CNN-BiLSTM": "cnn_bilstm_model5.h5",
    "CNN-Attention-LSTM": "cnn_attention_lstm_model5.h5",
    "Transformer": "transformer_model5.h5",
}


def _save_examples(name: str, clean_samples: np.ndarray, fgsm_samples: np.ndarray, pgd_samples: np.ndarray, labels: np.ndarray) -> None:
    """Persist clean and adversarial examples for one model family."""
    output_path = ADVERSARIAL_EXAMPLES_DIR / f"{name.lower().replace(' ', '_').replace('/', '_')}_examples.npz"
    np.savez_compressed(output_path, clean=clean_samples, fgsm=fgsm_samples, pgd=pgd_samples, y=labels)


def _predict_sklearn_metrics(model, X: np.ndarray, y: np.ndarray) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return build_metric_dict(y, y_pred, y_proba), y_pred, y_proba


def _evaluate_attack_bundle(
    y_true: np.ndarray,
    y_pred_clean_full: np.ndarray,
    y_proba_clean_full: np.ndarray,
    y_pred_attack_full: np.ndarray,
    y_proba_attack_full: np.ndarray,
    y_bot: np.ndarray,
    y_pred_clean_bot: np.ndarray,
    y_pred_attack_bot: np.ndarray,
) -> dict[str, object]:
    clean_metrics = build_metric_dict(y_true, y_pred_clean_full, y_proba_clean_full)
    attack_metrics = build_metric_dict(y_true, y_pred_attack_full, y_proba_attack_full)
    asr = attack_success_rate(
        y_bot,
        y_pred_clean_bot,
        y_pred_attack_bot,
        DEFAULT_THREAT_MODEL.source_classes,
        DEFAULT_THREAT_MODEL.target_class,
    )
    return {
        "accuracy": float(attack_metrics["accuracy"]),
        "precision": float(attack_metrics["precision"]),
        "recall": float(attack_metrics["recall"]),
        "f1_score": float(attack_metrics["f1_score"]),
        "roc_auc": float(attack_metrics["roc_auc"]),
        "accuracy_drop": float(clean_metrics["accuracy"] - attack_metrics["accuracy"]),
        "asr": float(asr["overall_asr"]),
        "asr_details": asr,
        "classification_report": attack_metrics["classification_report"],
    }


def _load_sequence_feature_names() -> list[str]:
    metadata = json.loads(SEQUENCE_METADATA_PATH.read_text(encoding="utf-8"))
    return list(metadata["feature_names"])


def _run_tabular_robustness(report: dict[str, object], epsilon_grid: list[float]) -> tuple[dict[str, list[dict[str, float]]], list[dict[str, float]], dict[str, dict[str, float]]]:
    split_artifacts = joblib.load(TABULAR_SPLIT_ARTIFACTS_PATH)
    X_train = split_artifacts["X_train_processed"]
    X_test = split_artifacts["X_test_processed"]
    y_train = split_artifacts["y_train"]
    y_test = split_artifacts["y_test"]
    feature_names = split_artifacts["feature_names"]

    bot_mask_test = np.isin(y_test, DEFAULT_THREAT_MODEL.source_classes)
    bot_mask_train = np.isin(y_train, DEFAULT_THREAT_MODEL.source_classes)
    X_bot_test = X_test[bot_mask_test]
    y_bot_test = y_test[bot_mask_test]
    X_bot_train = X_train[bot_mask_train]
    y_bot_train = y_train[bot_mask_train]
    target_labels_test = np.full(X_bot_test.shape[0], DEFAULT_THREAT_MODEL.target_class, dtype=np.int64)
    target_labels_train = np.full(X_bot_train.shape[0], DEFAULT_THREAT_MODEL.target_class, dtype=np.int64)

    surrogate = SurrogateModel()
    if SURROGATE_MODEL_PATH.exists():
        surrogate.load(SURROGATE_MODEL_PATH)
        if surrogate.input_dim != X_train.shape[1]:
            LOGGER.info(
                "Existing surrogate input_dim=%s does not match current tabular feature count=%s. Retraining surrogate.",
                surrogate.input_dim,
                X_train.shape[1],
            )
            surrogate = SurrogateModel()
    if surrogate.model is None:
        LOGGER.info("Training tabular surrogate model for Model5 robustness.")
        surrogate.train(X_train, y_train, epochs=40)
        surrogate.save(SURROGATE_MODEL_PATH)

    constraints = TabularDomainConstraints(feature_names=feature_names)
    x_bot_fgsm = fgsm_attack(X_bot_test, target_labels_test, surrogate, constraints, epsilon=DEFAULT_THREAT_MODEL.epsilon)
    x_bot_pgd = pgd_attack(X_bot_test, target_labels_test, surrogate, constraints, epsilon=DEFAULT_THREAT_MODEL.epsilon, n_steps=15)
    _save_examples("tabular", X_bot_test, x_bot_fgsm, x_bot_pgd, y_bot_test)

    X_test_fgsm = X_test.copy()
    X_test_pgd = X_test.copy()
    X_test_fgsm[bot_mask_test] = x_bot_fgsm
    X_test_pgd[bot_mask_test] = x_bot_pgd

    report["tabular"] = {}
    defense_rows: list[dict[str, float]] = []
    comparison_rows: dict[str, dict[str, float]] = {}
    epsilon_curves: dict[str, list[dict[str, float]]] = {}

    feature_squeezer = FeatureSqueezer(feature_names)
    X_test_pgd_squeezed = feature_squeezer.transform(X_test_pgd)

    for model_name, file_name in TABULAR_MODEL_FILES.items():
        model = joblib.load((Path(__file__).resolve().parents[1] / "outputs" / "boosting_models" / file_name))
        clean_metrics, y_pred_clean_full, y_proba_clean_full = _predict_sklearn_metrics(model, X_test, y_test)
        _, y_pred_fgsm_full, y_proba_fgsm_full = _predict_sklearn_metrics(model, X_test_fgsm, y_test)
        _, y_pred_pgd_full, y_proba_pgd_full = _predict_sklearn_metrics(model, X_test_pgd, y_test)
        y_pred_clean_bot = model.predict(X_bot_test)
        y_pred_fgsm_bot = model.predict(x_bot_fgsm)
        y_pred_pgd_bot = model.predict(x_bot_pgd)

        report["tabular"][model_name] = {
            "clean": {key: value for key, value in clean_metrics.items() if key != "classification_report"},
            "fgsm": _evaluate_attack_bundle(
                y_test,
                y_pred_clean_full,
                y_proba_clean_full,
                y_pred_fgsm_full,
                y_proba_fgsm_full,
                y_bot_test,
                y_pred_clean_bot,
                y_pred_fgsm_bot,
            ),
            "pgd": _evaluate_attack_bundle(
                y_test,
                y_pred_clean_full,
                y_proba_clean_full,
                y_pred_pgd_full,
                y_proba_pgd_full,
                y_bot_test,
                y_pred_clean_bot,
                y_pred_pgd_bot,
            ),
        }

        squeezed_metrics, _, _ = _predict_sklearn_metrics(model, X_test_pgd_squeezed, y_test)
        defense_rows.append(
            {
                "model": model_name,
                "defense": "FeatureSqueezing",
                "clean_accuracy": float(clean_metrics["accuracy"]),
                "adversarial_accuracy": float(report["tabular"][model_name]["pgd"]["accuracy"]),
                "hardened_accuracy": float(squeezed_metrics["accuracy"]),
                "defense_recovery": float(squeezed_metrics["accuracy"] - report["tabular"][model_name]["pgd"]["accuracy"]),
            }
        )

        comparison_rows[model_name] = {
            "clean": float(clean_metrics["accuracy"]),
            "attacked": float(report["tabular"][model_name]["pgd"]["accuracy"]),
        }

        def attack_for_epsilon(epsilon: float) -> dict[str, object]:
            x_bot_eps = pgd_attack(X_bot_test, target_labels_test, surrogate, constraints, epsilon=epsilon, n_steps=10)
            X_eps = X_test.copy()
            X_eps[bot_mask_test] = x_bot_eps
            _, y_pred_eps_full, y_proba_eps_full = _predict_sklearn_metrics(model, X_eps, y_test)
            y_pred_eps_bot = model.predict(x_bot_eps)
            return {
                "adversarial_metrics": build_metric_dict(y_test, y_pred_eps_full, y_proba_eps_full),
                "asr": attack_success_rate(y_bot_test, y_pred_clean_bot, y_pred_eps_bot, DEFAULT_THREAT_MODEL.source_classes, DEFAULT_THREAT_MODEL.target_class),
            }

        epsilon_curves[model_name] = epsilon_accuracy_curve(epsilon_grid, attack_for_epsilon)

    X_train_adv = pgd_attack(X_bot_train, target_labels_train, surrogate, constraints, epsilon=DEFAULT_THREAT_MODEL.epsilon, n_steps=10)
    X_train_aug, y_train_aug = augment_training_data(X_train, y_train, X_train_adv, y_bot_train)

    for model_name in TABULAR_MODEL_FILES:
        hardened_model = train_hardened_tabular_model(model_name, X_train_aug, y_train_aug)
        joblib.dump(hardened_model, HARDENED_MODELS_DIR / f"{model_name.lower().replace('-', '_')}_hardened.pkl")
        hardened_metrics, _, _ = _predict_sklearn_metrics(hardened_model, X_test_pgd, y_test)
        comparison_rows[model_name]["hardened"] = float(hardened_metrics["accuracy"])
        defense_rows.append(
            {
                "model": f"{model_name}-AdvTrain",
                "defense": "AdversarialTraining",
                "clean_accuracy": float(report["tabular"][model_name]["clean"]["accuracy"]),
                "adversarial_accuracy": float(report["tabular"][model_name]["pgd"]["accuracy"]),
                "hardened_accuracy": float(hardened_metrics["accuracy"]),
                "defense_recovery": float(hardened_metrics["accuracy"] - report["tabular"][model_name]["pgd"]["accuracy"]),
            }
        )

    return epsilon_curves, defense_rows, comparison_rows


def _run_sequence_robustness(report: dict[str, object], epsilon_grid: list[float]) -> tuple[dict[str, list[dict[str, float]]], list[dict[str, float]], dict[str, dict[str, float]]]:
    split_artifacts = np.load(SEQUENCE_SPLIT_ARTIFACTS_PATH)
    X_train = split_artifacts["X_train"].astype(np.float32)
    X_val = split_artifacts["X_val"].astype(np.float32)
    X_test = split_artifacts["X_test"].astype(np.float32)
    y_train = split_artifacts["y_train"].astype(np.int64)
    y_val = split_artifacts["y_val"].astype(np.int64)
    y_test = split_artifacts["y_test"].astype(np.int64)
    feature_names = _load_sequence_feature_names()

    bot_mask_test = np.isin(y_test, DEFAULT_THREAT_MODEL.source_classes)
    bot_mask_train = np.isin(y_train, DEFAULT_THREAT_MODEL.source_classes)
    X_bot_test = X_test[bot_mask_test]
    y_bot_test = y_test[bot_mask_test]
    X_bot_train = X_train[bot_mask_train]
    y_bot_train = y_train[bot_mask_train]
    target_labels_test = np.full(X_bot_test.shape[0], DEFAULT_THREAT_MODEL.target_class, dtype=np.int64)
    target_labels_train = np.full(X_bot_train.shape[0], DEFAULT_THREAT_MODEL.target_class, dtype=np.int64)

    constraints = SequenceDomainConstraints(feature_names=feature_names)
    report["sequence"] = {}
    defense_rows: list[dict[str, float]] = []
    comparison_rows: dict[str, dict[str, float]] = {}
    epsilon_curves: dict[str, list[dict[str, float]]] = {}

    dl_summary = load_json_if_exists(DL_SUMMARY_JSON)
    best_sequence_model = max(dl_summary.items(), key=lambda item: item[1]["accuracy"])[0] if dl_summary else "Transformer"
    builders = get_model_builders()
    squeezer = FeatureSqueezer(feature_names)

    for model_name, file_name in SEQUENCE_MODEL_FILES.items():
        model = load_model(DL_MODELS_DIR / file_name)
        clean_metrics, y_pred_clean_full, y_proba_clean_full = evaluate_dl_model(model, X_test, y_test)
        x_bot_fgsm = fgsm_sequence_attack(model, X_bot_test, target_labels_test, constraints, epsilon=DEFAULT_THREAT_MODEL.epsilon)
        x_bot_pgd = pgd_sequence_attack(model, X_bot_test, target_labels_test, constraints, epsilon=DEFAULT_THREAT_MODEL.epsilon, n_steps=10)
        _save_examples(model_name, X_bot_test, x_bot_fgsm, x_bot_pgd, y_bot_test)

        X_test_fgsm = X_test.copy()
        X_test_pgd = X_test.copy()
        X_test_fgsm[bot_mask_test] = x_bot_fgsm
        X_test_pgd[bot_mask_test] = x_bot_pgd

        fgsm_metrics, y_pred_fgsm_full, y_proba_fgsm_full = evaluate_dl_model(model, X_test_fgsm, y_test)
        pgd_metrics, y_pred_pgd_full, y_proba_pgd_full = evaluate_dl_model(model, X_test_pgd, y_test)
        y_pred_clean_bot = np.argmax(model.predict(X_bot_test, verbose=0), axis=1)
        y_pred_fgsm_bot = np.argmax(model.predict(x_bot_fgsm, verbose=0), axis=1)
        y_pred_pgd_bot = np.argmax(model.predict(x_bot_pgd, verbose=0), axis=1)

        report["sequence"][model_name] = {
            "clean": {key: value for key, value in clean_metrics.items() if key != "classification_report"},
            "fgsm": _evaluate_attack_bundle(
                y_test,
                y_pred_clean_full,
                y_proba_clean_full,
                y_pred_fgsm_full,
                y_proba_fgsm_full,
                y_bot_test,
                y_pred_clean_bot,
                y_pred_fgsm_bot,
            ),
            "pgd": _evaluate_attack_bundle(
                y_test,
                y_pred_clean_full,
                y_proba_clean_full,
                y_pred_pgd_full,
                y_proba_pgd_full,
                y_bot_test,
                y_pred_clean_bot,
                y_pred_pgd_bot,
            ),
        }

        comparison_rows[model_name] = {
            "clean": float(clean_metrics["accuracy"]),
            "attacked": float(pgd_metrics["accuracy"]),
        }

        if model_name == best_sequence_model:
            X_test_pgd_squeezed = squeezer.transform(X_test_pgd)
            squeezed_metrics, _, _ = evaluate_dl_model(model, X_test_pgd_squeezed, y_test)
            defense_rows.append(
                {
                    "model": model_name,
                    "defense": "FeatureSqueezing",
                    "clean_accuracy": float(clean_metrics["accuracy"]),
                    "adversarial_accuracy": float(pgd_metrics["accuracy"]),
                    "hardened_accuracy": float(squeezed_metrics["accuracy"]),
                    "defense_recovery": float(squeezed_metrics["accuracy"] - pgd_metrics["accuracy"]),
                }
            )

            def attack_for_epsilon(epsilon: float) -> dict[str, object]:
                x_bot_eps = pgd_sequence_attack(model, X_bot_test, target_labels_test, constraints, epsilon=epsilon, n_steps=8)
                X_eps = X_test.copy()
                X_eps[bot_mask_test] = x_bot_eps
                metrics_eps, y_pred_eps_full, y_proba_eps_full = evaluate_dl_model(model, X_eps, y_test)
                _ = metrics_eps
                y_pred_eps_bot = np.argmax(model.predict(x_bot_eps, verbose=0), axis=1)
                return {
                    "adversarial_metrics": build_metric_dict(y_test, y_pred_eps_full, y_proba_eps_full),
                    "asr": attack_success_rate(y_bot_test, y_pred_clean_bot, y_pred_eps_bot, DEFAULT_THREAT_MODEL.source_classes, DEFAULT_THREAT_MODEL.target_class),
                }

            epsilon_curves[model_name] = epsilon_accuracy_curve(epsilon_grid, attack_for_epsilon)

            X_train_adv = pgd_sequence_attack(model, X_bot_train, target_labels_train, constraints, epsilon=DEFAULT_THREAT_MODEL.epsilon, n_steps=8)
            X_train_aug, y_train_aug = augment_training_data(X_train, y_train, X_train_adv, y_bot_train)
            hardened_model = train_hardened_sequence_model(
                builders[model_name],
                input_shape=(X_train.shape[1], X_train.shape[2]),
                X_train=X_train_aug,
                y_train_onehot=to_categorical(y_train_aug, num_classes=3),
                X_val=X_val,
                y_val_onehot=to_categorical(y_val, num_classes=3),
                epochs=15,
                batch_size=64,
            )
            hardened_model.save(HARDENED_MODELS_DIR / f"{model_name.lower().replace('-', '_')}_hardened.h5", include_optimizer=True)
            hardened_metrics, _, _ = evaluate_dl_model(hardened_model, X_test_pgd, y_test)
            comparison_rows[model_name]["hardened"] = float(hardened_metrics["accuracy"])
            defense_rows.append(
                {
                    "model": f"{model_name}-AdvTrain",
                    "defense": "AdversarialTraining",
                    "clean_accuracy": float(clean_metrics["accuracy"]),
                    "adversarial_accuracy": float(pgd_metrics["accuracy"]),
                    "hardened_accuracy": float(hardened_metrics["accuracy"]),
                    "defense_recovery": float(hardened_metrics["accuracy"] - pgd_metrics["accuracy"]),
                }
            )

    return epsilon_curves, defense_rows, comparison_rows


def run_robustness_pipeline(epsilon_grid: list[float] | None = None) -> dict[str, object]:
    """Execute the full Model5 robustness pipeline."""
    ensure_model5_directories()
    epsilon_grid = epsilon_grid or EPSILON_GRID
    report: dict[str, object] = {"epsilon_grid": epsilon_grid}

    tabular_curves, tabular_defenses, tabular_comparison = _run_tabular_robustness(report, epsilon_grid)
    sequence_curves, sequence_defenses, sequence_comparison = _run_sequence_robustness(report, epsilon_grid)

    report["defenses"] = {}
    for row in tabular_defenses + sequence_defenses:
        report["defenses"][row["model"]] = row

    plot_epsilon_curve(tabular_curves, ADVERSARIAL_DIR / "tabular_epsilon_accuracy_curve.png", metric_name="accuracy")
    if sequence_curves:
        plot_epsilon_curve(sequence_curves, ADVERSARIAL_DIR / "sequence_epsilon_accuracy_curve.png", metric_name="accuracy")
    plot_clean_vs_attacked_vs_hardened({**tabular_comparison, **sequence_comparison}, ADVERSARIAL_DIR / "clean_vs_attacked_vs_hardened.png")
    plot_defense_recovery(tabular_defenses + sequence_defenses, ADVERSARIAL_DIR / "defense_recovery.png")

    save_report(report, ROBUSTNESS_SUMMARY_JSON)
    write_markdown_report(report, ROBUSTNESS_MARKDOWN_PATH)
    return report


def main() -> None:
    """Entry point for the Model5 robustness pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_robustness_pipeline()


if __name__ == "__main__":
    main()
