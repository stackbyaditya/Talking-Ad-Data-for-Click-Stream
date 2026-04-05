"""Robustness metrics for adversarial click-fraud evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np

from Model5.models.model5_utils import build_metric_dict


def attack_success_rate(
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_adv: np.ndarray,
    source_classes: list[int],
    target_class: int,
) -> Dict[str, object]:
    """Compute overall and per-class attack success rate."""
    successful_mask = np.zeros_like(y_true, dtype=bool)
    denominators: Dict[str, int] = {}
    per_class: Dict[str, float] = {}

    for source_class in source_classes:
        class_mask = y_true == source_class
        correctly_classified_mask = class_mask & (y_pred_clean == y_true)
        targeted_success_mask = correctly_classified_mask & (y_pred_adv == target_class)
        denominators[str(source_class)] = int(np.sum(correctly_classified_mask))
        per_class[str(source_class)] = (
            float(np.mean(targeted_success_mask[correctly_classified_mask])) if np.any(correctly_classified_mask) else 0.0
        )
        successful_mask |= targeted_success_mask

    eligible_mask = np.isin(y_true, source_classes) & (y_pred_clean == y_true)
    overall = float(np.mean(successful_mask[eligible_mask])) if np.any(eligible_mask) else 0.0
    return {
        "overall_asr": overall,
        "per_class_asr": per_class,
        "eligible_samples": int(np.sum(eligible_mask)),
        "successful_samples": int(np.sum(successful_mask)),
        "per_class_denominators": denominators,
    }


def accuracy_drop(clean_accuracy: float, adversarial_accuracy: float) -> float:
    """Compute the absolute accuracy drop."""
    return float(clean_accuracy - adversarial_accuracy)


def build_attack_result(
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_proba_clean: np.ndarray,
    y_pred_adv: np.ndarray,
    y_proba_adv: np.ndarray,
    source_classes: list[int],
    target_class: int,
) -> Dict[str, object]:
    """Build a consistent attack-evaluation bundle."""
    clean_metrics = build_metric_dict(y_true, y_pred_clean, y_proba_clean)
    adv_metrics = build_metric_dict(y_true, y_pred_adv, y_proba_adv)
    asr = attack_success_rate(y_true, y_pred_clean, y_pred_adv, source_classes, target_class)
    return {
        "clean_metrics": clean_metrics,
        "adversarial_metrics": adv_metrics,
        "accuracy_drop": accuracy_drop(clean_metrics["accuracy"], adv_metrics["accuracy"]),
        "asr": asr,
    }


def epsilon_accuracy_curve(
    epsilon_values: list[float],
    attack_callback,
) -> list[dict[str, float]]:
    """Collect accuracy and ASR values across an epsilon grid."""
    curve = []
    for epsilon in epsilon_values:
        result = attack_callback(epsilon)
        curve.append(
            {
                "epsilon": float(epsilon),
                "accuracy": float(result["adversarial_metrics"]["accuracy"]),
                "asr": float(result["asr"]["overall_asr"]),
            }
        )
    return curve
