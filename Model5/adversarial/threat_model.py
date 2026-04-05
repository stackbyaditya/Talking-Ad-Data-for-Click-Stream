"""Threat model configuration for adversarial click-fraud evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


def normalize_feature_name(feature_name: str) -> str:
    """Map processed feature names back to their raw base feature names."""
    if feature_name.startswith("num__"):
        return feature_name.split("num__", 1)[1]
    if feature_name.startswith("cat__"):
        return feature_name.split("cat__", 1)[1].split("_", 1)[0]
    return feature_name


@dataclass
class ThreatModel:
    """Encodes the adversary's capabilities and goals."""

    attacker_knowledge: str = "white_box"
    epsilon: float = 0.10
    target_class: int = 0
    source_classes: List[int] = field(default_factory=lambda: [1, 2])
    mutable_features: List[str] = field(
        default_factory=lambda: [
            "mouse_speed_mean",
            "mouse_speed_std",
            "mouse_path_length",
            "direction_change_count",
            "movement_std",
            "coordinate_entropy",
            "session_duration_sec",
            "request_interval_mean",
            "request_interval_std",
            "clicks_per_minute",
            "requests_per_minute",
            "success_rate",
            "burstiness",
            "click_interval_entropy",
            "movement_acceleration",
            "mouse_direction_entropy",
            "click_burst_score",
            "session_idle_ratio",
            "trajectory_smoothness",
            "interaction_variability",
            "behavioral_complexity",
        ]
    )
    immutable_features: List[str] = field(
        default_factory=lambda: [
            "browser",
            "operating_system",
            "device_type",
            "country",
            "region",
            "bot_likelihood_score",
            "anomaly_score",
        ]
    )
    feature_epsilons: Dict[str, float] = field(
        default_factory=lambda: {
            "request_interval_mean": 0.05,
            "request_interval_std": 0.05,
            "click_interval_entropy": 0.05,
            "mouse_speed_mean": 0.08,
            "mouse_speed_std": 0.08,
            "mouse_path_length": 0.12,
            "direction_change_count": 0.12,
            "coordinate_entropy": 0.12,
            "trajectory_smoothness": 0.08,
        }
    )

    def get_epsilon(self, feature_name: str) -> float:
        """Return the epsilon budget for one feature."""
        return self.feature_epsilons.get(normalize_feature_name(feature_name), self.epsilon)

    def is_mutable(self, feature_name: str) -> bool:
        """Check whether a processed or raw feature is mutable."""
        raw_name = normalize_feature_name(feature_name)
        return raw_name in self.mutable_features and raw_name not in self.immutable_features


DEFAULT_THREAT_MODEL = ThreatModel()
