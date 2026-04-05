"""Domain constraint enforcement for adversarial robustness evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Model5.adversarial.threat_model import DEFAULT_THREAT_MODEL, ThreatModel
from Model5.models.model5_config import SEQUENCE_FEATURE_METADATA_PATH, TABULAR_FEATURE_METADATA_PATH


class TabularDomainConstraints:
    """Project tabular adversarial examples onto a realistic processed feature space."""

    def __init__(
        self,
        feature_names: list[str],
        metadata_path: Path = TABULAR_FEATURE_METADATA_PATH,
        threat_model: ThreatModel = DEFAULT_THREAT_MODEL,
    ):
        self.feature_names = feature_names
        self.threat_model = threat_model
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.numeric_meta = self.metadata["numeric_feature_metadata"]
        self.numeric_feature_names = self.metadata["processed_numeric_feature_names"]
        self.numeric_indices = [self.feature_names.index(name) for name in self.numeric_feature_names]
        self.mutable_numeric_indices = []
        self.immutable_indices = []
        for idx, feature_name in enumerate(self.feature_names):
            if feature_name in self.numeric_meta and threat_model.is_mutable(feature_name):
                self.mutable_numeric_indices.append(idx)
            else:
                self.immutable_indices.append(idx)

        self._centers = np.asarray(
            [self.numeric_meta[name]["scaler_center"] for name in self.numeric_feature_names],
            dtype=np.float32,
        )
        self._scales = np.asarray(
            [max(self.numeric_meta[name]["scaler_scale"], 1e-6) for name in self.numeric_feature_names],
            dtype=np.float32,
        )

    def _inverse_scale(self, processed_numeric: np.ndarray) -> np.ndarray:
        return processed_numeric * self._scales + self._centers

    def _forward_scale(self, raw_numeric: np.ndarray) -> np.ndarray:
        return (raw_numeric - self._centers) / self._scales

    def _enforce_semantic_rules(self, raw_numeric: np.ndarray) -> np.ndarray:
        name_to_local_idx = {self.numeric_feature_names[i].split("num__", 1)[1]: i for i in range(len(self.numeric_feature_names))}

        def idx(name: str) -> int | None:
            return name_to_local_idx.get(name)

        non_negative = [
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
            "burstiness",
            "click_interval_entropy",
        ]
        for feature_name in non_negative:
            feature_idx = idx(feature_name)
            if feature_idx is not None:
                raw_numeric[:, feature_idx] = np.clip(raw_numeric[:, feature_idx], 0.0, None)

        success_rate_idx = idx("success_rate")
        if success_rate_idx is not None:
            raw_numeric[:, success_rate_idx] = np.clip(raw_numeric[:, success_rate_idx], 0.0, 1.0)

        clicks_idx = idx("clicks_per_minute")
        requests_idx = idx("requests_per_minute")
        if clicks_idx is not None and requests_idx is not None:
            raw_numeric[:, requests_idx] = np.maximum(raw_numeric[:, requests_idx], raw_numeric[:, clicks_idx])

        session_duration_idx = idx("session_duration_sec")
        request_mean_idx = idx("request_interval_mean")
        request_std_idx = idx("request_interval_std")
        if session_duration_idx is not None and request_mean_idx is not None:
            raw_numeric[:, session_duration_idx] = np.maximum(raw_numeric[:, session_duration_idx], raw_numeric[:, request_mean_idx])
        if session_duration_idx is not None and request_std_idx is not None:
            raw_numeric[:, request_std_idx] = np.minimum(raw_numeric[:, request_std_idx], raw_numeric[:, session_duration_idx])
        return raw_numeric

    def project(self, x_adv: np.ndarray, x_orig: np.ndarray) -> np.ndarray:
        """Project perturbed tabular samples back to the valid feature space."""
        x_proj = np.asarray(x_adv, dtype=np.float32).copy()
        x_proj[:, self.immutable_indices] = x_orig[:, self.immutable_indices]

        for feature_name in self.numeric_feature_names:
            feature_idx = self.feature_names.index(feature_name)
            meta = self.numeric_meta[feature_name]
            if not self.threat_model.is_mutable(feature_name):
                x_proj[:, feature_idx] = x_orig[:, feature_idx]
                continue

            epsilon = self.threat_model.get_epsilon(feature_name)
            processed_range = max(meta["processed_max"] - meta["processed_min"], 1e-6)
            delta_max = epsilon * processed_range
            x_proj[:, feature_idx] = np.clip(
                x_proj[:, feature_idx],
                x_orig[:, feature_idx] - delta_max,
                x_orig[:, feature_idx] + delta_max,
            )
            x_proj[:, feature_idx] = np.clip(x_proj[:, feature_idx], meta["processed_min"], meta["processed_max"])

        numeric_processed = x_proj[:, self.numeric_indices]
        numeric_raw = self._inverse_scale(numeric_processed)
        numeric_raw = self._enforce_semantic_rules(numeric_raw)
        x_proj[:, self.numeric_indices] = self._forward_scale(numeric_raw)
        x_proj[:, self.immutable_indices] = x_orig[:, self.immutable_indices]
        return x_proj


class SequenceDomainConstraints:
    """Project sequence adversarial examples onto observed training ranges."""

    def __init__(
        self,
        feature_names: list[str],
        metadata_path: Path = SEQUENCE_FEATURE_METADATA_PATH,
        threat_model: ThreatModel = DEFAULT_THREAT_MODEL,
    ):
        self.feature_names = feature_names
        self.threat_model = threat_model
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.feature_min = np.asarray(self.metadata["feature_min"], dtype=np.float32)
        self.feature_max = np.asarray(self.metadata["feature_max"], dtype=np.float32)

    def project(self, x_adv: np.ndarray, x_orig: np.ndarray, epsilon: float) -> np.ndarray:
        """Clip sequence perturbations relative to the original sample and train range."""
        x_proj = np.asarray(x_adv, dtype=np.float32).copy()
        feature_range = np.maximum(self.feature_max - self.feature_min, 1e-6)
        delta = epsilon * feature_range.reshape(1, 1, -1)
        x_proj = np.clip(x_proj, x_orig - delta, x_orig + delta)
        x_proj = np.clip(x_proj, self.feature_min.reshape(1, 1, -1), self.feature_max.reshape(1, 1, -1))
        return x_proj
