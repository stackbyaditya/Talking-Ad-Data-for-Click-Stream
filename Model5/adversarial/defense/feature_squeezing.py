"""Feature squeezing defense for tabular and sequence models."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class FeatureSqueezer:
    """Apply decimal-place reduction to continuous features."""

    def __init__(self, feature_names: list[str], precision_map: Optional[Dict[str, int]] = None, default_precision: int = 3):
        self.feature_names = feature_names
        self.default_precision = default_precision
        self.precision_map = precision_map or {
            "clicks_per_minute": 1,
            "requests_per_minute": 1,
            "direction_change_count": 0,
            "mouse_path_length": 1,
            "mouse_speed_mean": 3,
            "mouse_speed_std": 3,
            "request_interval_mean": 3,
            "request_interval_std": 3,
            "click_interval_entropy": 4,
            "coordinate_entropy": 4,
            "click_event": 0,
            "pause_event": 0,
        }

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Round each feature to its configured precision."""
        squeezed = np.asarray(X).copy()
        if squeezed.ndim == 2:
            for idx, feature_name in enumerate(self.feature_names):
                precision = self.precision_map.get(feature_name.split("__")[-1], self.default_precision)
                squeezed[:, idx] = np.round(squeezed[:, idx], decimals=precision)
            return squeezed
        if squeezed.ndim == 3:
            for idx, feature_name in enumerate(self.feature_names):
                precision = self.precision_map.get(feature_name, self.default_precision)
                squeezed[:, :, idx] = np.round(squeezed[:, :, idx], decimals=precision)
            return squeezed
        raise ValueError(f"Unsupported feature-squeezing input rank: {squeezed.ndim}")
