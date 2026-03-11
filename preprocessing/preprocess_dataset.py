"""Reusable preprocessing utilities for the realistic bot-detection dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


LOGGER = logging.getLogger(__name__)
DATASET_PATH = Path("data/processed/final_training_dataset_realistic.csv")
DROP_COLUMNS = [
    "session_id",
    "ip_address",
    "user_agent",
    "label_name",
    "source_click_time",
    "source_attributed_time",
]
BASE_CATEGORICAL_FEATURES = [
    "browser",
    "operating_system",
    "device_type",
    "country",
    "region",
]
BASE_NUMERIC_FEATURES = [
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
    "bot_likelihood_score",
    "anomaly_score",
]


def load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the realistic training dataset and enforce basic quality checks."""
    LOGGER.info("Loading dataset from %s", dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        LOGGER.info("Removing %s duplicate rows.", duplicate_count)
        df = df.drop_duplicates().reset_index(drop=True)

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values after deduplication.")
    return df


def build_feature_sets(df: pd.DataFrame) -> Dict[str, object]:
    """Construct X, y, and the feature type lists used by the preprocessor."""
    drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    X = df.drop(columns=["label"] + drop_cols)
    y = df["label"].copy()

    categorical_features = [col for col in BASE_CATEGORICAL_FEATURES if col in X.columns]
    numeric_features = [col for col in BASE_NUMERIC_FEATURES if col in X.columns]
    selected_features = numeric_features + categorical_features
    X = X[selected_features].copy()

    return {
        "X": X,
        "y": y,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "drop_cols": drop_cols,
    }


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Build the shared preprocessing transformer."""
    return ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def prepare_train_test_data(
    dataset_path: Path = DATASET_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Load, split, and preprocess the realistic training dataset."""
    df = load_dataset(dataset_path)
    feature_bundle = build_feature_sets(df)
    X = feature_bundle["X"]
    y = feature_bundle["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(
        numeric_features=feature_bundle["numeric_features"],
        categorical_features=feature_bundle["categorical_features"],
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    LOGGER.info("Fitting preprocessing pipeline on the training split.")
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    LOGGER.info("Preprocessing complete.")

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "pipeline": pipeline,
        "feature_names": feature_names,
        "categorical_features": feature_bundle["categorical_features"],
        "numeric_features": feature_bundle["numeric_features"],
    }
