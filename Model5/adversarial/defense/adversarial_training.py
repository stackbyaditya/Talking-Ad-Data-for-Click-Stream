"""Adversarial training helpers for hardened tabular and sequence models."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier

from Model5.models.deep_learning.dl_utils import build_callbacks
from Model5.models.model5_config import RANDOM_STATE


def augment_training_data(X_clean: np.ndarray, y_clean: np.ndarray, X_adv: np.ndarray, y_adv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate clean and adversarial training samples."""
    X_aug = np.concatenate([X_clean, X_adv], axis=0)
    y_aug = np.concatenate([y_clean, y_adv], axis=0)
    return X_aug, y_aug


def build_tabular_model(model_name: str):
    """Instantiate a hardened tabular model using Model4-compatible hyperparameters."""
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=3,
            n_jobs=1,
        )
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            objective="multiclass",
            num_class=3,
            n_jobs=1,
            verbosity=-1,
        )
    raise ValueError(f"Unsupported hardened tabular model: {model_name}")


def train_hardened_tabular_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray):
    """Fit one hardened tabular model on augmented training data."""
    model = build_tabular_model(model_name)
    model.fit(X_train, y_train)
    return model


def train_hardened_sequence_model(
    model_builder,
    input_shape: tuple[int, int],
    X_train: np.ndarray,
    y_train_onehot: np.ndarray,
    X_val: np.ndarray,
    y_val_onehot: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
):
    """Retrain one sequence model on adversarially augmented data."""
    model = model_builder(input_shape)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train,
        y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=build_callbacks(),
        verbose=0,
    )
    return model
