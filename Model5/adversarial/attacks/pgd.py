"""PGD attacks for Model5 tabular and sequence models."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def pgd_attack(
    X: np.ndarray,
    target_labels: np.ndarray,
    surrogate,
    constraints,
    epsilon: float,
    n_steps: int = 20,
    step_size: float | None = None,
    targeted: bool = True,
) -> np.ndarray:
    """Generate targeted or untargeted PGD adversarial examples for tabular data."""
    if step_size is None:
        step_size = epsilon / max(n_steps // 2, 1)
    x_adv = X.copy().astype(np.float32)
    for _ in range(n_steps):
        gradients = surrogate.get_gradients(x_adv, target_labels)
        direction = -np.sign(gradients) if targeted else np.sign(gradients)
        x_adv = x_adv + step_size * direction
        x_adv = constraints.project(x_adv, X)
    return x_adv


def pgd_sequence_attack(
    model,
    X: np.ndarray,
    target_labels: np.ndarray,
    constraints,
    epsilon: float,
    n_steps: int = 10,
    step_size: float | None = None,
    targeted: bool = True,
    batch_size: int = 128,
) -> np.ndarray:
    """Generate targeted or untargeted PGD adversarial examples for sequence models."""
    if step_size is None:
        step_size = epsilon / max(n_steps // 2, 1)
    all_batches = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        x_orig = X[start:stop].astype(np.float32)
        x_adv = x_orig.copy()
        y_batch = tf.convert_to_tensor(target_labels[start:stop], dtype=tf.int32)
        for _ in range(n_steps):
            x_tensor = tf.convert_to_tensor(x_adv, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(x_tensor)
                predictions = model(x_tensor, training=False)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
                loss = tf.reduce_mean(loss)
            gradients = tape.gradient(loss, x_tensor)
            direction = -tf.sign(gradients) if targeted else tf.sign(gradients)
            x_adv = x_adv + step_size * direction.numpy()
            x_adv = constraints.project(x_adv, x_orig, epsilon)
        all_batches.append(x_adv)
    return np.concatenate(all_batches, axis=0)
