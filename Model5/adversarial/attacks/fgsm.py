"""FGSM attacks for Model5 tabular and sequence models."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def fgsm_attack(
    X: np.ndarray,
    target_labels: np.ndarray,
    surrogate,
    constraints,
    epsilon: float,
    targeted: bool = True,
) -> np.ndarray:
    """Generate targeted or untargeted FGSM adversarial examples for tabular data."""
    gradients = surrogate.get_gradients(X, target_labels)
    direction = -np.sign(gradients) if targeted else np.sign(gradients)
    x_adv = X + epsilon * direction
    return constraints.project(x_adv, X)


def fgsm_sequence_attack(
    model,
    X: np.ndarray,
    target_labels: np.ndarray,
    constraints,
    epsilon: float,
    targeted: bool = True,
    batch_size: int = 128,
) -> np.ndarray:
    """Generate targeted or untargeted FGSM adversarial examples for sequence models."""
    outputs = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        x_batch = tf.convert_to_tensor(X[start:stop], dtype=tf.float32)
        y_batch = tf.convert_to_tensor(target_labels[start:stop], dtype=tf.int32)
        with tf.GradientTape() as tape:
            tape.watch(x_batch)
            predictions = model(x_batch, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, x_batch)
        direction = -tf.sign(gradients) if targeted else tf.sign(gradients)
        x_adv = x_batch + epsilon * direction
        outputs.append(constraints.project(x_adv.numpy(), X[start:stop], epsilon))
    return np.concatenate(outputs, axis=0)
