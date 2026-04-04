"""High-accuracy CNN-BiLSTM architecture for behavioural sequence classification."""

from __future__ import annotations

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GaussianNoise,
    Input,
    LSTM,
    MaxPooling1D,
    ReLU,
)


def build_high_accuracy_cnn_bilstm(
    input_shape: tuple[int, int],
    num_classes: int = 3,
) -> Model:
    """Build the experimental high-capacity CNN-BiLSTM classifier."""
    inputs = Input(shape=input_shape, name="sequence_input")
    x = GaussianNoise(0.05, name="gaussian_noise")(inputs)

    x = Conv1D(128, kernel_size=3, padding="same", name="conv1")(x)
    x = BatchNormalization(name="bn1")(x)
    x = ReLU(name="relu1")(x)
    x = MaxPooling1D(pool_size=2, name="pool1")(x)

    x = Conv1D(256, kernel_size=3, padding="same", name="conv2")(x)
    x = BatchNormalization(name="bn2")(x)
    x = ReLU(name="relu2")(x)
    x = MaxPooling1D(pool_size=2, name="pool2")(x)

    x = Bidirectional(LSTM(128, return_sequences=True), name="bilstm1")(x)
    x = Dropout(0.3, name="dropout1")(x)
    x = Bidirectional(LSTM(64), name="bilstm2")(x)

    x = Dense(128, activation="relu", name="dense1")(x)
    x = Dropout(0.4, name="dropout2")(x)
    outputs = Dense(num_classes, activation="softmax", name="classifier")(x)
    return Model(inputs=inputs, outputs=outputs, name="HighAccuracy-CNN-BiLSTM")
