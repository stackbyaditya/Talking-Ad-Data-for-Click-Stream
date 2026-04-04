"""Baseline deep learning architectures for the Model4 sequence experiment."""

from __future__ import annotations

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Attention,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    MaxPooling1D,
)


NUM_CLASSES = 3


def build_cnn_model(input_shape: tuple[int, int]) -> Model:
    """Create the CNN baseline model."""
    inputs = Input(shape=input_shape, name="cnn_input")
    x = Conv1D(64, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling1D()(x)
    x = Conv1D(128, kernel_size=3, activation="relu")(x)
    x = MaxPooling1D()(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="CNN")


def build_lstm_model(input_shape: tuple[int, int]) -> Model:
    """Create the stacked LSTM baseline."""
    inputs = Input(shape=input_shape, name="lstm_input")
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="LSTM")


def build_cnn_lstm_model(input_shape: tuple[int, int]) -> Model:
    """Create the CNN-LSTM hybrid model."""
    inputs = Input(shape=input_shape, name="cnn_lstm_input")
    x = Conv1D(64, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling1D()(x)
    x = Conv1D(128, kernel_size=3, activation="relu")(x)
    x = MaxPooling1D()(x)
    x = LSTM(64)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="CNN-LSTM")


def build_cnn_bilstm_model(input_shape: tuple[int, int]) -> Model:
    """Create the CNN-BiLSTM hybrid model."""
    inputs = Input(shape=input_shape, name="cnn_bilstm_input")
    x = Conv1D(64, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling1D()(x)
    x = Conv1D(128, kernel_size=3, activation="relu")(x)
    x = MaxPooling1D()(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="CNN-BiLSTM")


def build_cnn_attention_lstm_model(input_shape: tuple[int, int]) -> Model:
    """Create the CNN-Attention-LSTM hybrid model."""
    inputs = Input(shape=input_shape, name="cnn_attention_lstm_input")
    x = Conv1D(64, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling1D()(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Attention()([x, x])
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="CNN-Attention-LSTM")


def get_model_builders() -> dict[str, callable]:
    """Return the baseline model family used for Model4."""
    return {
        "CNN": build_cnn_model,
        "LSTM": build_lstm_model,
        "CNN-LSTM": build_cnn_lstm_model,
        "CNN-BiLSTM": build_cnn_bilstm_model,
        "CNN-Attention-LSTM": build_cnn_attention_lstm_model,
    }
