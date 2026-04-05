"""Deep-learning architectures for the Model5 sequence experiment."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Attention,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    LSTM,
    MaxPooling1D,
    MultiHeadAttention,
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


def _transformer_block(x: tf.Tensor, num_heads: int, key_dim: int, ff_dim: int, dropout_rate: float = 0.1) -> tf.Tensor:
    """Single Transformer encoder block."""
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn_out = Dropout(dropout_rate)(attn_out)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn_out]))

    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(x.shape[-1])(ffn)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, ffn]))
    return x


def build_transformer_model(input_shape: tuple[int, int]) -> Model:
    """Reuse the Model2 Transformer-style encoder for sequence classification."""
    inputs = Input(shape=input_shape, name="transformer_input")
    x = Dense(64)(inputs)
    x = LayerNormalization()(x)
    x = _transformer_block(x, num_heads=4, key_dim=16, ff_dim=128, dropout_rate=0.1)
    x = _transformer_block(x, num_heads=4, key_dim=16, ff_dim=128, dropout_rate=0.1)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="Transformer")


def get_model_builders() -> dict[str, callable]:
    """Return the baseline model family used for Model5."""
    return {
        "CNN": build_cnn_model,
        "LSTM": build_lstm_model,
        "CNN-LSTM": build_cnn_lstm_model,
        "CNN-BiLSTM": build_cnn_bilstm_model,
        "CNN-Attention-LSTM": build_cnn_attention_lstm_model,
        "Transformer": build_transformer_model,
    }
