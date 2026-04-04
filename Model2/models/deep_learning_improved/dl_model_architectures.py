"""Deep learning model architectures for behavioural sequence classification.

v2 improvements over v1:
- BatchNormalization after every Conv1D block for training stability.
- Residual (skip) connections in the CNN backbone.
- Transformer encoder model (multi-head self-attention) as a new architecture.
- Larger LSTM hidden sizes to match the longer (50-step) sequences.
- Ensemble model that averages softmax outputs of all five base models.
- All existing model names are preserved so train_dl_models.py needs
  minimal changes.
"""

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


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _conv_bn_block(x: tf.Tensor, filters: int, kernel_size: int = 3) -> tf.Tensor:
    """Conv1D → BatchNorm → ReLU."""
    x = Conv1D(filters, kernel_size=kernel_size, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    return x


def _residual_conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Two Conv-BN layers with a skip connection.

    If the channel dimension changes we project the residual with a 1×1 conv.
    """
    residual = x
    x = _conv_bn_block(x, filters, kernel_size=3)
    x = _conv_bn_block(x, filters, kernel_size=3)

    # project residual if channel count differs
    if residual.shape[-1] != filters:
        residual = Conv1D(filters, kernel_size=1, padding="same")(residual)
        residual = BatchNormalization()(residual)

    x = Add()([x, residual])
    return x


# ---------------------------------------------------------------------------
# 1. CNN  (+ BatchNorm + residual)
# ---------------------------------------------------------------------------

def build_cnn_model(input_shape: tuple[int, int]) -> Model:
    """CNN with residual blocks and BatchNorm."""
    inputs = Input(shape=input_shape, name="cnn_input")
    x = _residual_conv_block(inputs, 64)
    x = MaxPooling1D()(x)
    x = _residual_conv_block(x, 128)
    x = MaxPooling1D()(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="CNN")


# ---------------------------------------------------------------------------
# 2. LSTM  (wider hidden + LayerNorm)
# ---------------------------------------------------------------------------

def build_lstm_model(input_shape: tuple[int, int]) -> Model:
    """Stacked LSTM with increased capacity for 50-step sequences."""
    inputs = Input(shape=input_shape, name="lstm_input")
    x = LSTM(128, return_sequences=True)(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="LSTM")


# ---------------------------------------------------------------------------
# 3. CNN-LSTM  (residual CNN front-end + wider LSTM)
# ---------------------------------------------------------------------------

def build_cnn_lstm_model(input_shape: tuple[int, int]) -> Model:
    """Residual CNN front-end feeding a wider LSTM."""
    inputs = Input(shape=input_shape, name="cnn_lstm_input")
    x = _residual_conv_block(inputs, 64)
    x = MaxPooling1D()(x)
    x = _residual_conv_block(x, 128)
    x = MaxPooling1D()(x)
    x = LSTM(128)(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="CNN-LSTM")


# ---------------------------------------------------------------------------
# 4. CNN-BiLSTM  (residual CNN + wider BiLSTM)
# ---------------------------------------------------------------------------

def build_cnn_bilstm_model(input_shape: tuple[int, int]) -> Model:
    """Residual CNN front-end with Bidirectional LSTM."""
    inputs = Input(shape=input_shape, name="cnn_bilstm_input")
    x = _residual_conv_block(inputs, 64)
    x = MaxPooling1D()(x)
    x = _residual_conv_block(x, 128)
    x = MaxPooling1D()(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="CNN-BiLSTM")


# ---------------------------------------------------------------------------
# 5. CNN-Attention-LSTM  (residual CNN + multi-head attention + LSTM)
# ---------------------------------------------------------------------------

def build_cnn_attention_lstm_model(input_shape: tuple[int, int]) -> Model:
    """Residual CNN → Multi-Head Attention → LSTM with GlobalAvgPool."""
    inputs = Input(shape=input_shape, name="cnn_attention_lstm_input")
    x = _residual_conv_block(inputs, 64)
    x = MaxPooling1D()(x)
    # Multi-head self-attention (4 heads)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization()(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Attention()([x, x])
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="CNN-Attention-LSTM")


# ---------------------------------------------------------------------------
# 6. Transformer Encoder  (NEW architecture)
# ---------------------------------------------------------------------------

def _transformer_block(x: tf.Tensor, num_heads: int, key_dim: int, ff_dim: int, dropout_rate: float = 0.1) -> tf.Tensor:
    """Single Transformer encoder block: MHA → Add&Norm → FFN → Add&Norm."""
    # Multi-head self-attention
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn_out = Dropout(dropout_rate)(attn_out)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn_out]))

    # Position-wise FFN
    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(x.shape[-1])(ffn)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, ffn]))
    return x


def build_transformer_model(input_shape: tuple[int, int]) -> Model:
    """Pure Transformer encoder for sequence classification.

    Architecture: Linear projection → 2× Transformer blocks
                  → GlobalAvgPool → Dense → Softmax
    """
    inputs = Input(shape=input_shape, name="transformer_input")

    # Project input features to a uniform embedding dimension
    x = Dense(64)(inputs)               # (batch, seq_len, 64)
    x = LayerNormalization()(x)

    # Two stacked Transformer encoder blocks
    x = _transformer_block(x, num_heads=4, key_dim=16, ff_dim=128, dropout_rate=0.1)
    x = _transformer_block(x, num_heads=4, key_dim=16, ff_dim=128, dropout_rate=0.1)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="Transformer")


# ---------------------------------------------------------------------------
# 7. Ensemble (soft-voting over all base models)
# ---------------------------------------------------------------------------

def build_ensemble_model(base_models: list[Model], input_shape: tuple[int, int]) -> Model:
    """Average the softmax outputs of all base models (soft voting).

    Parameters
    ----------
    base_models : list of already-trained Keras Model objects.
    input_shape : (seq_len, feature_count) tuple.
    """
    inputs = Input(shape=input_shape, name="ensemble_input")

    proba_list = []
    for m in base_models:
        m.trainable = False          # freeze individual models
        proba_list.append(m(inputs, training=False))

    # Element-wise average across models → shape (batch, NUM_CLASSES)
    averaged = tf.keras.layers.Average()(proba_list)
    return Model(inputs=inputs, outputs=averaged, name="Ensemble")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def get_model_builders() -> dict[str, callable]:
    """Return builder functions for all trainable models."""
    return {
        "CNN": build_cnn_model,
        "LSTM": build_lstm_model,
        "CNN-LSTM": build_cnn_lstm_model,
        "CNN-BiLSTM": build_cnn_bilstm_model,
        "CNN-Attention-LSTM": build_cnn_attention_lstm_model,
        "Transformer": build_transformer_model,
    }
