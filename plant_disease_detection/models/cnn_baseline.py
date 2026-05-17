"""
Custom CNN baseline for plant disease classification.

A four-block convolutional network built with the Keras Functional API.
Suitable as a lightweight baseline before transfer-learning models.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from training.config import IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_SHAPE, LEARNING_RATE
from utils.logger import get_logger

logger = get_logger(__name__)


def _conv_block(
    x: tf.Tensor,
    filters: int,
    block_name: str,
    dropout_rate: float = 0.25,
) -> tf.Tensor:
    """
    One convolutional block: Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout.

    Args:
        x: Input feature map tensor.
        filters: Number of convolution filters for this block.
        block_name: Prefix for layer names (e.g. 'block1').
        dropout_rate: Dropout probability after pooling.

    Returns:
        Output tensor after the block.
    """
    # Conv2D — learns spatial filters (edges, textures, disease patterns)
    x = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name=f"{block_name}_conv",
    )(x)
    # BatchNormalization — stabilizes training and speeds convergence
    x = layers.BatchNormalization(name=f"{block_name}_bn")(x)
    # ReLU — introduces non-linearity
    x = layers.Activation("relu", name=f"{block_name}_relu")(x)
    # MaxPooling2D — downsamples feature maps, adds translation invariance
    x = layers.MaxPooling2D(pool_size=(2, 2), name=f"{block_name}_pool")(x)
    # Dropout — regularizes to reduce overfitting on leaf textures
    x = layers.Dropout(dropout_rate, name=f"{block_name}_dropout")(x)
    return x


def build_cnn_baseline(
    num_classes: int,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = LEARNING_RATE,
    compile_model: bool = True,
) -> keras.Model:
    """
    Build and optionally compile the custom CNN baseline model.

    Architecture:
        4 conv blocks (32, 64, 128, 256 filters) -> GlobalAveragePooling2D ->
        Dense(512) -> BatchNorm -> ReLU -> Dropout(0.5) -> softmax output.

    Args:
        num_classes: Number of disease classes.
        input_shape: (height, width, channels), default (224, 224, 3).
        learning_rate: Adam learning rate when compile_model is True.
        compile_model: If True, compile with Adam and categorical_crossentropy.

    Returns:
        tf.keras.Model instance.

    Raises:
        ValueError: If num_classes < 2 or input_shape is invalid.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    if len(input_shape) != 3:
        raise ValueError(f"input_shape must be (H, W, C), got {input_shape}")

    # Input layer — raw RGB leaf image
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Block 1 — 32 filters: low-level edges and color blobs
    x = _conv_block(inputs, filters=32, block_name="block1", dropout_rate=0.25)
    # Block 2 — 64 filters: simple textures and spots
    x = _conv_block(x, filters=64, block_name="block2", dropout_rate=0.25)
    # Block 3 — 128 filters: lesion shapes and vein patterns
    x = _conv_block(x, filters=128, block_name="block3", dropout_rate=0.30)
    # Block 4 — 256 filters: higher-level disease appearance
    x = _conv_block(x, filters=256, block_name="block4", dropout_rate=0.30)

    # GlobalAveragePooling2D — spatial summary without large fully-connected params
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Dense(512) — learns combinations of visual features for classification
    x = layers.Dense(512, use_bias=False, name="fc_dense")(x)
    x = layers.BatchNormalization(name="fc_bn")(x)
    x = layers.Activation("relu", name="fc_relu")(x)
    # Dropout(0.5) — strong regularization on the classifier head
    x = layers.Dropout(0.5, name="fc_dropout")(x)

    # Output — one probability per class (sums to 1.0)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_baseline")

    if compile_model:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info(
            "Compiled cnn_baseline: classes=%d, input=%s, lr=%s",
            num_classes,
            input_shape,
            learning_rate,
        )

    return model


def get_cnn_baseline(
    num_classes: int,
    input_shape: tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
) -> keras.Model:
    """
    Factory alias for build_cnn_baseline (compiled model).

    Args:
        num_classes: Number of output classes.
        input_shape: Model input spatial shape.

    Returns:
        Compiled CNN baseline model.
    """
    return build_cnn_baseline(
        num_classes=num_classes,
        input_shape=input_shape,
        compile_model=True,
    )
