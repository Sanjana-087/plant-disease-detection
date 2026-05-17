"""
ResNet50 transfer-learning model for plant disease classification.

Phase 1: train classifier head with frozen ImageNet backbone.
Phase 2: unfreeze top layers via unfreeze_top_layers() for fine-tuning.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from training.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INPUT_SHAPE,
    LEARNING_RATE,
    RESNET_UNFREEZE_LAYERS,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Used by split_dataset and inference pipelines — do not rescale to [0,1] before this
PREPROCESS_INPUT = preprocess_input


class ResNet50PlantModel:
    """
    ResNet50-based classifier with frozen backbone and fine-tuning support.

    Attributes:
        model: Compiled Keras model after build().
        base_model: ResNet50 feature extractor (include_top=False).
    """

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, int, int] = INPUT_SHAPE,
        learning_rate: float = LEARNING_RATE,
        weights: str = "imagenet",
    ) -> None:
        """
        Initialize builder (model is created when build() is called).

        Args:
            num_classes: Number of disease classes.
            input_shape: (height, width, channels).
            learning_rate: Adam learning rate for compile().
            weights: ResNet50 weights ('imagenet' or None).

        Raises:
            ValueError: If num_classes < 2.
        """
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.weights = weights
        self.model: keras.Model | None = None
        self.base_model: keras.Model | None = None

    def build(self, compile_model: bool = True) -> keras.Model:
        """
        Construct ResNet50 + custom head. Base layers are frozen for Phase 1.

        Args:
            compile_model: If True, compile with Adam and categorical_crossentropy.

        Returns:
            Compiled or uncompiled Keras model.
        """
        inputs = layers.Input(shape=self.input_shape, name="input_image")

        # ResNet50 backbone — pretrained on ImageNet, no top classification layer
        self.base_model = ResNet50(
            include_top=False,
            weights=self.weights,
            input_tensor=inputs,
            pooling=None,
        )
        # Freeze backbone for Phase 1 transfer learning
        self.base_model.trainable = False

        x = self.base_model.output
        # GlobalAveragePooling2D — aggregate spatial features into a vector
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        # Dense(512) — disease-specific representation on top of ImageNet features
        x = layers.Dense(512, activation="relu", name="fc_dense")(x)
        x = layers.BatchNormalization(name="fc_bn")(x)
        # Dropout — regularize the head before softmax
        x = layers.Dropout(0.4, name="fc_dropout")(x)
        outputs = layers.Dense(
            self.num_classes,
            activation="softmax",
            name="predictions",
        )(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name="resnet50_plant")

        if compile_model:
            self.compile()

        logger.info(
            "Built ResNet50 model: classes=%d, frozen base=%s",
            self.num_classes,
            not self.base_model.trainable,
        )
        return self.model

    def compile(self, learning_rate: float | None = None) -> None:
        """
        Compile the model with Adam and categorical crossentropy.

        Args:
            learning_rate: Optional override for Adam learning rate.

        Returns:
            None

        Raises:
            RuntimeError: If build() has not been called.
        """
        if self.model is None:
            raise RuntimeError("Call build() before compile().")

        lr = learning_rate if learning_rate is not None else self.learning_rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info("ResNet50 compiled with Adam lr=%s", lr)

    def unfreeze_top_layers(self, n: int = RESNET_UNFREEZE_LAYERS) -> None:
        """
        Unfreeze the top n layers of the ResNet50 backbone for Phase 2 fine-tuning.

        Args:
            n: Number of layers (from the end of base_model) to set trainable.

        Returns:
            None

        Raises:
            RuntimeError: If build() has not been called.
            ValueError: If n is not positive.
        """
        if self.base_model is None or self.model is None:
            raise RuntimeError("Call build() before unfreeze_top_layers().")
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        self.base_model.trainable = True
        total_layers = len(self.base_model.layers)
        freeze_until = max(0, total_layers - n)

        for layer in self.base_model.layers[:freeze_until]:
            layer.trainable = False
        for layer in self.base_model.layers[freeze_until:]:
            layer.trainable = True

        trainable_layers = sum(1 for layer in self.base_model.layers if layer.trainable)
        logger.info(
            "Unfroze top %d/%d ResNet50 layers (freeze_until=%d, trainable=%d).",
            n,
            total_layers,
            freeze_until,
            trainable_layers,
        )

    def get_model(self) -> keras.Model:
        """
        Return the Keras model instance.

        Returns:
            Built Keras model.

        Raises:
            RuntimeError: If build() has not been called.
        """
        if self.model is None:
            raise RuntimeError("Call build() before get_model().")
        return self.model


def build_resnet50_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = LEARNING_RATE,
    compile_model: bool = True,
) -> keras.Model:
    """
    Build and return a compiled ResNet50 transfer-learning model.

    Args:
        num_classes: Number of output classes.
        input_shape: Input tensor shape (H, W, C).
        learning_rate: Adam learning rate.
        compile_model: Whether to compile the model.

    Returns:
        tf.keras.Model with frozen ResNet50 backbone.
    """
    builder = ResNet50PlantModel(
        num_classes=num_classes,
        input_shape=input_shape,
        learning_rate=learning_rate,
    )
    return builder.build(compile_model=compile_model)


def unfreeze_resnet50_top_layers(
    model: keras.Model,
    n: int = RESNET_UNFREEZE_LAYERS,
) -> keras.Model:
    """
    Unfreeze the top n layers of the ResNet50 base inside a compiled model.

    Args:
        model: Model returned by build_resnet50_model().
        n: Number of backbone layers to unfreeze from the end.

    Returns:
        The same model instance (modified in place).

    Raises:
        ValueError: If no ResNet50 base layer is found in the model.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "resnet50" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find ResNet50 base model inside the given model.")

    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - n)

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True

    logger.info("Unfroze top %d ResNet50 layers via unfreeze_resnet50_top_layers().", n)
    return model


def get_resnet50(
    num_classes: int,
    input_shape: tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
) -> keras.Model:
    """
    Factory alias: compiled ResNet50 plant disease model.

    Args:
        num_classes: Number of classes.
        input_shape: Spatial input shape.

    Returns:
        Compiled Keras model.
    """
    return build_resnet50_model(
        num_classes=num_classes,
        input_shape=input_shape,
        compile_model=True,
    )
