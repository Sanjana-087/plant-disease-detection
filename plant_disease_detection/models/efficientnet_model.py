"""
EfficientNet B0/B3 transfer-learning models for plant disease classification.

Phase 1: train classifier head with frozen ImageNet backbone.
Phase 2: unfreeze top layers via unfreeze_top_layers() for fine-tuning.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

from training.config import (
    EFFICIENTNET_UNFREEZE_LAYERS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INPUT_SHAPE,
    LEARNING_RATE,
)
from utils.logger import get_logger

logger = get_logger(__name__)

PREPROCESS_INPUT = preprocess_input

_EFFICIENTNET_BACKBONES = {
    "B0": EfficientNetB0,
    "B3": EfficientNetB3,
}


def _normalize_version(version: str) -> str:
    """
    Normalize version string to 'B0' or 'B3'.

    Args:
        version: User-provided version (e.g. 'b0', 'efficientnetb3').

    Returns:
        'B0' or 'B3'.

    Raises:
        ValueError: If version is not supported.
    """
    key = version.upper().replace("EFFICIENTNET", "").strip()
    if key in ("B0", "0"):
        return "B0"
    if key in ("B3", "3"):
        return "B3"
    raise ValueError(
        f"Unsupported EfficientNet version '{version}'. Use 'B0' or 'B3'."
    )


class EfficientNetPlantModel:
    """
    EfficientNet-based classifier with frozen backbone and fine-tuning support.

    Supports EfficientNetB0 and EfficientNetB3 via the version parameter.
    """

    def __init__(
        self,
        num_classes: int,
        version: str = "B0",
        input_shape: tuple[int, int, int] = INPUT_SHAPE,
        learning_rate: float = LEARNING_RATE,
        weights: str = "imagenet",
    ) -> None:
        """
        Initialize EfficientNet builder.

        Args:
            num_classes: Number of disease classes.
            version: 'B0' or 'B3' (also accepts 'efficientnetb0', etc.).
            input_shape: (height, width, channels).
            learning_rate: Adam learning rate when compiling.
            weights: Backbone weights ('imagenet' or None).

        Raises:
            ValueError: If num_classes < 2 or version is invalid.
        """
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.num_classes = num_classes
        self.version = _normalize_version(version)
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.weights = weights
        self.model: keras.Model | None = None
        self.base_model: keras.Model | None = None

    def build(self, compile_model: bool = True) -> keras.Model:
        """
        Build EfficientNet + custom head with frozen backbone for Phase 1.

        Args:
            compile_model: If True, compile with Adam and categorical_crossentropy.

        Returns:
            Keras model instance.
        """
        inputs = layers.Input(shape=self.input_shape, name="input_image")

        backbone_cls = _EFFICIENTNET_BACKBONES[self.version]
        self.base_model = backbone_cls(
            include_top=False,
            weights=self.weights,
            input_tensor=inputs,
            pooling=None,
        )
        self.base_model.trainable = False

        x = self.base_model.output
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.Dense(256, activation="relu", name="fc_dense")(x)
        x = layers.BatchNormalization(name="fc_bn")(x)
        x = layers.Dropout(0.3, name="fc_dropout")(x)
        outputs = layers.Dense(
            self.num_classes,
            activation="softmax",
            name="predictions",
        )(x)

        model_name = f"efficientnet{self.version.lower()}_plant"
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        if compile_model:
            self.compile()

        logger.info(
            "Built EfficientNet%s: classes=%d, frozen base=%s",
            self.version,
            self.num_classes,
            not self.base_model.trainable,
        )
        return self.model

    def compile(self, learning_rate: float | None = None) -> None:
        """
        Compile with Adam optimizer and categorical crossentropy.

        Args:
            learning_rate: Optional Adam learning rate override.

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
        logger.info("EfficientNet%s compiled with Adam lr=%s", self.version, lr)

    def unfreeze_top_layers(self, n: int = EFFICIENTNET_UNFREEZE_LAYERS) -> None:
        """
        Unfreeze the top n backbone layers for Phase 2 fine-tuning.

        Args:
            n: Number of layers from the end of base_model to train.

        Returns:
            None

        Raises:
            RuntimeError: If build() was not called.
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
            "EfficientNet%s: unfroze top %d/%d layers (freeze_until=%d, trainable=%d).",
            self.version,
            n,
            total_layers,
            freeze_until,
            trainable_layers,
        )

    def get_model(self) -> keras.Model:
        """
        Return the built Keras model.

        Returns:
            Keras model.

        Raises:
            RuntimeError: If build() has not been called.
        """
        if self.model is None:
            raise RuntimeError("Call build() before get_model().")
        return self.model


def build_efficientnet_model(
    num_classes: int,
    version: str = "B0",
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = LEARNING_RATE,
    compile_model: bool = True,
) -> keras.Model:
    """
    Build a compiled EfficientNet B0 or B3 plant disease classifier.

    Args:
        num_classes: Number of output classes.
        version: 'B0' or 'B3'.
        input_shape: Input shape (H, W, C).
        learning_rate: Adam learning rate.
        compile_model: Whether to compile the model.

    Returns:
        tf.keras.Model with frozen EfficientNet backbone.
    """
    builder = EfficientNetPlantModel(
        num_classes=num_classes,
        version=version,
        input_shape=input_shape,
        learning_rate=learning_rate,
    )
    return builder.build(compile_model=compile_model)


def unfreeze_efficientnet_top_layers(
    model: keras.Model,
    n: int = EFFICIENTNET_UNFREEZE_LAYERS,
) -> keras.Model:
    """
    Unfreeze the top n layers of an EfficientNet base inside a Keras model.

    Args:
        model: Model from build_efficientnet_model().
        n: Number of backbone layers to unfreeze.

    Returns:
        The same model (modified in place).

    Raises:
        ValueError: If no EfficientNet backbone is found.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find EfficientNet base model in the given model.")

    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - n)

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True

    logger.info("Unfroze top %d EfficientNet layers.", n)
    return model


def get_efficientnet_b0(
    num_classes: int,
    input_shape: tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
) -> keras.Model:
    """
    Factory alias for EfficientNetB0.

    Args:
        num_classes: Number of classes.
        input_shape: Spatial input shape.

    Returns:
        Compiled EfficientNetB0 model.
    """
    return build_efficientnet_model(
        num_classes=num_classes,
        version="B0",
        input_shape=input_shape,
        compile_model=True,
    )


def get_efficientnet_b3(
    num_classes: int,
    input_shape: tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
) -> keras.Model:
    """
    Factory alias for EfficientNetB3.

    Args:
        num_classes: Number of classes.
        input_shape: Spatial input shape.

    Returns:
        Compiled EfficientNetB3 model.
    """
    return build_efficientnet_model(
        num_classes=num_classes,
        version="B3",
        input_shape=input_shape,
        compile_model=True,
    )
