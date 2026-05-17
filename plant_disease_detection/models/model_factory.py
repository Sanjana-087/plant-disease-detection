"""
Model factory — single entry point to instantiate any supported architecture.
"""

import tensorflow as tf
from tensorflow import keras

from models.cnn_baseline import build_cnn_baseline
from models.efficientnet_model import build_efficientnet_model
from models.resnet50_model import build_resnet50_model
from training.config import IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_SHAPE, LEARNING_RATE, MODEL_NAMES
from utils.logger import get_logger

logger = get_logger(__name__)

# Aliases map user-facing names to canonical keys
_MODEL_ALIASES: dict[str, str] = {
    "cnn": "cnn_baseline",
    "cnn_baseline": "cnn_baseline",
    "baseline": "cnn_baseline",
    "resnet": "resnet50",
    "resnet50": "resnet50",
    "resnet-50": "resnet50",
    "efficientnetb0": "efficientnetb0",
    "efficientnet_b0": "efficientnetb0",
    "efficientnet-b0": "efficientnetb0",
    "efficientnetb3": "efficientnetb3",
    "efficientnet_b3": "efficientnetb3",
    "efficientnet-b3": "efficientnetb3",
    "efficientnet": "efficientnetb0",
}


def _normalize_model_name(model_name: str) -> str:
    """
    Normalize a model name string to a canonical factory key.

    Args:
        model_name: User-provided model identifier.

    Returns:
        Canonical name in MODEL_NAMES.

    Raises:
        ValueError: If the name is not recognized.
    """
    key = model_name.lower().strip().replace(" ", "_")
    canonical = _MODEL_ALIASES.get(key, key)

    if canonical not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {', '.join(MODEL_NAMES)}"
        )
    return canonical


def get_model(
    model_name: str,
    num_classes: int,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = LEARNING_RATE,
    compile_model: bool = True,
    print_summary: bool = True,
) -> keras.Model:
    """
    Build and return a compiled model by name.

    Args:
        model_name: One of 'cnn_baseline', 'resnet50', 'efficientnetb0',
            'efficientnetb3' (aliases accepted).
        num_classes: Number of output disease classes.
        input_shape: (height, width, channels), default (224, 224, 3).
        learning_rate: Adam learning rate passed to builders.
        compile_model: If True, model is compiled before return.
        print_summary: If True, print Keras model summary to stdout.

    Returns:
        Compiled tf.keras.Model ready for training.

    Raises:
        ValueError: If model_name is unknown or num_classes < 2.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    canonical = _normalize_model_name(model_name)
    logger.info(
        "Creating model '%s' (classes=%d, input_shape=%s)",
        canonical,
        num_classes,
        input_shape,
    )

    if canonical == "cnn_baseline":
        model = build_cnn_baseline(
            num_classes=num_classes,
            input_shape=input_shape,
            learning_rate=learning_rate,
            compile_model=compile_model,
        )
    elif canonical == "resnet50":
        model = build_resnet50_model(
            num_classes=num_classes,
            input_shape=input_shape,
            learning_rate=learning_rate,
            compile_model=compile_model,
        )
    elif canonical == "efficientnetb0":
        model = build_efficientnet_model(
            num_classes=num_classes,
            version="B0",
            input_shape=input_shape,
            learning_rate=learning_rate,
            compile_model=compile_model,
        )
    elif canonical == "efficientnetb3":
        model = build_efficientnet_model(
            num_classes=num_classes,
            version="B3",
            input_shape=input_shape,
            learning_rate=learning_rate,
            compile_model=compile_model,
        )
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {', '.join(MODEL_NAMES)}"
        )

    if print_summary:
        print(f"\n{'=' * 60}")
        print(f"Model Summary: {canonical}")
        print(f"{'=' * 60}")
        model.summary()
        print(f"{'=' * 60}\n")

    total_params = model.count_params()
    logger.info("Model '%s' ready — total parameters: %s", canonical, f"{total_params:,}")
    return model


def list_available_models() -> list[str]:
    """
    Return the list of supported model names.

    Returns:
        List of canonical model name strings.
    """
    return list(MODEL_NAMES)


def is_transfer_learning_model(model_name: str) -> bool:
    """
    Check whether a model uses a frozen pretrained backbone in Phase 1.

    Args:
        model_name: Model identifier.

    Returns:
        True for resnet50 and efficientnet variants.
    """
    canonical = _normalize_model_name(model_name)
    return canonical in ("resnet50", "efficientnetb0", "efficientnetb3")
