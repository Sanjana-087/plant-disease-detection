"""
TensorFlow/Keras image augmentation pipelines for training and evaluation.

Training applies randomized geometric and photometric transforms. Validation
and test use only resize and optional rescaling (model-specific preprocessing
such as ResNet's preprocess_input is applied in the data pipeline when needed).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from training.config import AUGMENTATION, IMAGE_HEIGHT, IMAGE_WIDTH, RANDOM_SEED
from utils.logger import get_logger

logger = get_logger(__name__)


def _rotation_factor(degrees: float) -> float:
    """
    Convert degrees to Keras RandomRotation factor (fraction of 2*pi).

    Args:
        degrees: Maximum rotation in degrees (e.g. 30 for +/-30 degrees).

    Returns:
        Factor passed to layers.RandomRotation.
    """
    return degrees / 360.0


def get_training_augmentation(
    image_size: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
    seed: int = RANDOM_SEED,
    rescale: bool = True,
) -> keras.Sequential:
    """
    Build the training augmentation model with random transforms.

    Pipeline order: resize (for crop headroom) -> flip -> rotation -> zoom ->
    translation -> random crop -> brightness -> contrast -> rescale.

    Args:
        image_size: Target (height, width) after augmentation.
        seed: Random seed for reproducible augmentation.
        rescale: If True, scale pixel values to [0, 1] via Rescaling(1/255).
            Set False when using ResNet/EfficientNet preprocess_input later.

    Returns:
        tf.keras.Sequential model applied to batched images in [0, 255] float32.
    """
    height, width = image_size
    aug = AUGMENTATION

    # Resize slightly larger so RandomCrop can sample diverse regions
    resize_h = max(height, int(height / aug["crop_height_factor"]))
    resize_w = max(width, int(width / aug["crop_width_factor"]))

    zoom = aug["zoom_range"]
    zoom_factors = (-zoom, zoom) if isinstance(zoom, (int, float)) else tuple(zoom)

    layer_list: list[layers.Layer] = [
        # Resize up before crop — gives room for RandomCrop to vary composition
        layers.Resizing(
            resize_h,
            resize_w,
            interpolation="bilinear",
            name="resize_for_crop",
        ),
        # RandomFlip — mirrors leaves horizontally (natural symmetry for many plants)
        layers.RandomFlip(
            mode=aug.get("random_flip", "horizontal"),
            seed=seed,
            name="random_flip",
        ),
        # RandomRotation — +/-30 degrees simulates camera angle variation
        layers.RandomRotation(
            factor=_rotation_factor(aug["rotation_range"]),
            fill_mode="reflect",
            seed=seed,
            name="random_rotation",
        ),
        # RandomZoom — +/-20% scale mimics distance / focal length changes
        layers.RandomZoom(
            height_factor=zoom_factors,
            width_factor=zoom_factors,
            fill_mode="reflect",
            seed=seed,
            name="random_zoom",
        ),
        # RandomTranslation — small shifts improve robustness to off-center leaves
        layers.RandomTranslation(
            height_factor=aug["translation_height_factor"],
            width_factor=aug["translation_width_factor"],
            fill_mode="reflect",
            seed=seed,
            name="random_translation",
        ),
        # RandomCrop — extracts target_size patch from the enlarged canvas
        layers.RandomCrop(
            height=height,
            width=width,
            seed=seed,
            name="random_crop",
        ),
        # RandomBrightness — lighting changes (sunlight, shade, camera exposure)
        layers.RandomBrightness(
            factor=aug["brightness_range"],
            seed=seed,
            name="random_brightness",
        ),
        # RandomContrast — emphasizes or softens disease texture patterns
        layers.RandomContrast(
            factor=aug["contrast_range"],
            seed=seed,
            name="random_contrast",
        ),
    ]

    if rescale:
        # Rescale — maps [0, 255] inputs to [0, 1] for CNN baseline training
        layer_list.append(
            layers.Rescaling(1.0 / 255.0, name="rescale_to_01")
        )

    model = keras.Sequential(layer_list, name="training_augmentation")
    logger.info(
        "Built training augmentation: %s -> %s (rescale=%s)",
        (resize_h, resize_w),
        image_size,
        rescale,
    )
    return model


def get_validation_augmentation(
    image_size: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
    rescale: bool = True,
) -> keras.Sequential:
    """
    Build validation augmentation: resize and optional rescale only.

    No random transforms — ensures consistent metrics across epochs.

    Args:
        image_size: Target (height, width).
        rescale: If True, apply Rescaling(1/255).

    Returns:
        tf.keras.Sequential model for validation data.
    """
    height, width = image_size
    layer_list: list[layers.Layer] = [
        # Resize — fixed spatial dimensions required by the network
        layers.Resizing(
            height,
            width,
            interpolation="bilinear",
            name="resize",
        ),
    ]
    if rescale:
        # Normalize — same scaling as training for CNN (no random noise)
        layer_list.append(
            layers.Rescaling(1.0 / 255.0, name="rescale_to_01")
        )

    model = keras.Sequential(layer_list, name="validation_augmentation")
    logger.info("Built validation augmentation: resize %s, rescale=%s", image_size, rescale)
    return model


def get_test_augmentation(
    image_size: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
    rescale: bool = True,
) -> keras.Sequential:
    """
    Build test augmentation (identical to validation: resize + normalize).

    Args:
        image_size: Target (height, width).
        rescale: If True, apply Rescaling(1/255).

    Returns:
        tf.keras.Sequential model for test data.
    """
    height, width = image_size
    layer_list: list[layers.Layer] = [
        layers.Resizing(height, width, interpolation="bilinear", name="resize"),
    ]
    if rescale:
        layer_list.append(layers.Rescaling(1.0 / 255.0, name="rescale_to_01"))

    return keras.Sequential(layer_list, name="test_augmentation")


def apply_augmentation(
    images: tf.Tensor,
    augmentation_model: keras.Sequential,
    training: bool = True,
) -> tf.Tensor:
    """
    Apply a Sequential augmentation model to a batch of images.

    Args:
        images: Float tensor (batch, H, W, 3) in [0, 255] or already rescaled.
        augmentation_model: Model from get_training/validation_augmentation.
        training: Passed to layers that behave differently in train vs inference
            (kept for API symmetry; Keras preprocessing layers ignore this).

    Returns:
        Augmented tensor with the same batch dimension.
    """
    return augmentation_model(images, training=training)


def get_augmentation_for_model(
    model_name: str,
    training: bool = True,
    image_size: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
) -> keras.Sequential:
    """
    Return augmentation appropriate for the model family.

    CNN baseline uses rescale to [0, 1]. ResNet and EfficientNet skip rescale
    here because their preprocess_input is applied when loading images.

    Args:
        model_name: One of cnn_baseline, resnet50, efficientnetb0, efficientnetb3.
        training: If True, return training pipeline; else validation/test.
        image_size: Target (height, width).

    Returns:
        tf.keras.Sequential augmentation model.

    Raises:
        ValueError: If model_name is not recognized.
    """
    name = model_name.lower().strip()
    transfer_models = {"resnet50", "efficientnetb0", "efficientnetb3"}
    rescale = name not in transfer_models

    if name not in {"cnn_baseline", *transfer_models}:
        raise ValueError(
            f"Unknown model_name '{model_name}' for augmentation selection."
        )

    if training:
        return get_training_augmentation(image_size=image_size, rescale=rescale)
    return get_validation_augmentation(image_size=image_size, rescale=rescale)
