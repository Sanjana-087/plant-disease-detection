"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for model explainability.

Highlights image regions that most influenced the model's prediction for a given class.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.logger import get_logger

logger = get_logger(__name__)


def get_last_conv_layer(model: keras.Model) -> str:
    """
    Auto-detect the name of the last convolutional layer in a Keras model.

    Searches top-level layers and nested backbone models (ResNet, EfficientNet).

    Args:
        model: Trained or untrained Keras model.

    Returns:
        Name of the last Conv2D layer.

    Raises:
        ValueError: If no convolutional layer is found.
    """
    conv_layers: list[keras.layers.Layer] = []

    def _collect(layer: keras.layers.Layer) -> None:
        if isinstance(layer, keras.layers.Conv2D):
            conv_layers.append(layer)
        if isinstance(layer, keras.Model):
            for sub in layer.layers:
                _collect(sub)

    for layer in model.layers:
        _collect(layer)

    if not conv_layers:
        raise ValueError("No Conv2D layer found in the model for Grad-CAM.")

    last_layer = conv_layers[-1]
    logger.debug("Last conv layer for Grad-CAM: %s", last_layer.name)
    return last_layer.name


def _find_layer_by_name(model: keras.Model, layer_name: str) -> keras.layers.Layer:
    """
    Find a layer in a model (including nested models) by name.

    Args:
        model: Keras model to search.
        layer_name: Target layer name.

    Returns:
        Matching layer object.

    Raises:
        ValueError: If the layer is not found.
    """
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
        if isinstance(layer, keras.Model):
            try:
                return layer.get_layer(layer_name)
            except ValueError:
                continue
    try:
        return model.get_layer(layer_name)
    except ValueError as exc:
        raise ValueError(f"Layer '{layer_name}' not found in model.") from exc


def _make_gradcam_model(model: keras.Model, last_conv_layer_name: str) -> keras.Model:
    """
    Build a sub-model returning conv feature maps and predictions.

    Args:
        model: Full classification model.
        last_conv_layer_name: Name of the target convolution layer.

    Returns:
        Model with outputs [conv_feature_maps, predictions].
    """
    last_conv_layer = _find_layer_by_name(model, last_conv_layer_name)
    return keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
        name="grad_cam_submodel",
    )


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert an image array to uint8 RGB for display.

    Args:
        image: Array (H, W, 3) in [0, 1] or [0, 255].

    Returns:
        uint8 RGB image.
    """
    if image.max() <= 1.0:
        return (np.clip(image, 0, 1) * 255.0).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def generate_grad_cam(
    model: keras.Model,
    image_array: np.ndarray,
    class_index: int,
    last_conv_layer_name: str | None = None,
    display_image: np.ndarray | None = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap overlaid on the input image.

    Uses tf.GradientTape to compute gradients of the target class score with
    respect to the last convolutional feature maps.

    Args:
        model: Trained Keras classifier.
        image_array: Model input — shape (H, W, 3) or (1, H, W, 3), preprocessed
            as required by the model (e.g. ResNet preprocess_input).
        class_index: Target class index for visualization.
        last_conv_layer_name: Conv layer name; auto-detected if None.
        display_image: Optional uint8/float RGB image for overlay (before preprocess).
            If None, image_array is used (works when values are in [0, 255]).
        alpha: Blend factor for heatmap overlay.

    Returns:
        RGB image as uint8 numpy array (H, W, 3) with heatmap overlay.

    Raises:
        ValueError: If inputs are invalid or gradients are None.
        ImportError: If opencv-python is not installed.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for Grad-CAM overlay.") from exc

    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer(model)

    if image_array.ndim == 3:
        batch_img = np.expand_dims(image_array, axis=0).astype(np.float32)
    else:
        batch_img = image_array.astype(np.float32)

    if display_image is not None:
        if display_image.ndim == 4:
            display_image = display_image[0]
        original_uint8 = _to_uint8_rgb(display_image)
    else:
        display = batch_img[0]
        original_uint8 = _to_uint8_rgb(display)

    if original_uint8.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB image with 3 channels, got shape {original_uint8.shape}"
        )

    grad_model = _make_gradcam_model(model, last_conv_layer_name)
    conv_outputs, predictions = grad_model(batch_img)

    with tf.GradientTape() as tape:
        tape.watch(conv_outputs)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradients are None — cannot compute Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs_np = conv_outputs.numpy()[0]
    pooled_grads_np = pooled_grads.numpy()

    heatmap = np.zeros(conv_outputs_np.shape[:2], dtype=np.float32)
    for i, weight in enumerate(pooled_grads_np):
        heatmap += weight * conv_outputs_np[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_resized = cv2.resize(
        heatmap,
        (original_uint8.shape[1], original_uint8.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlaid = cv2.addWeighted(original_uint8, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlaid


def generate_grad_cam_from_path(
    model: keras.Model,
    image_path: str,
    class_index: int,
    preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    target_size: tuple[int, int] = (224, 224),
    last_conv_layer_name: str | None = None,
) -> np.ndarray:
    """
    Load an image from disk, preprocess, and generate Grad-CAM overlay.

    Args:
        model: Trained Keras model.
        image_path: Path to image file.
        class_index: Target class index.
        preprocess_fn: Optional batch preprocess (e.g. resnet_preprocess on (1,H,W,3)).
        target_size: (width, height) for resizing.
        last_conv_layer_name: Optional conv layer name.

    Returns:
        RGB overlaid image as uint8 numpy array.

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If the image cannot be loaded.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for Grad-CAM.") from exc

    from pathlib import Path

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, target_size)
    original_uint8 = rgb.copy()

    img = rgb.astype(np.float32)
    batch = np.expand_dims(img, axis=0)
    if preprocess_fn is not None:
        batch = preprocess_fn(batch)

    return generate_grad_cam(
        model,
        batch,
        class_index,
        last_conv_layer_name=last_conv_layer_name,
        display_image=original_uint8,
    )
