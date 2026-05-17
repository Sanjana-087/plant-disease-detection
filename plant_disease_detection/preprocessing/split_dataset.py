"""
Train/validation/test splitting and tf.data.Dataset construction.

Splits image paths with stratified sampling, then builds lazy-loading TensorFlow
datasets with augmentation applied only on the training split.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from preprocessing.augmentation import get_augmentation_for_model
from training.config import (
    BATCH_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    get_num_classes,
)
from utils.logger import get_logger

logger = get_logger(__name__)

AUTOTUNE = tf.data.AUTOTUNE


def stratified_train_val_test_split(
    image_paths: list[str],
    labels: list[int],
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT,
    random_state: int = RANDOM_SEED,
) -> tuple[
    tuple[list[str], list[int]],
    tuple[list[str], list[int]],
    tuple[list[str], list[int]],
]:
    """
    Split paths and labels into train, validation, and test sets.

    Uses two-stage stratified train_test_split so class proportions are preserved.

    Args:
        image_paths: List of image file paths.
        labels: Integer-encoded labels parallel to image_paths.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for test (default 0.15).
        random_state: Random seed for reproducibility.

    Returns:
        Three tuples: (X_train, y_train), (X_val, y_val), (X_test, y_test).

    Raises:
        ValueError: If ratios do not sum to 1.0 or inputs are empty/mismatched.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if len(image_paths) != len(labels):
        raise ValueError(
            f"Path/label length mismatch: {len(image_paths)} vs {len(labels)}"
        )
    if len(image_paths) == 0:
        raise ValueError("Cannot split an empty dataset.")

    paths = np.array(image_paths)
    y = np.array(labels)

    # Ensure stratification is possible (each class needs at least 2 samples)
    unique, counts = np.unique(y, return_counts=True)
    if np.any(counts < 2):
        logger.warning(
            "Some classes have fewer than 2 samples; stratification may fail. "
            "Affected classes: %s",
            unique[counts < 2].tolist(),
        )

    holdout_ratio = val_ratio + test_ratio

    try:
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            paths,
            y,
            test_size=holdout_ratio,
            stratify=y,
            random_state=random_state,
        )
        relative_test_size = test_ratio / holdout_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_holdout,
            y_holdout,
            test_size=relative_test_size,
            stratify=y_holdout,
            random_state=random_state,
        )
    except ValueError as exc:
        logger.warning("Stratified split failed (%s); using random split.", exc)
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            paths,
            y,
            test_size=holdout_ratio,
            random_state=random_state,
        )
        relative_test_size = test_ratio / holdout_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_holdout,
            y_holdout,
            test_size=relative_test_size,
            random_state=random_state,
        )

    train_tuple = (X_train.tolist(), y_train.tolist())
    val_tuple = (X_val.tolist(), y_val.tolist())
    test_tuple = (X_test.tolist(), y_test.tolist())

    print_split_summary(train_tuple, val_tuple, test_tuple)
    return train_tuple, val_tuple, test_tuple


def print_split_summary(
    train_data: tuple[list[str], list[int]],
    val_data: tuple[list[str], list[int]],
    test_data: tuple[list[str], list[int]],
) -> None:
    """
    Log dataset split sizes and percentages.

    Args:
        train_data: (paths, labels) for training.
        val_data: (paths, labels) for validation.
        test_data: (paths, labels) for test.

    Returns:
        None
    """
    n_train = len(train_data[0])
    n_val = len(val_data[0])
    n_test = len(test_data[0])
    total = n_train + n_val + n_test

    logger.info("=" * 50)
    logger.info("DATASET SPLIT SUMMARY")
    logger.info("=" * 50)
    logger.info("Train : %6d  (%5.1f%%)", n_train, 100.0 * n_train / total)
    logger.info("Val   : %6d  (%5.1f%%)", n_val, 100.0 * n_val / total)
    logger.info("Test  : %6d  (%5.1f%%)", n_test, 100.0 * n_test / total)
    logger.info("Total : %6d", total)
    logger.info("=" * 50)


def _get_preprocess_fn(model_name: str) -> Callable[[tf.Tensor], tf.Tensor] | None:
    """
    Return model-specific preprocess_input or None for CNN baseline.

    Args:
        model_name: Registered model name.

    Returns:
        Callable that preprocesses a float image batch, or None.
    """
    name = model_name.lower().strip()
    if name == "resnet50":
        return resnet_preprocess
    if name in ("efficientnetb0", "efficientnetb3"):
        return efficientnet_preprocess
    return None


def _load_image_from_path(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Decode an image file path into a float32 RGB tensor in [0, 255].

    Args:
        path: String tensor file path.
        label: Scalar label tensor.

    Returns:
        Tuple (image, label).

    Raises:
        No exception — corrupt files return a black placeholder image.
    """
    try:
        file_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(file_bytes, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = image * 255.0
        image.set_shape([None, None, 3])
    except Exception:
        image = tf.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, 3], dtype=tf.float32)
    return image, label


def _apply_preprocess(
    image: tf.Tensor,
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor] | None,
) -> tf.Tensor:
    """
    Apply transfer-learning preprocess_input when configured.

    Args:
        image: Image tensor after augmentation (float32).
        preprocess_fn: e.g. resnet_preprocess, or None to skip.

    Returns:
        Preprocessed image tensor.
    """
    if preprocess_fn is None:
        return image
    return preprocess_fn(image)


def _make_one_hot(label: tf.Tensor, num_classes: int) -> tf.Tensor:
    """
    Convert a scalar class index to a one-hot vector.

    Args:
        label: Scalar integer label.
        num_classes: Number of classes.

    Returns:
        One-hot float32 tensor of shape (num_classes,).
    """
    return tf.one_hot(tf.cast(label, tf.int32), depth=num_classes)


def create_dataset_from_paths(
    image_paths: list[str],
    labels: list[int],
    augmentation_model: tf.keras.Sequential,
    batch_size: int = BATCH_SIZE,
    num_classes: int | None = None,
    shuffle: bool = False,
    shuffle_buffer: int | None = None,
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
    one_hot: bool = True,
    cache: bool = True,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset that loads images lazily from file paths.

    Args:
        image_paths: List of image file paths.
        labels: Integer labels.
        augmentation_model: Keras Sequential (training or val/test).
        batch_size: Batch size for training or evaluation.
        num_classes: Number of classes for one-hot encoding.
        shuffle: Whether to shuffle before batching (training).
        shuffle_buffer: Shuffle buffer size; defaults to min(len, 10000).
        preprocess_fn: Optional ResNet/EfficientNet preprocess_input.
        one_hot: If True, yield one-hot labels for categorical crossentropy.
        cache: If True, cache dataset in memory after first epoch.

    Returns:
        Batched and prefetched tf.data.Dataset of (image, label).

    Raises:
        ValueError: If paths and labels differ in length or num_classes is invalid.
    """
    if len(image_paths) != len(labels):
        raise ValueError("image_paths and labels must have the same length.")

    if num_classes is None:
        num_classes = get_num_classes()

    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def _process(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image, label = _load_image_from_path(path, label)
        image = augmentation_model(image, training=shuffle)
        image = _apply_preprocess(image, preprocess_fn)
        if one_hot:
            label = _make_one_hot(label, num_classes)
        return image, label

    dataset = path_ds.map(_process, num_parallel_calls=AUTOTUNE)

    if shuffle:
        buffer = shuffle_buffer or min(len(image_paths), 10000)
        dataset = dataset.shuffle(buffer, seed=RANDOM_SEED, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)
    if cache:
        dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def _resolve_num_classes(num_classes: int | None, labels: list[int]) -> int:
    """
    Resolve number of classes from config or label indices.

    Args:
        num_classes: Explicit class count, or None to infer.
        labels: Encoded label list.

    Returns:
        Number of classes.

    Raises:
        ValueError: If labels are empty and config has no class info.
    """
    if num_classes is not None:
        return num_classes
    try:
        return get_num_classes()
    except ValueError:
        if not labels:
            raise ValueError("Cannot infer num_classes from empty labels.") from None
        return int(max(labels)) + 1


def build_datasets_from_splits(
    train_data: tuple[list[str], list[int]],
    val_data: tuple[list[str], list[int]],
    test_data: tuple[list[str], list[int]],
    model_name: str = "cnn_baseline",
    batch_size: int = BATCH_SIZE,
    num_classes: int | None = None,
    image_size: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Build tf.data.Dataset objects from pre-split path/label tuples.

    Args:
        train_data: (paths, labels) for training.
        val_data: (paths, labels) for validation.
        test_data: (paths, labels) for test.
        model_name: Model name for augmentation/preprocess selection.
        batch_size: Batch size.
        num_classes: Number of classes for one-hot labels.
        image_size: (height, width) target size.

    Returns:
        Tuple (train_dataset, val_dataset, test_dataset).
    """
    all_labels = train_data[1] + val_data[1] + test_data[1]
    n_classes = _resolve_num_classes(num_classes, all_labels)

    preprocess_fn = _get_preprocess_fn(model_name)
    train_aug = get_augmentation_for_model(model_name, training=True, image_size=image_size)
    val_aug = get_augmentation_for_model(model_name, training=False, image_size=image_size)

    train_ds = create_dataset_from_paths(
        train_data[0],
        train_data[1],
        augmentation_model=train_aug,
        batch_size=batch_size,
        num_classes=n_classes,
        shuffle=True,
        preprocess_fn=preprocess_fn,
    )
    val_ds = create_dataset_from_paths(
        val_data[0],
        val_data[1],
        augmentation_model=val_aug,
        batch_size=batch_size,
        num_classes=n_classes,
        shuffle=False,
        preprocess_fn=preprocess_fn,
    )
    test_ds = create_dataset_from_paths(
        test_data[0],
        test_data[1],
        augmentation_model=val_aug,
        batch_size=batch_size,
        num_classes=n_classes,
        shuffle=False,
        preprocess_fn=preprocess_fn,
    )

    logger.info(
        "Built tf.data datasets for model '%s' (batch_size=%d, classes=%d)",
        model_name,
        batch_size,
        n_classes,
    )
    return train_ds, val_ds, test_ds


def build_datasets(
    image_paths: list[str],
    labels: list[int],
    model_name: str = "cnn_baseline",
    batch_size: int = BATCH_SIZE,
    num_classes: int | None = None,
    image_size: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Split data and return train, validation, and test tf.data.Dataset objects.

    Training set receives random augmentation; val and test receive resize only.
    Transfer-learning models get preprocess_input after augmentation.

    Args:
        image_paths: All image paths from DataLoader.
        labels: Integer-encoded labels.
        model_name: Model name for augmentation/preprocess selection.
        batch_size: Batch size.
        num_classes: Number of classes; inferred from config if None.
        image_size: (height, width) target size.

    Returns:
        Tuple (train_dataset, val_dataset, test_dataset).

    Raises:
        ValueError: If splitting or dataset creation fails.
    """
    train_data, val_data, test_data = stratified_train_val_test_split(image_paths, labels)
    return build_datasets_from_splits(
        train_data,
        val_data,
        test_data,
        model_name=model_name,
        batch_size=batch_size,
        num_classes=num_classes,
        image_size=image_size,
    )


def split_and_build_datasets(
    image_paths: list[str],
    labels: list[int],
    model_name: str = "cnn_baseline",
    batch_size: int = BATCH_SIZE,
) -> tuple[
    tuple[list[str], list[int]],
    tuple[list[str], list[int]],
    tuple[list[str], list[int]],
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
]:
    """
    Split paths/labels and build TensorFlow datasets in one call.

    Args:
        image_paths: Image file paths.
        labels: Encoded labels.
        model_name: Model name for preprocessing pipeline.
        batch_size: Batch size.

    Returns:
        Tuple containing:
            (X_train, y_train), (X_val, y_val), (X_test, y_test),
            train_ds, val_ds, test_ds
    """
    train_data, val_data, test_data = stratified_train_val_test_split(image_paths, labels)
    train_ds, val_ds, test_ds = build_datasets_from_splits(
        train_data,
        val_data,
        test_data,
        model_name=model_name,
        batch_size=batch_size,
    )
    return train_data, val_data, test_data, train_ds, val_ds, test_ds
