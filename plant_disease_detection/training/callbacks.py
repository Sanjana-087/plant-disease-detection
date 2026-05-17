"""
Keras training callbacks for plant disease model training.

Provides ModelCheckpoint, EarlyStopping, learning-rate reduction, CSV logging,
and TensorBoard in a single factory function.
"""

import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from training.config import (
    EARLY_STOPPING_PATIENCE,
    MIN_LEARNING_RATE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    RESULTS_PATH,
    SAVED_MODELS_PATH,
    TENSORBOARD_PATH,
    get_model_checkpoint_path,
    get_training_log_csv_path,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def get_callbacks(
    model_name: str,
    results_path: str | None = None,
    saved_models_path: str | None = None,
    monitor: str = "val_accuracy",
    mode: str = "max",
) -> list[tf.keras.callbacks.Callback]:
    """
    Return standard training callbacks for a named model.

    Includes:
        - ModelCheckpoint: best val_accuracy -> saved_models/{model}_best.keras
        - EarlyStopping: patience=10, restore_best_weights=True
        - ReduceLROnPlateau: factor=0.5, patience=5, min_lr=1e-7
        - CSVLogger: results/{model_name}_training_log.csv
        - TensorBoard: results/tensorboard/{model_name}/

    Args:
        model_name: Canonical model name (e.g. 'resnet50').
        results_path: Directory for CSV logs and TensorBoard. Defaults to config.
        saved_models_path: Directory for checkpoint files. Defaults to config.
        monitor: Metric to monitor for checkpoint and early stopping.
        mode: 'max' for accuracy, 'min' for loss.

    Returns:
        List of Keras Callback instances.

    Raises:
        ValueError: If model_name is empty.
    """
    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")

    name = model_name.strip().lower()
    results_dir = Path(results_path or RESULTS_PATH)
    models_dir = Path(saved_models_path or SAVED_MODELS_PATH)
    tensorboard_dir = Path(TENSORBOARD_PATH) / name

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = get_model_checkpoint_path(name)
    csv_log_path = get_training_log_csv_path(name)

    callbacks: list[tf.keras.callbacks.Callback] = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),
        CSVLogger(
            filename=csv_log_path,
            separator=",",
            append=False,
        ),
        TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch",
        ),
    ]

    logger.info("Callbacks for '%s':", name)
    logger.info("  Checkpoint : %s", checkpoint_path)
    logger.info("  CSV log    : %s", csv_log_path)
    logger.info("  TensorBoard: %s", tensorboard_dir)

    return callbacks


def get_phase_callbacks(
    model_name: str,
    phase: int,
    results_path: str | None = None,
) -> list[tf.keras.callbacks.Callback]:
    """
    Return callbacks with phase-specific CSV and TensorBoard subfolders.

    Useful for two-phase transfer learning (frozen base, then fine-tune).

    Args:
        model_name: Model name.
        phase: Training phase number (1 or 2).
        results_path: Base results directory.

    Returns:
        List of Keras callbacks with phase suffix on log paths.
    """
    phase_name = f"{model_name.strip().lower()}_phase{phase}"
    base_callbacks = get_callbacks(phase_name, results_path=results_path)

    # Replace CSV and TensorBoard paths to include original model name + phase
    results_dir = Path(results_path or RESULTS_PATH)
    csv_path = os.path.join(results_dir, f"{model_name}_phase{phase}_training_log.csv")
    tb_dir = Path(TENSORBOARD_PATH) / model_name.strip().lower() / f"phase{phase}"
    tb_dir.mkdir(parents=True, exist_ok=True)

    for callback in base_callbacks:
        if isinstance(callback, CSVLogger):
            callback.filename = csv_path
        elif isinstance(callback, TensorBoard):
            callback.log_dir = str(tb_dir)

    return base_callbacks
