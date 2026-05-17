"""
Main training script for plant disease detection models.

Supports CLI usage and importable train_model() for notebooks.
Transfer-learning models use two-phase training: frozen backbone, then fine-tune.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from models.efficientnet_model import unfreeze_efficientnet_top_layers
from models.model_factory import get_model, is_transfer_learning_model
from models.resnet50_model import unfreeze_resnet50_top_layers
from preprocessing.data_loader import DataLoader
from preprocessing.split_dataset import build_datasets
from training.callbacks import get_callbacks, get_phase_callbacks
from training.config import (
    BATCH_SIZE,
    EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    LEARNING_RATE,
    PHASE1_EPOCHS,
    get_dataset_path,
    get_final_model_path,
    get_training_history_path,
    set_tensorflow_seed,
)
from utils.logger import get_logger, log_system_info, setup_root_logger

logger = get_logger(__name__)


def _merge_histories(
    history1: keras.callbacks.History,
    history2: keras.callbacks.History | None = None,
) -> dict[str, list[float]]:
    """
    Merge one or two Keras History objects into a single serializable dict.

    Args:
        history1: History from phase 1 or single-phase training.
        history2: Optional history from phase 2.

    Returns:
        Dict mapping metric name to list of float values.
    """
    merged: dict[str, list[float]] = {}
    for hist in (history1, history2):
        if hist is None:
            continue
        for key, values in hist.history.items():
            merged.setdefault(key, [])
            merged[key].extend(float(v) for v in values)
    return merged


def _save_history(history_dict: dict[str, list[float]], model_name: str) -> str:
    """
    Save training history to JSON.

    Args:
        history_dict: Merged metrics dictionary.
        model_name: Model name for filename.

    Returns:
        Path to saved JSON file.
    """
    path = get_training_history_path(model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    logger.info("Training history saved to %s", path)
    return path


def _unfreeze_model_for_phase2(model: keras.Model, model_name: str) -> None:
    """
    Unfreeze top backbone layers for transfer-learning fine-tuning.

    Args:
        model: Compiled Keras model.
        model_name: Canonical model name.

    Returns:
        None
    """
    name = model_name.lower()
    if name == "resnet50":
        unfreeze_resnet50_top_layers(model)
    elif name in ("efficientnetb0", "efficientnetb3"):
        unfreeze_efficientnet_top_layers(model)
    else:
        raise ValueError(f"Cannot unfreeze backbone for model '{model_name}'.")


def train_model(
    model_name: str,
    dataset_path: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    fine_tune_lr: float = FINE_TUNE_LEARNING_RATE,
    phase1_epochs: int = PHASE1_EPOCHS,
    dataset_type: str = "auto",
    validate_images: bool = True,
    print_summary: bool = True,
) -> keras.Model:
    """
    Train a plant disease model end-to-end.

    Pipeline: DataLoader -> stratified split -> tf.data -> model -> fit.
    Transfer-learning models run phase 1 (frozen) then phase 2 (fine-tune).

    Args:
        model_name: One of cnn_baseline, resnet50, efficientnetb0, efficientnetb3.
        dataset_path: Root directory of the image dataset.
        epochs: Total training epochs (split across phases for TL models).
        batch_size: Training batch size.
        learning_rate: Adam LR for phase 1 / single-phase training.
        fine_tune_lr: Adam LR for phase 2 fine-tuning.
        phase1_epochs: Epochs with frozen backbone before unfreezing.
        dataset_type: Passed to DataLoader ('auto', 'plantvillage', etc.).
        validate_images: Skip corrupt images when True.
        print_summary: Print Keras model summary on creation.

    Returns:
        Trained Keras model (weights from final fit).

    Raises:
        FileNotFoundError: If dataset_path is missing.
        ValueError: If no images are found or epochs < 1.
    """
    set_tensorflow_seed()

    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    dataset_path = str(Path(dataset_path).resolve())
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    logger.info("Starting training: model=%s, dataset=%s", model_name, dataset_path)

    # Load paths and encoded labels
    loader = DataLoader(
        root_dir=dataset_path,
        dataset_name=dataset_type,
        validate_images=validate_images,
    )
    image_paths, labels = loader.load()
    num_classes = len(loader.class_names)

    # Build tf.data pipelines with model-specific preprocessing
    train_ds, val_ds, _test_ds = build_datasets(
        image_paths,
        labels,
        model_name=model_name,
        batch_size=batch_size,
        num_classes=num_classes,
    )

    # Create and compile model
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        learning_rate=learning_rate,
        print_summary=print_summary,
    )

    canonical = model_name.lower().strip()
    use_two_phase = is_transfer_learning_model(canonical) and epochs > phase1_epochs
    phase2_epochs = max(0, epochs - phase1_epochs)

    if use_two_phase:
        print(f"\n--- Phase 1: frozen backbone ({phase1_epochs} epochs) ---\n")
        callbacks_p1 = get_phase_callbacks(canonical, phase=1)
        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=phase1_epochs,
            callbacks=callbacks_p1,
            verbose=1,
        )

        print(f"\n--- Phase 2: fine-tuning top layers ({phase2_epochs} epochs) ---\n")
        _unfreeze_model_for_phase2(model, canonical)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks_p2 = get_phase_callbacks(canonical, phase=2)
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=phase2_epochs,
            callbacks=callbacks_p2,
            verbose=1,
        )
        history_dict = _merge_histories(history1, history2)
    else:
        if is_transfer_learning_model(canonical) and epochs <= phase1_epochs:
            logger.warning(
                "epochs <= phase1_epochs: training transfer model with frozen backbone only."
            )
        callbacks = get_callbacks(canonical)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        history_dict = _merge_histories(history)

    # Persist artifacts
    _save_history(history_dict, canonical)
    final_path = get_final_model_path(canonical)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    model.save(final_path)

    print("Training complete. Model saved.")
    logger.info("Final model saved to %s", final_path)
    return model


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a plant disease detection model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name: cnn_baseline, resnet50, efficientnetb0, efficientnetb3.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset root OR name: plantvillage, plantdoc, arecanut.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Total training epochs (default {EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        dest="batch_size",
        help=f"Batch size (default {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        dest="learning_rate",
        help=f"Learning rate (default {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=FINE_TUNE_LEARNING_RATE,
        dest="fine_tune_lr",
        help=f"Phase 2 learning rate (default {FINE_TUNE_LEARNING_RATE}).",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=PHASE1_EPOCHS,
        dest="phase1_epochs",
        help=f"Frozen-backbone epochs for transfer models (default {PHASE1_EPOCHS}).",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="auto",
        dest="dataset_type",
        help="Dataset layout: auto, plantvillage, plantdoc, arecanut.",
    )
    parser.add_argument(
        "--no-validate-images",
        action="store_true",
        help="Skip PIL validation when scanning images (faster).",
    )
    return parser.parse_args(argv)


def _resolve_dataset_path(dataset_arg: str) -> str:
    """
    Resolve CLI --dataset to a filesystem path.

    Args:
        dataset_arg: Path string or dataset name (plantvillage, etc.).

    Returns:
        Absolute dataset directory path.

    Raises:
        FileNotFoundError: If path does not exist and name is not a known dataset.
    """
    path = Path(dataset_arg)
    if path.is_dir():
        return str(path.resolve())

    try:
        return get_dataset_path(dataset_arg)
    except ValueError as exc:
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_arg}' and is not a known dataset name."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for training.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Exit code 0 on success, 1 on failure.
    """
    setup_root_logger()
    log_system_info(logger)

    args = _parse_args(argv)
    try:
        dataset_path = _resolve_dataset_path(args.dataset)
        train_model(
            model_name=args.model,
            dataset_path=dataset_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            fine_tune_lr=args.fine_tune_lr,
            phase1_epochs=args.phase1_epochs,
            dataset_type=args.dataset_type,
            validate_images=not args.no_validate_images,
        )
        return 0
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
