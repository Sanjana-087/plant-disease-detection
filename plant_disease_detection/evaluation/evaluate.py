"""
Full model evaluation on a held-out test set.

Loads a saved .keras model, runs predictions, prints metrics, saves plots/CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from preprocessing.data_loader import DataLoader
from preprocessing.augmentation import get_augmentation_for_model
from preprocessing.split_dataset import (
    _get_preprocess_fn,
    create_dataset_from_paths,
    stratified_train_val_test_split,
)
from training.config import (
    BATCH_SIZE,
    RESULTS_PATH,
    get_dataset_path,
    set_tensorflow_seed,
)
from utils.logger import get_logger, setup_root_logger
from utils.metrics import compute_all_metrics, predictions_to_labels
from utils.visualization import plot_confusion_matrix

logger = get_logger(__name__)


def load_model_from_path(model_path: str) -> keras.Model:
    """
    Load a saved Keras model with error handling.

    Args:
        model_path: Path to .keras or SavedModel directory.

    Returns:
        Loaded tf.keras.Model.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If loading fails.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = keras.models.load_model(str(path))
        logger.info("Loaded model from %s", model_path)
        return model
    except Exception as exc:
        raise ValueError(f"Failed to load model from '{model_path}': {exc}") from exc


def predict_dataset(
    model: keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a batched tf.data.Dataset and collect true/predicted labels.

    Args:
        model: Trained Keras classifier.
        dataset: Batched dataset yielding (images, labels).

    Returns:
        Tuple (y_true, y_pred) as integer numpy arrays.
    """
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []

    for batch_images, batch_labels in dataset:
        try:
            probs = model.predict(batch_images, verbose=0)
        except Exception as exc:
            logger.warning("Predict failed on batch (%s); skipping batch.", exc)
            continue

        y_pred_list.append(predictions_to_labels(probs))
        labels_np = batch_labels.numpy()
        if labels_np.ndim > 1 and labels_np.shape[-1] > 1:
            y_true_list.append(np.argmax(labels_np, axis=1))
        else:
            y_true_list.append(labels_np.astype(int).ravel())

    if not y_true_list:
        raise ValueError("No predictions generated — test dataset may be empty.")

    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


def _print_metrics(metrics: dict, class_names: list[str]) -> None:
    """
    Print formatted evaluation metrics to stdout.

    Args:
        metrics: Output of compute_all_metrics().
        class_names: List of class names.

    Returns:
        None
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy          : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted)   : {metrics['recall_weighted']:.4f}")
    print(f"F1 (weighted)       : {metrics['f1_weighted']:.4f}")
    print(f"F1 (macro)          : {metrics['f1_macro']:.4f}")
    print("\n--- Per-class metrics ---")
    for row in metrics["per_class_metrics"]:
        print(
            f"  {row['class_name']:<40} "
            f"P={row['precision']:.3f} R={row['recall']:.3f} "
            f"F1={row['f1_score']:.3f} support={row['support']}"
        )
    print("\n--- Classification report ---")
    print(metrics["classification_report"])
    print("=" * 60 + "\n")


def save_per_class_metrics_csv(
    per_class_metrics: list[dict],
    model_name: str,
    results_path: str | None = None,
) -> str:
    """
    Save per-class metrics to CSV.

    Args:
        per_class_metrics: List of per-class metric dicts.
        model_name: Model name for filename.
        results_path: Output directory (default: config RESULTS_PATH).

    Returns:
        Path to saved CSV file.
    """
    out_dir = Path(results_path or RESULTS_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{model_name}_metrics.csv"
    pd.DataFrame(per_class_metrics).to_csv(csv_path, index=False)
    logger.info("Per-class metrics saved to %s", csv_path)
    return str(csv_path)


def evaluate_model(
    model_path: str,
    dataset_path: str,
    model_name: str = "model",
    batch_size: int = BATCH_SIZE,
    dataset_type: str = "auto",
    validate_images: bool = True,
    results_path: str | None = None,
) -> dict:
    """
    Evaluate a saved model on the test split of a dataset.

    Args:
        model_path: Path to .keras model file.
        dataset_path: Root directory of images (class subfolders).
        model_name: Name used for output filenames and plots.
        batch_size: Inference batch size.
        dataset_type: DataLoader layout hint (auto, plantvillage, etc.).
        validate_images: Validate images when scanning dataset.
        results_path: Directory for PNG/CSV/JSON outputs.

    Returns:
        Dictionary of all metrics from compute_all_metrics(), plus paths to artifacts.

    Raises:
        FileNotFoundError: If model or dataset is missing.
        ValueError: If evaluation cannot complete.
    """
    set_tensorflow_seed()
    results_dir = Path(results_path or RESULTS_PATH)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_path(model_path)

    loader = DataLoader(
        root_dir=dataset_path,
        dataset_name=dataset_type,
        validate_images=validate_images,
    )
    image_paths, labels = loader.load()
    class_names = loader.class_names
    num_classes = len(class_names)

    _, _, test_data = stratified_train_val_test_split(image_paths, labels)

    test_aug = get_augmentation_for_model(model_name, training=False)
    test_ds = create_dataset_from_paths(
        test_data[0],
        test_data[1],
        augmentation_model=test_aug,
        batch_size=batch_size,
        num_classes=num_classes,
        shuffle=False,
        preprocess_fn=_get_preprocess_fn(model_name),
    )

    y_true, y_pred = predict_dataset(model, test_ds)
    metrics = compute_all_metrics(y_true, y_pred, class_names, normalize_cm=True)

    _print_metrics(metrics, class_names)

    cm_path = results_dir / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(
        metrics["confusion_matrix_normalized"],
        class_names,
        model_name,
        str(cm_path),
        normalized=True,
    )

    csv_path = save_per_class_metrics_csv(
        metrics["per_class_metrics"],
        model_name,
        results_path=str(results_dir),
    )

    summary_path = results_dir / f"{model_name}_eval_summary.json"
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "dataset_path": str(Path(dataset_path).resolve()),
        "accuracy": metrics["accuracy"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "f1_macro": metrics["f1_macro"],
        "num_test_samples": int(len(y_true)),
        "confusion_matrix_plot": str(cm_path),
        "metrics_csv": csv_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    metrics["artifacts"] = {
        "confusion_matrix_plot": str(cm_path),
        "metrics_csv": csv_path,
        "summary_json": str(summary_path),
    }
    logger.info("Evaluation complete for '%s'", model_name)
    return metrics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for evaluate.py."""
    parser = argparse.ArgumentParser(description="Evaluate a plant disease model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .keras model.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset root path or name (plantvillage, plantdoc, arecanut).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="model",
        help="Name for output files (default: model).",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, dest="batch_size")
    parser.add_argument("--dataset-type", type=str, default="auto", dest="dataset_type")
    parser.add_argument("--no-validate-images", action="store_true")
    return parser.parse_args(argv)


def _resolve_dataset_path(dataset_arg: str) -> str:
    """Resolve dataset CLI argument to a directory path."""
    path = Path(dataset_arg)
    if path.is_dir():
        return str(path.resolve())
    return get_dataset_path(dataset_arg)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for model evaluation.

    Returns:
        Exit code 0 on success, 1 on failure.
    """
    setup_root_logger()
    args = _parse_args(argv)
    try:
        dataset_path = _resolve_dataset_path(args.dataset)
        evaluate_model(
            model_path=args.model_path,
            dataset_path=dataset_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
            dataset_type=args.dataset_type,
            validate_images=not args.no_validate_images,
        )
        return 0
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
