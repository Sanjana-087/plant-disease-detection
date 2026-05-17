"""
Cross-dataset evaluation: train on PlantVillage, test on PlantDoc (mapped classes).

Measures domain shift when deploying a PlantVillage-trained model on PlantDoc images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.evaluate import load_model_from_path, predict_dataset
from preprocessing.augmentation import get_augmentation_for_model
from preprocessing.data_loader import DataLoader
from preprocessing.split_dataset import (
    _get_preprocess_fn,
    create_dataset_from_paths,
    stratified_train_val_test_split,
)
from training.config import (
    BATCH_SIZE,
    CROSS_DATASET_CLASS_MAP,
    PLANTDOC_PATH,
    PLANTVILLAGE_PATH,
    RESULTS_PATH,
    set_tensorflow_seed,
)
from utils.logger import get_logger, setup_root_logger
from utils.metrics import compute_accuracy, compute_precision_recall_f1

logger = get_logger(__name__)


def _plantdoc_to_plantvillage() -> dict[str, str]:
    """
    Reverse map: PlantDoc folder name -> PlantVillage class name.

    Returns:
        Dict mapping PlantDoc class strings to PlantVillage class strings.
    """
    return {plantdoc: plantvillage for plantvillage, plantdoc in CROSS_DATASET_CLASS_MAP.items()}


def _get_mapped_model_indices(class_to_idx: dict[str, int]) -> dict[str, int]:
    """
    PlantVillage class names in the cross-dataset map -> model output index.

    Args:
        class_to_idx: Mapping from training label JSON.

    Returns:
        Dict plantvillage_class_name -> model index.

    Raises:
        ValueError: If a mapped class is missing from the trained model.
    """
    mapped: dict[str, int] = {}
    missing: list[str] = []
    for pv_class in CROSS_DATASET_CLASS_MAP:
        if pv_class in class_to_idx:
            mapped[pv_class] = int(class_to_idx[pv_class])
        else:
            missing.append(pv_class)

    if missing:
        logger.warning(
            "Mapped PlantVillage classes not in model label map (skipped): %s",
            missing,
        )
    if not mapped:
        raise ValueError(
            "No overlapping classes found between CROSS_DATASET_CLASS_MAP and the model."
        )
    return mapped


def _filter_samples_for_mapped_classes(
    image_paths: list[str],
    labels: list[int],
    loader_class_names: list[str],
    pv_to_model_idx: dict[str, int],
    plantdoc_mode: bool = False,
) -> tuple[list[str], list[int]]:
    """
    Keep only samples whose class is in the cross-dataset mapping.

    Args:
        image_paths: Image file paths.
        labels: Integer labels from DataLoader (dataset-local encoding).
        loader_class_names: class_names from DataLoader.
        pv_to_model_idx: PlantVillage class -> model output index.
        plantdoc_mode: If True, map PlantDoc folder names via reverse mapping.

    Returns:
        Filtered (paths, model_indices) using the trained model's label indices.
    """
    doc_to_pv = _plantdoc_to_plantvillage()
    filtered_paths: list[str] = []
    filtered_model_labels: list[int] = []

    for path, label_idx in zip(image_paths, labels):
        local_name = loader_class_names[int(label_idx)]

        if plantdoc_mode:
            if local_name not in doc_to_pv:
                continue
            pv_name = doc_to_pv[local_name]
        else:
            pv_name = local_name
            if pv_name not in pv_to_model_idx:
                continue

        if pv_name not in pv_to_model_idx:
            continue

        filtered_paths.append(path)
        filtered_model_labels.append(pv_to_model_idx[pv_name])

    return filtered_paths, filtered_model_labels


def _evaluate_mapped_subset(
    model,
    image_paths: list[str],
    model_labels: list[int],
    model_name: str,
    batch_size: int,
    num_classes: int,
) -> dict[str, float]:
    """
    Build test dataset on mapped samples and return accuracy + F1.

    Args:
        model: Loaded Keras model.
        image_paths: Filtered image paths.
        model_labels: Labels as model output indices.
        model_name: Architecture name for preprocessing.
        batch_size: Batch size.
        num_classes: Full model class count (for one-hot).

    Returns:
        Dict with accuracy and f1_score.

    Raises:
        ValueError: If no samples remain after filtering.
    """
    if not image_paths:
        raise ValueError("No samples available for evaluation after class mapping.")

    test_aug = get_augmentation_for_model(model_name, training=False)
    dataset = create_dataset_from_paths(
        image_paths,
        model_labels,
        augmentation_model=test_aug,
        batch_size=batch_size,
        num_classes=num_classes,
        shuffle=False,
        preprocess_fn=_get_preprocess_fn(model_name),
    )

    y_true, y_pred = predict_dataset(model, dataset)
    accuracy = compute_accuracy(y_true, y_pred)
    f1_metrics = compute_precision_recall_f1(y_true, y_pred, average="weighted")

    return {
        "accuracy": accuracy,
        "f1_score": f1_metrics["f1_score"],
        "num_samples": len(y_true),
    }


def run_cross_dataset_evaluation(
    model_path: str,
    model_name: str = "resnet50",
    plantvillage_path: str | None = None,
    plantdoc_path: str | None = None,
    batch_size: int = BATCH_SIZE,
    label_mapping_path: str | None = None,
    results_path: str | None = None,
) -> dict:
    """
    Evaluate a PlantVillage-trained model on PlantVillage and PlantDoc test splits.

    Only classes present in CROSS_DATASET_CLASS_MAP are used. Computes domain shift
    as PlantVillage test accuracy minus PlantDoc accuracy (and F1).

    Args:
        model_path: Path to saved .keras model.
        model_name: Architecture for preprocessing (must match training).
        plantvillage_path: PlantVillage dataset root.
        plantdoc_path: PlantDoc dataset root.
        batch_size: Inference batch size.
        label_mapping_path: JSON from training with class_to_idx.
        results_path: Directory for cross_dataset_results.csv.

    Returns:
        Dict with plantvillage/plantdoc metrics and domain shift scores.

    Raises:
        FileNotFoundError: If model or datasets are missing.
        ValueError: If evaluation cannot proceed.
    """
    set_tensorflow_seed()
    pv_root = plantvillage_path or PLANTVILLAGE_PATH
    pd_root = plantdoc_path or PLANTDOC_PATH
    out_dir = Path(results_path or RESULTS_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_path(model_path)
    mapping = DataLoader.load_label_mapping(label_mapping_path)
    class_to_idx = mapping["class_to_idx"]
    num_classes = int(mapping.get("num_classes", len(class_to_idx)))

    pv_to_model_idx = _get_mapped_model_indices(class_to_idx)
    logger.info("Evaluating %d overlapping classes.", len(pv_to_model_idx))

    # PlantVillage test split (mapped classes only)
    pv_loader = DataLoader(pv_root, dataset_name="plantvillage", validate_images=True)
    pv_paths, pv_labels = pv_loader.load()
    _, _, pv_test = stratified_train_val_test_split(pv_paths, pv_labels)
    pv_test_paths, pv_test_model_labels = _filter_samples_for_mapped_classes(
        pv_test[0],
        pv_test[1],
        pv_loader.class_names,
        pv_to_model_idx,
        plantdoc_mode=False,
    )
    pv_metrics = _evaluate_mapped_subset(
        model,
        pv_test_paths,
        pv_test_model_labels,
        model_name,
        batch_size,
        num_classes,
    )

    # PlantDoc (mapped classes only)
    pd_loader = DataLoader(pd_root, dataset_name="plantdoc", validate_images=True)
    pd_paths, pd_labels = pd_loader.load()
    _, _, pd_test = stratified_train_val_test_split(pd_paths, pd_labels)
    pd_test_paths, pd_test_model_labels = _filter_samples_for_mapped_classes(
        pd_test[0],
        pd_test[1],
        pd_loader.class_names,
        pv_to_model_idx,
        plantdoc_mode=True,
    )
    pd_metrics = _evaluate_mapped_subset(
        model,
        pd_test_paths,
        pd_test_model_labels,
        model_name,
        batch_size,
        num_classes,
    )

    domain_shift_acc = pv_metrics["accuracy"] - pd_metrics["accuracy"]
    domain_shift_f1 = pv_metrics["f1_score"] - pd_metrics["f1_score"]

    results = {
        "plantvillage": pv_metrics,
        "plantdoc": pd_metrics,
        "domain_shift_accuracy": domain_shift_acc,
        "domain_shift_f1": domain_shift_f1,
        "num_mapped_classes": len(pv_to_model_idx),
    }

    _print_comparison_table(pv_metrics, pd_metrics, domain_shift_acc, domain_shift_f1)
    csv_path = _save_results_csv(results, out_dir)
    results["csv_path"] = str(csv_path)

    return results


def _print_comparison_table(
    pv_metrics: dict,
    pd_metrics: dict,
    domain_shift_acc: float,
    domain_shift_f1: float,
) -> None:
    """
    Print formatted comparison table to stdout.

    Args:
        pv_metrics: PlantVillage test metrics.
        pd_metrics: PlantDoc metrics.
        domain_shift_acc: Accuracy difference (PV - PlantDoc).
        domain_shift_f1: F1 difference (PV - PlantDoc).

    Returns:
        None
    """
    pv_acc = pv_metrics["accuracy"] * 100
    pd_acc = pd_metrics["accuracy"] * 100
    pv_f1 = pv_metrics["f1_score"] * 100
    pd_f1 = pd_metrics["f1_score"] * 100
    shift_acc = domain_shift_acc * 100
    shift_f1 = domain_shift_f1 * 100

    print("\n" + "=" * 50)
    print("CROSS-DATASET EVALUATION (Mapped Classes Only)")
    print("=" * 50)
    print(f"| {'Dataset':<14} | {'Accuracy':^10} | {'F1-Score':^10} |")
    print(f"|{'-' * 16}|{'-' * 12}|{'-' * 12}|")
    print(f"| {'PlantVillage':<14} | {pv_acc:>9.2f}% | {pv_f1:>9.2f}% |")
    print(f"| {'PlantDoc':<14} | {pd_acc:>9.2f}% | {pd_f1:>9.2f}% |")
    print(f"| {'Domain Shift':<14} | {shift_acc:>+9.2f}% | {shift_f1:>+9.2f}% |")
    print("=" * 50)
    print(f"PlantVillage test samples: {pv_metrics['num_samples']}")
    print(f"PlantDoc test samples    : {pd_metrics['num_samples']}")
    print("=" * 50 + "\n")


def _save_results_csv(results: dict, out_dir: Path) -> Path:
    """
    Save comparison table to results/cross_dataset_results.csv.

    Args:
        results: Results dict from run_cross_dataset_evaluation.
        out_dir: Output directory.

    Returns:
        Path to CSV file.
    """
    rows = [
        {
            "Dataset": "PlantVillage",
            "Accuracy": results["plantvillage"]["accuracy"],
            "F1-Score": results["plantvillage"]["f1_score"],
            "Num_Samples": results["plantvillage"]["num_samples"],
        },
        {
            "Dataset": "PlantDoc",
            "Accuracy": results["plantdoc"]["accuracy"],
            "F1-Score": results["plantdoc"]["f1_score"],
            "Num_Samples": results["plantdoc"]["num_samples"],
        },
        {
            "Dataset": "Domain Shift",
            "Accuracy": results["domain_shift_accuracy"],
            "F1-Score": results["domain_shift_f1"],
            "Num_Samples": "",
        },
    ]
    csv_path = out_dir / "cross_dataset_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Cross-dataset results saved to %s", csv_path)
    return csv_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-dataset evaluation (PlantVillage -> PlantDoc).",
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--plantvillage-path", type=str, default=None)
    parser.add_argument("--plantdoc-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, dest="batch_size")
    parser.add_argument("--label-mapping", type=str, default=None, dest="label_mapping")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for cross-dataset evaluation.

    Returns:
        Exit code 0 on success, 1 on failure.
    """
    setup_root_logger()
    args = _parse_args(argv)
    try:
        run_cross_dataset_evaluation(
            model_path=args.model_path,
            model_name=args.model_name,
            plantvillage_path=args.plantvillage_path,
            plantdoc_path=args.plantdoc_path,
            batch_size=args.batch_size,
            label_mapping_path=args.label_mapping,
        )
        return 0
    except Exception as exc:
        logger.exception("Cross-dataset evaluation failed: %s", exc)
        print(f"Cross-dataset evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
