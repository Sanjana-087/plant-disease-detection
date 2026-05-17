"""
Reusable metric computation functions for model evaluation.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from utils.logger import get_logger

logger = get_logger(__name__)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        Accuracy in [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def compute_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
    labels: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute precision, recall, and F1 for the given averaging mode.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        average: sklearn average mode ('weighted', 'macro', 'micro', etc.).
        labels: Optional list of label indices to include.

    Returns:
        Dict with keys precision, recall, f1_score.
    """
    kwargs = {"average": average, "zero_division": 0}
    if labels is not None:
        kwargs["labels"] = labels

    return {
        "precision": float(precision_score(y_true, y_pred, **kwargs)),
        "recall": float(recall_score(y_true, y_pred, **kwargs)),
        "f1_score": float(f1_score(y_true, y_pred, **kwargs)),
    }


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> list[dict[str, float | str]]:
    """
    Compute precision, recall, F1, and support for each class.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Human-readable class names (index-aligned).

    Returns:
        List of dicts, one per class, with name and metric fields.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class: list[dict[str, float | str]] = []
    for name in class_names:
        if name not in report:
            continue
        stats = report[name]
        per_class.append(
            {
                "class_name": name,
                "precision": float(stats["precision"]),
                "recall": float(stats["recall"]),
                "f1_score": float(stats["f1-score"]),
                "support": int(stats["support"]),
            }
        )
    return per_class


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = False,
    labels: list[int] | None = None,
) -> np.ndarray:
    """
    Compute confusion matrix (optionally row-normalized).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        normalize: If True, normalize by true class counts (rows sum to 1).
        labels: Optional ordered label list.

    Returns:
        2D numpy array confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm = cm.astype(np.float64) / row_sums
    return cm


def get_classification_report_string(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> str:
    """
    Return sklearn classification report as a formatted string.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Target class names.

    Returns:
        Multi-line classification report string.
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    normalize_cm: bool = True,
) -> dict:
    """
    Compute full evaluation metrics bundle for programmatic use.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: List of class names.
        normalize_cm: Whether to include normalized confusion matrix.

    Returns:
        Dictionary with accuracy, weighted/macro F1, confusion matrices,
        per-class metrics, and classification report text.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    weighted = compute_precision_recall_f1(y_true, y_pred, average="weighted")
    macro = compute_precision_recall_f1(y_true, y_pred, average="macro")

    metrics = {
        "accuracy": compute_accuracy(y_true, y_pred),
        "precision_weighted": weighted["precision"],
        "recall_weighted": weighted["recall"],
        "f1_weighted": weighted["f1_score"],
        "precision_macro": macro["precision"],
        "recall_macro": macro["recall"],
        "f1_macro": macro["f1_score"],
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred, normalize=False),
        "confusion_matrix_normalized": compute_confusion_matrix(
            y_true, y_pred, normalize=normalize_cm
        ),
        "per_class_metrics": compute_per_class_metrics(y_true, y_pred, class_names),
        "classification_report": get_classification_report_string(
            y_true, y_pred, class_names
        ),
    }

    logger.info(
        "Metrics — accuracy: %.4f, F1 (weighted): %.4f",
        metrics["accuracy"],
        metrics["f1_weighted"],
    )
    return metrics


def predictions_to_labels(y_prob: np.ndarray) -> np.ndarray:
    """
    Convert model probability outputs to class index predictions.

    Args:
        y_prob: Array of shape (n_samples, n_classes) or (n_samples,).

    Returns:
        Integer label array of shape (n_samples,).
    """
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        return y_prob.astype(int)
    return np.argmax(y_prob, axis=1)
