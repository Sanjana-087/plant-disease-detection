"""
Plotting utilities for training curves, confusion matrices, and comparisons.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.logger import get_logger

logger = get_logger(__name__)

PLOT_DPI = 200


def _ensure_parent_dir(save_path: str) -> None:
    """Create parent directory for save_path if needed."""
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_training_history(
    history_dict: dict[str, list[float]],
    model_name: str,
    save_path: str,
) -> str:
    """
    Plot training/validation accuracy and loss curves side by side.

    Args:
        history_dict: Dict with keys like accuracy, val_accuracy, loss, val_loss.
        model_name: Model name for plot title.
        save_path: Output PNG path.

    Returns:
        Absolute path to saved figure.
    """
    _ensure_parent_dir(save_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy subplot
    if "accuracy" in history_dict:
        axes[0].plot(history_dict["accuracy"], label="Train", linewidth=2)
    if "val_accuracy" in history_dict:
        axes[0].plot(history_dict["val_accuracy"], label="Validation", linewidth=2)
    axes[0].set_title(f"{model_name} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss subplot
    if "loss" in history_dict:
        axes[1].plot(history_dict["loss"], label="Train", linewidth=2)
    if "val_loss" in history_dict:
        axes[1].plot(history_dict["val_loss"], label="Validation", linewidth=2)
    axes[1].set_title(f"{model_name} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info("Training history plot saved to %s", save_path)
    return str(Path(save_path).resolve())


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    model_name: str,
    save_path: str,
    normalized: bool = False,
) -> str:
    """
    Plot an annotated confusion matrix heatmap.

    Args:
        cm: 2D confusion matrix array.
        class_names: Class labels for axes.
        model_name: Title prefix.
        save_path: Output PNG path.
        normalized: If True, title indicates normalized matrix.

    Returns:
        Absolute path to saved figure.
    """
    _ensure_parent_dir(save_path)

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.4), 8))
    fmt = ".2f" if normalized else "d"
    title_suffix = " (Normalized)" if normalized else ""

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalized else "Count"},
    )
    ax.set_title(f"{model_name} — Confusion Matrix{title_suffix}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info("Confusion matrix plot saved to %s", save_path)
    return str(Path(save_path).resolve())


def plot_class_distribution(
    label_counts: dict[str, int],
    save_path: str,
    title: str = "Class Distribution",
) -> str:
    """
    Plot a bar chart of samples per class.

    Args:
        label_counts: Mapping class_name -> image count.
        save_path: Output PNG path.
        title: Chart title.

    Returns:
        Absolute path to saved figure.
    """
    _ensure_parent_dir(save_path)

    classes = sorted(label_counts.keys())
    counts = [label_counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.35), 6))
    bars = ax.bar(range(len(classes)), counts, color="seagreen", edgecolor="black", alpha=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Number of Images")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info("Class distribution plot saved to %s", save_path)
    return str(Path(save_path).resolve())


def plot_model_comparison(
    results_dict: dict[str, dict[str, float]],
    save_path: str,
    metrics: tuple[str, str] = ("accuracy", "f1_score"),
) -> str:
    """
    Plot grouped bar chart comparing multiple models on accuracy and F1.

    Args:
        results_dict: {model_name: {metric_name: value, ...}, ...}.
            Values may be in [0,1] or [0,100]; auto-scaled if max <= 1.
        save_path: Output PNG path.
        metrics: Tuple of two metric keys to compare.

    Returns:
        Absolute path to saved figure.
    """
    _ensure_parent_dir(save_path)

    model_names = list(results_dict.keys())
    metric_a, metric_b = metrics

    values_a = [results_dict[m].get(metric_a, 0.0) for m in model_names]
    values_b = [results_dict[m].get(metric_b, 0.0) for m in model_names]

    # Scale to percentage if values look like fractions
    if max(values_a + values_b, default=0) <= 1.0:
        values_a = [v * 100 for v in values_a]
        values_b = [v * 100 for v in values_b]
        ylabel = "Score (%)"
    else:
        ylabel = "Score"

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, values_a, width, label=metric_a.replace("_", " ").title(), color="#2ecc71")
    ax.bar(x + width / 2, values_b, width, label=metric_b.replace("_", " ").title(), color="#3498db")

    ax.set_ylabel(ylabel)
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info("Model comparison plot saved to %s", save_path)
    return str(Path(save_path).resolve())
