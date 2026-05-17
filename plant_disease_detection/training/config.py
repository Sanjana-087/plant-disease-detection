"""
Central configuration for the Plant Disease Detection System.

All paths are resolved relative to the project root. Import this module
from training scripts, preprocessing, evaluation, and the Streamlit app.
"""

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project root (plant_disease_detection/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# TensorFlow seed is set when TF is imported (see set_tensorflow_seed())
try:
    import tensorflow as tf

    tf.random.set_seed(RANDOM_SEED)
except ImportError:
    pass


def set_tensorflow_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set TensorFlow random seeds for reproducible training.

    Args:
        seed: Integer seed value. Defaults to RANDOM_SEED.
    """
    import tensorflow as tf

    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


# ---------------------------------------------------------------------------
# Image and training hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
IMAGE_HEIGHT, IMAGE_WIDTH = IMAGE_SIZE
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5

# Two-phase transfer learning
PHASE1_EPOCHS = 10  # Frozen base
PHASE2_EPOCHS = EPOCHS - PHASE1_EPOCHS  # Fine-tuning (remaining epochs)

# ---------------------------------------------------------------------------
# Dataset splits (must sum to 1.0)
# ---------------------------------------------------------------------------
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6, (
    "Train/val/test splits must sum to 1.0"
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATASET_PATH = str(DATASET_ROOT / "plantvillage")
PLANTVILLAGE_PATH = str(DATASET_ROOT / "plantvillage")
PLANTDOC_PATH = str(DATASET_ROOT / "plantdoc")
ARECANUT_PATH = str(DATASET_ROOT / "arecanut")

SAVED_MODELS_PATH = str(PROJECT_ROOT / "saved_models")
RESULTS_PATH = str(PROJECT_ROOT / "results")
TENSORBOARD_PATH = str(Path(RESULTS_PATH) / "tensorboard")

# Sub-paths for artifacts
LABEL_MAPPING_PATH = str(Path(RESULTS_PATH) / "label_mapping.json")
CLASS_NAMES_PATH = str(Path(RESULTS_PATH) / "class_names.json")

# Supported dataset folder names (under dataset/)
DATASET_NAMES = ("plantvillage", "plantdoc", "arecanut")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_NAMES = [
    "cnn_baseline",
    "resnet50",
    "efficientnetb0",
    "efficientnetb3",
]

DEFAULT_MODEL_NAME = "cnn_baseline"

# ---------------------------------------------------------------------------
# Class labels (populated after dataset scan via update_class_info)
# ---------------------------------------------------------------------------
NUM_CLASSES: int = 0
CLASS_NAMES: list[str] = []


def update_class_info(class_names: list[str]) -> None:
    """
    Update global NUM_CLASSES and CLASS_NAMES after scanning a dataset.

    Args:
        class_names: Sorted list of human-readable class / disease names.
    """
    global NUM_CLASSES, CLASS_NAMES
    CLASS_NAMES = list(class_names)
    NUM_CLASSES = len(CLASS_NAMES)


def get_num_classes() -> int:
    """
    Return the current number of classes.

    Returns:
        Number of disease classes. 0 if dataset has not been scanned yet.

    Raises:
        ValueError: If NUM_CLASSES is still 0 and no classes were configured.
    """
    if NUM_CLASSES <= 0:
        raise ValueError(
            "NUM_CLASSES is not set. Run DataLoader and update_class_info() "
            "or scan the dataset before training."
        )
    return NUM_CLASSES


# ---------------------------------------------------------------------------
# Data augmentation (training only; val/test use resize + normalize)
# ---------------------------------------------------------------------------
AUGMENTATION = {
    "random_flip": "horizontal",
    "rotation_range": 30,  # degrees, ±30
    "zoom_range": 0.2,  # ±20%
    "brightness_range": 0.2,
    "contrast_range": 0.2,
    "translation_height_factor": 0.1,
    "translation_width_factor": 0.1,
    "crop_height_factor": 0.9,
    "crop_width_factor": 0.9,
}

# Validation / test: no random augmentations
NORMALIZE_MEAN = [0.0, 0.0, 0.0]  # Model-specific preprocess in data pipeline
NORMALIZE_STD = [1.0, 1.0, 1.0]

# ---------------------------------------------------------------------------
# Training callbacks
# ---------------------------------------------------------------------------
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7

# ResNet50 / EfficientNet fine-tuning
RESNET_UNFREEZE_LAYERS = 30
EFFICIENTNET_UNFREEZE_LAYERS = 20

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD_DEFAULT = 0.5
TOP_K_PREDICTIONS = 3

# ---------------------------------------------------------------------------
# Cross-dataset class mapping (PlantVillage → PlantDoc overlapping classes)
# Keys: PlantVillage-style names; values: PlantDoc folder names.
# Extend this dict when adding more overlapping disease pairs.
# ---------------------------------------------------------------------------
CROSS_DATASET_CLASS_MAP: dict[str, str] = {
    "Apple___Apple_scab": "Apple Scab Leaf",
    "Apple___Black_rot": "Apple Rot Leaf",
    "Apple___healthy": "Apple Healthy Leaf",
    "Corn_(maize)___Common_rust_": "Corn Rust Leaf",
    "Corn_(maize)___healthy": "Corn Healthy Leaf",
    "Grape___Black_rot": "Grape Black Rot Leaf",
    "Grape___healthy": "Grape Healthy Leaf",
    "Potato___Early_blight": "Potato Early Blight Leaf",
    "Potato___healthy": "Potato Healthy Leaf",
    "Tomato___Early_blight": "Tomato Early Blight Leaf",
    "Tomato___healthy": "Tomato Healthy Leaf",
}

# ---------------------------------------------------------------------------
# File extensions accepted for images
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# tf.data performance
# ---------------------------------------------------------------------------
AUTOTUNE = -1  # tf.data.AUTOTUNE when TensorFlow is available


def ensure_directories() -> None:
    """
    Create results, saved_models, and dataset subfolders if missing.

    Returns:
        None
    """
    for path in (
        SAVED_MODELS_PATH,
        RESULTS_PATH,
        TENSORBOARD_PATH,
        DATASET_ROOT,
        PLANTVILLAGE_PATH,
        PLANTDOC_PATH,
        ARECANUT_PATH,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


def get_dataset_path(dataset_name: str) -> str:
    """
    Resolve the filesystem path for a named dataset.

    Args:
        dataset_name: One of 'plantvillage', 'plantdoc', 'arecanut'.

    Returns:
        Absolute path string to the dataset directory.

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    name = dataset_name.lower().strip()
    mapping = {
        "plantvillage": PLANTVILLAGE_PATH,
        "plantdoc": PLANTDOC_PATH,
        "arecanut": ARECANUT_PATH,
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {', '.join(DATASET_NAMES)}"
        )
    return mapping[name]


def get_model_checkpoint_path(model_name: str) -> str:
    """
    Path for the best checkpoint (val_accuracy) during training.

    Args:
        model_name: Registered model name (e.g. 'resnet50').

    Returns:
        Full path to .keras checkpoint file.
    """
    return os.path.join(SAVED_MODELS_PATH, f"{model_name}_best.keras")


def get_final_model_path(model_name: str) -> str:
    """
    Path for the final saved model after training completes.

    Args:
        model_name: Registered model name.

    Returns:
        Full path to final .keras model file.
    """
    return os.path.join(SAVED_MODELS_PATH, f"{model_name}_final.keras")


def get_training_history_path(model_name: str) -> str:
    """
    Path for JSON training history export.

    Args:
        model_name: Registered model name.

    Returns:
        Full path to history JSON file.
    """
    return os.path.join(RESULTS_PATH, f"{model_name}_history.json")


def get_training_log_csv_path(model_name: str) -> str:
    """
    Path for CSVLogger epoch metrics.

    Args:
        model_name: Registered model name.

    Returns:
        Full path to training log CSV.
    """
    return os.path.join(RESULTS_PATH, f"{model_name}_training_log.csv")


# Create essential directories on import
ensure_directories()
