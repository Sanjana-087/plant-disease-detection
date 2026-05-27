"""
Helper functions for the Streamlit plant disease detection app.

Note: Named app_helpers.py (not utils.py) to avoid shadowing the project utils/ package
when Streamlit adds the app/ folder to sys.path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from training.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LABEL_MAPPING_PATH,
    SAVED_MODELS_PATH,
    TOP_K_PREDICTIONS,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def list_saved_models(models_dir: str | None = None) -> list[str]:
    """
    List available .keras model files in saved_models/.

    Args:
        models_dir: Directory to scan. Defaults to SAVED_MODELS_PATH.

    Returns:
        Sorted list of model file paths.
    """
    directory = Path(models_dir or SAVED_MODELS_PATH)
    if not directory.is_dir():
        return []
    models = sorted(directory.glob("*.keras"))
    return [str(p) for p in models]


def infer_model_type(model_path: str) -> str:
    """
    Guess preprocessing family from the model filename.

    Args:
        model_path: Path to .keras file.

    Returns:
        One of 'resnet50', 'efficientnet', 'cnn_baseline'.
    """
    name = Path(model_path).stem.lower()
    if "resnet" in name:
        return "resnet50"
    if "efficientnet" in name or "efficient" in name:
        if "b3" in name:
            return "efficientnetb3"
        return "efficientnetb0"
    return "cnn_baseline"


def load_class_names(mapping_path: str | None = None) -> list[str]:
    """
    Load class names from label_mapping.json or class_names.json.

    Args:
        mapping_path: Optional path to label_mapping.json.

    Returns:
        List of class name strings.

    Raises:
        FileNotFoundError: If no mapping file exists.
    """
    path = Path(mapping_path or LABEL_MAPPING_PATH)
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        names = data.get("class_names", [])
        if names:
            return list(names)

    alt = path.parent / "class_names.json"
    if alt.is_file():
        with open(alt, encoding="utf-8") as f:
            data = json.load(f)
        return list(data.get("class_names", []))

    raise FileNotFoundError(
        f"Class names not found. Train a model first or place label_mapping.json at {path}."
    )


def load_keras_model(model_path: str) -> keras.Model:
    """
    Load a Keras model from disk with error handling.

    Args:
        model_path: Path to .keras file.

    Returns:
        Loaded tf.keras.Model.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If loading fails.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        model = keras.models.load_model(model_path)
        logger.info("Loaded model: %s", model_path)
        return model
    except Exception as exc:
        raise ValueError(f"Failed to load model: {exc}") from exc


def preprocess_image(
    image: Image.Image,
    model_type: str,
    target_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize and preprocess a PIL image for model inference.

    Args:
        image: PIL Image (RGB).
        model_type: Preprocessing family from infer_model_type().
        target_size: (width, height).

    Returns:
        Tuple (preprocessed_batch, display_rgb_uint8) where batch is (1, H, W, 3).
    """
    image = image.convert("RGB")
    image = image.resize(target_size)
    display_rgb = np.array(image, dtype=np.uint8)
    img = display_rgb.astype(np.float32)

    batch = np.expand_dims(img, axis=0)
    model_type = model_type.lower()

    if model_type == "resnet50":
        batch = resnet_preprocess(batch)
    elif model_type in ("efficientnetb0", "efficientnetb3", "efficientnet"):
        batch = efficientnet_preprocess(batch)
    else:
        batch = batch / 255.0

    return batch, display_rgb


def predict_image(
    model: keras.Model,
    image_batch: np.ndarray,
    class_names: list[str],
    top_k: int = TOP_K_PREDICTIONS,
) -> dict:
    """
    Run inference and return top-k predictions with confidence scores.

    Args:
        model: Loaded Keras classifier.
        image_batch: Preprocessed batch (1, H, W, 3).
        class_names: Ordered class labels matching model outputs.
        top_k: Number of top predictions to return.

    Returns:
        Dict with predicted_class, confidence, class_index, and top_predictions list.

    Raises:
        ValueError: If prediction fails.
    """
    try:
        probabilities = model.predict(image_batch, verbose=0)[0]
    except Exception as exc:
        raise ValueError(f"Prediction failed: {exc}") from exc

    if len(probabilities) != len(class_names):
        logger.warning(
            "Class count mismatch: model outputs %d, class_names has %d.",
            len(probabilities),
            len(class_names),
        )

    top_indices = np.argsort(probabilities)[::-1][:top_k]
    top_predictions = [
        {
            "class_name": class_names[i] if i < len(class_names) else f"Class_{i}",
            "confidence": float(probabilities[i]),
            "class_index": int(i),
        }
        for i in top_indices
    ]

    best = top_predictions[0]
    return {
        "predicted_class": best["class_name"],
        "confidence": best["confidence"],
        "class_index": best["class_index"],
        "top_predictions": top_predictions,
        "probabilities": probabilities.tolist(),
    }


def is_healthy_class(class_name: str) -> bool:
    """
    Return True if the class name indicates a healthy plant.

    Args:
        class_name: Disease or health label string.

    Returns:
        True if 'healthy' appears in the name (case-insensitive).
    """
    return "healthy" in class_name.lower()


def format_class_name(class_name: str) -> str:
    """
    Convert PlantVillage-style names to readable labels.

    Args:
        class_name: Raw class string (e.g. Tomato___Early_blight).

    Returns:
        Human-readable formatted name.
    """
    return class_name.replace("___", " — ").replace("_", " ").strip()


# Static disease information for the UI (PlantVillage classes)
DISEASE_INFO: dict[str, dict[str, str]] = {
    "Apple___Apple_scab": {
        "description": "Fungal disease causing olive-green to dark brown velvety spots on leaves and fruit.",
        "symptoms": "Circular scabby lesions on leaves; leaves may yellow and drop early.",
        "treatment": "Apply fungicides (captan, myclobutanil); remove fallen leaves; plant resistant varieties.",
    },
    "Apple___Black_rot": {
        "description": "Fungal disease affecting fruit, leaves, and branches of apple trees.",
        "symptoms": "Brown expanding leaf spots; frog-eye lesions; fruit rot with concentric rings.",
        "treatment": "Prune infected wood; apply fungicides; improve orchard sanitation.",
    },
    "Apple___healthy": {
        "description": "No significant disease detected on the apple leaf sample.",
        "symptoms": "Uniform green color; no spots, lesions, or necrosis.",
        "treatment": "Continue regular monitoring, balanced fertilization, and proper irrigation.",
    },
    "Tomato___Early_blight": {
        "description": "Common fungal disease caused by Alternaria solani affecting tomatoes.",
        "symptoms": "Brown concentric rings (target spots) on lower leaves; yellowing around lesions.",
        "treatment": "Use chlorothalonil or mancozeb; rotate crops; remove infected debris.",
    },
    "Tomato___healthy": {
        "description": "Tomato leaf appears healthy with no major disease indicators.",
        "symptoms": "Bright green foliage without target spots or wilting.",
        "treatment": "Maintain spacing for airflow; avoid overhead watering on leaves.",
    },
    "Potato___Early_blight": {
        "description": "Fungal leaf blight that reduces potato tuber yield when severe.",
        "symptoms": "Dark brown angular leaf spots with yellow halos; lower leaves affected first.",
        "treatment": "Apply fungicide sprays; use certified seed; practice crop rotation.",
    },
    "Corn_(maize)___Common_rust_": {
        "description": "Fungal rust disease producing pustules on corn leaves.",
        "symptoms": "Small cinnamon-brown raised pustules on upper and lower leaf surfaces.",
        "treatment": "Plant resistant hybrids; fungicides if severe; timely field scouting.",
    },
    "Grape___Black_rot": {
        "description": "Serious fungal disease of grapes affecting leaves, shoots, and fruit.",
        "symptoms": "Tan leaf spots with dark borders; black shriveled fruit mummies.",
        "treatment": "Fungicide program from bud break; remove mummified berries and infected canes.",
    },
}


def get_disease_info(class_name: str) -> dict[str, str]:
    """
    Return disease information card content for a predicted class.

    Args:
        class_name: Predicted class name string.

    Returns:
        Dict with description, symptoms, and treatment keys.
    """
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]

    readable = format_class_name(class_name)
    healthy = is_healthy_class(class_name)
    return {
        "description": (
            f"Predicted class: {readable}. "
            + ("Leaf appears healthy." if healthy else "Disease indicators may be present.")
        ),
        "symptoms": "Refer to agricultural extension guides for this crop and disease combination.",
        "treatment": (
            "Continue monitoring and consult a local agronomist for treatment specific to your region."
        ),
    }
