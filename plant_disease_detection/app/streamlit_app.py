"""
Streamlit web application for Plant Disease Detection.

Run from project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Fix imports before other project modules: Streamlit adds app/ to sys.path,
# which makes app/utils.py shadow the project utils/ package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
_root = str(PROJECT_ROOT)
if sys.path[0] != _root:
    if _root in sys.path:
        sys.path.remove(_root)
    sys.path.insert(0, _root)
_app = str(APP_DIR)
while _app in sys.path:
    sys.path.remove(_app)

import streamlit as st
from PIL import Image

from app.app_helpers import (
    format_class_name,
    get_disease_info,
    infer_model_type,
    is_healthy_class,
    list_saved_models,
    load_class_names,
    load_keras_model,
    predict_image,
    preprocess_image,
)
from evaluation.grad_cam import generate_grad_cam
from training.config import CONFIDENCE_THRESHOLD_DEFAULT

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def cached_load_model(model_path: str):
    """
    Load and cache a Keras model for the Streamlit session.

    Args:
        model_path: Path to .keras file.

    Returns:
        Loaded tf.keras.Model.
    """
    return load_keras_model(model_path)


@st.cache_data(show_spinner=False)
def cached_class_names():
    """
    Load and cache class names from training artifacts.

    Returns:
        List of class name strings.
    """
    return load_class_names()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


def _add_to_history(record: dict) -> None:
    """Append a prediction record; keep only the last 10."""
    st.session_state.prediction_history.insert(0, record)
    st.session_state.prediction_history = st.session_state.prediction_history[:10]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🌿 Plant Disease AI")
    st.markdown("---")

    available_models = list_saved_models()
    if not available_models:
        st.warning(
            "No trained models found in `saved_models/`.\n\n"
            "Train a model first:\n"
            "`python -m training.train --model cnn_baseline --dataset plantvillage`"
        )
        selected_model_path = None
    else:
        model_labels = [Path(p).name for p in available_models]
        default_idx = 0
        choice = st.selectbox("Select model", model_labels, index=default_idx)
        selected_model_path = available_models[model_labels.index(choice)]

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.5,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD_DEFAULT,
        step=0.05,
        help="Predictions below this confidence are flagged as uncertain.",
    )

    show_gradcam = st.toggle("Grad-CAM visualization", value=False)

    st.markdown("---")
    st.caption("Research-grade plant leaf disease classifier")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown("# 🌿 Plant Disease Detection System")
st.markdown(
    "*Upload a leaf image to detect diseases using deep learning "
    "(CNN, ResNet50, EfficientNet).*"
)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP",
)

col_left, col_right = st.columns(2)

with col_left:
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption="Uploaded leaf image", use_container_width=True)
        except Exception as exc:
            st.error(f"Could not open image: {exc}")
            pil_image = None
    else:
        st.info("👆 Upload an image to begin.")
        pil_image = None

detect_clicked = st.button("🔬 Detect Disease", type="primary", disabled=pil_image is None)

with col_right:
    if detect_clicked and pil_image is not None:
        if selected_model_path is None:
            st.error("Please train and save a model before running detection.")
        else:
            try:
                model = cached_load_model(selected_model_path)
                class_names = cached_class_names()
                model_type = infer_model_type(selected_model_path)

                batch, display_rgb = preprocess_image(pil_image, model_type)
                result = predict_image(model, batch, class_names)

                predicted = result["predicted_class"]
                confidence = result["confidence"]
                healthy = is_healthy_class(predicted)
                above_threshold = confidence >= confidence_threshold

                # Result styling
                color = "#27ae60" if healthy else "#e74c3c"
                st.markdown(
                    f"<h2 style='color:{color};'>{format_class_name(predicted)}</h2>",
                    unsafe_allow_html=True,
                )

                if not above_threshold:
                    st.warning(
                        f"Confidence {confidence:.1%} is below threshold "
                        f"{confidence_threshold:.0%}. Result may be uncertain."
                    )

                st.progress(min(confidence, 1.0), text=f"Confidence: {confidence:.1%}")

                st.subheader("Top 3 predictions")
                for rank, pred in enumerate(result["top_predictions"], start=1):
                    st.write(
                        f"**{rank}.** {format_class_name(pred['class_name'])} "
                        f"— {pred['confidence']:.1%}"
                    )

                info = get_disease_info(predicted)
                with st.expander("📋 Disease information", expanded=True):
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown(f"**Symptoms:** {info['symptoms']}")
                    st.markdown(f"**Treatment:** {info['treatment']}")

                gradcam_image = None
                if show_gradcam:
                    try:
                        with st.spinner("Generating Grad-CAM..."):
                            gradcam_image = generate_grad_cam(
                                model,
                                batch,
                                class_index=result["class_index"],
                                display_image=display_rgb,
                            )
                        st.image(
                            gradcam_image,
                            caption="Grad-CAM — regions influencing the prediction",
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.warning(f"Grad-CAM could not be generated: {exc}")

                _add_to_history(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "class": format_class_name(predicted),
                        "confidence": f"{confidence:.1%}",
                        "model": Path(selected_model_path).name,
                    }
                )

            except FileNotFoundError as exc:
                st.error(f"Missing file: {exc}")
            except Exception as exc:
                st.error(f"Detection failed: {exc}")

    elif pil_image is not None and not detect_clicked:
        st.markdown("### Ready to analyze")
        st.write("Click **Detect Disease** to run the model on your uploaded image.")

# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📜 Prediction history (last 10)")

if st.session_state.prediction_history:
    st.table(
        {
            "Time": [r["timestamp"] for r in st.session_state.prediction_history],
            "Prediction": [r["class"] for r in st.session_state.prediction_history],
            "Confidence": [r["confidence"] for r in st.session_state.prediction_history],
            "Model": [r["model"] for r in st.session_state.prediction_history],
        }
    )
else:
    st.caption("No predictions yet.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This tool is for educational and research purposes only. "
    "It is not a substitute for professional agricultural diagnosis. "
    "Always verify results with a qualified agronomist before treatment decisions."
)
