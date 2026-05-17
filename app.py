import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    "plant_disease_detection/saved_models/cnn_baseline_final.keras"
)

# Class names
class_names = [
    "Apple___healthy",
    "Tomato___Early_blight"
]

# Title
st.title("🌿 Plant Disease Detection")

st.write("Upload a plant leaf image to predict disease.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Show image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize image
    image = image.resize((224, 224))

    # Convert to array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.subheader("Prediction")

    st.success(
        f"{predicted_class} ({confidence:.2f}% confidence)"
    )