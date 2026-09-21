import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Steel Microstructure Classification",
    page_icon="🔬",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_my_model():
    return load_model("steel_microstructure_cnn (2).keras")

try:
    model = load_my_model()
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# Class Names
class_names = [
    "CPJ Alloy",
    "HR Alloy",
    "P92 Alloy"
]

# Sidebar
st.sidebar.title("🔬 Steel Classification")
st.sidebar.markdown("""
### Project Overview

This AI model classifies steel microstructure images into:

- CPJ Alloy
- HR Alloy
- P92 Alloy

### Supported Formats
- JPG
- JPEG
- PNG
- BMP

### Model
CNN Deep Learning Model
""")

# Main Title
st.title("🔬 Steel Microstructure Classification Dashboard")

st.markdown("""
This application uses a trained Convolutional Neural Network (CNN)
to classify steel microstructure images.

Upload an image below and get instant predictions.
""")

# Upload Image
uploaded_file = st.file_uploader(
    "📤 Upload Steel Microstructure Image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img = image.convert("RGB")
    img = img.resize((128, 128))

    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

    with col2:

        st.subheader("Prediction Result")

        st.success(f"Predicted Class: {predicted_class}")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.subheader("Class Description")

        descriptions = {
            "CPJ Alloy":
            "Complex Phase Steel Alloy used in automotive applications.",

            "HR Alloy":
            "Hot Rolled Steel Alloy with characteristic microstructure.",

            "P92 Alloy":
            "High-temperature creep-resistant alloy used in power plants."
        }

        st.info(descriptions[predicted_class])

# Footer
st.markdown("---")
st.markdown(
    """
    **Final Year Project**

    Steel Microstructure Classification using Deep Learning (CNN)

    Developed using Streamlit & TensorFlow
    """
)
