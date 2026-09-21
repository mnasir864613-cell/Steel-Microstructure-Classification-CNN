import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Steel Microstructure Classification",
    page_icon="🔬",
    layout="wide"
)

# Load model
try:
    model = load_model("steel_microstructure_cnn (2).keras")
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"Model Error: {e}")
    st.stop()

# Classes
class_names = [
    "CPJ Alloy",
    "HR Alloy",
    "P92 Alloy"
]

# Sidebar
st.sidebar.title("🔬 Steel Classification")
st.sidebar.info(
    """
    Upload a steel microstructure image.

    Supported Formats:
    - JPG
    - JPEG
    - PNG
    - BMP
    """
)

# Main title
st.title("🔬 Steel Microstructure Classification Dashboard")

st.markdown("""
This AI model classifies steel microstructure images into:

- CPJ Alloy
- HR Alloy
- P92 Alloy
""")

uploaded_file = st.file_uploader(
    "📤 Upload Steel Microstructure Image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file:

    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.convert("RGB")
    img = img.resize((128, 128))

    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    with col2:
        st.subheader("Prediction Result")
        st.success(predicted_class)

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.subheader("Class Probabilities")

        for i, cls in enumerate(class_names):
            st.progress(float(prediction[0][i]))
            st.write(
                f"{cls}: {prediction[0][i]*100:.2f}%"
            )

st.markdown("---")
st.caption("Developed using TensorFlow + Streamlit")
