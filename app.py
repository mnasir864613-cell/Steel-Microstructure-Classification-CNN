import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Steel Microstructure Classification",
    page_icon="🔬",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_my_model():
    return load_model("steel_microstructure_cnn (2).keras")

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# =========================
# CLASS NAMES
# =========================
class_names = [
    "CPJ Alloy",
    "HR Alloy",
    "P92 Alloy"
]

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🎓 FYP Dashboard")

st.sidebar.markdown("""
### Student
**Nagina Bibi**

### Project
Steel Microstructure Classification

### Model
CNN Deep Learning Model

### Classes
✅ CPJ Alloy

✅ HR Alloy

✅ P92 Alloy

### Technology
- TensorFlow
- Keras
- Streamlit
""")

# =========================
# HEADER
# =========================
st.markdown("""
<div style="
background: linear-gradient(90deg,#1f4e79,#0e7490);
padding:25px;
border-radius:15px;
text-align:center;
color:white;
">

<h1>🔬 Steel Microstructure Classification Dashboard</h1>

<h3>Deep Learning Based CNN Model</h3>

<p style="font-size:22px;">
🎓 Developed & Trained By <b>Nagina Bibi</b>
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# =========================
# PROJECT OVERVIEW
# =========================
st.info("""
This application uses a trained Convolutional Neural Network (CNN)
to classify steel microstructure images into:

• CPJ Alloy

• HR Alloy

• P92 Alloy
""")

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "📤 Upload Steel Microstructure Image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(
            image,
            use_container_width=True
        )

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

        st.subheader("🎯 Prediction Result")

        st.success(
            f"Predicted Class: {predicted_class}"
        )

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

        st.subheader("📈 Prediction Probabilities")

        prob_df = pd.DataFrame(
            {
                "Probability (%)":
                [float(x) * 100 for x in prediction[0]]
            },
            index=class_names
        )

        st.bar_chart(prob_df)

        descriptions = {

            "CPJ Alloy":
            "Complex Phase Steel Alloy used in automotive applications.",

            "HR Alloy":
            "Hot Rolled Steel Alloy with characteristic microstructure.",

            "P92 Alloy":
            "High-temperature creep-resistant alloy used in power plants."
        }

        st.subheader("📖 Alloy Description")
        st.info(descriptions[predicted_class])

# =========================
# MODEL PERFORMANCE
# =========================
st.markdown("---")

st.header("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "accuracy_loss.png",
        caption="Training Accuracy & Loss Curves",
        use_container_width=True
    )

with col2:
    st.image(
        "confusion_matrix.png",
        caption="Confusion Matrix",
        use_container_width=True
    )

# =========================
# DATASET IMAGES
# =========================
st.markdown("---")

st.header("🖼 Dataset Visualization")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "sample_images.png",
        caption="Sample Dataset Images",
        use_container_width=True
    )

with col2:
    st.image(
        "augmented_images.png",
        caption="Augmented Dataset Images",
        use_container_width=True
    )

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown("""
<div style="text-align:center">

## 🎓 Final Year Project

### Steel Microstructure Classification Using CNN

Developed & Trained By **Nagina Bibi**

Department of Computer Science

Powered by TensorFlow • Keras • Streamlit

</div>
""", unsafe_allow_html=True)
