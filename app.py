
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image, ImageOps

st.set_page_config(
    page_title="DigitAI | Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 750;
        letter-spacing: -1.5px;
        text-align: center;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #A0A0A0;
        font-size: 16px;
        margin-bottom: 35px;
    }

    div[data-testid="stMetric"] {
        background: #242424;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #383838;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        min-height: 45px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load trained CNN
@st.cache_resource
def load_digit_model():
    return tf.keras.models.load_model(
        "notebooks/digit_recognizer.keras"
    )

model = load_digit_model()

st.markdown(
    '<div class="main-title">DigitAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Handwritten Digit Recognition powered by Deep Learning</div>',
    unsafe_allow_html=True
)
st.markdown("""
<div style="
    background: #222222;
    border: 1px solid #383838;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 28px;
">
    <h3 style="margin-top: 0;">Recognize handwritten digits with AI</h3>
    <p style="color: #B0B0B0;">
        Upload an image of a handwritten number from 0 to 9.
        Our Convolutional Neural Network (CNN) analyzes the
        image and predicts the most likely digit.
    </p>
</div>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)
st.caption(
    "Supported formats: PNG, JPG, JPEG | "
    "Recommended: one clearly written digit on a plain background."
)

if uploaded_file is not None:

    original_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(original_image, width=250)

    if st.button("Predict Digit"):

        # Convert to grayscale
        gray_image = ImageOps.grayscale(original_image)

        # Crop central region to remove notebook edges
        width, height = gray_image.size

        gray_image = gray_image.crop((
            int(width * 0.15),
            int(height * 0.10),
            int(width * 0.85),
            int(height * 0.90)
        ))

        # Improve contrast
        gray_image = ImageOps.autocontrast(gray_image)

        # Invert: dark handwriting becomes white
        inverted_image = ImageOps.invert(gray_image)

        # Remove faint background noise
        binary_image = inverted_image.point(
            lambda pixel: 255 if pixel > 80 else 0
        )

        # Find digit boundaries
        bbox = binary_image.getbbox()

        if bbox is None:
            st.error("No digit detected. Please upload another image.")
            st.stop()

        # Crop around the digit
        digit = binary_image.crop(bbox)

        # Resize while preserving proportions
        digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

        # Center on a black 28x28 canvas
        canvas = Image.new("L", (28, 28), 0)

        x_offset = (28 - digit.width) // 2
        y_offset = (28 - digit.height) // 2

        canvas.paste(digit, (x_offset, y_offset))

        # Display processed image
        st.subheader("Processed Image")
        st.image(canvas.resize((280, 280)), width=280)

        # Prepare model input
        image_array = np.array(canvas).astype("float32") / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        # Predict
        predictions = model.predict(image_array, verbose=0)

        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        st.success(f"Predicted Digit: {predicted_digit}")
        st.metric("Prediction Confidence", f"{confidence:.2f}%")

      # Show prediction probabilities
        st.subheader("Prediction Probabilities")

        probability_df = pd.DataFrame({
          "Digit": [str(i) for i in range(10)],
          "Confidence (%)": predictions[0] * 100
        })

        probability_df = probability_df.sort_values(
          by="Confidence (%)",
        ascending=False
        )

        st.bar_chart(
          probability_df.set_index("Digit")
        )

        st.caption(
          "Probabilities show how strongly the model favors each digit."
        )
        st.divider()

st.subheader("About DigitAI")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Model**")
    st.write("Convolutional Neural Network (CNN)")

with col2:
    st.markdown("**Dataset**")
    st.write("MNIST Handwritten Digits")

with col3:
    st.markdown("**Framework**")
    st.write("TensorFlow and Keras")

st.caption(
    "DigitAI | Built with Python, TensorFlow and Streamlit"
)