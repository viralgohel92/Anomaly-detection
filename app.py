import streamlit as st
import numpy as np
from PIL import Image
from keras.layers import TFSMLayer

st.title("Anomaly Detection with Teachable Machine")

# Load model once at the top
@st.cache_resource
def load_model():
    return TFSMLayer("model/", call_endpoint="serving_default")

model = load_model()

uploaded_file = st.file_uploader("Upload an image to check", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize and preprocess
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model(image_array)
    output_tensor = next(iter(prediction.values()))
    prediction_array = output_tensor.numpy()

    st.write("Prediction Probabilities:", prediction_array)

    predicted_class = np.argmax(prediction_array)
    class_labels = ["Normal", "Anomaly"]

    st.write(f"Predicted Class: **{class_labels[predicted_class]}**")

    if predicted_class == 1:
        st.error("⚠️ Anomaly Detected!")
    else:
        st.success("✅ good Product")
