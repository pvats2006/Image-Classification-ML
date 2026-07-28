import tempfile
import joblib
import numpy as np
import streamlit as st
from PIL import Image

from src.config import MODEL_DIR, CLASS_NAMES
from src.preprocess import load_image
from src.features import extract_features

st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="centered",
)

st.title("🌸 Flower Classification")
st.write("Upload an image and let the model predict the flower.")

scaler = joblib.load(MODEL_DIR / "scaler.pkl")
model = joblib.load(MODEL_DIR / "best_model.pkl")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True,
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)

        img = load_image(temp.name)

    features = extract_features(img)

    features = features.reshape(1, -1)

    features = scaler.transform(features)

    probabilities = model.predict_proba(features)[0]

    predicted_index = np.argmax(probabilities)

    prediction = CLASS_NAMES[predicted_index]

    confidence = probabilities[predicted_index]

    st.success(f"Prediction: {prediction}")

    st.info(f"Confidence: {confidence:.2%}")

    st.subheader("Class Probabilities")

    for cls, prob in zip(CLASS_NAMES, probabilities):
        st.progress(float(prob))
        st.write(f"{cls}: {prob:.2%}")