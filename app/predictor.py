import joblib
import numpy as np

from src.config import MODEL_DIR, CLASS_NAMES
from src.preprocess import load_image
from src.features import extract_features

scaler = joblib.load(MODEL_DIR / "scaler.pkl")
model = joblib.load(MODEL_DIR / "best_model.pkl")


def predict_image(image_path: str):

    image = load_image(image_path)

    features = extract_features(image)
    features = features.reshape(1, -1)
    features = scaler.transform(features)

    probabilities = model.predict_proba(features)[0]

    predicted_index = np.argmax(probabilities)

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": round(float(probabilities[predicted_index]), 4),
    }