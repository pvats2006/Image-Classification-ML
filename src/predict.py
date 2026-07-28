import argparse
import joblib
import numpy as np

from src.config import MODEL_DIR, CLASS_NAMES
from src.preprocess import load_image
from src.features import extract_features


def predict(image_path):

    print("=" * 60)
    print("Loading Model...")
    print("=" * 60)

    scaler = joblib.load(
        MODEL_DIR / "scaler.pkl"
    )

    model = joblib.load(
        MODEL_DIR / "best_model.pkl"
    )

    print("Loading Image...")

    image = load_image(image_path)

    features = extract_features(image)

    features = features.reshape(1, -1)

    features = scaler.transform(features)

    prediction = model.predict(features)[0]

    class_name = CLASS_NAMES[int(prediction)]

    print("\nPrediction")
    print("-" * 40)
    print(f"Predicted Class : {class_name}")

    return class_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image",
    )

    args = parser.parse_args()

    predict(args.image)