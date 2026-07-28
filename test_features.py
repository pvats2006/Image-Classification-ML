from pathlib import Path

from src.preprocess import load_image
from features import extract_features

image_path = list(Path("data/raw/flowers/daisy").glob("*"))[0]

image = load_image(image_path)

features = extract_features(image)

print("=" * 60)

for name, value in features.items():
    print(f"{name:<10} : {value.shape}")

print("=" * 60)