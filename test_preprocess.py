from pathlib import Path

from src.preprocess import load_image
from src.preprocess import resize_image
from src.preprocess import convert_to_gray

image_path = Path("data/raw/flowers/daisy").glob("*")

image_path = list(image_path)[0]

print(image_path)

image = load_image(image_path)

print("Original Shape:", image.shape)

image = resize_image(image)

print("Resized Shape:", image.shape)

gray = convert_to_gray(image)

print("Gray Shape:", gray.shape)