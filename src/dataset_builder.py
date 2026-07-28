from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import RAW_DATA_DIR
from src.config import CLASSES

from src.preprocess import load_image
from src.features import extract_features


def build_dataset():

    X = []
    y = []

    for label, class_name in enumerate(CLASSES):

        folder = RAW_DATA_DIR / class_name

        print(f"\nProcessing {class_name}")

        images = list(folder.glob("*"))

        for image_path in tqdm(images):

            try:

                image = load_image(image_path)

                feature_vector = extract_features(image)

                X.append(feature_vector)

                y.append(label)

            except Exception as e:

                print(image_path, e)

    X = np.array(X)

    y = np.array(y)

    print()

    print("Dataset Created")

    print("X Shape :", X.shape)

    print("y Shape :", y.shape)

    np.save("data/X.npy", X)

    np.save("data/y.npy", y)


if __name__ == "__main__":

    build_dataset()