import cv2
import numpy as np

from skimage.feature import hog
from skimage.feature import local_binary_pattern

from src.preprocess import resize_image
from src.preprocess import convert_to_gray


# --------------------------
# COLOR HISTOGRAM
# --------------------------

def extract_color_histogram(image, bins=8):
    """
    Extract normalized color histogram.
    """

    hist = cv2.calcHist(
        [image],
        [0, 1, 2],
        None,
        [bins, bins, bins],
        [0, 256, 0, 256, 0, 256],
    )

    hist = cv2.normalize(hist, hist)

    return hist.flatten()


# --------------------------
# HOG
# --------------------------

def extract_hog(image):

    gray = convert_to_gray(image)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return features.astype("float32")

# --------------------------
# LBP
# --------------------------

def extract_lbp(image):

    gray = convert_to_gray(image)

    lbp = local_binary_pattern(
        gray,
        P=8,
        R=1,
        method="uniform",
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        range=(0, 10),
    )

    hist = hist.astype("float")

    hist /= hist.sum()

    return hist


# --------------------------
# ORB
# --------------------------

def extract_orb(image):

    gray = convert_to_gray(image)

    orb = cv2.ORB_create(nfeatures=100)

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None:
        return np.zeros(3200)

    return descriptors.flatten()


# --------------------------
# COMBINE
# --------------------------


def normalize_feature(feature):

    feature = feature.astype("float32")

    norm = np.linalg.norm(feature)

    if norm > 0:
        feature = feature / norm

    return feature


def extract_features(image):

    image = resize_image(image)

    color = normalize_feature(
        extract_color_histogram(image)
    )

    hog_feature = normalize_feature(
        extract_hog(image)
    )

    lbp = normalize_feature(
        extract_lbp(image)
    )

    features = np.concatenate([
        color,
        hog_feature,
        lbp
    ])

    return features.astype("float32")