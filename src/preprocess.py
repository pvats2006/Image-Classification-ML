import cv2

from src.config import IMAGE_SIZE

def load_image(image_path):
    """
    Read image using OpenCV.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    return image


def resize_image(image):
    """
    Resize image.
    """

    return cv2.resize(image, IMAGE_SIZE)


def convert_to_gray(image):
    """
    Convert BGR to Gray.
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def save_image(path, image):
    """
    Save image.
    """

    cv2.imwrite(str(path), image)