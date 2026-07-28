from pathlib import Path


def create_directory(path):
    """
    Create directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def print_header(title):
    """
    Print formatted title.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def list_images(folder):
    """
    Return all image paths.
    """
    image_extensions = [".jpg", ".jpeg", ".png"]

    images = []

    for ext in image_extensions:
        images.extend(Path(folder).rglob(f"*{ext}"))

    return sorted(images)