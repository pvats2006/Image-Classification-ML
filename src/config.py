from pathlib import Path

# ==============================
# PROJECT PATHS
# ==============================

# Root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "flowers"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TEST_DATA_DIR = DATA_DIR / "test"

# Models
MODEL_DIR = BASE_DIR / "models"

# Reports
REPORT_DIR = BASE_DIR / "reports"

# ==============================
# IMAGE SETTINGS
# ==============================

IMAGE_SIZE = (224, 224)

# Flower classes
CLASS_NAMES= [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip",
]

# Random seed
RANDOM_STATE = 42