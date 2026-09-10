"""Central project configuration."""

from pathlib import Path

FRUIT = "grape"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_TEST_DIR = DATA_DIR / "external_test"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# label: 0 = rotten, 1 = fresh
CLASSES = {"rotten": 0, "fresh": 1}
CLASS_NAMES = {v: k for k, v in CLASSES.items()}

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

MODEL_PATH = MODELS_DIR / "grape_quality_classifier.keras"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing.json"
LABELS_CSV = PROCESSED_DIR / "labels.csv"
