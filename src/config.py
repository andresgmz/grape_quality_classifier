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

# El dataset viene pre-aumentado: 25 variantes de cada foto original.
# Nos quedamos con unas pocas por foto (el resto de la variacion la genera
# nuestra propia capa de augmentation al entrenar).
VARIANTS_PER_PHOTO = 4

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
# Margen para que el EarlyStopping se dispare: en TF 2.11 restore_best_weights
# solo restaura los mejores pesos si el entrenamiento se corta antes del final.
EPOCHS = 60
SEED = 42
# Con solo 236 fotos originales, un test del 10% son 24 fotos: demasiado
# pocas para que la metrica signifique algo. 20% da ~47 fotos.
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

MODEL_PATH = MODELS_DIR / "grape_quality_classifier.keras"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing.json"
LABELS_CSV = PROCESSED_DIR / "labels.csv"
