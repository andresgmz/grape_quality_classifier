"""Image preprocessing: resize, normalization and data augmentation.

The parameters used here are stored next to the model (step 5 of the
assignment) so they can be reused at inference time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import config, dataset

AUTOTUNE = tf.data.AUTOTUNE
NORMALIZATION = "rescale_1/255"


def load_image(path: Path) -> np.ndarray:
    """Load an image, resize it to IMG_SIZE and normalize it.

    Devuelve un array (alto, ancho, 3) con valores en [0, 1]. Es la misma
    transformacion que usa el pipeline de entrenamiento, para que inferencia
    y entrenamiento no se desalineen.
    """
    raw = tf.io.read_file(str(path))
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, config.IMG_SIZE)
    return (image / 255.0).numpy()


def _decode(path, label):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, config.IMG_SIZE)
    image.set_shape((*config.IMG_SIZE, 3))
    return image / 255.0, tf.cast(label, tf.float32)


def augmentation_layers() -> tf.keras.Sequential:
    """Data augmentation layers (flip, rotation, zoom).

    Solo se aplica en entrenamiento. Compensa que usamos pocas variantes por
    foto y ayuda a que el modelo no memorice la orientacion exacta del racimo.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )


def build_dataset(split: str, frame=None, shuffle: bool = False) -> tf.data.Dataset:
    """tf.data.Dataset para un split concreto de labels.csv."""
    frame = dataset.load_labels() if frame is None else frame
    part = frame[frame["split"] == split]
    if part.empty:
        raise ValueError(f"El split '{split}' esta vacio en labels.csv")

    paths = [str(config.DATASET_DIR / image) for image in part["image"]]
    labels = part["label"].to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE)
    return ds.batch(config.BATCH_SIZE).prefetch(AUTOTUNE)


def build_datasets():
    """Return (train_ds, val_ds) ready for training."""
    frame = dataset.load_labels()
    train_ds = build_dataset("train", frame, shuffle=True)
    val_ds = build_dataset("val", frame)
    return train_ds, val_ds


def save_preprocessing(
    path: Path = config.PREPROCESSING_PATH, decision_threshold: float = 0.5
) -> Path:
    """Serialize img_size, normalization, classes and decision threshold."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "img_size": list(config.IMG_SIZE),
        "normalization": NORMALIZATION,
        "classes": config.CLASSES,
        "variants_per_photo": config.VARIANTS_PER_PHOTO,
        "decision_threshold": float(decision_threshold),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_preprocessing(path: Path = config.PREPROCESSING_PATH) -> dict:
    """Read the preprocessing JSON back."""
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Corre antes: python -m src.train")
    return json.loads(path.read_text(encoding="utf-8"))


def decision_threshold(path: Path = config.PREPROCESSING_PATH) -> float:
    """Umbral por encima del cual se declara 'fresh'.

    Se calibra en validacion al entrenar. Es mayor que 0.5 a proposito: dejar
    pasar una uva podrida como fresca es mas grave que descartar una sana, asi
    que se exige mas evidencia para decir 'fresh'.
    """
    return float(load_preprocessing(path).get("decision_threshold", 0.5))
