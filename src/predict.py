"""Inference on new images (step 4 of the assignment).

Usage:
    python -m src.predict <image_path> [<image_path> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import config, preprocessing

_model = None


def load_model(path: Path = config.MODEL_PATH):
    """Load the trained model (se cachea entre llamadas)."""
    global _model
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Corre antes: python -m src.train")
    if _model is None:
        _model = tf.keras.models.load_model(path)
    return _model


def predict(image_path: Path) -> dict:
    """Return label, class_name and confidence for one image."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen {image_path}")

    # Releemos el preprocessing guardado junto al modelo para aplicar en
    # inferencia exactamente la misma transformacion que en entrenamiento.
    preprocessing.load_preprocessing()

    model = load_model()
    image = preprocessing.load_image(image_path)
    probability = float(model.predict(np.expand_dims(image, 0), verbose=0)[0][0])

    label = int(probability >= 0.5)
    return {
        "image": image_path.name,
        "label": label,
        "class_name": config.CLASS_NAMES[label],
        "confidence": probability if label == 1 else 1 - probability,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m src.predict <image_path> [<image_path> ...]")
        return 1

    for raw_path in argv[1:]:
        result = predict(Path(raw_path))
        print(
            f"{result['image']:<55} -> {result['class_name']:<6} "
            f"(label={result['label']}, confianza={result['confidence']:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
