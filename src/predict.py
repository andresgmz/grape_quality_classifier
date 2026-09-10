"""Inference on new images (step 4 of the assignment)."""

from __future__ import annotations

import sys
from pathlib import Path

from . import config


def load_model(path: Path = config.MODEL_PATH):
    """Load the trained model. TODO: implement."""
    raise NotImplementedError


def predict(image_path: Path) -> dict:
    """Return label, class_name and confidence for one image. TODO."""
    raise NotImplementedError


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m src.predict <image_path>")
        return 1
    print(predict(Path(argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
