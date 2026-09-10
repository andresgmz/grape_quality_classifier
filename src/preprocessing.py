"""Image preprocessing: resize, normalization and data augmentation.

The parameters used here are stored next to the model (step 5 of the
assignment) so they can be reused at inference time.
"""

from __future__ import annotations

from pathlib import Path

from . import config


def load_image(path: Path):
    """Load an image, resize it to IMG_SIZE and normalize it. TODO."""
    raise NotImplementedError


def build_datasets():
    """Return (train_ds, val_ds) ready for training. TODO: implement."""
    raise NotImplementedError


def augmentation_layers():
    """Data augmentation layers (flip, rotation, zoom). TODO: implement."""
    raise NotImplementedError


def save_preprocessing(path: Path = config.PREPROCESSING_PATH) -> None:
    """Serialize img_size, normalization and classes to JSON. TODO."""
    raise NotImplementedError


def load_preprocessing(path: Path = config.PREPROCESSING_PATH) -> dict:
    """Read the preprocessing JSON back. TODO: implement."""
    raise NotImplementedError
