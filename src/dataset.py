"""Dataset building from data/raw/.

Produces `data/processed/labels.csv` with the columns:
    image -> relative file path
    label -> 0 = rotten, 1 = fresh
"""

from __future__ import annotations

from pathlib import Path

from . import config


def list_images(root: Path = config.RAW_DIR) -> list[tuple[Path, int]]:
    """Walk raw/<class>/ and return (path, label) pairs. TODO: implement."""
    raise NotImplementedError


def build_labels_csv(output: Path = config.LABELS_CSV) -> Path:
    """Write labels.csv into data/processed/. TODO: implement."""
    raise NotImplementedError


def split_train_val_test():
    """Split the dataset into train/val/test. TODO: implement."""
    raise NotImplementedError


def main() -> None:
    build_labels_csv()


if __name__ == "__main__":
    main()
