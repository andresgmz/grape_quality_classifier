"""Dataset building from data/raw/.

Produces `data/processed/labels.csv` with the columns:
    image      -> file path relative to data/raw/
    label      -> 0 = rotten, 1 = fresh
    class_name -> 'rotten' / 'fresh'
    base_id    -> foto original de la que sale la variante aumentada
    split      -> 'train' / 'val' / 'test'

El split se hace **por base_id**, no por archivo: el dataset viene pre-aumentado
(25 variantes de cada foto), asi que repartir archivos al azar pondria variantes
de la misma foto en train y en test e inflaria la precision. Ver data/README.md.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from . import config
from .prepare_data import base_image_id

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def list_images(root: Path = config.DATASET_DIR) -> list[tuple[Path, int]]:
    """Walk raw/<class>/ and return (path, label) pairs."""
    pairs: list[tuple[Path, int]] = []
    for class_name, label in config.CLASSES.items():
        folder = root / class_name
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                pairs.append((path, label))
    return pairs


def subsample_variants(frame: pd.DataFrame, per_photo: int = None) -> pd.DataFrame:
    """Se queda con N variantes de cada foto original.

    Conserva **todas** las fotos distintas (que es lo que aporta informacion real)
    y recorta la redundancia de las 25 variantes sinteticas por foto.
    """
    per_photo = config.VARIANTS_PER_PHOTO if per_photo is None else per_photo
    if per_photo <= 0:
        return frame

    rng = random.Random(config.SEED)
    keep: list[str] = []
    for _, group in frame.groupby("base_id", sort=True):
        variants = sorted(group["image"])
        rng.shuffle(variants)
        keep.extend(variants[:per_photo])

    return frame[frame["image"].isin(set(keep))].reset_index(drop=True)


def split_train_val_test(frame: pd.DataFrame) -> pd.DataFrame:
    """Asigna la columna 'split' agrupando por base_id y estratificando por clase."""
    rng = random.Random(config.SEED)
    assignments: dict[str, str] = {}

    for label in sorted(frame["label"].unique()):
        groups = sorted(frame.loc[frame["label"] == label, "base_id"].unique())
        rng.shuffle(groups)

        n = len(groups)
        n_test = max(1, round(n * config.TEST_SPLIT))
        n_val = max(1, round(n * config.VAL_SPLIT))

        for i, group in enumerate(groups):
            if i < n_test:
                assignments[group] = "test"
            elif i < n_test + n_val:
                assignments[group] = "val"
            else:
                assignments[group] = "train"

    return frame.assign(split=frame["base_id"].map(assignments))


def build_labels_csv(output: Path = config.LABELS_CSV) -> Path:
    """Write labels.csv into data/processed/."""
    pairs = list_images()
    if not pairs:
        raise FileNotFoundError(
            f"No hay imagenes en {config.DATASET_DIR}. Esa carpeta debe tener "
            f"una subcarpeta por clase: {', '.join(config.CLASSES)}."
        )

    frame = pd.DataFrame(
        {
            "image": [str(p.relative_to(config.DATASET_DIR)).replace("\\", "/") for p, _ in pairs],
            "label": [label for _, label in pairs],
        }
    )
    frame["class_name"] = frame["label"].map(config.CLASS_NAMES)
    frame["base_id"] = frame["image"].map(lambda p: base_image_id(Path(p).name))
    frame = subsample_variants(frame)
    frame = split_train_val_test(frame)

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def load_labels(path: Path = config.LABELS_CSV) -> pd.DataFrame:
    """Lee labels.csv. Lanza un error claro si todavia no se genero."""
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Corre antes: python -m src.dataset")
    return pd.read_csv(path)


def summarize(frame: pd.DataFrame) -> str:
    """Resumen legible: imagenes y fotos originales por split y clase."""
    lines = [f"{'split':<6} {'clase':<7} {'imagenes':>9} {'fotos':>7}"]
    for split in ("train", "val", "test"):
        part = frame[frame["split"] == split]
        for class_name in config.CLASSES:
            sub = part[part["class_name"] == class_name]
            lines.append(
                f"{split:<6} {class_name:<7} {len(sub):>9} {sub['base_id'].nunique():>7}"
            )
    lines.append(
        f"{'TOTAL':<6} {'':<7} {len(frame):>9} {frame['base_id'].nunique():>7}"
    )
    return "\n".join(lines)


def main() -> None:
    output = build_labels_csv()
    frame = pd.read_csv(output)
    print(f"labels.csv -> {output}")
    print(summarize(frame))


if __name__ == "__main__":
    main()
