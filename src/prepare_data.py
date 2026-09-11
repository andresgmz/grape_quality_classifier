"""Extract the black round grape subset from the GrapeNet archive into data/raw/.

See `data/README.md` for the expected source layout and the known caveats
(the dataset ships pre-augmented, so splits must be grouped by base image).

Usage:
    python -m src.prepare_data [ruta_a_Grapes_Dataset]
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

from . import config

# Source subsets inside the GrapeNet archive -> destination class folder.
SUBSETS = {
    "black/round_fresh": "fresh",
    "black/round_rotten": "rotten",
}

DEFAULT_ARCHIVE_ROOT = (
    Path.home()
    / "Downloads"
    / "GrapeNet An Image Dataset for Grape Variety and Qu"
    / "GrapeNet An Image Dataset for Grape Variety and Qu"
    / "Grapes_Dataset"
)

# Las variantes se llaman <base>_<transformacion>_iter<n>.jpg
AUG_SUFFIX = re.compile(r"^(?P<base>.+?)_(?P<aug>[a-z_]+)_iter\d+$")


def base_image_id(file_name: str) -> str:
    """Strip augmentation suffixes to group variants of the same photo.

    'black_round_fresh_IMG_20250412_115529_115529_gamma_iter3.jpg'
        -> 'black_round_fresh_IMG_20250412_115529_115529'

    Si el nombre no trae sufijo de aumentacion se devuelve tal cual (sin
    extension), de modo que la foto cuenta como su propio grupo.
    """
    stem = Path(file_name).stem
    match = AUG_SUFFIX.match(stem)
    return match.group("base") if match else stem


def extract_subset(archive: Path, destination: Path) -> int:
    """Unzip one subset into data/raw/<class>/. Devuelve cuantas imagenes escribio."""
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            name = Path(member).name
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            target = destination / name
            if target.exists():  # re-ejecutable sin volver a descomprimir todo
                continue
            target.write_bytes(zf.read(member))
            written += 1
    return written


def prepare(archive_root: Path = DEFAULT_ARCHIVE_ROOT) -> None:
    """Extract every subset in SUBSETS into data/raw/."""
    if not archive_root.exists():
        raise FileNotFoundError(
            f"No encuentro el dataset en {archive_root}. "
            "Pasa la ruta a Grapes_Dataset como argumento."
        )

    for subset, class_name in SUBSETS.items():
        folder = archive_root / subset
        zips = sorted(folder.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"No hay .zip en {folder}")

        destination = config.RAW_DIR / class_name
        total = 0
        for archive in zips:
            total += extract_subset(archive, destination)

        existing = len(list(destination.glob("*.jpg")))
        print(f"{class_name:>6}: {total} imagenes nuevas ({existing} en total en {destination})")


def main() -> None:
    archive_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ARCHIVE_ROOT
    prepare(archive_root)


if __name__ == "__main__":
    main()
