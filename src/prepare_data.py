"""Extract the black round grape subset from the GrapeNet archive into data/raw/.

See `data/README.md` for the expected source layout and the known caveats
(the dataset ships pre-augmented, so splits must be grouped by base image).
"""

from __future__ import annotations

from pathlib import Path

from . import config

# Source subsets inside the GrapeNet archive -> destination class folder.
SUBSETS = {
    "black/round_fresh": "fresh",
    "black/round_rotten": "rotten",
}


def extract_subset(archive: Path, subset: str, destination: Path) -> None:
    """Unzip one subset into data/raw/<class>/. TODO: implement."""
    raise NotImplementedError


def base_image_id(file_name: str) -> str:
    """Strip augmentation suffixes to group variants of the same photo. TODO."""
    raise NotImplementedError


def prepare(archive_root: Path) -> None:
    """Extract every subset in SUBSETS into data/raw/. TODO: implement."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
