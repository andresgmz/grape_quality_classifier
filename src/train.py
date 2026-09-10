"""Model training and artifact saving."""

from __future__ import annotations

from . import config, model, preprocessing


def train():
    """Train the CNN and return (model, history). TODO: implement."""
    raise NotImplementedError


def save_artifacts(trained_model) -> None:
    """Save the model (.keras) and the preprocessing config (.json). TODO."""
    raise NotImplementedError


def main() -> None:
    trained_model, _ = train()
    save_artifacts(trained_model)


if __name__ == "__main__":
    main()
