"""Model evaluation: metrics, confusion matrix and training curves."""

from __future__ import annotations

from . import config


def evaluate(model, dataset):
    """Accuracy, precision, recall and F1. TODO: implement."""
    raise NotImplementedError


def confusion_matrix(model, dataset):
    """Compute and plot the confusion matrix. TODO: implement."""
    raise NotImplementedError


def plot_history(history):
    """Loss/accuracy curves per epoch. TODO: implement."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
