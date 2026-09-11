"""Model evaluation: metrics, confusion matrix and training curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana grafica: solo guardamos PNG
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    precision_recall_fscore_support,
)

from . import config, dataset, preprocessing
from .train import HISTORY_PATH

CONFUSION_PATH = config.REPORTS_DIR / "confusion_matrix.png"
CURVES_PATH = config.REPORTS_DIR / "training_curves.png"
METRICS_PATH = config.REPORTS_DIR / "metrics.json"


def predict_split(model: tf.keras.Model, split: str = "test"):
    """Devuelve (y_true, y_pred, frame) para un split."""
    frame = dataset.load_labels()
    part = frame[frame["split"] == split].reset_index(drop=True)
    ds = preprocessing.build_dataset(split, frame)

    probabilities = model.predict(ds, verbose=0).ravel()
    return part["label"].to_numpy(), (probabilities >= 0.5).astype(int), part


def evaluate(model: tf.keras.Model, split: str = "test") -> dict:
    """Accuracy, precision, recall and F1."""
    y_true, y_pred, part = predict_split(model, split)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # Precision por foto original: cada foto aporta varias variantes, asi que
    # votamos entre ellas. Da una lectura mas honesta del tamaño real del test.
    per_photo = part.assign(pred=y_pred).groupby("base_id").agg(
        label=("label", "first"), vote=("pred", lambda s: int(s.mean() >= 0.5))
    )

    return {
        "split": split,
        "n_images": int(len(y_true)),
        "n_photos": int(part["base_id"].nunique()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy_per_photo": float(accuracy_score(per_photo["label"], per_photo["vote"])),
    }


def confusion_matrix(model: tf.keras.Model, split: str = "test") -> np.ndarray:
    """Compute and plot the confusion matrix."""
    y_true, y_pred, _ = predict_split(model, split)
    matrix = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])
    names = [config.CLASS_NAMES[0], config.CLASS_NAMES[1]]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=names)
    ax.set_yticks([0, 1], labels=names)
    ax.set_xlabel("prediccion")
    ax.set_ylabel("real")
    ax.set_title(f"Matriz de confusion ({split})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")

    CONFUSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CONFUSION_PATH, dpi=120)
    plt.close(fig)
    return matrix


def plot_history(history: dict = None) -> Path:
    """Loss/accuracy curves per epoch."""
    if history is None:
        if not HISTORY_PATH.exists():
            raise FileNotFoundError(f"Falta {HISTORY_PATH}. Corre antes: python -m src.train")
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(10, 4))
    ax_acc.plot(history["accuracy"], label="train")
    ax_acc.plot(history["val_accuracy"], label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoca")
    ax_acc.legend()

    ax_loss.plot(history["loss"], label="train")
    ax_loss.plot(history["val_loss"], label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoca")
    ax_loss.legend()

    CURVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CURVES_PATH, dpi=120)
    plt.close(fig)
    return CURVES_PATH


def main() -> None:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Falta {config.MODEL_PATH}. Corre antes: python -m src.train")

    model = tf.keras.models.load_model(config.MODEL_PATH)

    metrics = {split: evaluate(model, split) for split in ("val", "test")}
    matrix = confusion_matrix(model, "test")
    plot_history()

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for split, values in metrics.items():
        print(
            f"[{split}] accuracy={values['accuracy']:.3f} precision={values['precision']:.3f} "
            f"recall={values['recall']:.3f} f1={values['f1']:.3f} "
            f"(por foto: {values['accuracy_per_photo']:.3f}, "
            f"{values['n_images']} imgs / {values['n_photos']} fotos)"
        )
    print(f"matriz de confusion:\n{matrix}")
    print(f"graficas -> {CONFUSION_PATH.name}, {CURVES_PATH.name}")
    print(f"metricas -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
