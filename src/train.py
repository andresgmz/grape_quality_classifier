"""Model training and artifact saving."""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from . import config, model as model_module, preprocessing

HISTORY_PATH = config.MODELS_DIR / "history.json"


def train():
    """Train the CNN and return (model, history)."""
    tf.keras.utils.set_random_seed(config.SEED)

    train_ds, val_ds = preprocessing.build_datasets()
    cnn = model_module.compile_model(model_module.build_model())

    # Early stopping: con un dataset chico el sobreajuste llega rapido y no
    # tiene sentido gastar las 20 epocas completas si la validacion deja de mejorar.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        )
    ]

    history = cnn.fit(
        train_ds, validation_data=val_ds, epochs=config.EPOCHS, callbacks=callbacks
    )
    return cnn, history


def save_artifacts(trained_model: tf.keras.Model, history=None) -> None:
    """Save the model (.keras) and the preprocessing config (.json)."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trained_model.save(config.MODEL_PATH)
    preprocessing.save_preprocessing()

    if history is not None:
        HISTORY_PATH.write_text(
            json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=2),
            encoding="utf-8",
        )

    print(f"modelo         -> {config.MODEL_PATH}")
    print(f"preprocessing  -> {config.PREPROCESSING_PATH}")
    if history is not None:
        print(f"historial      -> {HISTORY_PATH}")


def main() -> None:
    trained_model, history = train()
    save_artifacts(trained_model, history)


if __name__ == "__main__":
    main()
