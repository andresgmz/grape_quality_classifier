"""Model training and artifact saving."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import config, dataset, model as model_module, preprocessing

HISTORY_PATH = config.MODELS_DIR / "history.json"


def train():
    """Train the CNN and return (model, history)."""
    tf.keras.utils.set_random_seed(config.SEED)

    train_ds, val_ds = preprocessing.build_datasets()
    cnn = model_module.compile_model(model_module.build_model())

    # Early stopping sobre val_loss, no val_accuracy: con 60 imagenes de
    # validacion la accuracy avanza a saltos de 1,7% y corta el entrenamiento
    # antes de tiempo. La perdida es una senal mas fina.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        )
    ]

    history = cnn.fit(
        train_ds, validation_data=val_ds, epochs=config.EPOCHS, callbacks=callbacks
    )
    return cnn, history


def choose_threshold(trained_model: tf.keras.Model) -> float:
    """Calibra en VALIDACION el umbral para declarar 'fresh'.

    Dejar pasar una uva podrida como fresca es el error caro; descartar una
    sana solo cuesta fruta. Asi que se busca el umbral mas bajo que no deje
    escapar ninguna podrida, y entre los empatados el de mejor accuracy. El
    umbral se elige en validacion, nunca en test.
    """
    frame = dataset.load_labels()
    y_true = frame[frame["split"] == "val"]["label"].to_numpy()
    probabilities = trained_model.predict(
        preprocessing.build_dataset("val", frame), verbose=0
    ).ravel()

    mejor, mejor_clave = 0.5, None
    for threshold in np.arange(0.30, 0.96, 0.01):
        predicted = (probabilities >= threshold).astype(int)
        escapan = int(((y_true == 0) & (predicted == 1)).sum())
        accuracy = float((predicted == y_true).mean())
        clave = (-escapan, accuracy)  # primero minimizar escapes, luego accuracy
        if mejor_clave is None or clave > mejor_clave:
            mejor, mejor_clave = round(float(threshold), 2), clave

    escapan, accuracy = -mejor_clave[0], mejor_clave[1]
    print(f"umbral calibrado -> {mejor:.2f} (val: {escapan} podridas escapan, accuracy {accuracy:.3f})")
    return mejor


def save_artifacts(trained_model: tf.keras.Model, history=None) -> None:
    """Save the model (.keras) and the preprocessing config (.json)."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trained_model.save(config.MODEL_PATH)
    preprocessing.save_preprocessing(decision_threshold=choose_threshold(trained_model))

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
