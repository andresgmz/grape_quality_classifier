"""Binary classification CNN (fresh vs. rotten grapes)."""

from __future__ import annotations

import tensorflow as tf

from . import config, preprocessing


def build_model(input_shape=None) -> tf.keras.Model:
    """Simple CNN: stacked Conv2D + MaxPooling with a dense head.

    Salida: una sola neurona con sigmoide (probabilidad de 'fresh'), porque la
    tarea es binaria. Tres bloques conv bastan para 128x128 y un dataset chico;
    mas capas solo aumentarian el sobreajuste.
    """
    input_shape = input_shape or (*config.IMG_SIZE, 3)

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            preprocessing.augmentation_layers(),
            tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="grape_quality_classifier",
    )


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    """Compile with optimizer, binary loss and metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
