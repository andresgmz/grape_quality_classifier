"""Binary classification CNN (fresh vs. rotten grapes).

Dos arquitecturas, elegidas con `config.MODEL_KIND`:

- "transfer": MobileNetV2 preentrenada en ImageNet con la base congelada.
  Es la opcion por defecto. Con pocos cientos de imagenes, unas features ya
  entrenadas sobre millones de fotos generalizan mucho mejor que una red
  aprendida desde cero, sobre todo ante fondos e iluminaciones nuevos.
- "cnn": red propia de tres bloques convolucionales, entrenada desde cero.
  Se conserva como linea base para comparar.
"""

from __future__ import annotations

import tensorflow as tf

from . import config, preprocessing


def build_simple_cnn(input_shape) -> tf.keras.Model:
    """Stacked Conv2D + MaxPooling with a dense head (linea base)."""
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
        name="grape_cnn",
    )


def build_transfer_model(input_shape) -> tf.keras.Model:
    """MobileNetV2 congelada + cabeza binaria."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            preprocessing.augmentation_layers(),
            # El pipeline entrega [0, 1] y MobileNetV2 espera [-1, 1].
            tf.keras.layers.Rescaling(2.0, offset=-1.0),
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="grape_transfer",
    )


def build_model(input_shape=None, kind: str = None) -> tf.keras.Model:
    """Devuelve la arquitectura indicada por config.MODEL_KIND."""
    input_shape = input_shape or (*config.IMG_SIZE, 3)
    kind = kind or config.MODEL_KIND

    if kind == "transfer":
        return build_transfer_model(input_shape)
    if kind == "cnn":
        return build_simple_cnn(input_shape)
    raise ValueError(f"MODEL_KIND desconocido: {kind!r}. Usa 'transfer' o 'cnn'.")


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    """Compile with optimizer, binary loss and metrics.

    La cabeza sobre una base congelada tolera un learning rate mas bajo que la
    red desde cero, que necesita avanzar mas rapido.
    """
    learning_rate = 1e-4 if config.MODEL_KIND == "transfer" else 1e-3
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
