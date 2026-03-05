from typing import Sequence

import keras
from keras import layers


def build_image_classifier(
    num_classes: int,
    img_size: int = 64,
    hidden_channels: Sequence[int] | None = None,
) -> keras.Model:
    if hidden_channels is None:
        hidden_channels = [32, 64, 128]

    inputs = keras.Input(shape=(img_size, img_size, 1), name="image")

    x = inputs
    for i, c in enumerate(hidden_channels):
        x = layers.Conv2D(c, 3, padding="same", activation="relu", name=f"conv_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool_{i}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="image_classifier")
    return model
