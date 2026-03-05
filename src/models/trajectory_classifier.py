from typing import Sequence

import keras
from keras import layers


def build_trajectory_classifier(
    num_classes: int,
    seq_len: int = 100,
    hidden_sizes: Sequence[int] | None = None,
) -> keras.Model:
    if hidden_sizes is None:
        hidden_sizes = [64, 128]

    inputs = keras.Input(shape=(seq_len, 2), name="trajectory")

    x = inputs
    for i, h in enumerate(hidden_sizes):
        x = layers.Conv1D(
            filters=h,
            kernel_size=5,
            padding="same",
            activation="relu",
            name=f"conv_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="trajectory_classifier")
    return model

