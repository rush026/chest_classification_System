# ==========================================================
# src/model_arch.py
# Shared model architecture builder.
# Used both for training and for loading weights in the API,
# so the architecture always matches exactly - avoids Keras
# version serialization issues with load_model().
# ==========================================================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (224, 224)
NUM_CLASSES = 4


def build_model(num_classes: int = NUM_CLASSES, img_size=IMG_SIZE):
    """Builds the MobileNetV2-based classifier. Must match the
    architecture used during training in Colab exactly."""
    base_model = MobileNetV2(
        weights=None,  # weights are loaded afterwards via load_weights
        include_top=False,
        input_shape=(*img_size, 3),
    )

    inputs = layers.Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model