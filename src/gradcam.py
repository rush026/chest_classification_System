# ==========================================================
# src/gradcam.py
# Grad-CAM explainability for the chest X-ray CNN classifier
#
# Usage:
#   python src/gradcam.py path/to/xray_image.png
#
# Produces: gradcam_output.png showing the original image,
# the heatmap, and the overlay side by side.
# ==========================================================

import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---- CONFIG ----
MODEL_PATH = os.path.join("models", "covid_chest_xray_model.keras")
CLASS_INDICES_PATH = os.path.join("models", "class_indices.json")
IMG_SIZE = (224, 224)


def get_last_conv_layer_name(model):
    """Get the last conv layer name for Grad-CAM.
    For MobileNetV2, the last activation after conv is always 'out_relu'.
    We find the nested base model name dynamically."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            # This is the MobileNetV2 base model
            return layer.name, "out_relu"
    raise ValueError("No nested base model found.")


    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index.numpy()), predictions.numpy()[0]


def overlay_heatmap(img_path, heatmap, alpha=0.4):
    """Overlay the heatmap on the original image."""
    img = load_img(img_path, target_size=IMG_SIZE)
    img = img_to_array(img)

    heatmap_resized = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    overlay = jet_heatmap * alpha + img
    overlay = tf.keras.preprocessing.image.array_to_img(overlay)
    return img / 255.0, heatmap_resized, overlay


def main(image_path):
    # ---- Load model and class names ----
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    # ---- Preprocess image ----
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, nested_model_name=None, pred_index=None):
    """Generate Grad-CAM using tf.GradientTape on the outer model directly."""
    base_model = model.get_layer(nested_model_name) if nested_model_name else model

    # Build a sub-model from base_model input to its last conv output
    conv_layer = base_model.get_layer(last_conv_layer_name)
    base_grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[conv_layer.output, base_model.output]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        # Run base model to get conv outputs and base model output
        conv_outputs, base_output = base_grad_model(img_tensor)
        tape.watch(conv_outputs)

        # Run rest of outer model (pooling + dense layers) on base output
        x = base_output
        reached_base = False
        for layer in model.layers:
            if layer.name == nested_model_name:
                reached_base = True
                continue
            if reached_base:
                x = layer(x)

        predictions = x
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_outputs[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index.numpy()), predictions.numpy()[0]
