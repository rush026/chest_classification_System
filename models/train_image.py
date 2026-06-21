# ==========================================================
# src/train_image_model.py
# Trains the chest X-ray CNN classifier and saves it for the API
#
# NOTE: Run this in an environment with GPU access (recommended)
# and the COVID-19 Radiography Database downloaded into
# data/COVID-19_Radiography_Dataset/
# Kaggle dataset: tawsifurrahman/covid19-radiography-database
# ==========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

# ---- 1. CONFIG ----
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
DATA_DIR = os.path.join("data", "COVID-19_Radiography_Dataset")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---- 2. DATA GENERATORS ----
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="training", shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="validation", shuffle=False
)

class_names = list(train_gen.class_indices.keys())
print(f"Classes found: {class_names}")

# Save class index mapping for the API
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(train_gen.class_indices, f)

# ---- 3. BUILD MODEL (DenseNet121 transfer learning) ----
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
               loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
]

# ---- 4. TRAIN (frozen base) ----
history = model.fit(train_gen, validation_data=val_gen, epochs=15, callbacks=callbacks)

# ---- 5. FINE-TUNE ----
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
               loss="categorical_crossentropy", metrics=["accuracy"])

history_fine = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)

# ---- 6. EVALUATE ----
val_gen.reset()
y_true = val_gen.classes
y_pred = np.argmax(model.predict(val_gen, verbose=1), axis=1)
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("image_confusion_matrix.png")
plt.close()

# ---- 7. SAVE MODEL ----
model.save(os.path.join(MODEL_DIR, "covid_chest_xray_model.keras"))
print(f"\nSaved: {MODEL_DIR}/covid_chest_xray_model.keras")
print(f"Saved: {MODEL_DIR}/class_indices.json -> {train_gen.class_indices}")