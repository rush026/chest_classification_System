import tensorflow as tf
import numpy as np
from PIL import Image

# Model load
model = tf.keras.models.load_model(
    "models/covid_chest_xray_model.keras"
)

# Image load
img = Image.open("sample_xray.png")
img = img.resize((224, 224))

img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

# Prediction
prediction = model.predict(img)

print(prediction)
print("Predicted Class:", np.argmax(prediction))

print(model.summary())