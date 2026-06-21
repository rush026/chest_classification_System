# ==========================================================
# api/main.py
# FastAPI service for the COVID-19 Chest Classification System
#
# Run locally with:
#   uvicorn api.main:app --reload
# Then open: http://127.0.0.1:8000/docs
# ==========================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    ClinicalData,
    ClinicalPredictionResponse,
    ImagePredictionResponse,
)

# ---- CONFIG ----
# MODEL_DIR resolves to <project_root>/models regardless of where uvicorn is launched from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_SIZE = (224, 224)

app = FastAPI(
    title="COVID-19 Chest Classification System",
    description="Predicts COVID-19 likelihood from chest X-ray images and/or clinical vitals.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- LOAD CLINICAL MODEL (loaded at startup) ----
clinical_model = None
clinical_scaler = None
clinical_meta = None

clinical_model_path = os.path.join(MODEL_DIR, "clinical_model.pkl")
if os.path.exists(clinical_model_path):
    clinical_model = joblib.load(clinical_model_path)
    clinical_scaler = joblib.load(os.path.join(MODEL_DIR, "clinical_scaler.pkl"))
    clinical_meta = joblib.load(os.path.join(MODEL_DIR, "clinical_feature_meta.pkl"))

# ---- LOAD IMAGE MODEL (loaded at startup, lazy-imported to avoid TF cost if unused) ----
image_model = None
class_indices = None
idx_to_class = None

image_weights_path = os.path.join(MODEL_DIR, "covid_chest_xray_model.weights.h5")
image_model_path = os.path.join(MODEL_DIR, "covid_chest_xray_model.keras")
class_indices_path = os.path.join(MODEL_DIR, "class_indices.json")

if os.path.exists(class_indices_path) and (os.path.exists(image_weights_path) or os.path.exists(image_model_path)):
    import sys
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    with open(class_indices_path) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    num_classes = len(class_indices)

    if os.path.exists(image_weights_path):
        from src.model_arch import build_model, IMG_SIZE as ARCH_IMG_SIZE
        image_model = build_model(num_classes=num_classes, img_size=ARCH_IMG_SIZE)
        image_model.load_weights(image_weights_path)
    else:
        import tensorflow as tf
        image_model = tf.keras.models.load_model(image_model_path)
        
@app.get("/")
def root():
    return {
        "message": "COVID-19 Chest Classification System API",
        "endpoints": ["/predict/clinical", "/predict/image", "/docs"],
        "clinical_model_loaded": clinical_model is not None,
        "image_model_loaded": image_model is not None,
    }


@app.post("/predict/clinical", response_model=ClinicalPredictionResponse)
def predict_clinical(data: ClinicalData):
    """Predict COVID-19 likelihood from clinical vitals and symptoms."""
    if clinical_model is None:
        raise HTTPException(status_code=503, detail="Clinical model not loaded. Train it first with train_clinical.py")

    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # Reorder/select columns to match training feature order
    feature_cols = clinical_meta["feature_columns"]
    numeric_cols = clinical_meta["numeric_columns"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    df = df[feature_cols]
    df[numeric_cols] = clinical_scaler.transform(df[numeric_cols])

    proba = clinical_model.predict_proba(df)[0, 1]
    prediction = "Positive" if proba >= 0.5 else "Negative"

    return ClinicalPredictionResponse(
        prediction=prediction,
        probability_positive=round(float(proba), 4),
    )


@app.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Predict chest condition class from an uploaded X-ray image."""
    if image_model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded. Train it first with train_image_model.py")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = image_model.predict(arr)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = idx_to_class[pred_idx]

    class_probs = {idx_to_class[i]: round(float(p), 4) for i, p in enumerate(preds)}

    return ImagePredictionResponse(
        predicted_class=pred_class,
        confidence=round(float(preds[pred_idx]), 4),
        class_probabilities=class_probs,
    )


@app.post("/predict/image/gradcam")
async def predict_image_gradcam(file: UploadFile = File(...)):
    """Predict chest condition class AND return a Grad-CAM heatmap overlay (base64 PNG)."""
    if image_model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded. Train it first with train_image_model.py")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    import base64
    import tempfile
    import sys
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    from src.gradcam import get_last_conv_layer_name, make_gradcam_heatmap, overlay_heatmap

    contents = await file.read()

    # Save to a temp file because gradcam helpers expect a file path
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        img = Image.open(BytesIO(contents)).convert("RGB").resize(IMG_SIZE)
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)

        nested_model_name, last_conv_layer_name = get_last_conv_layer_name(image_model)
        heatmap, pred_idx, probs = make_gradcam_heatmap(
            arr, image_model, last_conv_layer_name, nested_model_name
        )

        _, _, overlay_img = overlay_heatmap(tmp_path, heatmap)

        buf = BytesIO()
        overlay_img.save(buf, format="PNG")
        overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        pred_class = idx_to_class[pred_idx]
        class_probs = {idx_to_class[i]: round(float(p), 4) for i, p in enumerate(probs)}

        return {
            "predicted_class": pred_class,
            "confidence": round(float(probs[pred_idx]), 4),
            "class_probabilities": class_probs,
            "gradcam_overlay_png_base64": overlay_b64,
        }
    finally:
        os.remove(tmp_path)