# ==========================================================
# src/train_clinical.py
# Trains the clinical/tabular model and saves it for the API
# ==========================================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = os.path.join("data", "clean_clinical_data.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---- 1. Load data ----
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["target"])
y = df["target"]

numeric_cols = ['temperature', 'pulse', 'sys', 'dia', 'rr', 'sats', 'age']
numeric_cols = [c for c in numeric_cols if c in X.columns]

# ---- 2. Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 3. Scale numeric features ----
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ---- 4. Train Random Forest (class_weight balanced for imbalance) ----
model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced",
    max_depth=10, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# ---- 5. Evaluate ----
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ---- 6. Save model, scaler, and feature column order ----
joblib.dump(model, os.path.join(MODEL_DIR, "clinical_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "clinical_scaler.pkl"))
joblib.dump(
    {"feature_columns": list(X.columns), "numeric_columns": numeric_cols},
    os.path.join(MODEL_DIR, "clinical_feature_meta.pkl")
)

print("\nSaved:")
print(f" - {MODEL_DIR}/clinical_model.pkl")
print(f" - {MODEL_DIR}/clinical_scaler.pkl")
print(f" - {MODEL_DIR}/clinical_feature_meta.pkl")