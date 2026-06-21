# ==========================================================
# api/schemas.py
# Pydantic models defining the API request/response shapes
# ==========================================================

from pydantic import BaseModel, Field
from typing import Optional


class ClinicalData(BaseModel):
    """Clinical vitals and symptom data for a patient."""
    age: float = Field(..., example=45)
    temperature: float = Field(..., example=37.2)
    pulse: float = Field(..., example=88)
    sys: float = Field(..., example=120)
    dia: float = Field(..., example=80)
    rr: float = Field(..., example=18)
    sats: float = Field(..., example=97)

    # Risk factors / comorbidities (0 = No, 1 = Yes)
    high_risk_exposure_occupation: int = 0
    high_risk_interactions: int = 0
    diabetes: int = 0
    chd: int = 0
    htn: int = 0
    cancer: int = 0
    asthma: int = 0
    copd: int = 0

    # Symptoms (0 = No, 1 = Yes)
    cough: int = 0
    fever: int = 0
    sob: int = 0
    fatigue: int = 0
    headache: int = 0
    loss_of_smell: int = 0
    loss_of_taste: int = 0
    sore_throat: int = 0


class ClinicalPredictionResponse(BaseModel):
    prediction: str
    probability_positive: float


class ImagePredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: dict


class CombinedPredictionResponse(BaseModel):
    image_result: Optional[ImagePredictionResponse] = None
    clinical_result: Optional[ClinicalPredictionResponse] = None