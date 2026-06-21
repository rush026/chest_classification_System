# ==========================================================
# src/severity.py
# Computes a 0-100 COVID-19 risk/severity score from:
#   - Clinical model probability
#   - Image model confidence + predicted class
#   - Raw vitals (SpO2, respiratory rate, temperature, pulse)
# ==========================================================

from dataclasses import dataclass
from typing import Optional


@dataclass
class SeverityResult:
    score: float              # 0-100
    level: str                # Low / Moderate / High / Critical
    color: str                # for UI display
    explanation: list         # list of contributing factors


def compute_severity(
    clinical_prob: Optional[float] = None,
    image_pred_class: Optional[str] = None,
    image_confidence: Optional[float] = None,
    sats: Optional[float] = None,
    rr: Optional[float] = None,
    temperature: Optional[float] = None,
    pulse: Optional[float] = None,
) -> SeverityResult:
    """
    Compute a composite severity score from available signals.
    Each component contributes a weighted sub-score.
    """
    score = 0.0
    max_score = 0.0
    factors = []

    # ---- 1. Clinical model probability (weight: 35) ----
    if clinical_prob is not None:
        contribution = clinical_prob * 35
        score += contribution
        max_score += 35
        if clinical_prob >= 0.7:
            factors.append(f"Clinical vitals strongly suggest COVID-19 ({clinical_prob:.0%} probability)")
        elif clinical_prob >= 0.4:
            factors.append(f"Clinical vitals show moderate COVID-19 risk ({clinical_prob:.0%} probability)")
        else:
            factors.append(f"Clinical vitals show low COVID-19 risk ({clinical_prob:.0%} probability)")

    # ---- 2. Image model prediction (weight: 35) ----
    if image_pred_class is not None and image_confidence is not None:
        class_weights = {
            "COVID": 1.0,
            "Viral Pneumonia": 0.7,
            "Lung_Opacity": 0.5,
            "Normal": 0.0,
        }
        weight = class_weights.get(image_pred_class, 0.5)
        contribution = weight * image_confidence * 35
        score += contribution
        max_score += 35
        if image_pred_class == "COVID":
            factors.append(f"Chest X-ray classified as COVID-19 ({image_confidence:.0%} confidence)")
        elif image_pred_class == "Normal":
            factors.append(f"Chest X-ray appears Normal ({image_confidence:.0%} confidence)")
        else:
            factors.append(f"Chest X-ray shows {image_pred_class} ({image_confidence:.0%} confidence)")

    # ---- 3. SpO2 / Oxygen saturation (weight: 15) ----
    if sats is not None:
        max_score += 15
        if sats < 90:
            score += 15
            factors.append(f"Critically low oxygen saturation (SpO2: {sats:.0f}%)")
        elif sats < 94:
            score += 10
            factors.append(f"Low oxygen saturation (SpO2: {sats:.0f}%)")
        elif sats < 96:
            score += 5
            factors.append(f"Slightly low oxygen saturation (SpO2: {sats:.0f}%)")

    # ---- 4. Respiratory rate (weight: 8) ----
    if rr is not None:
        max_score += 8
        if rr > 30:
            score += 8
            factors.append(f"Severe tachypnea (RR: {rr:.0f} breaths/min)")
        elif rr > 24:
            score += 5
            factors.append(f"Elevated respiratory rate (RR: {rr:.0f} breaths/min)")
        elif rr > 20:
            score += 2
            factors.append(f"Mildly elevated respiratory rate (RR: {rr:.0f} breaths/min)")

    # ---- 5. Temperature (weight: 4) ----
    if temperature is not None:
        max_score += 4
        if temperature >= 39.5:
            score += 4
            factors.append(f"High fever (Temp: {temperature:.1f}°C)")
        elif temperature >= 38.0:
            score += 2
            factors.append(f"Fever present (Temp: {temperature:.1f}°C)")

    # ---- 6. Pulse (weight: 3) ----
    if pulse is not None:
        max_score += 3
        if pulse > 110:
            score += 3
            factors.append(f"Elevated heart rate (Pulse: {pulse:.0f} bpm)")
        elif pulse > 100:
            score += 1
            factors.append(f"Mildly elevated heart rate (Pulse: {pulse:.0f} bpm)")

    # Normalize to 0-100
    if max_score > 0:
        normalized = (score / max_score) * 100
    else:
        normalized = 0.0

    # Determine level
    if normalized >= 75:
        level, color = "Critical", "#d32f2f"
    elif normalized >= 50:
        level, color = "High", "#f57c00"
    elif normalized >= 25:
        level, color = "Moderate", "#fbc02d"
    else:
        level, color = "Low", "#388e3c"

    if not factors:
        factors.append("Insufficient data to determine contributing factors.")

    return SeverityResult(
        score=round(normalized, 1),
        level=level,
        color=color,
        explanation=factors,
    )