# ==========================================================
# src/report.py
# Generates a downloadable PDF clinical report
# pip install reportlab
# ==========================================================

import io
import base64
from datetime import datetime
from typing import Optional
from src.severity import SeverityResult

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_report_pdf(
    patient_name: str,
    age: float,
    vitals: dict,
    symptoms: dict,
    clinical_result: Optional[dict] = None,
    image_result: Optional[dict] = None,
    severity: Optional[SeverityResult] = None,
    gradcam_base64: Optional[str] = None,
    chatbot_explanation: Optional[str] = None,
) -> bytes:
    """Generate a PDF clinical report and return as bytes."""

    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab not installed. Run: pip install reportlab")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=18, spaceAfter=6)
    h2_style = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceAfter=4, textColor=colors.HexColor("#1565C0"))
    normal = styles["Normal"]
    small = ParagraphStyle("small", parent=normal, fontSize=9, textColor=colors.grey)

    story = []

    # ---- HEADER ----
    story.append(Paragraph("COVID-19 Chest Classification Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", small))
    story.append(Paragraph("⚠️ This report is for educational/research purposes only — not a medical diagnosis.", small))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1565C0")))
    story.append(Spacer(1, 0.3*cm))

    # ---- PATIENT INFO ----
    story.append(Paragraph("Patient Information", h2_style))
    patient_data = [
        ["Name", patient_name, "Age", f"{int(age)} years"],
        ["Date", datetime.now().strftime("%Y-%m-%d"), "Report ID", f"RPT-{datetime.now().strftime('%H%M%S')}"],
    ]
    pt = Table(patient_data, colWidths=[3*cm, 6*cm, 3*cm, 5*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E3F2FD")),
        ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#E3F2FD")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.4*cm))

    # ---- VITALS ----
    story.append(Paragraph("Clinical Vitals", h2_style))
    vitals_data = [["Parameter", "Value", "Normal Range"]] + [
        ["Temperature", f"{vitals.get('temperature', 'N/A')} °C", "36.1 – 37.2 °C"],
        ["Pulse", f"{vitals.get('pulse', 'N/A')} bpm", "60 – 100 bpm"],
        ["Systolic BP", f"{vitals.get('sys', 'N/A')} mmHg", "90 – 120 mmHg"],
        ["Diastolic BP", f"{vitals.get('dia', 'N/A')} mmHg", "60 – 80 mmHg"],
        ["Respiratory Rate", f"{vitals.get('rr', 'N/A')} breaths/min", "12 – 20 breaths/min"],
        ["SpO2", f"{vitals.get('sats', 'N/A')} %", "≥ 95 %"],
    ]
    vt = Table(vitals_data, colWidths=[5*cm, 5*cm, 7*cm])
    vt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1565C0")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(vt)
    story.append(Spacer(1, 0.4*cm))

    # ---- SEVERITY SCORE ----
    if severity:
        story.append(Paragraph("Risk / Severity Assessment", h2_style))
        sev_color = colors.HexColor(severity.color)
        sev_data = [
            ["Severity Score", f"{severity.score}/100", "Level", severity.level],
        ]
        st2 = Table(sev_data, colWidths=[4*cm, 4*cm, 4*cm, 5*cm])
        st2.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,0), colors.HexColor("#E3F2FD")),
            ("BACKGROUND", (1,0), (1,0), sev_color),
            ("TEXTCOLOR", (1,0), (1,0), colors.white),
            ("FONTNAME", (0,0), (-1,-1), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 11),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("PADDING", (0,0), (-1,-1), 8),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ]))
        story.append(st2)
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Contributing factors:", normal))
        for factor in severity.explanation:
            story.append(Paragraph(f"• {factor}", normal))
        story.append(Spacer(1, 0.4*cm))

    # ---- CLINICAL MODEL RESULT ----
    if clinical_result:
        story.append(Paragraph("Clinical Model Prediction", h2_style))
        cd = [
            ["Prediction", clinical_result.get("prediction", "N/A")],
            ["COVID-19 Probability", f"{clinical_result.get('probability_positive', 0):.1%}"],
        ]
        ct = Table(cd, colWidths=[6*cm, 11*cm])
        ct.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E3F2FD")),
            ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("PADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(ct)
        story.append(Spacer(1, 0.4*cm))

    # ---- IMAGE MODEL RESULT ----
    if image_result:
        story.append(Paragraph("Chest X-Ray Analysis", h2_style))
        id_ = [
            ["Predicted Class", image_result.get("predicted_class", "N/A")],
            ["Confidence", f"{image_result.get('confidence', 0):.1%}"],
        ]
        for cls, prob in image_result.get("class_probabilities", {}).items():
            id_.append([f"  {cls}", f"{prob:.1%}"])
        it = Table(id_, colWidths=[6*cm, 11*cm])
        it.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E3F2FD")),
            ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("PADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(it)
        story.append(Spacer(1, 0.4*cm))

    # ---- GRAD-CAM IMAGE ----
    if gradcam_base64:
        story.append(Paragraph("Grad-CAM Chest X-Ray Overlay", h2_style))
        story.append(Paragraph("Highlighted regions indicate areas the model focused on for its prediction.", small))
        img_bytes = base64.b64decode(gradcam_base64)
        img_buffer = io.BytesIO(img_bytes)
        rl_img = RLImage(img_buffer, width=10*cm, height=10*cm)
        story.append(rl_img)
        story.append(Spacer(1, 0.4*cm))

    # ---- AI EXPLANATION ----
    if chatbot_explanation:
        story.append(Paragraph("AI-Generated Explanation", h2_style))
        story.append(Paragraph(chatbot_explanation, normal))
        story.append(Spacer(1, 0.4*cm))

    # ---- DISCLAIMER ----
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "DISCLAIMER: This report was generated by an AI-based educational research tool. "
        "It is NOT a substitute for professional medical diagnosis, advice, or treatment. "
        "Always consult a qualified healthcare provider for medical decisions.",
        ParagraphStyle("disclaimer", parent=normal, fontSize=8, textColor=colors.grey)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()