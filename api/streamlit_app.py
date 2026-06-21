# ==========================================================
# app/streamlit_app.py
# Streamlit frontend - Full featured version with:
#   - Clinical vitals prediction
#   - Chest X-ray prediction + Grad-CAM
#   - Disease severity score
#   - PDF report generation
#   - Claude AI chatbot explanation
#   - Patient history dashboard
# ==========================================================

import streamlit as st
import requests
import base64
import sys
import os
from io import BytesIO
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.severity import compute_severity
from src.history import save_prediction, get_all_predictions, get_patient_trend, init_db
from src.report import generate_report_pdf, REPORTLAB_AVAILABLE

API_URL = "http://127.0.0.1:8000"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

st.set_page_config(page_title="Chest Classification System", layout="wide", page_icon="🫁")

# Init DB
init_db()

# ---- SIDEBAR ----
with st.sidebar:
    st.title("🫁 Chest Classification System")
    st.caption("Educational/research tool")
    st.divider()

    # API status
    try:
        status = requests.get(f"{API_URL}/", timeout=3).json()
        clinical_ready = status.get("clinical_model_loaded", False)
        image_ready = status.get("image_model_loaded", False)
        st.success("Online")
    except:
        st.error(f"API offline\nStart: `uvicorn api.main:app --reload`")
        clinical_ready = False
        image_ready = False

   # st.markdown(f"Clinical model: {'✅' if clinical_ready else '❌'}")
    #st.markdown(f"Image model: {'✅' if image_ready else '❌'}")
    st.divider()

    # Patient info (shared across tabs)
    st.subheader("Patient Info")
    patient_name = st.text_input("Patient Name", value="Anonymous")
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    st.divider()

   # OpenAI API key
    st.subheader("AI Explanation (OpenAI)")
    api_key_input = st.text_input("OpenAI API Key", type="password",
                                   value=os.environ.get("OPENAI_API_KEY", ""),
                                   help="Get yours at platform.openai.com/api-keys")
# ---- MAIN TABS ----
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Clinical + X-Ray Prediction",
    "📊 Patient History Dashboard",
    "💬 Ask About Results",
    "ℹ️ About"
])

# ==========================================================
# TAB 1: PREDICTION
# ==========================================================
with tab1:
    col_left, col_right = st.columns([1, 1])

    # ---- LEFT: Clinical vitals ----
    with col_left:
        st.subheader("Clinical Vitals & Symptoms")

        c1, c2 = st.columns(2)
        with c1:
            temperature = st.number_input("Temperature (°C)", 30.0, 43.0, 37.0, 0.1)
            pulse = st.number_input("Pulse (bpm)", 30, 200, 80)
            sys_bp = st.number_input("Systolic BP", 60, 220, 120)
        with c2:
            dia_bp = st.number_input("Diastolic BP", 40, 140, 80)
            rr = st.number_input("Respiratory Rate", 5, 60, 16)
            sats = st.number_input("SpO2 (%)", 50, 100, 98)

        st.markdown("**Symptoms**")
        s1, s2 = st.columns(2)
        with s1:
            cough = st.checkbox("Cough")
            fever = st.checkbox("Fever")
            sob = st.checkbox("Shortness of breath")
            fatigue = st.checkbox("Fatigue")
        with s2:
            headache = st.checkbox("Headache")
            loss_of_smell = st.checkbox("Loss of smell")
            loss_of_taste = st.checkbox("Loss of taste")
            sore_throat = st.checkbox("Sore throat")

        with st.expander("Risk factors / Comorbidities"):
            r1, r2 = st.columns(2)
            with r1:
                high_risk_exp = st.checkbox("High-risk occupation")
                high_risk_int = st.checkbox("High-risk interactions")
                diabetes = st.checkbox("Diabetes")
                chd = st.checkbox("Heart disease")
            with r2:
                htn = st.checkbox("Hypertension")
                cancer = st.checkbox("Cancer")
                asthma = st.checkbox("Asthma")
                copd = st.checkbox("COPD")

    # ---- RIGHT: X-Ray upload ----
    with col_right:
        st.subheader("Chest X-Ray")
        uploaded_file = st.file_uploader("Upload X-ray image", type=["png", "jpg", "jpeg"])
        show_gradcam = st.checkbox("Show Grad-CAM explanation", value=True)
        if uploaded_file:
            st.image(Image.open(uploaded_file), caption="Uploaded X-Ray", use_container_width=True)

    st.divider()

    # ---- PREDICT BUTTON ----
    notes = st.text_area("Notes (optional)", placeholder="Add any clinical notes here...")
    predict_btn = st.button("🔍 Run Full Prediction", type="primary", use_container_width=True)

    if predict_btn:
        clinical_result = None
        image_result = None
        gradcam_b64 = None

        vitals = {
            "temperature": temperature, "pulse": pulse,
            "sys": sys_bp, "dia": dia_bp, "rr": rr, "sats": sats
        }

        # -- Clinical prediction --
        if clinical_ready:
            with st.spinner("Running clinical model..."):
                payload = {
                    "age": age, "temperature": temperature, "pulse": pulse,
                    "sys": sys_bp, "dia": dia_bp, "rr": rr, "sats": sats,
                    "cough": int(cough), "fever": int(fever), "sob": int(sob),
                    "fatigue": int(fatigue), "headache": int(headache),
                    "loss_of_smell": int(loss_of_smell), "loss_of_taste": int(loss_of_taste),
                    "sore_throat": int(sore_throat),
                    "high_risk_exposure_occupation": int(high_risk_exp),
                    "high_risk_interactions": int(high_risk_int),
                    "diabetes": int(diabetes), "chd": int(chd),
                    "htn": int(htn), "cancer": int(cancer),
                    "asthma": int(asthma), "copd": int(copd),
                }
                try:
                    resp = requests.post(f"{API_URL}/predict/clinical", json=payload, timeout=15)
                    resp.raise_for_status()
                    clinical_result = resp.json()
                except Exception as e:
                    st.error(f"Clinical model error: {e}")

        # -- Image prediction --
        if image_ready and uploaded_file:
            endpoint = "/predict/image/gradcam" if show_gradcam else "/predict/image"
            with st.spinner("Running image model..."):
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
                    resp = requests.post(f"{API_URL}{endpoint}", files=files, timeout=60)
                    resp.raise_for_status()
                    image_result = resp.json()
                    gradcam_b64 = image_result.pop("gradcam_overlay_png_base64", None)
                except Exception as e:
                    st.error(f"Image model error: {e}")

        # -- Severity score --
        severity = compute_severity(
            clinical_prob=clinical_result.get("probability_positive") if clinical_result else None,
            image_pred_class=image_result.get("predicted_class") if image_result else None,
            image_confidence=image_result.get("confidence") if image_result else None,
            sats=sats, rr=rr, temperature=temperature, pulse=pulse,
        )
        # Save context for Q&A tab
        context_lines = [f"Patient: {patient_name}, Age: {age}"]
        context_lines.append(f"Vitals: Temp={temperature}°C, Pulse={pulse}bpm, SpO2={sats}%, RR={rr}, BP={sys_bp}/{dia_bp}")
        if clinical_result:
            context_lines.append(f"Clinical model: {clinical_result['prediction']} ({clinical_result['probability_positive']:.1%} probability)")
        if image_result:
            context_lines.append(f"Chest X-ray: {image_result['predicted_class']} ({image_result['confidence']:.1%} confidence)")
        context_lines.append(f"Severity score: {severity.score}/100 ({severity.level})")
        context_lines.append(f"Contributing factors: {'; '.join(severity.explanation)}")
        st.session_state.last_result_context = "\n".join(context_lines)
        st.session_state.chat_history = []  # reset chat for new results

        # -- Claude AI explanation --
        ai_explanation = None
        if api_key_input:
            with st.spinner("Generating AI explanation..."):
                prompt_parts = [f"Patient: {patient_name}, Age: {age}"]
                prompt_parts.append(f"Vitals: Temp={temperature}°C, Pulse={pulse}bpm, SpO2={sats}%, RR={rr}")
                if clinical_result:
                    prompt_parts.append(f"Clinical model: {clinical_result['prediction']} ({clinical_result['probability_positive']:.1%} probability)")
                if image_result:
                    prompt_parts.append(f"Chest X-ray: {image_result['predicted_class']} ({image_result['confidence']:.1%} confidence)")
                prompt_parts.append(f"Severity: {severity.level} ({severity.score}/100)")

                prompt = "\n".join(prompt_parts) + "\n\nPlease explain these results in simple, clear language for a healthcare professional. Include what the findings suggest, key risk factors, and recommended next steps. Keep it concise (3-4 paragraphs)."

                try:
                    resp = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key_input}", "content-type": "application/json"},
                        json={
                            "model": "gpt-4o-mini",
                            "max_tokens": 500,
                            "messages": [{"role": "user", "content": prompt}]
                        },
                        timeout=30
                    )
                    resp.raise_for_status()
                    ai_explanation = resp.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    st.warning(f"AI explanation unavailable: {e}")

        # ---- DISPLAY RESULTS ----
        st.divider()
        st.subheader("Results")

        # Severity score banner
        sev_col1, sev_col2, sev_col3 = st.columns(3)
        sev_col1.metric("Severity Score", f"{severity.score}/100")
        sev_col2.metric("Risk Level", severity.level)
        if clinical_result:
            sev_col3.metric("COVID Probability", f"{clinical_result['probability_positive']:.1%}")

        # Contributing factors
        with st.expander("Severity contributing factors"):
            for factor in severity.explanation:
                st.write(f"• {factor}")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if clinical_result:
                st.markdown("**Clinical Model**")
                pred = clinical_result["prediction"]
                prob = clinical_result["probability_positive"]
                if pred == "Positive":
                    st.error(f"Prediction: {pred} ({prob:.1%})")
                else:
                    st.success(f"Prediction: {pred} ({prob:.1%})")
                st.progress(prob)

            if image_result:
                st.markdown("**X-Ray Model**")
                st.info(f"Class: {image_result['predicted_class']} ({image_result['confidence']:.1%})")
                for cls, prob in image_result.get("class_probabilities", {}).items():
                    st.write(f"{cls}: {prob:.1%}")
                    st.progress(prob)

        with res_col2:
            if gradcam_b64:
                st.markdown("**Grad-CAM Overlay**")
                overlay = Image.open(BytesIO(base64.b64decode(gradcam_b64)))
                st.image(overlay, use_container_width=True)

      

        # ---- SAVE TO HISTORY ----
        record_id = save_prediction(
            patient_name=patient_name,
            age=age,
            vitals=vitals,
            clinical_result=clinical_result,
            image_result=image_result,
            severity_score=severity.score,
            severity_level=severity.level,
            notes=notes,
        )
        st.caption(f"Saved to history (Record #{record_id})")

        # ---- PDF REPORT ----
        # ---- PDF REPORT ----
        # ---- PDF REPORT (auto-generated immediately, no extra button) ----
        if REPORTLAB_AVAILABLE:
            st.divider()
            st.subheader("📄 Download Report")
            with st.spinner("Generating PDF..."):
                pdf_bytes = generate_report_pdf(
                    patient_name=patient_name,
                    age=age,
                    vitals=vitals,
                    symptoms={"cough": cough, "fever": fever, "sob": sob, "fatigue": fatigue},
                    clinical_result=clinical_result,
                    image_result=image_result,
                    severity=severity,
                    gradcam_base64=gradcam_b64,
                    chatbot_explanation=None,
                )
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"covid_report_{patient_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )
        else:
            st.info("Install reportlab for PDF reports: `pip install reportlab`")


# ==========================================================
# TAB 2: PATIENT HISTORY DASHBOARD
# ==========================================================
with tab2:
    st.subheader("📊 Patient History Dashboard")

    col_filter, col_btn = st.columns([3, 1])
    with col_filter:
        search_name = st.text_input("Search by patient name (leave blank for all)")
    with col_btn:
        st.write("")
        refresh = st.button("🔄 Refresh")

    records = get_all_predictions(search_name if search_name else None)

    if not records:
        st.info("No prediction records yet. Run a prediction to see history here.")
    else:
        st.markdown(f"**{len(records)} record(s) found**")

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        positive_count = sum(1 for r in records if r.get("clinical_prediction") == "Positive")
        covid_xray = sum(1 for r in records if r.get("image_prediction") == "COVID")
        avg_severity = sum(r.get("severity_score") or 0 for r in records) / len(records)
        critical_count = sum(1 for r in records if r.get("severity_level") in ["High", "Critical"])

        m1.metric("Total Records", len(records))
        m2.metric("Clinical Positives", positive_count)
        m3.metric("COVID X-Rays", covid_xray)
        m4.metric("Avg Severity Score", f"{avg_severity:.1f}")

        st.divider()

        # Records table
        import pandas as pd
        df = pd.DataFrame(records)
        display_cols = ["id", "timestamp", "patient_name", "age",
                        "sats", "clinical_prediction", "clinical_probability",
                        "image_prediction", "severity_score", "severity_level"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)

        # Trend chart for specific patient
        if search_name:
            trend = get_patient_trend(search_name)
            if len(trend) > 1:
                st.subheader(f"Severity Trend: {search_name}")
                trend_df = pd.DataFrame(trend)
                trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"])
                trend_df = trend_df.set_index("timestamp")
                st.line_chart(trend_df[["severity_score", "sats"]])
                # ==========================================================
# TAB 3: ASK ABOUT RESULTS (Q&A Chatbot)
# ==========================================================
with tab3:
    st.subheader("💬 Ask Questions About Your Results")

    if "last_result_context" not in st.session_state:
        st.session_state.last_result_context = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.session_state.last_result_context is None:
        st.info("Run a prediction in the first tab to enable Q&A about your results.")
    else:
        st.caption("Ask questions about the most recent prediction results.")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_question = st.chat_input("Ask a question about these results...")

        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if not OPENAI_API_KEY:
                        answer = "No OpenAI API key configured. Add OPENAI_API_KEY to your .env file."
                    else:
                        system_context = (
                            "You are a medical AI assistant explaining COVID-19 screening results "
                            "to a healthcare professional. Here are the patient's latest results:\n\n"
                            f"{st.session_state.last_result_context}\n\n"
                            "Answer the user's question clearly and concisely based on this data. "
                            "Always remind them this is not a substitute for professional medical diagnosis "
                            "if the question is about treatment or diagnosis decisions."
                        )
                        messages = [{"role": "system", "content": system_context}]
                        messages += st.session_state.chat_history

                        try:
                            resp = requests.post(
                                "https://api.openai.com/v1/chat/completions",
                                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "content-type": "application/json"},
                                json={"model": "gpt-4o-mini", "max_tokens": 400, "messages": messages},
                                timeout=30
                            )
                            resp.raise_for_status()
                            answer = resp.json()["choices"][0]["message"]["content"]
                        except Exception as e:
                            answer = f"Error: {e}"

                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ==========================================================
# TAB 4: ABOUT
# ==========================================================
with tab4:
    st.subheader("About This Project")
    st.markdown("""
    ### COVID-19 Chest Classification System

    A multimodal AI system that combines **clinical vitals** and **chest X-ray imaging**
    to assess COVID-19 risk, built as a data science portfolio project.

    **Models used:**
    - **Clinical model**: Random Forest trained on ~94,000 patient records from the
      COVID-19 Clinical Data Repository (Carbon Health & Braid Health)
    - **Image model**: MobileNetV2 (transfer learning) trained on the COVID-19
      Radiography Database (Kaggle) — 4 classes: COVID, Normal, Lung Opacity, Viral Pneumonia

    **Features:**
    - Clinical vitals + symptom-based COVID-19 probability
    - Chest X-ray classification with Grad-CAM explainability
    - Composite severity score (0–100) from multiple signals
    - Downloadable PDF clinical report
    - Claude AI-powered result explanation
    - Patient history dashboard with trend tracking

    **Limitations:**
    - The clinical dataset has ~1.4% positive rate — high class imbalance
    - Image and clinical datasets are not patient-matched
    - This is an educational tool, NOT a medical diagnostic device

    **Tech stack:** Python, TensorFlow/Keras, scikit-learn, FastAPI, Streamlit, SQLite, ReportLab, Claude API

    ---
    ⚠️ *This system is for educational and portfolio purposes only.
    It is not approved for clinical use and should not be used for medical decision-making.*
    """)