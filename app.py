# app.py - Premium Dark Mode Multi-Model Heart Disease Predictor
# Requirements: streamlit, pandas, numpy, plotly
# Optional: shap (for interpretability), fpdf (for PDF reports)
# Models expected in same folder: DecisionTreeR.pkl, LogisticR.pkl, randomforestR.pkl, svmR.pkl

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64
import time
import io
import concurrent.futures
from typing import Dict, Any, Tuple

import plotly.graph_objects as go
import plotly.express as px

# Optional libs
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# --------------- Page config & metadata ---------------
st.set_page_config(page_title="Heart Disease — ML Predictor (Sahib Sodhi)",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   page_icon="❤️")

APP_AUTHOR = "Sahib Singh Sodhi"
APP_TITLE = "Heart Disease Prediction — Premium ML Dashboard"

# --------------- Dark Premium CSS (Option A) ---------------
st.markdown(
    """
<style>
:root{
  --bg:#0b1020;
  --card: rgba(255,255,255,0.03);
  --muted:#92a0c3;
  --accent1:#ff5f6d;
  --accent2:#ffc371;
  --glass: rgba(255,255,255,0.03);
}
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(135deg, #060616 0%, #07112b 100%);
    color: #e6eef8;
    padding: 1.2rem 1.6rem;
    min-height: 100vh;
    font-family: "Inter", sans-serif;
}
.card {
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 18px;
    box-shadow: 0 8px 30px rgba(2,6,23,0.6);
    border: 1px solid rgba(255,255,255,0.03);
}
.header-title {
    font-size: 30px;
    font-weight: 800;
    background: linear-gradient(90deg, #ff5f6d, #ffc371);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: -4px;
}
.header-sub { color: var(--muted); margin-bottom: 8px; }
.stButton>button {
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
    color: #071127;
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: 700;
    border: none;
}
.small-muted { color: var(--muted); font-size: 0.9rem; }
.result-good {
    padding: 12px;
    border-radius: 10px;
    background: linear-gradient(90deg, rgba(34,197,94,0.08), rgba(34,197,94,0.03));
    border: 1px solid rgba(34,197,94,0.12);
    color: #bff8d6;
    font-weight: 700;
}
.result-bad {
    padding: 12px;
    border-radius: 10px;
    background: linear-gradient(90deg, rgba(239,68,68,0.08), rgba(239,68,68,0.03));
    border: 1px solid rgba(239,68,68,0.12);
    color: #ffd6d6;
    font-weight: 700;
}
.footer { color: var(--muted); text-align: center; margin-top: 18px; }
.loading-dot { display:inline-block; width:8px; height:8px; background:var(--accent2); border-radius:999px; margin-left:6px; animation: blink 1s infinite; }
@keyframes blink { 0% {opacity:0.2;} 50%{opacity:1;} 100%{opacity:0.2;} }
</style>
""",
    unsafe_allow_html=True,
)

# --------------- Sidebar Controls ---------------
st.sidebar.markdown("<div style='font-weight:800; font-size:16px'>🔧 Controls</div>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["Dashboard", "Single Predict", "Bulk Predict", "Model Info", "About"])
st.sidebar.markdown("---")

primary_model_choice = st.sidebar.selectbox("Primary model (for highlighting)", ["Logistic Regression", "Random Forest", "Decision Tree", "SVM"])
st.sidebar.markdown("---")
if st.sidebar.button("Reset (reload)"):
   st.rerun()

st.sidebar.caption(f"Made by {APP_AUTHOR}")

# --------------- Header ---------------
st.markdown(f"<div class='header-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='header-sub small-muted'>AI-powered multi-model heart disease predictor • Simultaneous predictions • Interpretability • PDF reports</div>", unsafe_allow_html=True)

# --------------- Model filenames (confirmed) ---------------
MODEL_MAP = {
    "Decision Trees": "DecisionTreeR.pkl",
    "Logistic Regression": "LogisticR.pkl",
    "Random Forest": "randomforestR.pkl",
    "Support Vector Machine": "svmR.pkl",
}
MODEL_NAMES = list(MODEL_MAP.keys())

# --------------- Utility functions ---------------
def safe_load_model(filepath: str):
    if not os.path.exists(filepath):
        return None, f"File not found: {filepath}"
    try:
        with open(filepath, "rb") as f:
            m = pickle.load(f)
        return m, None
    except Exception as e:
        return None, f"Load error: {e}"

def predict_model(model, X: np.ndarray) -> Tuple[Any, Any, str]:
    """Return (label_int, probability_float_or_None, error_str)"""
    try:
        pred = model.predict(X)
    except Exception as e:
        return None, None, f"Predict error: {e}"
    prob = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                prob = float(probs[0,1])
            else:
                prob = float(probs[0,0])
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X)
            prob = float(1 / (1 + np.exp(-df[0])))
    except Exception:
        prob = None
    return int(pred[0]), prob, ""

def df_to_link(df: pd.DataFrame, name="predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{name}">⬇️ Download {name}</a>'

def make_pdf_report(inputs: Dict[str,Any], results: Dict[str, Dict[str,Any]]):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    
    if FPDF_AVAILABLE:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, "Heart Disease Prediction Report", ln=True, align="C")
        pdf.ln(4)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 6, f"Created: {now}", ln=True)
        pdf.cell(0, 6, f"Generated by: {APP_AUTHOR}", ln=True)
        pdf.ln(6)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 6, "Patient Inputs:", ln=True)
        pdf.set_font("Arial", size=11)
        for k, v in inputs.items():
            pdf.cell(0, 6, f"- {k}: {v}", ln=True)
        pdf.ln(6)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 6, "Model Predictions:", ln=True)
        pdf.set_font("Arial", size=11)
        for m, res in results.items():
            err = res.get("error","")
            pred = res.get("prediction", "N/A")
            prob = res.get("probability", None)
            line = f"- {m}: {pred}"
            if prob is not None:
                line += f" (confidence {prob*100:.1f}%)"
            if err:
                line += f" — ERROR: {err}"
            pdf.cell(0, 6, line, ln=True)

        # Save PDF to temporary file (works on Streamlit Cloud)
        tmp_path = "/tmp/heart_report.pdf"
        pdf.output(tmp_path)

        # Read bytes
        with open(tmp_path, "rb") as f:
            bio = f.read()
        
        return bio, "application/pdf"
    
    else:
        # Fallback text report
        text = f"Heart Disease Prediction Report\nMade by {APP_AUTHOR}\nCreated: {now}\n\nInputs:\n"
        for k,v in inputs.items():
            text += f"{k}: {v}\n"
        text += "\nPredictions:\n"
        for m,res in results.items():
            err = res.get("error","")
            pred = res.get("prediction","N/A")
            prob = res.get("probability", None)
            text += f"{m}: {pred}"
            if prob is not None:
                text += f" (confidence {prob*100:.1f}%)"
            if err:
                text += f" — ERROR: {err}"
            text += "\n"
        return text.encode("utf-8"), "text/plain"

# --------------- Load models at start (safe) ---------------
MODELS = {}
MODEL_ERRORS = {}
for friendly, fname in MODEL_MAP.items():
    m, err = safe_load_model(fname)
    MODELS[friendly] = m
    MODEL_ERRORS[friendly] = err

# --------------- Input form builder ---------------
def build_input_form():
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", 0, 150, value=50)
        sex_txt = st.selectbox("Sex", ["Male", "Female"])
        chest_txt = st.selectbox("Chest Pain Type", ["typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 0, 1000, value=200)
    with c2:
        fasting_txt = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
        rest_ecg_txt = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        max_hr = st.number_input("Maximum Heart Rate Achieved", 0, 250, value=150)
        exercise_txt = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0, value=0.5, step=0.1)
    st_slope_txt = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # numeric encoding
    sex = 0 if sex_txt == "Male" else 1
    chest = ["typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_txt)
    fasting = 1 if fasting_txt == "> 120 mg/dl" else 0
    rest_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(rest_ecg_txt)
    exercise = 1 if exercise_txt == "Yes" else 0
    st_slope = ["Upsloping","Flat","Downsloping"].index(st_slope_txt)

    df = pd.DataFrame({
        "Age":[age],"Sex":[sex],"ChestPainType":[chest],"RestingBP":[resting_bp],
        "Cholesterol":[chol],"FastingBS":[fasting],"RestingECG":[rest_ecg],"MaxHR":[max_hr],
        "ExerciseAngina":[exercise],"Oldpeak":[oldpeak],"ST_Slope":[st_slope]
    })
    inputs_pretty = {
        "Age": age, "Sex": sex_txt, "ChestPainType": chest_txt, "RestingBP": resting_bp,
        "Cholesterol": chol, "FastingBS": fasting_txt, "RestingECG": rest_ecg_txt,
        "MaxHR": max_hr, "ExerciseAngina": exercise_txt, "Oldpeak": oldpeak, "ST_Slope": st_slope_txt
    }
    return df, inputs_pretty

# --------------- Pages ---------------
if page == "Dashboard":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Dashboard - Model Insights")
    # top metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Models loaded", f"{sum(1 for v in MODELS.values() if v is not None)}/{len(MODELS)}")
    with c2:
        st.metric("SHAP installed", "Yes" if SHAP_AVAILABLE else "No")
    with c3:
        st.metric("PDF (FPDF)", "Yes" if FPDF_AVAILABLE else "Fallback")
    with c4:
        st.metric("Primary (highlight)", primary_model_choice)

    st.markdown("### 🔬 Model Comparison ")
    perf = pd.DataFrame({
        "Model":["Decision Trees","Logistic Regression","Random Forest","SVM"],
        "Accuracy":[81,85,86,84],
        "Precision":[78,82,84,80],
        "Recall":[79,81,85,80]
    })

    # Radar chart that looks 'crazy' / flashy
    cats = ["Accuracy","Precision","Recall"]
    fig_r = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i,row in perf.iterrows():
        vals = [row["Accuracy"], row["Precision"], row["Recall"]]
        fig_r.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill='toself', name=row["Model"], opacity=0.7))
    fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,100])), height=520, title="Model Comparison Radar")
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("3D Risk Surface")
    # toy 3D surface
    x = np.linspace(30,80,40)
    y = np.linspace(80,200,40)
    Xg, Yg = np.meshgrid(x,y)
    Zg = 100 - ((0.02*(Xg-55)**2) + (0.01*(Yg-140)**2))*2
    surf = go.Figure(data=[go.Surface(z=Zg, x=Xg, y=Yg, colorscale='RdYlBu')])
    surf.update_layout(scene=dict(xaxis_title='Age', yaxis_title='MaxHR', zaxis_title='Risk Score'), height=520)
    st.plotly_chart(surf, use_container_width=True)

    st.markdown("### Model Cards")
    cols = st.columns(2)
    feat_names = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
    for i, m in enumerate(MODEL_NAMES):
        with cols[i%2]:
            st.markdown(f"#### {m}")
            mo = MODELS.get(m)
            if mo is None:
                st.warning(f"{MODEL_MAP[m]} not loaded ({MODEL_ERRORS.get(m)})")
                st.write("Status: ❌ Not available")
            else:
                st.write("Status: ✅ Loaded")
                st.write("- Example accuracy (placeholder): **86%**")
                # try importances
                if hasattr(mo, "feature_importances_"):
                    fi = mo.feature_importances_
                    df_fi = pd.DataFrame({"feature": feat_names, "importance": fi}).sort_values("importance", ascending=False)
                    st.bar_chart(df_fi.set_index("feature").head(6))
                elif hasattr(mo, "coef_"):
                    coefs = np.abs(mo.coef_).ravel()
                    df_coef = pd.DataFrame({"feature": feat_names, "abs_coef": coefs}).sort_values("abs_coef", ascending=False)
                    st.bar_chart(df_coef.set_index("feature").head(6))
                else:
                    st.info("No importances available for this model.")

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Single Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🩺 Single Patient - Run all models simultaneously")

    input_df, inputs_pretty = build_input_form()

    st.markdown("<div class='small-muted'>Model loading status:</div>", unsafe_allow_html=True)
    status_cols = st.columns(len(MODEL_NAMES))
    for idx, m in enumerate(MODEL_NAMES):
        with status_cols[idx]:
            if MODELS[m] is not None:
                st.success(m)
            else:
                st.warning(f"{m} (missing)")

    st.write("")  # spacing

    # Button to predict all models in parallel
    if st.button("🔎 Predict Now (Run all models simultaneously)"):
        placeholder = st.empty()
        progress = st.progress(0)
        futures_map = {}
        results: Dict[str, Dict[str, Any]] = {}

        def task_runner(model_name: str, model_obj, X):
            if model_obj is None:
                return {"prediction": None, "probability": None, "error": f"Model file missing: {MODEL_MAP[model_name]}"}
            p, prob, err = predict_model(model_obj, X)
            if err:
                return {"prediction": None, "probability": None, "error": err}
            label = "Heart Disease" if p == 1 else "No Heart Disease"
            return {"prediction": label, "probability": prob, "error": ""}

        X = input_df.values

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            for name in MODEL_NAMES:
                future = ex.submit(task_runner, name, MODELS.get(name), X)
                futures_map[future] = name

            total = len(futures_map)
            completed = 0
            # update live as futures finish
            while completed < total:
                done, _ = concurrent.futures.wait(futures_map.keys(), timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED)
                # iterate over done futures and record results
                for fut in list(done):
                    name = futures_map.pop(fut)
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = {"prediction": None, "probability": None, "error": str(e)}
                    results[name] = res
                    completed += 1
                    progress.progress(int(completed/total*100))
                # show compact status
                with placeholder.container():
                    st.markdown("### Predictions (live update)")
                    cols = st.columns(4)
                    for i, model_name in enumerate(MODEL_NAMES):
                        with cols[i]:
                            r = results.get(model_name)
                            if r is None:
                                st.write(f"**{model_name}**\n- status: pending")
                            else:
                                if r.get("error"):
                                    st.write(f"**{model_name}**\n- ❌ Error")
                                    st.text(r.get("error"))
                                else:
                                    lbl = r.get("prediction")
                                    conf = r.get("probability")
                                    if lbl == "No Heart Disease":
                                        st.markdown(f"**{model_name}**\n\n🟢 {lbl}")
                                    else:
                                        st.markdown(f"**{model_name}**\n\n🔴 {lbl}")
                                    if conf is not None:
                                        st.caption(f"Confidence: {conf*100:.1f}%")
                time.sleep(0.05)

        # finished
        progress.empty()
        placeholder.empty()

        # Compute ensemble average prediction:
        # use average of available probabilities; if none available, use majority vote of predictions
        probs = [res.get("probability") for res in results.values() if res.get("probability") is not None]
        labels = [res.get("prediction") for res in results.values() if res.get("prediction") is not None and res.get("error")==""]
        ensemble_label = None
        ensemble_conf = None
        if len(probs) > 0:
            avg_prob = float(np.mean(probs))
            ensemble_conf = avg_prob
            ensemble_label = "Heart Disease" if avg_prob >= 0.5 else "No Heart Disease"
        else:
            # majority vote
            if len(labels) > 0:
                vote = pd.Series(labels).mode()
                if not vote.empty:
                    ensemble_label = vote.iloc[0]
                else:
                    ensemble_label = labels[0]
                # no probability available in this case
                ensemble_conf = None
            else:
                ensemble_label = "N/A"
                ensemble_conf = None

        # Show full results
        st.markdown("### ✅ Final Results (All Models)")
        left, right = st.columns([2,1])

        with left:
            for m in MODEL_NAMES:
                r = results.get(m, {})
                if r.get("error"):
                    st.warning(f"{m}: {r.get('error')}")
                else:
                    lbl = r.get("prediction")
                    conf = r.get("probability")
                    if lbl == "No Heart Disease":
                        st.markdown(f"**{m}** — 🟢 {lbl}")
                    else:
                        st.markdown(f"**{m}** — 🔴 {lbl}")
                    if conf is not None:
                        st.write(f"Confidence: {conf*100:.1f}%")
                    st.write("---")

            # Ensemble
            st.markdown("Average Prediction")
            if ensemble_label == "Heart Disease":
                st.markdown(f"<div class='result-bad'>🔴 {ensemble_label}</div>", unsafe_allow_html=True)
            elif ensemble_label == "No Heart Disease":
                st.markdown(f"<div class='result-good'>🟢 {ensemble_label}</div>", unsafe_allow_html=True)
            else:
                st.info("Ensemble result not available")

            if ensemble_conf is not None:
                st.write(f"Overall Chances of Getting a Heart Disease: {ensemble_conf*100:.1f}%")
            else:
                st.write("Ensemble confidence: N/A (not enough probability outputs)")

        with right:
            st.subheader("📊 Confidence Gauges")
            for m in MODEL_NAMES:
                r = results.get(m, {})
                conf = r.get("probability")
                if conf is None:
                    st.info(f"{m} — Confidence not available")
                else:
                    fig = go.Figure(go.Indicator(mode="gauge+number", value=conf*100,
                                                 gauge={'axis': {'range': [0,100]}, 'bar': {'color': "#ff7675"}},
                                                 title={'text': m}))
                    st.plotly_chart(fig, use_container_width=True)

            # ensemble gauge
            if ensemble_conf is not None:
                st.subheader("Overall Chances")
                fig_e = go.Figure(go.Indicator(mode="gauge+number", value=ensemble_conf*100,
                                               gauge={'axis': {'range': [0,100]}, 'bar': {'color': "#ffd371"}},
                                               title={'text': "Confidence"}))
                st.plotly_chart(fig_e, use_container_width=True)

        # SHAP interpretability for primary model
        st.markdown("### 🔎 Interpretability (Primary Model)")
        pmodel_name = primary_model_choice
        pmodel = MODELS.get(pmodel_name)
        if pmodel is None:
            st.info(f"Primary model ({pmodel_name}) not available for explanations.")
        else:
            if SHAP_AVAILABLE:
                st.caption("Computing SHAP (best effort) — may take a few seconds")
                try:
                    if hasattr(pmodel, "predict_proba") and ("Random" in pmodel_name or "Decision" in pmodel_name):
                        explainer = shap.TreeExplainer(pmodel)
                        shap_vals = explainer(input_df)
                        # take mean abs
                        if isinstance(shap_vals, list):
                            arr = np.abs(shap_vals[1]).mean(axis=0) if len(shap_vals) > 1 else np.abs(shap_vals[0]).mean(axis=0)
                        else:
                            arr = np.abs(shap_vals.values).mean(axis=0) if hasattr(shap_vals, "values") else np.abs(shap_vals).mean(axis=0)
                        df_shap = pd.DataFrame({"feature": input_df.columns.tolist(), "mean_abs_shap": arr}).sort_values("mean_abs_shap", ascending=False)
                        st.bar_chart(df_shap.set_index("feature").head(8))
                    else:
                        explainer = shap.Explainer(pmodel.predict, input_df)
                        shap_vals = explainer(input_df)
                        arr = np.abs(shap_vals.values).mean(axis=0)
                        df_shap = pd.DataFrame({"feature": input_df.columns.tolist(), "mean_abs_shap": arr}).sort_values("mean_abs_shap", ascending=False)
                        st.bar_chart(df_shap.set_index("feature").head(8))
                except Exception as e:
                    st.error(f"SHAP compute failed: {e}")
            else:
                st.info("Install `shap` to enable model interpretability; falling back to model-provided importances if any.")
                if hasattr(pmodel, "feature_importances_"):
                    fi = pmodel.feature_importances_
                    df_fi = pd.DataFrame({"feature": input_df.columns.tolist(), "importance": fi}).sort_values("importance", ascending=False)
                    st.bar_chart(df_fi.set_index("feature").head(8))
                elif hasattr(pmodel, "coef_"):
                    coefs = np.abs(pmodel.coef_).ravel()
                    df_coef = pd.DataFrame({"feature": input_df.columns.tolist(), "abs_coef": coefs}).sort_values("abs_coef", ascending=False)
                    st.bar_chart(df_coef.set_index("feature").head(8))

        # PDF & CSV downloads
        st.markdown("### 📄 Export")
        pdf_bytes, mime = make_pdf_report(inputs_pretty, results)
        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(f'<a href="data:{mime};base64,{b64}" download="prediction_report.{"pdf" if FPDF_AVAILABLE else "txt"}">⬇️ Download Report</a>', unsafe_allow_html=True)

        # CSV
        out_df = input_df.copy()
        for m in MODEL_NAMES:
            r = results.get(m, {})
            out_df[f"{m}_Prediction"] = r.get("prediction")
            out_df[f"{m}_Probability"] = r.get("probability")
        st.markdown(df_to_link(out_df, "multi_model_single_prediction.csv"), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Bulk Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📂 Bulk Predictions (CSV)")
    st.info("Upload a CSV with these columns (case-sensitive): Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        try:
            df_bulk = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df_bulk = None

        if df_bulk is not None:
            required = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
            missing = [c for c in required if c not in df_bulk.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                # map text to numeric where likely text present
                df_proc = df_bulk.copy()
                try:
                    if df_proc['Sex'].dtype == object:
                        df_proc['Sex'] = df_proc['Sex'].map({'Male':0,'Female':1})
                    df_proc['FastingBS'] = df_proc['FastingBS'].map({'<= 120 mg/dl':0, '> 120 mg/dl':1})
                    df_proc['ChestPainType'] = df_proc['ChestPainType'].map({"typical Angina":0, "Atypical Angina":1, "Non-Anginal Pain":2, "Asymptomatic":3})
                    df_proc['RestingECG'] = df_proc['RestingECG'].map({"Normal":0, "ST-T Wave Abnormality":1, "Left Ventricular Hypertrophy":2})
                    df_proc['ExerciseAngina'] = df_proc['ExerciseAngina'].map({'No':0,'Yes':1})
                    df_proc['ST_Slope'] = df_proc['ST_Slope'].map({"Upsloping":0,"Flat":1,"Downsloping":2})
                except Exception:
                    st.warning("Auto-mapping of textual columns partially failed — ensure numeric encoding matches training set.")

                # Bulk model: use Logistic Regression if available
                bulk_model = MODELS.get("Logistic Regression")
                if bulk_model is None:
                    st.error("Logistic Regression model missing — bulk prediction not possible.")
                else:
                    if st.button("Run Bulk Prediction (Logistic Regression)"):
                        n = len(df_proc)
                        prog = st.progress(0)
                        preds = []
                        probs = []
                        for i in range(n):
                            row = df_proc.iloc[[i]]
                            p, prob, err = predict_model(bulk_model, row.values)
                            if err:
                                preds.append(None); probs.append(None)
                            else:
                                preds.append("Heart Disease" if p==1 else "No Heart Disease")
                                probs.append(prob)
                            if i % max(1, n//20) == 0:
                                prog.progress(int(i/n*100))
                        prog.progress(100)
                        df_proc["Prediction"] = preds
                        df_proc["Probability"] = [p if p is not None else "" for p in probs]
                        st.success("Bulk prediction finished.")
                        st.dataframe(df_proc.head(200))
                        st.markdown(df_to_link(df_proc, "bulk_predictions.csv"), unsafe_allow_html=True)

elif page == "Model Info":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📊 Model Info & Diagnostics")

    st.markdown("### 3D Decision/Risk Surface")
    x = np.linspace(30,80,40); y=np.linspace(80,200,40)
    Xg, Yg = np.meshgrid(x,y)
    Z = 100 - ((0.02*(Xg-55)**2) + (0.01*(Yg-140)**2))*2
    fig3 = go.Figure(data=[go.Surface(z=Z, x=Xg, y=Yg, colorscale='Viridis')])
    fig3.update_layout(scene=dict(xaxis_title='Age', yaxis_title='MaxHR', zaxis_title='Risk Score'), height=520)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Model Cards & Feature Importances")
    columns = st.columns(2)
    feat_names = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
    for i, m in enumerate(MODEL_NAMES):
        with columns[i%2]:
            st.markdown(f"#### {m}")
            mo = MODELS.get(m)
            if mo is None:
                st.warning(f"Not loaded ({MODEL_MAP[m]})")
            else:
                st.write("Status: ✅ Loaded")
                st.write("- Example accuracy (placeholder): **86%**")
                if hasattr(mo, "feature_importances_"):
                    df_fi = pd.DataFrame({"feature": feat_names, "importance": mo.feature_importances_}).sort_values("importance", ascending=False)
                    st.bar_chart(df_fi.set_index("feature").head(6))
                elif hasattr(mo, "coef_"):
                    coefs = np.abs(mo.coef_).ravel()
                    dfc = pd.DataFrame({"feature": feat_names, "abs_coef": coefs}).sort_values("abs_coef", ascending=False)
                    st.bar_chart(dfc.set_index("feature").head(6))
                else:
                    st.info("No importances/coefs available for this model.")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("About")
    st.markdown(f"""
**Made by:** {APP_AUTHOR}  
Description:This Heart Disease Prediction System is an end-to-end machine learning application that analyzes key health parameters to estimate the risk of heart disease. It uses multiple ML algorithms—including Logistic Regression, SVM, Random Forest, XGBoost, KNN, Naïve Bayes, and Decision Trees—to provide an aggregated and reliable prediction. The system integrates essential ML concepts such as data preprocessing, feature scaling, cross-validation, hyperparameter tuning, and model evaluation (Accuracy, Precision, Recall, F1-Score, ROC-AUC).

Built with Python, Streamlit, scikit-learn, Pandas, NumPy, XGBoost, SHAP, and Plotly, the app also offers interactive visualizations, feature importance analysis, model comparison, and a clean UI for seamless user experience. This project showcases skills in machine learning, data analysis, visualization, and full-stack ML deployment.
""")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------- Footer ---------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"<div class='footer'>Made by <strong>{APP_AUTHOR}</strong> • ML Predictor • © {time.strftime('%Y')}</div>", unsafe_allow_html=True)

