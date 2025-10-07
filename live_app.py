# ======================================================
# ✈️ X-PLANE PREDICTIVE MAINTENANCE STREAMLIT APP
# ======================================================
import os
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score
)
from tensorflow.keras.models import load_model
# ---------- CONFIG / PATHS ----------
XGB_MODEL_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\models\xplane_xgboost.pkl"
LSTM_MODEL_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\models\xplane_lstm.h5"
SCALER_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\models\lstm_scaler.pkl"
DATA_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\data\processed\xplane_features.csv"
DEFAULT_LSTM_TIMESTEPS = 50
# ======================================================
# CACHED HELPERS
# ======================================================
@st.cache_resource
def load_xgb_model(path=XGB_MODEL_PATH):
    if not os.path.exists(path):
        return None, 0.5
    data = joblib.load(path)
    if isinstance(data, dict):
        model = data.get("model", data.get("model_object", None)) or data
        threshold = data.get("threshold", 0.5)
    elif isinstance(data, (tuple, list)):
        try:
            model, threshold = data[0], data[1]
        except Exception:
            model, threshold = data, 0.5
    else:
        model, threshold = data, 0.5
    return model, float(threshold)
@st.cache_resource
def load_lstm_model(path=LSTM_MODEL_PATH):
    if not os.path.exists(path):
        return None
    return load_model(path)
@st.cache_resource
def load_scaler(path=SCALER_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None
# ======================================================
# STREAM SIMULATION
# ======================================================
def live_stream(file_path = DATA_PATH):
    df_iter = pd.read_csv(file_path,chunksize = 1)
    for row in df_iter:
        yield row

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def clean_tabular_for_xgb(df, model=None):
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    for c in df.select_dtypes(include=["object"]).columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            pass
    df = df.select_dtypes(include=[np.number])
    if model is not None:
        expected = None
        try:
            expected = list(model.feature_names_in_)
        except Exception:
            expected = None
        if expected:
            for col in expected:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[expected]
    return df
def sliding_windows(X, timesteps=50):
    Xs = []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
    if len(Xs) == 0:
        return np.empty((0, timesteps, X.shape[1]))
    return np.stack(Xs, axis=0)
def plot_confusion(cm, labels=["0", "1"], title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.title(title)
    plt.tight_layout()
    return fig
def plot_roc(y_true, y_proba, label="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    return fig, roc_auc
# ======================================================
# STREAMLIT APP CONFIG
# ======================================================
st.set_page_config(page_title="✈️ X-Plane Predictive Maintenance", layout="wide")
st.sidebar.header("Mode Selection")
mode = st.sidebar.radio("Choose mode", ["📡 Real-Time Streaming", "📊 Interactive Batch Analysis"])
with st.sidebar.expander("📘 About This Dashboard"):
    st.markdown("""
    ### ✈️ X-Plane Predictive Maintenance Dashboard
    This dashboard simulates **real-time engine health monitoring** for aircraft systems using live data from X-Plane.

    #### 🧩 Parameters:
    - **RPM**: Engine revolutions per minute — reflects power output.
    - **N1 / N2**: Turbine speeds (low & high pressure turbine speed).
    - **EGT**: Exhaust Gas Temperature — a key early failure indicator.
    - **Oil Temp / Pressure**: Critical for lubrication and cooling.
    - **Fuel Pressure**: Indicates consistent flow; sudden drops can hint at pump or line faults.

    #### 🎯 Failure Probability Threshold Meter (default values, can be modified from the slider below):
    - 🟢 0.00 – 0.50 → Stable (Engine healthy)
    - 🟡 0.51 – 0.70 → Low Risk (Potential warning signs)
    - 🔴 0.71 – 1.00 → High Risk (Immediate inspection advised)

    #### 💡 Powered by:
    - **XGBoost** (for static feature-based health scoring)
    - **LSTM (Long Short Term Memory Neural Network)** (for temporal failure prediction)

    **Goal:** Predict failures before they happen — transforming maintenance from Reactive to Predictive.
    """)

# Load models
xgb_model, saved_threshold = load_xgb_model()
lstm_model = load_lstm_model()
scaler = load_scaler()
# ======================================================
# 📡 REAL-TIME STREAMING MODE
# ======================================================
if mode == "📡 Real-Time Streaming":
    st.title("📡 Real-Time Predictive Maintenance Dashboard")
    # Sidebar controls
    st.sidebar.subheader("🔧 Stream Controls")
    refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 2)
    start_stream = st.sidebar.button("▶ Start Live Streaming")
    # Threshold zones
    st.sidebar.subheader("🎯 Risk Zone Thresholds")
    green_threshold = st.sidebar.slider("🟢 Green Zone (Safe up to)", 0.0, 1.0, 0.5, 0.01)
    yellow_threshold = st.sidebar.slider("🟡 Yellow Zone (Caution up to)", green_threshold, 1.0, 0.7, 0.01)
    red_threshold = 1.0
    # Placeholders
    gauge_placeholder = st.empty()
    gauge_col, status_col = st.columns([2, 1])
    status_placeholder = st.empty()
    status = st.empty()
    chart_xgb = st.line_chart()
    chart_lstm = st.line_chart()
    # Gauge function
    def update_gauge(prob,last_prob=[0]):
        prob = max(0,min(prob,1.0))
        smooth_prob = last_prob[0] + (prob - last_prob[0])*1
        last_prob[0] = smooth_prob
        if smooth_prob < green_threshold:
            color = 'green'
            status_text = "🟢 STABLE: "
            description = "Engine is operating normally! 😊"
        elif smooth_prob < yellow_threshold:
            color = 'yellow'
            status_text = "⚠️ LOW RISK: "
            description = "Model detected minor anomalies! "
        else:
            color = 'red'
            status_text = "🔴 HIGH RISK: "
            description = "Potential failure detected! Consider replacing the part before failure! "
        fig = go.Figure(go.Indicator(
            mode = 'gauge+number',
            value = smooth_prob,
            delta = {'reference':last_prob[0],'increasing':{'color':'red'},'decreasing':{'color':'green'}},
            domain = {'x':[0,1],'y':[0,1]},
            title = {'text':'Failure Probability','font':{'size':22}},
            gauge = {
                'axis':{'range':[0,1]},
                'bar':{'color':color},
                'steps':[
                    {'range':[0,green_threshold],'color':'lightgreen'},
                    {'range':[green_threshold,yellow_threshold],'color':'orange'},
                    {'range':[yellow_threshold,red_threshold],'color':'salmon'}
                ],
                'threshold':{
                    'line':{'color':'black','width':3},
                    'thickness':0.8,
                    'value':smooth_prob
                }
            }
        ))
        fig.update_layout(
            height=250,
            margin=dict(t=10, b=10, l=10, r=10),
            transition = {'duration':10,'easing':'cubic-in-out'})
        gauge_placeholder.plotly_chart(
            fig, use_container_width=True,key = f"gauge_{int(time.time()*1000)}")
        status_placeholder.markdown(
             f"<h3 style='color:{color}'>{status_text}{description}</h3>",
        unsafe_allow_html=True
        )
    # Live streaming loop
    def run_dashboard():
        seq_buffer = []
        for row in live_stream():
            features = row.drop(columns=["failure", "Unnamed: 34"], errors="ignore")
            # Engine readings
            engine_rpm = row['rpm_1engin'].values[0] if 'rpm_1engin' in row else 0
            n1 = row['N1__1_pcnt'].values[0] if 'N1__1_pcnt' in row else 0
            n2 = row['N1__2_pcnt'].values[0] if 'N1__2_pcnt' in row else 0
            egt1 = row['EGT_1__deg'].values[0] if 'EGT_1__deg' in row else 0
            egt2 = row['EGT_2__deg'].values[0] if 'EGT_2__deg' in row else 0
            oil_temp1 = row['OILT1__deg'].values[0] if 'OILT1__deg' in row else 0
            oil_temp2 = row['OILT2__deg'].values[0] if 'OILT2__deg' in row else 0
            fuel_pressure = row['FUEP1__psi'].values[0] if 'FUEP1__psi' in row else 0
            # XGBoost inference
            try:
                xgb_prob = xgb_model.predict_proba(features)[0][1]
            except:
                xgb_prob = 0.0
            # LSTM inference
            try:
                scaled = scaler.transform(features)
                seq_buffer.append(scaled.flatten())
                if len(seq_buffer) >= 50:
                    X_seq = np.array(seq_buffer[-50:]).reshape(1, 50, features.shape[1])
                    lstm_prob = float(lstm_model.predict(X_seq, verbose=0)[0][0])
                else:
                    lstm_prob = 0.0
            except:
                lstm_prob = 0.0
            # Combined probability
            combined_prob = max(xgb_prob, lstm_prob)
            update_gauge(combined_prob)
            status.write(f"""
            **Engine RPM**: {engine_rpm:.2f}  
            **N1**: {n1:.2f}%  
            **N2**: {n2:.2f}%   
            **Oil Temp (Engine 1)**: {oil_temp1:.2f} °C  
            **Oil Temp (Engine 2)**: {oil_temp2:.2f} °C  
            **EGT (Engine 1)**: {egt1:.2f} °C   
            **EGT (Engine 2)**: {egt2:.2f} °C   
            **Fuel Pressure**: {fuel_pressure:.2f} psi  
            **Failure Probability (XGBoost) TOP GRAPH**: {xgb_prob:.2f}  
            **Failure Probability (LSTM) BOTTOM GRAPH**: {lstm_prob:.2f}  
            """)
            chart_xgb.add_rows({"XGBoost Failure Probability": [xgb_prob]})
            chart_lstm.add_rows({"LSTM Failure Probability": [lstm_prob]})
            time.sleep(refresh_rate)
    if start_stream:
        run_dashboard()
# ======================================================
# 📊 INTERACTIVE BATCH ANALYSIS MODE
# ======================================================
if mode == "📊 Interactive Batch Analysis":
    st.title("📊 Interactive Batch Analysis")
    uploaded = st.file_uploader("Upload X-Plane Processed CSV", type=["csv"])
    model_choice = st.selectbox("Select Model", ["XGBoost", "LSTM", "Both"], index=0)
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"✅ Loaded file with {df.shape[0]} rows and {df.shape[1]} columns")
        st.dataframe(df.head())
        if model_choice in ["XGBoost", "Both"]:
            st.subheader("XGBoost Analysis")
            X = df.drop(columns=["failure", "Unnamed: 34"], errors="ignore")
            y = df["failure"] if "failure" in df.columns else None
            try:
                proba = xgb_model.predict_proba(X)[:, 1]
                preds = (proba >= 0.5).astype(int)
                if y is not None:
                    cm = confusion_matrix(y, preds)
                    st.pyplot(plot_confusion(cm, ["No Failure", "Failure"], "XGBoost Confusion Matrix"))
                    fig_roc, auc_val = plot_roc(y, proba, "XGBoost")
                    st.pyplot(fig_roc)
                    st.success(f"ROC-AUC: {auc_val:.3f}")
            except Exception as e:
                st.error(f"XGBoost inference failed: {e}")
        if model_choice in ["LSTM", "Both"]:
            st.subheader("LSTM Analysis")
            df_num = df.select_dtypes(include=[np.number]).drop(columns=["failure"], errors="ignore")
            y = df["failure"] if "failure" in df.columns else None
            try:
                X_scaled = scaler.transform(df_num)
                X_seq = sliding_windows(X_scaled, timesteps=DEFAULT_LSTM_TIMESTEPS)
                proba = lstm_model.predict(X_seq).ravel()
                preds = (proba >= 0.5).astype(int)
                if y is not None:
                    y_true = y[DEFAULT_LSTM_TIMESTEPS: DEFAULT_LSTM_TIMESTEPS + len(preds)]
                    cm = confusion_matrix(y_true, preds)
                    st.pyplot(plot_confusion(cm, ["No Failure", "Failure"], "LSTM Confusion Matrix"))
                    fig_roc, auc_val = plot_roc(y_true, proba, "LSTM")
                    st.pyplot(fig_roc)
                    st.success(f"ROC-AUC: {auc_val:.3f}")
            except Exception as e:
                st.error(f"LSTM inference failed: {e}")
