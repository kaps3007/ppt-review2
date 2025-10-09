# ======================================================
# ✈️ X-PLANE PREDICTIVE MAINTENANCE STREAMLIT APP (Unified + Enhanced)
# ======================================================
import os
import time
from datetime import datetime
import io
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model
from zoneinfo import ZoneInfo


# ---------- CONFIG / PATHS ----------
XGB_MODEL_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\models\xplane_xgboost.pkl"
LSTM_MODEL_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\models\xplane_lstm.h5"
SCALER_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\models\lstm_scaler.pkl"
DATA_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\data\processed\xplane_features.csv"
DEFAULT_LSTM_TIMESTEPS = 50
LOG_OUT_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\data\live_log.csv"

# ---------- APP CONFIG ----------
st.set_page_config(page_title="✈️ X-Plane Predictive Maintenance", layout="wide")

# ---------- CACHED HELPERS ----------
@st.cache_resource
def load_xgb_model(path=XGB_MODEL_PATH):
    if not os.path.exists(path):
        return None, 0.5
    data = joblib.load(path)
    if isinstance(data, dict):
        model = data.get("model", data.get("model_object", None)) or data
        threshold = data.get("threshold", 0.5)
    elif isinstance(data, (tuple, list)):
        model, threshold = data[0], data[1] if len(data) > 1 else 0.5
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

# ---------- UTILITIES ----------
def live_stream(file_path=DATA_PATH):
    if not os.path.exists(file_path):
        return
    for row in pd.read_csv(file_path, chunksize=1):
        yield row

def clean_features_for_model(row_df, drop_cols=("failure",)):
    df = row_df.copy()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.select_dtypes(include=[np.number])

def sliding_windows(X, timesteps=50):
    Xs = [X[i:i+timesteps] for i in range(len(X)-timesteps)]
    return np.stack(Xs, axis=0) if Xs else np.empty((0, timesteps, X.shape[1]))

def plot_confusion(cm, labels=["0", "1"], title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i,j], ha="center", va="center", color="black")
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_roc(y_true, y_proba, label="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.2f})")
    ax.plot([0,1],[0,1], color="grey", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    return fig, roc_auc

def identify_top_contributors(xgb_model, scaler, features_df, top_k=3):
    if xgb_model is None or scaler is None:
        return None
    try:
        feat_names = list(xgb_model.feature_names_in_)
    except Exception:
        feat_names = None
    if feat_names is None:
        return None
    importances = getattr(xgb_model, "feature_importances_", np.ones(len(feat_names)))
    mean, scale = getattr(scaler, "mean_", None), getattr(scaler, "scale_", None)
    if mean is None or scale is None:
        return None
    row_vals = np.array([float(features_df.get(col, 0)) for col in feat_names])
    z = (row_vals - mean) / np.where(scale==0, 1e-6, scale)
    scores = np.abs(z) * np.abs(importances)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{"feature": feat_names[i], "value": row_vals[i], "score": scores[i]} for i in top_idx]

# ---------- UI ----------
with st.sidebar.expander("📘 About This Dashboard"):
    st.markdown("""
    ### ✈️ X-Plane Predictive Maintenance Dashboard
    This dashboard simulates **real-time engine health monitoring** for aircraft systems using live data from X-Plane 11.

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

st.sidebar.header("Mode Selection")
mode = st.sidebar.radio("Choose mode", ["📡 Real-Time Streaming", "📊 Interactive Batch Analysis"])

xgb_model, saved_threshold = load_xgb_model()
lstm_model = load_lstm_model()
scaler = load_scaler()

# ---------- REAL-TIME STREAMING ----------
if mode == "📡 Real-Time Streaming":
    st.title("📡 Real-Time Predictive Maintenance Dashboard")

    st.sidebar.subheader("🔧 Stream Controls")
    refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 0.5, 10.0, 1.0, 0.5)
    start_stream = st.sidebar.button("▶ Start Live Streaming")
    stop_stream = st.sidebar.button("■ Stop Live Streaming")

    st.sidebar.subheader("🎯 Risk Zone Thresholds")
    green_threshold = st.sidebar.slider("🟢 Green Zone", 0.0, 1.0, 0.5, 0.01)
    yellow_threshold = st.sidebar.slider("🟡 Yellow Zone", green_threshold, 1.0, 0.75, 0.01)

    # Logging Controls
    st.sidebar.subheader("📥 Logging")
    if "live_log_df" not in st.session_state:
        st.session_state.live_log_df = pd.DataFrame(columns=["timestamp","xgb_prob","lstm_prob","combined_prob","zone"])
    log_button = st.sidebar.button("Toggle Logging")
    if log_button:
        st.session_state["log_enabled"] = not st.session_state.get("log_enabled", False)
        st.success("Logging Enabled" if st.session_state["log_enabled"] else "Logging Disabled")

    # Layout
    col_left, col_right = st.columns([2,1])
    with col_left:
        gauge_ph = st.empty()
        chart_xgb = st.line_chart(pd.DataFrame(columns=["xgb_prob"]))
        chart_lstm = st.line_chart(pd.DataFrame(columns=["lstm_prob"]))
    with col_right:
        status_area = st.empty()
        faulty_area = st.empty()

    if "stream_running" not in st.session_state:
        st.session_state.stream_running = False

    def render_gauge(prob, g_thresh, y_thresh):
        """Animated cinematic gauge with glowing background that reacts to failure probability."""
        prob = float(np.clip(prob, 0.0, 1.0))

        # Determine zone colors + glow intensity
        if prob <= g_thresh:
            bar_color = "#15FF00"       # bright green
            bg_color = "rgba(0, 200, 0, 0.5)"  # subtle green
            pulse_strength = 0.1
        elif prob <= y_thresh:
            bar_color = "#FFD700"       # amber
            bg_color = "rgba(255, 215, 0, 0.25)"
            pulse_strength = 0.3
        else:
            bar_color = "#FF4C4C"       # bright red
            bg_color = "rgba(255, 0, 0, 0.3)"
            pulse_strength = 0.6

        pulse_phase = (time.time() * 2.5) % (2 * np.pi) 
        pulse_alpha = 0.25 + pulse_strength * (0.5 + 0.5 * np.sin(pulse_phase)) 
        glow_rgba = f"rgba(255, 0, 0, {pulse_alpha:.2f})" if prob > y_thresh else bg_color

        # Build the gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={'font': {'color': 'white', 'size': 44}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Failure Probability", 'font': {'size': 22, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 1], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                'bar': {'color': bar_color, 'thickness': 0.35},
                'borderwidth': 3,
                'bordercolor': "#000000",
                'steps': [
                    {'range': [0, g_thresh], 'color': '#003300'},
                    {'range': [g_thresh, y_thresh], 'color': '#705900'},
                    {'range': [y_thresh, 1.0], 'color': '#4D0000'}
                ],
                'threshold': {
                    'line': {'color': "#000000", 'width': 5},
                    'thickness': 0.8,
                    'value': prob
                }
            }
        ))

        # Set layout with dynamic glow background
        fig.update_layout(
            height=360,
            margin=dict(t=60, b=40, l=40, r=40),
            paper_bgcolor=glow_rgba,     # 💡 soft pulsating background behind the meter
            plot_bgcolor="#0E1117",      # consistent dashboard tone
            font={'color': 'white'},
            transition={'duration': 500, 'easing': 'cubic-in-out'}
        )

        gauge_ph.plotly_chart(fig, use_container_width=True, key=f"gauge_{time.time_ns()}")

    def zone_label(prob, g_thresh, y_thresh):
        if prob <= g_thresh:
            return "🟢 STABLE","green","Engine is operating normally! 😊"
        elif prob <= y_thresh:
            return "🟡 LOW RISK","gold","Model has detected minor anomalies! "
        else:
            return "🔴 HIGH RISK","red","Potential failure detected! Consider replacing the part before failure! "

    if start_stream:
        st.session_state.stream_running = True
    if stop_stream:
        st.session_state.stream_running = False

    if st.session_state.stream_running:
        last_prob = 0.0
        for row in live_stream():
            if not st.session_state.stream_running:
                break
            features = clean_features_for_model(row)
            try:
                xgb_prob = float(xgb_model.predict_proba(features)[0][1])
            except Exception: xgb_prob = 0.0
            try:
                scaled = scaler.transform(features)
                if "seq_buf" not in st.session_state:
                    st.session_state.seq_buf = []
                st.session_state.seq_buf.append(scaled.flatten())
                lstm_prob = float(lstm_model.predict(
                    np.array(st.session_state.seq_buf[-DEFAULT_LSTM_TIMESTEPS:]).reshape(1,DEFAULT_LSTM_TIMESTEPS,features.shape[1]),
                    verbose=0)[0][0]) if len(st.session_state.seq_buf)>=DEFAULT_LSTM_TIMESTEPS else 0.0
            except Exception: lstm_prob = 0.0

            combined = (xgb_prob + lstm_prob)
            smooth = last_prob + (combined - last_prob)
            last_prob = smooth

            render_gauge(smooth, green_threshold, yellow_threshold)
            chart_xgb.add_rows(pd.DataFrame({"xgb_prob":[xgb_prob]}))
            chart_lstm.add_rows(pd.DataFrame({"lstm_prob":[lstm_prob]}))

            zone_txt, color, desc = zone_label(smooth, green_threshold, yellow_threshold)
            try:
                engine_rpm = float(row['rpm_1engin'].values[0]) if 'rpm_1engin' in row.columns else 0.0
                n1 = float(row['N1__1_pcnt'].values[0]) if 'N1__1_pcnt' in row.columns else 0.0
                n2 = float(row['N1__2_pcnt'].values[0]) if 'N1__2_pcnt' in row.columns else 0.0
                oil_temp1 = float(row['OILT1__deg'].values[0]) if 'OILT1__deg' in row.columns else 0.0
                oil_temp2 = float(row['OILT2__deg'].values[0]) if 'OILT2__deg' in row.columns else 0.0
                egt1 = float(row['EGT_1__deg'].values[0]) if 'EGT_1__deg' in row.columns else 0.0
                egt2 = float(row['EGT_2__deg'].values[0]) if 'EGT_2__deg' in row.columns else 0.0
                fuel_pressure = float(row['FUEP1__psi'].values[0]) if 'FUEP1__psi' in row.columns else 0.0
            except Exception:
                engine_rpm = n1 = n2 = oil_temp1 = oil_temp2 = egt1 = egt2 = fuel_pressure = 0.0
            status_area.markdown(f"""
                <div style="padding:12px;border-radius:12px;background:rgba(255,255,255,0.05);
                    border-left:6px solid {color};box-shadow:0 0 25px {color}80;">
                <h3 style="margin:0;color:{color};font-size:22px">{zone_txt}</h3>
                <p style="margin:4px 0;font-size:16px;color:white">{desc}</p>
                <p style="margin:4px 0;color:lightgray">
                Combined Probability: <b style="color:{color}">{smooth:.3f}</b></p>

                <hr style="border:1px solid rgba(255,255,255,0.1)">
                <h4 style="color:white;margin-bottom:4px;">Telemetry Data</h4>
                <ul style="list-style:none;padding-left:8px;color:#dcdcdc;font-size:15px;line-height:1.5;">
                    <li><b>Engine RPM:</b> {engine_rpm:.2f}</li>
                    <li><b>N1:</b> {n1:.2f}% | <b>N2:</b> {n2:.2f}%</li>
                    <li><b>Oil Temp (Eng 1):</b> {oil_temp1:.2f} °C | <b>Oil Temp (Eng 2):</b> {oil_temp2:.2f} °C</li>
                    <li><b>EGT (Eng 1):</b> {egt1:.2f} °C | <b>EGT (Eng 2):</b> {egt2:.2f} °C</li>
                    <li><b>Fuel Pressure:</b> {fuel_pressure:.2f} psi</li>
                    <li><b>Failure Probability (XGBoost – Top Graph):</b> {xgb_prob:.2f}</li>
                    <li><b>Failure Probability (LSTM – Bottom Graph):</b> {lstm_prob:.2f}</li>
                </ul>
                </div>
            """, unsafe_allow_html=True)

            if st.session_state.get("log_enabled", False):
                st.session_state.live_log_df = pd.concat([st.session_state.live_log_df, pd.DataFrame([{
                    "timestamp": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
                    "xgb_prob": xgb_prob,
                    "lstm_prob": lstm_prob,
                    "combined_prob": smooth,
                    "zone": zone_txt
                }])], ignore_index=True)
                st.session_state.live_log_df.tail(1).to_csv(LOG_OUT_PATH, mode="a", header=not os.path.exists(LOG_OUT_PATH), index=False)
            time.sleep(refresh_rate)
        st.info("Stream stopped.")
    
# ---------- BATCH ANALYSIS ----------
if mode == "📊 Interactive Batch Analysis":
    st.title("📊 Interactive Batch Analysis")
    uploaded = st.file_uploader("Upload processed X-Plane CSV", type=["csv"])
    model_choice = st.selectbox("Select Model", ["XGBoost", "LSTM", "Both"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
        st.dataframe(df.head())
        if model_choice in ("XGBoost","Both"):
            st.subheader("XGBoost Analysis")
            X = clean_features_for_model(df)
            y = df["failure"] if "failure" in df.columns else None
            proba = xgb_model.predict_proba(X)[:,1]
            preds = (proba>=0.5).astype(int)
            if y is not None:
                cm = confusion_matrix(y, preds)
                st.pyplot(plot_confusion(cm,["NoFail","Fail"],"XGB Confusion"))
                fig_roc, aucv = plot_roc(y, proba, "XGBoost")
                st.pyplot(fig_roc)
                st.success(f"ROC-AUC: {aucv:.3f}")
        if model_choice in ("LSTM","Both"):
            st.subheader("LSTM Analysis")
            df_num = df.select_dtypes(include=[np.number])
            y = df["failure"] if "failure" in df.columns else None
            expected_features = getattr(scaler, "feature_names_in_", None)
            if expected_features is not None:
                # Add missing columns
                for col in expected_features:
                    if col not in df_num.columns:
                        df_num[col] = 0.0
                # Drop extra columns not seen during training
                df_num = df_num[expected_features]
            else:
                # fallback: ensure consistent number of columns
                df_num = df_num.iloc[:, :scaler.mean_.shape[0]]
            X_scaled = scaler.transform(df_num)
            X_seq = sliding_windows(X_scaled, DEFAULT_LSTM_TIMESTEPS)
            proba = lstm_model.predict(X_seq).ravel()
            preds = (proba>=0.5).astype(int)
            if y is not None:
                y_true = y[DEFAULT_LSTM_TIMESTEPS:DEFAULT_LSTM_TIMESTEPS+len(preds)]
                cm = confusion_matrix(y_true, preds)
                st.pyplot(plot_confusion(cm,["NoFail","Fail"],"LSTM Confusion"))
                fig_roc, aucv = plot_roc(y_true, proba, "LSTM")
                st.pyplot(fig_roc)
                st.success(f"ROC-AUC: {aucv:.3f}")

# ---------- FOOTER ----------
st.markdown("---")
if "live_log_df" in st.session_state and not st.session_state.live_log_df.empty:
    csv_bytes = st.session_state.live_log_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Log (CSV)", csv_bytes, "live_log.csv", "text/csv")
st.caption("🛫 Unified Predictive Maintenance Dashboard | XGBoost + LSTM | Real-time + Batch Analysis + Logging + Fault Insights")
