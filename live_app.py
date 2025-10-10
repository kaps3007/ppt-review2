# ======================================================
# ✈️ X-PLANE PREDICTIVE MAINTENANCE STREAMLIT APP
# Unified + Enhanced with full 34-feature Telemetry Simulator
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

# ---------------- CONFIG / PATHS ----------------
XGB_MODEL_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\models\xplane_xgboost.pkl"
LSTM_MODEL_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\models\xplane_lstm.h5"
SCALER_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\models\lstm_scaler.pkl"
DATA_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\data\processed\xplane_features.csv"
df = pd.read_csv(DATA_PATH)
p1_max = df["power_1hp"].max()
p2_max = df["power_2hp"].max()
t1_max = df["thrst_1lb"].max()
t2_max = df["thrst_2lb"].max()
r1_max = df["rpm_1engin"].max()
r2_max = df["rpm_2engin"].max()
e1_max = df["EGT_1__deg"].max()
e2_max = df["EGT_2__deg"].max()
ot1_max = df["OILT1__deg"].max()
ot2_max = df["OILT2__deg"].max()
fp1_max = df["FUEP1__psi"].max()
fp2_max = df["FUEP2__psi"].max()

DEFAULT_LSTM_TIMESTEPS = 50
LOG_OUT_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\data\live_log.csv"

# Full 34-feature order (as you provided)
FULL_FEATURE_LIST = [
    "fact_sec","fsim_sec","frame_time","__cpu_time","_gpu_time_","_grndratio","_flitratio",
    "_Vind_kias","_Vind_keas","Vtrue_ktas","Vtrue_ktgs","_Vind__mph","Vtruemphas","Vtruemphgs",
    "power_1hp","power_2hp","thrst_1lb","thrst_2lb","rpm_1engin","rpm_2engin",
    "N1__1_pcnt","N1__2_pcnt","N2__1_pcnt","N2__2_pcnt",
    "EGT_1__deg","EGT_2__deg","OILT1__deg","OILT2__deg",
    "FUEP1__psi","FUEP2__psi","batt1__amp","batt2__amp","batt1_volt","batt2_volt"
]

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
    """Simple generator reading CSV rows one-by-one (chunksize=1)."""
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

# ---------- UI: Sidebar About ----------
with st.sidebar.expander("📘 About This Dashboard"):
    st.markdown("""
    ### ✈️ X-Plane Predictive Maintenance Dashboard
    This dashboard simulates **real-time engine health monitoring** for aircraft systems using live data from X-Plane 11.

    #### 🧩 Parameters:
    - **RPM**: Engine revolutions per minute — reflects power output.
    - **N1 / N2**: Turbine speeds.
    - **EGT**: Exhaust Gas Temperature.
    - **Oil Temp / Pressure**: Critical for lubrication and cooling.
    - **Fuel Pressure**: Indicates consistent flow.

    #### 🎯 Failure Probability Threshold Meter (modify below):
    - 🟢 0.00 – 0.50 → Stable
    - 🟡 0.51 – 0.70 → Low Risk
    - 🔴 0.71 – 1.00 → High Risk

    **Powered by:**
    - **XGBoost** (feature-based)
    - **LSTM** (temporal)
    """)

st.sidebar.header("Mode Selection")
mode = st.sidebar.radio("Choose mode", ["📡 Real-Time Streaming", "📊 Interactive Batch Analysis"])

# Load models/scaler
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

    # ---------- Layout (placeholders) ----------
    col_left, col_right = st.columns([2,1])
    with col_left:
        gauge_ph = st.empty()
        chart_xgb = st.line_chart(pd.DataFrame(columns=["xgb_prob"]))
        chart_lstm = st.line_chart(pd.DataFrame(columns=["lstm_prob"]))
    with col_right:
        status_area = st.empty()
        faulty_area = st.empty()

    # Ensure placeholders are in session_state so simulator can reuse
    st.session_state.setdefault("placeholders", {})
    st.session_state["placeholders"].setdefault("gauge_ph", gauge_ph)
    st.session_state["placeholders"].setdefault("chart_xgb", chart_xgb)
    st.session_state["placeholders"].setdefault("chart_lstm", chart_lstm)
    st.session_state["placeholders"].setdefault("status_area", status_area)

    # Utility: render gauge
    def render_gauge(prob, g_thresh, y_thresh):
        prob = float(np.clip(prob, 0.0, 1.0))
        if prob <= g_thresh:
            bar_color = "#15FF00"
            bg_color = "rgba(0, 200, 0, 0.5)"
            pulse_strength = 0.1
        elif prob <= y_thresh:
            bar_color = "#FFD700"
            bg_color = "rgba(255, 215, 0, 0.25)"
            pulse_strength = 0.3
        else:
            bar_color = "#FF4C4C"
            bg_color = "rgba(255, 0, 0, 0.3)"
            pulse_strength = 0.6

        pulse_phase = (time.time() * 2.5) % (2 * np.pi)
        pulse_alpha = 0.25 + pulse_strength * (0.5 + 0.5 * np.sin(pulse_phase))
        glow_rgba = f"rgba(255, 0, 0, {pulse_alpha:.2f})" if prob > y_thresh else bg_color

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
        fig.update_layout(
            height=360,
            margin=dict(t=60, b=40, l=40, r=40),
            paper_bgcolor=glow_rgba,
            plot_bgcolor="#0E1117",
            font={'color': 'white'},
            transition={'duration': 500, 'easing': 'cubic-in-out'}
        )

        # Use stored placeholder
        ph = st.session_state["placeholders"].get("gauge_ph", None)
        if ph is not None:
            ph.plotly_chart(fig, use_container_width=True, key=f"gauge_{time.time_ns()}")

    def zone_label(prob, g_thresh, y_thresh):
        if prob <= g_thresh:
            return "🟢 STABLE","green","Engine is operating normally! 😊"
        elif prob <= y_thresh:
            return "🟡 LOW RISK","gold","Model has detected minor anomalies!"
        else:
            return "🔴 HIGH RISK","red","Potential failure detected! Consider inspection."

    # ---------- SIMULATOR: START ----------
    # Place simulator after layout & placeholders so it can update the same charts/gauge/status
    sim_exp = st.sidebar.expander("🕹️ Telemetry Simulator (Interactive)", expanded=False)

    with sim_exp:
        st.markdown("Use the sliders and toggles to simulate telemetry. Predictions use both XGBoost + LSTM like real streaming.")
        throttle = st.slider("Throttle Level (%)", 0, 100, 50, 1, help="Global throttle level.", key="sim_throttle")
        auto_map = st.checkbox("Auto-map throttle -> power/thrust/RPM", value=True, key="sim_auto_map",
                               help="When enabled, changing throttle will automatically change some telemetry values.")
        st.markdown("**Simulated Failure Modes**")
        f_overheat = st.checkbox("Engine Overheat", value=False, key="sim_fail_overheat")
        f_fuel_restrict = st.checkbox("Fuel Restriction", value=False, key="sim_fail_fuel_restrict")
        f_elec_pump = st.checkbox("Electronic Fuel Pump Failure", value=False, key="sim_fail_pump")

        st.markdown("---")
        # Defaults for the telemetry features (set sensible defaults)
        feature_defaults = {
            "fact_sec": 0.0,"fsim_sec":0.0,"frame_time":0.0,"__cpu_time":0.0,"_gpu_time_":0.0,
            "_grndratio":0.0,"_flitratio":0.0,"_Vind_kias":0.0,"_Vind_keas":0.0,"Vtrue_ktas":0.0,
            "Vtrue_ktgs":0.0,"_Vind__mph":0.0,"Vtruemphas":0.0,"Vtruemphgs":0.0,
            "power_1hp": 0.0, "power_2hp": 0.0,
            "thrst_1lb": 0.0, "thrst_2lb": 0.0,
            "rpm_1engin": 0.0, "rpm_2engin": 0.0,
            "N1__1_pcnt": 0.0, "N1__2_pcnt": 0.0,
            "N2__1_pcnt": 0.0, "N2__2_pcnt": 0.0,
            "EGT_1__deg": 0.0, "EGT_2__deg": 0.0,
            "OILT1__deg": 0.0, "OILT2__deg": 0.0,
            "FUEP1__psi": 0.0, "FUEP2__psi": 0.0,
            "batt1__amp": 0.0, "batt2__amp": 0.0,
            "batt1_volt": 0.0, "batt2_volt": 0.0
        }

        # Initialize sim_vals if absent
        if "sim_vals" not in st.session_state:
            st.session_state.sim_vals = feature_defaults.copy()
            st.session_state.sim_seq = []

        # Create sliders for each of the 34 features. We'll display them in two columns for compactness.
        col_a, col_b = st.columns(2)
        with col_a:
            # Show some of the features in the left column (mainly aerodynamics + engine1)
            fact_sec = st.number_input("fact_sec", value=float(st.session_state.sim_vals.get("fact_sec",0.0)), step=0.1, key="sim_fact_sec")
            fsim_sec = st.number_input("fsim_sec", value=float(st.session_state.sim_vals.get("fsim_sec",0.0)), step=0.1, key="sim_fsim_sec")
            frame_time = st.number_input("frame_time", value=float(st.session_state.sim_vals.get("frame_time",0.0)), step=0.01, key="sim_frame_time")
            __cpu_time = st.number_input("__cpu_time", value=float(st.session_state.sim_vals.get("__cpu_time",0.0)), step=0.01, key="sim_cpu_time")
            _gpu_time_ = st.number_input("_gpu_time_", value=float(st.session_state.sim_vals.get("_gpu_time_",0.0)), step=0.01, key="sim_gpu_time")
            _grndratio = st.number_input("_grndratio", value=float(st.session_state.sim_vals.get("_grndratio",0.0)), step=0.01, key="sim_grndratio")
            _flitratio = st.number_input("_flitratio", value=float(st.session_state.sim_vals.get("_flitratio",0.0)), step=0.01, key="sim_flitratio")
            _Vind_kias = st.number_input("_Vind_kias", value=float(st.session_state.sim_vals.get("_Vind_kias",0.0)), step=0.1, key="sim_Vind_kias")
            _Vind_keas = st.number_input("_Vind_keas", value=float(st.session_state.sim_vals.get("_Vind_keas",0.0)), step=0.1, key="sim_Vind_keas")
            Vtrue_ktas = st.number_input("Vtrue_ktas", value=float(st.session_state.sim_vals.get("Vtrue_ktas",0.0)), step=0.1, key="sim_Vtrue_ktas")
            Vtrue_ktgs = st.number_input("Vtrue_ktgs", value=float(st.session_state.sim_vals.get("Vtrue_ktgs",0.0)), step=0.1, key="sim_Vtrue_ktgs")
            _Vind__mph = st.number_input("_Vind__mph", value=float(st.session_state.sim_vals.get("_Vind__mph",0.0)), step=0.1, key="sim_Vind_mph")
            Vtruemphas = st.number_input("Vtruemphas", value=float(st.session_state.sim_vals.get("Vtruemphas",0.0)), step=0.1, key="sim_Vtruemphas")
        with col_b:
            Vtruemphgs = st.number_input("Vtruemphgs", value=float(st.session_state.sim_vals.get("Vtruemphgs",0.0)), step=0.1, key="sim_Vtruemphgs")
            # Engine 1 & 2 primary controls
            p1 = st.number_input("power_1hp", 0.0, p1_max, float(st.session_state.sim_vals["power_1hp"]), help="Engine 1 power (hp).", key="sim_power_1hp")
            p2 = st.number_input("power_2hp", 0.0, p2_max, float(st.session_state.sim_vals["power_2hp"]), help="Engine 2 power (hp).", key="sim_power_2hp")
            thr1 = st.number_input("thrst_1lb", 0.0, t1_max, float(st.session_state.sim_vals["thrst_1lb"]), help="Thrust engine 1 (lb).", key="sim_thrst_1lb")
            thr2 = st.number_input("thrst_2lb", 0.0, t2_max, float(st.session_state.sim_vals["thrst_2lb"]), help="Thrust engine 2 (lb).", key="sim_thrst_2lb")
            rpm1 = st.number_input("rpm_1engin", 0.0, r1_max, float(st.session_state.sim_vals["rpm_1engin"]), help="RPM engine 1.", key="sim_rpm_1engin")
            rpm2 = st.number_input("rpm_2engin", 0.0, r2_max, float(st.session_state.sim_vals["rpm_2engin"]), help="RPM engine 2.", key="sim_rpm_2engin")
            n1_1 = st.number_input("N1__1_pcnt", 0.0, 100.0, float(st.session_state.sim_vals["N1__1_pcnt"]), help="N1 % engine 1.", key="sim_N1_1")
            n1_2 = st.number_input("N1__2_pcnt", 0.0, 100.0, float(st.session_state.sim_vals["N1__2_pcnt"]), help="N1 % engine 2.", key="sim_N1_2")
            n2_1 = st.number_input("N2__1_pcnt", 0.0, 100.0, float(st.session_state.sim_vals["N2__1_pcnt"]), help="N2 % engine 1.", key="sim_N2_1")
            n2_2 = st.number_input("N2__2_pcnt", 0.0, 100.0, float(st.session_state.sim_vals["N2__2_pcnt"]), help="N2 % engine 2.", key="sim_N2_2")
            egt1 = st.number_input("EGT_1__deg", 0.0, e1_max, float(st.session_state.sim_vals["EGT_1__deg"]), help="EGT engine 1 (°C).", key="sim_EGT_1")
            egt2 = st.number_input("EGT_2__deg", 0.0, e2_max, float(st.session_state.sim_vals["EGT_2__deg"]), help="EGT engine 2 (°C).", key="sim_EGT_2")
            oilt1 = st.number_input("OILT1__deg", -40.0, ot1_max, float(st.session_state.sim_vals["OILT1__deg"]), help="Oil temp engine 1 (°C).", key="sim_OILT1")
            oilt2 = st.number_input("OILT2__deg", -40.0, ot2_max, float(st.session_state.sim_vals["OILT2__deg"]), help="Oil temp engine 2 (°C).", key="sim_OILT2")
            fuep1 = st.number_input("FUEP1__psi", 0.0, fp1_max, float(st.session_state.sim_vals["FUEP1__psi"]), help="Fuel press engine 1 (psi).", key="sim_FUEP1")
            fuep2 = st.number_input("FUEP2__psi", 0.0, fp2_max, float(st.session_state.sim_vals["FUEP2__psi"]), help="Fuel press engine 2 (psi).", key="sim_FUEP2")

        # battery fields (below the two columns, full width)
        b1_amp = st.number_input("batt1__amp", value=float(st.session_state.sim_vals["batt1__amp"]), step=0.1, key="sim_batt1_amp")
        b2_amp = st.number_input("batt2__amp", value=float(st.session_state.sim_vals["batt2__amp"]), step=0.1, key="sim_batt2_amp")
        b1_volt = st.number_input("batt1_volt", value=float(st.session_state.sim_vals["batt1_volt"]), step=0.01, key="sim_batt1_volt")
        b2_volt = st.number_input("batt2_volt", value=float(st.session_state.sim_vals["batt2_volt"]), step=0.01, key="sim_batt2_volt")

        # Reset button
        if st.button("🔄 Reset Simulator", key="sim_reset"):
            st.session_state.sim_vals = feature_defaults.copy()
            st.session_state.sim_seq = []
            st.session_state["last_sim_prob"] = 0.0
            st.rerun()

        # Update session_state.sim_vals with current values
        st.session_state.sim_vals.update({
            "fact_sec": fact_sec, "fsim_sec": fsim_sec, "frame_time": frame_time, "__cpu_time": __cpu_time, "_gpu_time_": _gpu_time_, "_grndratio": _grndratio, "_flitratio": _flitratio,
            "_Vind_kias": _Vind_kias, "_Vind_keas": _Vind_keas, "Vtrue_ktas": Vtrue_ktas, "Vtrue_ktgs": Vtrue_ktgs, "_Vind__mph": _Vind__mph, "Vtruemphas": Vtruemphas, "Vtruemphgs": Vtruemphgs,
            "power_1hp": p1, "power_2hp": p2,
            "thrst_1lb": thr1, "thrst_2lb": thr2,
            "rpm_1engin": rpm1, "rpm_2engin": rpm2,
            "N1__1_pcnt": n1_1, "N1__2_pcnt": n1_2,
            "N2__1_pcnt": n2_1, "N2__2_pcnt": n2_2,
            "EGT_1__deg": egt1, "EGT_2__deg": egt2,
            "OILT1__deg": oilt1, "OILT2__deg": oilt2,
            "FUEP1__psi": fuep1, "FUEP2__psi": fuep2,
            "batt1__amp": b1_amp, "batt2__amp": b2_amp,
            "batt1_volt": b1_volt, "batt2_volt": b2_volt
        })

        # Apply simple throttle mapping if enabled
        if auto_map:
            s = 0.5 + 0.55 * (throttle / 100.0)
            st.session_state.sim_vals["power_1hp"] = max(0.0, feature_defaults["power_1hp"] * s if st.session_state.get("sim_auto_map_first_apply", True) else st.session_state.sim_vals["power_1hp"] * s)
            st.session_state.sim_vals["power_2hp"] = max(0.0, feature_defaults["power_2hp"] * s if st.session_state.get("sim_auto_map_first_apply", True) else st.session_state.sim_vals["power_2hp"] * s)
            st.session_state.sim_vals["thrst_1lb"] = max(0.0, feature_defaults["thrst_1lb"] * s if st.session_state.get("sim_auto_map_first_apply", True) else st.session_state.sim_vals["thrst_1lb"] * s)
            st.session_state.sim_vals["thrst_2lb"] = max(0.0, feature_defaults["thrst_2lb"] * s if st.session_state.get("sim_auto_map_first_apply", True) else st.session_state.sim_vals["thrst_2lb"] * s)
            st.session_state.sim_vals["rpm_1engin"] = max(0.0, feature_defaults["rpm_1engin"] * (0.8 + 0.4*(throttle/100.0)))
            st.session_state.sim_vals["rpm_2engin"] = max(0.0, feature_defaults["rpm_2engin"] * (0.8 + 0.4*(throttle/100.0)))
            st.session_state["sim_auto_map_first_apply"] = False

        # Apply failure-mode influences
        if f_overheat:
            st.session_state.sim_vals["EGT_1__deg"] += 150.0
            st.session_state.sim_vals["EGT_2__deg"] += 150.0
            st.session_state.sim_vals["OILT1__deg"] += 20.0
            st.session_state.sim_vals["OILT2__deg"] += 20.0
        if f_fuel_restrict:
            st.session_state.sim_vals["FUEP1__psi"] *= 0.5
            st.session_state.sim_vals["FUEP2__psi"] *= 0.5
            st.session_state.sim_vals["power_1hp"] *= 0.7
            st.session_state.sim_vals["power_2hp"] *= 0.7
        if f_elec_pump:
            st.session_state.sim_vals["FUEP1__psi"] *= 0.3
            st.session_state.sim_vals["FUEP2__psi"] *= 0.3

        # Build sim_row in FULL_FEATURE_LIST order
        sim_row = {k: float(st.session_state.sim_vals.get(k, 0.0)) for k in FULL_FEATURE_LIST}

        # Run inference for simulator using same logic as streaming (XGB + LSTM)
        # XGBoost inference: prepare DataFrame matching feature names
        try:
            sim_df = pd.DataFrame([sim_row])
            # Reindex to model feature order if model provides it
            try:
                feat_names = list(xgb_model.feature_names_in_)
                sim_df = sim_df.reindex(columns=feat_names, fill_value=0.0)
            except Exception:
                # if xgb_model doesn't expose feature_names_in_, attempt to reindex to FULL_FEATURE_LIST
                sim_df = sim_df.reindex(columns=FULL_FEATURE_LIST, fill_value=0.0)
            if xgb_model is not None:
                # Some saved XGB objects might expect a numpy array; other times DataFrame is OK.
                try:
                    xgb_prob = float(xgb_model.predict_proba(sim_df)[0][1])
                except Exception:
                    # fallback to raw values
                    xgb_prob = float(xgb_model.predict_proba(sim_df.values)[0][1])
            else:
                xgb_prob = 0.0
        except Exception as e:
            xgb_prob = 0.0

        # LSTM inference: scale and maintain sim_seq
        try:
            # Determine scaler expected order
            if scaler is not None:
                expected = getattr(scaler, "feature_names_in_", None)
                if expected is not None:
                    arr = np.array([sim_row.get(c, 0.0) for c in expected], dtype=float).reshape(1, -1)
                else:
                    # Use FULL_FEATURE_LIST but LSTM typically trained on subset; ensure consistent column count
                    arr = np.array([sim_row.get(c, 0.0) for c in FULL_FEATURE_LIST], dtype=float).reshape(1, -1)
                arr_scaled = scaler.transform(arr).reshape(-1)
            else:
                # No scaler: just use raw features but ensure correct length
                arr_scaled = np.array([sim_row.get(c, 0.0) for c in FULL_FEATURE_LIST], dtype=float).reshape(-1)

            if "sim_seq" not in st.session_state:
                st.session_state.sim_seq = []
            st.session_state.sim_seq.append(arr_scaled)
            if len(st.session_state.sim_seq) > DEFAULT_LSTM_TIMESTEPS:
                st.session_state.sim_seq = st.session_state.sim_seq[-DEFAULT_LSTM_TIMESTEPS:]

            seq_len = len(st.session_state.sim_seq)
            if seq_len < DEFAULT_LSTM_TIMESTEPS:
                pad_count = DEFAULT_LSTM_TIMESTEPS - seq_len
                pad = [st.session_state.sim_seq[0]] * pad_count
                seq_arr = np.stack(pad + st.session_state.sim_seq, axis=0)
            else:
                seq_arr = np.stack(st.session_state.sim_seq[-DEFAULT_LSTM_TIMESTEPS:], axis=0)

            lstm_input = seq_arr.reshape(1, seq_arr.shape[0], seq_arr.shape[1])
            if lstm_model is not None:
                lstm_prob = float(lstm_model.predict(lstm_input, verbose=0)[0][0])
            else:
                lstm_prob = 0.0
        except Exception as e:
            lstm_prob = 0.0

        # Combine & smooth (reuse same smoothing as streaming)
        combined_prob = float((xgb_prob + lstm_prob))
        last_sim_prob = st.session_state.get("last_sim_prob", combined_prob)
        smooth_sim_prob = last_sim_prob + (combined_prob - last_sim_prob)
        st.session_state["last_sim_prob"] = smooth_sim_prob

        # Update gauge, charts, and status (use placeholders)
        try:
            render_gauge(smooth_sim_prob, green_threshold, yellow_threshold)
        except Exception:
            pass
        try:
            st.session_state["placeholders"]["chart_xgb"].add_rows(pd.DataFrame({"xgb_prob":[xgb_prob]}))
        except Exception:
            pass
        try:
            st.session_state["placeholders"]["chart_lstm"].add_rows(pd.DataFrame({"lstm_prob":[lstm_prob]}))
        except Exception:
            pass
        try:
            zone_txt, color, desc = zone_label(smooth_sim_prob, green_threshold, yellow_threshold)
            st.session_state["placeholders"]["status_area"].markdown(f"""
                <div style="padding:12px;border-radius:12px;background:rgba(255,255,255,0.03);
                    border-left:6px solid {color};box-shadow:0 0 18px {color}60;">
                <h3 style="margin:0;color:{color};font-size:22px">{zone_txt} (Simulator)</h3>
                <p style="margin:4px 0;font-size:14px;color:white">{desc}</p>
                <p style="margin:4px 0;color:lightgray">
                Combined Probability: <b style="color:{color}">{smooth_sim_prob:.3f}</b></p>
                <hr style="border:1px solid rgba(255,255,255,0.06)">
                <h4 style="color:white;margin-bottom:4px;">Simulator Telemetry</h4>
                <ul style="list-style:none;padding-left:8px;color:#dcdcdc;font-size:14px;line-height:1.4;">
                    <li><b>Throttle:</b> {throttle:.0f}%</li>
                    <li><b>Power (1/2 hp):</b> {st.session_state.sim_vals['power_1hp']:.1f} / {st.session_state.sim_vals['power_2hp']:.1f}</li>
                    <li><b>Thrust (1/2 lb):</b> {st.session_state.sim_vals['thrst_1lb']:.1f} / {st.session_state.sim_vals['thrst_2lb']:.1f}</li>
                    <li><b>RPM (1/2):</b> {st.session_state.sim_vals['rpm_1engin']:.0f} / {st.session_state.sim_vals['rpm_2engin']:.0f}</li>
                    <li><b>EGT (1/2 °C):</b> {st.session_state.sim_vals['EGT_1__deg']:.1f} / {st.session_state.sim_vals['EGT_2__deg']:.1f}</li>
                    <li><b>Fuel Press (1/2 psi):</b> {st.session_state.sim_vals['FUEP1__psi']:.1f} / {st.session_state.sim_vals['FUEP2__psi']:.1f}</li>
                </ul>
                </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass
    # ---------- SIMULATOR: END ----------

    # Manage stream start/stop state
    if "stream_running" not in st.session_state:
        st.session_state.stream_running = False

    if start_stream:
        st.session_state.stream_running = True
    if stop_stream:
        st.session_state.stream_running = False

    # Real streaming loop (reads file rows) - only run if enabled
    if st.session_state.stream_running:
        last_prob = st.session_state.get("last_prob", 0.0)
        for row in live_stream():
            if not st.session_state.stream_running:
                break
            features = clean_features_for_model(row)
            # XGB
            try:
                # align columns if possible
                if xgb_model is not None:
                    try:
                        feat_names = list(xgb_model.feature_names_in_)
                        features = features.reindex(columns=feat_names, fill_value=0.0)
                    except Exception:
                        pass
                xgb_prob = float(xgb_model.predict_proba(features)[0][1]) if xgb_model is not None else 0.0
            except Exception:
                xgb_prob = 0.0
            # LSTM
            try:
                scaled = scaler.transform(features)
                if "seq_buf" not in st.session_state:
                    st.session_state.seq_buf = []
                st.session_state.seq_buf.append(scaled.flatten())
                lstm_prob = float(lstm_model.predict(
                    np.array(st.session_state.seq_buf[-DEFAULT_LSTM_TIMESTEPS:]).reshape(1,DEFAULT_LSTM_TIMESTEPS,features.shape[1]),
                    verbose=0)[0][0]) if len(st.session_state.seq_buf)>=DEFAULT_LSTM_TIMESTEPS else 0.0
            except Exception:
                lstm_prob = 0.0

            combined = (xgb_prob + lstm_prob)
            smooth = last_prob + (combined - last_prob)
            last_prob = smooth
            st.session_state["last_prob"] = last_prob

            # update visuals
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
            try:
                proba = xgb_model.predict_proba(X)[:,1]
            except Exception:
                # attempt alignment
                try:
                    feat_names = list(xgb_model.feature_names_in_)
                    X = X.reindex(columns=feat_names, fill_value=0.0)
                    proba = xgb_model.predict_proba(X)[:,1]
                except Exception:
                    proba = np.zeros(X.shape[0])
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
                for col in expected_features:
                    if col not in df_num.columns:
                        df_num[col] = 0.0
                df_num = df_num[expected_features]
            else:
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
