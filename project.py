"""
WattsEye - Production Streamlit App
- Loads pretrained IsolationForest model (watts_eye_iforest.pkl) if available
- If requested, allows retraining (explicit user action)
- Preprocess -> Feature Engineer -> Predict (batch) -> Rule overlay -> Dashboard
- Shows infographics / explanation panels during loading so judges immediately grasp the project's purpose
"""

import os
from pathlib import Path
import io
import traceback

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

import streamlit as st
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
MODEL_PATH = "watts_eye_iforest.pkl"   # expected pretrained model file (joblib dump of (model, features))
DEFAULT_MAX_FILES = 42

st.set_page_config(page_title="WattsEye - Production", layout="wide")
st.title("⚡ WattsEye ")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Run Controls")

# Option to choose between upload or path
data_mode = st.sidebar.radio(
    "Select data source:",
    ("Upload CSV files", "Use folder path (local directory)")
)

uploaded_files = None
data_folder = None
if data_mode == "Upload CSV files":
    use_uploaded = True
    uploaded_files = st.sidebar.file_uploader("Upload CSVs (multi)", accept_multiple_files=True, type=["csv"])
else:
    use_uploaded = False
    data_folder = st.sidebar.text_input("Enter full folder path", value=str(Path.cwd()))

st.sidebar.markdown("---")
st.sidebar.subheader("Model / Execution Mode")
load_pretrained = st.sidebar.checkbox("Load pretrained model (recommended)", value=True)
allow_retrain = st.sidebar.checkbox("Allow retrain (only if you want to retrain here)", value=False)
train_if_no_model = st.sidebar.checkbox("Train automatically if no pretrained model found", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Options")
sample_percent = st.sidebar.slider("Sampling percent (per CSV, for speed)", 1, 100, 5)
batch_size = st.sidebar.number_input("Batch size for predictions", min_value=1000, max_value=200000, value=50000, step=1000)
contamination = st.sidebar.number_input("Model contamination (if training)", min_value=0.0001, max_value=0.1, value=0.01, format="%.4f")

st.sidebar.markdown("---")
run_infer = st.sidebar.button("▶ Run Inference (Load data & detect anomalies)")
run_train = st.sidebar.button("🧠 Train & Save Model ")


st.sidebar.markdown("---")
st.sidebar.info("Develped by Team 404 Not Found.")

# -------------------------
# Infographic / Quick Pitch (top)
# -------------------------
def show_infographic():
    st.markdown("## Quick project summary")
    col1, col2 = st.columns([1,2])
    with col1:
        st.image(
            # small inline schematic using matplotlib as image
            create_simple_infographic(),
            caption="WattsEye — Detects suspicious electricity usage (theft, faults, abnormal loads)",
            use_container_width=True
        )
    with col2:
        st.markdown("**What this project does :**")
        st.write("WattsEye uses an AI anomaly detector + rule-based checks to flag suspicious electricity consumption at the household level — fast, explainable, and deployable.")
        st.markdown("**Why it matters :**")
        st.write("- Helps distribution companies (like QESCO) find theft/tampering and abnormal loads.")
        st.write("- Hybrid approach: AI = adaptability, Rules = explainability for field inspections.")
        st.write(" Upload a CSV (or select local folder) and click *Run Inference* — flagged anomalies will be shown and downloadable as CSV.")
    st.markdown("---")

def create_simple_infographic():
    # generate a small matplotlib diagram and return as an image buffer
    fig, ax = plt.subplots(figsize=(4,3))
    stages = ["Raw CSVs", "Preprocess", "Features", "AI Detector", "Rules", "Alerts"]
    x = np.arange(len(stages))
    heights = [1, 0.9, 0.8, 1.5, 1.0, 1.3]
    ax.barh(x, heights)
    ax.set_yticks(x)
    ax.set_yticklabels(stages)
    ax.invert_yaxis()
    ax.set_xlabel("Processing pipeline (conceptual)")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# Show infographic at top
show_infographic()

# -------------------------
# Helper functions (preprocess, features, rules, batch predict)
# -------------------------
def detect_power_column_safe(df):
    priority_cols = ['usage', 'power', 'kw', 'consumption', 'load', 'kwh']
    for col in df.columns:
        col_lower = col.lower()
        if any(p in col_lower for p in priority_cols):
            return col
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return numeric_cols[0] if len(numeric_cols)>0 else None

def preprocess_df(df):
    df = df.copy()
    # timestamp detection
    time_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
    timestamp_col = time_cols[0] if time_cols else df.columns[0]
    if timestamp_col != "Date_Time":
        df = df.rename(columns={timestamp_col: "Date_Time"})
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors='coerce')
    df = df.dropna(subset=["Date_Time"])
    # power col
    power_col = detect_power_column_safe(df)
    if not power_col:
        raise RuntimeError("No power column found.")
    if power_col != "Usage_kW":
        df = df.rename(columns={power_col: "Usage_kW"})
    df = df.dropna(subset=["Usage_kW"])
    df = df[df["Usage_kW"] >= 0]
    # minimal keep
    if "house_id" not in df.columns:
        df["house_id"] = "house_unk"
    df = df[["Date_Time","Usage_kW","house_id"]].copy()
    df = df.sort_values("Date_Time").reset_index(drop=True)
    return df

def feature_engineer_df(df):
    df = df.copy()
    df["hour"] = df["Date_Time"].dt.hour
    df["weekday"] = df["Date_Time"].dt.weekday
    df["date"] = df["Date_Time"].dt.date
    df["rolling_mean_15"] = df.groupby("house_id")["Usage_kW"].transform(lambda x: x.rolling(window=15, min_periods=1).mean())
    df["daily_kWh_approx"] = df.groupby(["house_id","date"])["Usage_kW"].transform("mean")
    return df

# QESCO rule thresholds (tweakable)
DAILY_EXTREME = 25.0
INSTANT_EXTREME = 3.0

def apply_rules(df):
    df = df.copy()
    df["extreme_daily"] = (df["daily_kWh_approx"] > DAILY_EXTREME).astype(int)
    df["extreme_instant"] = (df["Usage_kW"] > INSTANT_EXTREME).astype(int)
    df["zero_consumption"] = (df["Usage_kW"] == 0).astype(int)
    return df

def build_final_alerts(df):
    df = df.copy()
    # quantile on anomaly score
    ai_threshold = df["anomaly_score"].quantile(0.98) if len(df)>1000 else df["anomaly_score"].max()
    df["final_alert"] = (
        (df["extreme_daily"]==1) |
        (df["extreme_instant"]==1) |
        ((df["anomaly_flag"]==1) & (df["anomaly_score"]>ai_threshold))
    ).astype(int)
    df["severity"] = 0.0
    mask = df["final_alert"]==1
    df.loc[mask,"severity"] = df.loc[mask, "extreme_daily"]*0.4 + df.loc[mask,"extreme_instant"]*0.4 + df.loc[mask,"anomaly_score"]*0.2
    def action(s, a):
        if a==0: return "Normal"
        if s>0.6: return "URGENT"
        if s>0.3: return "HIGH"
        return "MEDIUM"
    df["action"] = df.apply(lambda r: action(r["severity"], r["final_alert"]), axis=1)
    return df

def apply_model_batches(df, iso, features, batch_size=50000):
    n = len(df)
    scores = np.zeros(n, dtype=np.float32)
    flags = np.zeros(n, dtype=np.int8)
    # ensure features exist
    for f in features:
        if f not in df.columns:
            df[f] = 0
    for i in range(0, n, batch_size):
        end = min(i+batch_size, n)
        X = df.iloc[i:end][features].fillna(0).values
        scores[i:end] = -iso.decision_function(X)
        preds = iso.predict(X)
        flags[i:end] = (preds == -1).astype(np.int8)
    df_res = df.copy()
    df_res["anomaly_score"] = scores
    df_res["anomaly_flag"] = flags
    return df_res

# -------------------------
# Model loading/training helpers
# -------------------------
def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            model, features = joblib.load(path)
            return model, features
        except Exception as e:
            st.warning(f"Failed to load model at {path}: {e}")
            return None, None
    return None, None

def train_and_save_model(df, model_path=MODEL_PATH, contamination_val=0.01, sample_size=30000):
    # prepare features
    df = df.copy()
    features = ["Usage_kW","hour","weekday","rolling_mean_15"]
    df = df.dropna(subset=features)
    if len(df)>sample_size:
        df_train = df.sample(n=sample_size, random_state=42)
    else:
        df_train = df
    X_train = df_train[features].values
    st.info(f"Training IsolationForest on {len(X_train):,} samples (contamination={contamination_val})")
    iso = IsolationForest(n_estimators=100, contamination=contamination_val, max_samples=256, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    joblib.dump((iso, features), model_path)
    st.success(f"Model trained and saved to {model_path}")
    return iso, features

# -------------------------
# Data loading utilities
# -------------------------
def load_from_folder(folder_path, sample_percent=5, max_files=DEFAULT_MAX_FILES):
    p = Path(folder_path)
    if not p.exists():
        raise FileNotFoundError("Folder not found")
    files = sorted(list(p.glob("*.csv")))[:max_files]
    parts = []
    for f in files:
        try:
            df_temp = pd.read_csv(f)
            if len(df_temp)>1000 and sample_percent < 100:
                df_temp = df_temp.sample(frac=sample_percent/100.0, random_state=42)
            df_temp["house_id"] = f.stem
            parts.append(df_temp)
        except Exception as e:
            st.warning(f"Skipped {f.name}: {e}")
    if not parts:
        raise RuntimeError("No CSVs loaded from folder")
    df = pd.concat(parts, ignore_index=True)
    return df

def load_from_uploads(uploaded_files, sample_percent=5):
    parts = []
    for uf in uploaded_files:
        try:
            df_temp = pd.read_csv(uf)
            if len(df_temp)>1000 and sample_percent < 100:
                df_temp = df_temp.sample(frac=sample_percent/100.0, random_state=42)
            df_temp["house_id"] = Path(uf.name).stem
            parts.append(df_temp)
        except Exception as e:
            st.warning(f"Skipped upload {getattr(uf,'name',str(uf))}: {e}")
    if not parts:
        raise RuntimeError("No uploaded CSVs loaded")
    df = pd.concat(parts, ignore_index=True)
    return df

# -------------------------
# Main execution triggered by buttons
# -------------------------
def run_inference_pipeline():
    # 1. load model (if requested)
    iso_model = None
    features_used = None
    model_loaded = False
    if load_pretrained:
        iso_model, features_used = load_model(MODEL_PATH)
        if iso_model is not None:
            model_loaded = True
            st.success("✅ Pretrained model loaded.")
        else:
            st.warning("No pretrained model found or failed to load.")

    # 2. load data
    try:
        if use_uploaded:
            if not uploaded_files:
                st.error("Please upload files first.")
                return
            df_raw = load_from_uploads(uploaded_files, sample_percent)
        else:
            df_raw = load_from_folder(data_folder, sample_percent)
        st.info(f"Loaded {len(df_raw):,} raw records.")
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.caption(traceback.format_exc())
        return

    # show preview
    st.subheader("Data Preview (after initial read)")
    st.dataframe(df_raw.head(5))

    # 3. preprocess + features
    try:
        df_clean = preprocess_df(df_raw)
        df_feat = feature_engineer_df(df_clean)
        st.success("Preprocessing & feature engineering done.")
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.caption(traceback.format_exc())
        return

    # 4. train model if needed
    if (not model_loaded) and (train_if_no_model or allow_retrain):
        if allow_retrain or train_if_no_model:
            try:
                iso_model, features_used = train_and_save_model(df_feat, MODEL_PATH, contamination, sample_size=30000)
                model_loaded = True
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.caption(traceback.format_exc())
                return
    elif not model_loaded:
        st.warning("No model loaded. Either enable retrain or provide pretrained model file.")
        return

    # 5. apply model in batches
    try:
        df_scored = apply_model_batches(df_feat, iso_model, features_used, batch_size=int(batch_size))
        st.success("Model scoring complete.")
    except Exception as e:
        st.error(f"Scoring failed: {e}")
        st.caption(traceback.format_exc())
        return

    # 6. rules + final alerts
    df_ruled = apply_rules(df_scored)
    df_final = build_final_alerts(df_ruled)
    st.success("Final alerts generated.")

    # 7. dashboard & download
    anomalies = df_final[df_final["final_alert"]==1].copy()
    st.markdown("### Results Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df_final):,}")
    with col2:
        st.metric("Anomalies Found", f"{len(anomalies):,}")
    with col3:
        rate = len(anomalies)/len(df_final)*100 if len(df_final)>0 else 0
        st.metric("Anomaly Rate", f"{rate:.4f}%")

    st.markdown("### Priority Breakdown")
    st.bar_chart(anomalies["action"].value_counts())

    st.markdown("### Top Houses")
    house_summary = anomalies.groupby("house_id").agg({"severity":"max","final_alert":"count","action":lambda x: x.value_counts().index[0]}).rename(columns={"final_alert":"count"}).sort_values("count",ascending=False).head(15)
    st.dataframe(house_summary)

    st.markdown("### Sample Flagged Records")
    st.dataframe(anomalies.sort_values("severity",ascending=False).head(20))

    # download anomalies
    csv_buf = io.StringIO()
    anomalies.to_csv(csv_buf, index=False)
    st.download_button("📥 Download anomalies CSV", data=csv_buf.getvalue().encode(), file_name="watts_eye_anomalies.csv", mime="text/csv")

# Buttons
if run_infer:
    run_inference_pipeline()

if run_train and allow_retrain:
    # retrain explicitly using provided data (must be uploaded or folder)
    try:
        if use_uploaded:
            if not uploaded_files:
                st.error("Upload files first to train.")
            else:
                df_raw = load_from_uploads(uploaded_files, sample_percent)
        else:
            df_raw = load_from_folder(data_folder, sample_percent)
        df_clean = preprocess_df(df_raw)
        df_feat = feature_engineer_df(df_clean)
        train_and_save_model(df_feat, MODEL_PATH, contamination, sample_size=30000)
    except Exception as e:
        st.error(f"Retrain failed: {e}")
        st.caption(traceback.format_exc())

# Show footer help
st.markdown("---")
st.caption("⚡ WattsEye Production — Loads pretrained model by default. Use retrain only if you intentionally want to update the model.")
