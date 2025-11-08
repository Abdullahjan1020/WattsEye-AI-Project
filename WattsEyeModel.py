"""
WATTS EYE - MODEL TRAINING SCRIPT
---------------------------------
This script loads all CSV files, preprocesses the data,
trains an Isolation Forest model for anomaly detection,
and saves the trained model to disk.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

# =====================================================
# CONFIGURATION
# =====================================================
DATA_FOLDER = r"C:\Users\bolan\.cache\kagglehub\datasets\eishahassan\pakistan-residential-electricity-consumption\versions\1"
MODEL_PATH = "watts_eye_iforest.pkl"
SAMPLE_PERCENT = 5          # % sampling from each CSV (for memory)
MAX_FILES = 42              # number of houses (CSV files)
TRAIN_SAMPLE_SIZE = 30000   # rows for model training
CONTAMINATION = 0.01        # expected anomaly ratio

# =====================================================
# 1️⃣ LOAD ALL CSV FILES
# =====================================================
def load_all_data(data_folder, sample_percent=5, max_files=42):
    csv_files = list(Path(data_folder).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the given folder.")

    csv_files = csv_files[:max_files]
    print(f"📂 Found {len(csv_files)} CSV files. Loading...")

    li = []
    for f in csv_files:
        try:
            df_temp = pd.read_csv(f)
            if len(df_temp) > 1000:
                df_temp = df_temp.sample(frac=sample_percent/100, random_state=42)
            df_temp["house_id"] = f.stem
            li.append(df_temp)
        except Exception as e:
            print(f"⚠️ Skipped {f.name}: {e}")
            continue

    df = pd.concat(li, ignore_index=True)
    print(f"✅ Loaded {len(df):,} total rows from {len(li)} houses.")
    return df


# =====================================================
# 2️⃣ PREPROCESSING
# =====================================================
def preprocess_data(df):
    df = df.copy()

    # Detect timestamp column
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    timestamp_col = time_cols[0] if time_cols else df.columns[0]
    if timestamp_col != "Date_Time":
        df = df.rename(columns={timestamp_col: "Date_Time"})
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
    df = df.dropna(subset=["Date_Time"])

    # Detect power column
    power_cols = [col for col in df.columns if any(x in col.lower() for x in ["usage", "power", "kw", "load", "consumption"])]
    power_col = power_cols[0] if power_cols else None
    if not power_col:
        raise ValueError("No power usage column found.")
    if power_col != "Usage_kW":
        df = df.rename(columns={power_col: "Usage_kW"})

    df = df.dropna(subset=["Usage_kW"])
    df = df[df["Usage_kW"] >= 0]  # remove invalid negatives

    # Keep essential columns
    df = df[["Date_Time", "Usage_kW", "house_id"]].copy()
    df = df.sort_values("Date_Time").reset_index(drop=True)

    print("✅ Preprocessing complete.")
    return df


# =====================================================
# 3️⃣ FEATURE ENGINEERING
# =====================================================
def feature_engineer(df):
    df = df.copy()
    df["hour"] = df["Date_Time"].dt.hour
    df["weekday"] = df["Date_Time"].dt.weekday
    df["date"] = df["Date_Time"].dt.date

    # Rolling mean & daily usage
    df["rolling_mean_15"] = df.groupby("house_id")["Usage_kW"].transform(
        lambda x: x.rolling(window=15, min_periods=1).mean())
    df["daily_kWh_approx"] = df.groupby(["house_id", "date"])["Usage_kW"].transform("mean")

    print("✅ Feature engineering done.")
    return df


# =====================================================
# 4️⃣ TRAIN MODEL
# =====================================================
def train_model(df, sample_size=30000, contamination=0.01):
    features = ["Usage_kW", "hour", "weekday", "rolling_mean_15"]
    df = df.dropna(subset=features)

    if len(df) > sample_size:
        df_train = df.sample(n=sample_size, random_state=42)
    else:
        df_train = df

    X_train = df_train[features].values

    print(f"🧠 Training Isolation Forest on {len(X_train):,} samples...")
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        max_samples=256,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train)
    print("✅ Model training complete.")
    return iso, features


# =====================================================
# 5️⃣ SAVE MODEL
# =====================================================
def save_model(model, features, path):
    joblib.dump((model, features), path)
    print(f"💾 Model saved to: {path}")


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    df_raw = load_all_data(DATA_FOLDER, SAMPLE_PERCENT, MAX_FILES)
    df_clean = preprocess_data(df_raw)
    df_feat = feature_engineer(df_clean)
    model, features_used = train_model(df_feat, TRAIN_SAMPLE_SIZE, CONTAMINATION)
    save_model(model, features_used, MODEL_PATH)
    print("🎉 Training pipeline finished successfully.")
