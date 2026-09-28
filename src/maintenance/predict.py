"""
predict.py - Reusable Maintenance Failure Prediction API for Alibaba Cluster Trace 2017.
Supports both single-record dicts, raw DataFrames, preprocessed DataFrames, and batch CSV paths.
"""

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PIPELINE_PATH = os.path.join(BASE_DIR, "models", "maintenance", "preprocessing_pipeline.joblib")
DEFAULT_MODEL_PATH    = os.path.join(BASE_DIR, "models", "maintenance", "random_forest_maintenance.joblib")

MAINTENANCE_RAW_FEATURES = [
    "CPU_utilization",
    "Memory_utilization",
    "Disk_utilization",
    "Load_1min",
    "Load_5min",
    "Load_15min",
    "Duration_sec",
    "Server_Type"
]


def load_maintenance_pipeline(pipeline_path=DEFAULT_PIPELINE_PATH):
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"[PREDICT ERROR]: Maintenance pipeline not found at '{pipeline_path}'.")
    return joblib.load(pipeline_path)


def load_maintenance_model(model_path=DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[PREDICT ERROR]: Maintenance model not found at '{model_path}'.")
    return joblib.load(model_path)


def _preprocess_maintenance_input(df_raw, pipeline):
    """Applies scaler + OHE from the Alibaba maintenance pipeline."""
    feature_names = pipeline.get("feature_names", [])

    # Handle case where input is already preprocessed (contains exact feature_names)
    if all(c in df_raw.columns for c in feature_names):
        return df_raw[feature_names].values.astype(float)

    scaler   = pipeline.get("scaler")
    ohe      = pipeline.get("one_hot_encoder")
    num_cols = pipeline.get("numeric_cols", [])
    cat_cols = pipeline.get("categorical_cols", [])

    df = df_raw.copy()

    # Canonicalize telemetry / raw column names to expected model feature names
    rename_map = {
        "cpu_utilization": "CPU_utilization",
        "ram_utilization": "Memory_utilization",
        "memory_utilization": "Memory_utilization",
        "disk_utilization": "Disk_utilization",
        "load_1min": "Load_1min",
        "load_5min": "Load_5min",
        "load_15min": "Load_15min",
        "duration_sec": "Duration_sec",
        "server_type": "Server_Type",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Feature Engineering: Resource Pressure
    if 'CPU_utilization' in df.columns and 'Memory_utilization' in df.columns:
        df['Resource_pressure'] = (pd.to_numeric(df['CPU_utilization'], errors='coerce') * pd.to_numeric(df['Memory_utilization'], errors='coerce')) / 100.0

    # Fill defaults for missing numeric columns
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Fill defaults for missing categorical columns
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "ComputeNode"
        df[c] = df[c].astype(str).fillna("ComputeNode")

    if scaler is not None and len(num_cols) > 0:
        num_scaled = scaler.transform(df[num_cols])
    else:
        num_scaled = df[num_cols].values if len(num_cols) > 0 else np.empty((len(df), 0))

    if ohe is not None and len(cat_cols) > 0:
        cat_enc = ohe.transform(df[cat_cols])
    else:
        cat_enc = np.empty((len(df), 0))

    X = np.hstack([num_scaled, cat_enc]) if cat_enc.shape[1] > 0 else num_scaled
    return X


def predict_failure(input_data,
                    model_path=DEFAULT_MODEL_PATH,
                    pipeline_path=DEFAULT_PIPELINE_PATH):
    """
    Predicts machine/server failure probability for Alibaba Cluster Trace 2017 data.
    
    Supports:
    1. Single-record mode: input_data is a dict with feature values.
       Returns: {"failure_probability": float, "prediction": int, "risk_level": str}
    2. Batch mode: input_data is a CSV path or multi-row DataFrame.
       Returns: DataFrame with 'failure_probability' and 'prediction' columns.
    """
    pipeline = load_maintenance_pipeline(pipeline_path)
    model    = load_maintenance_model(model_path)

    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
        X  = _preprocess_maintenance_input(df, pipeline)
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else float(pred)

        if prob < 0.30:
            risk = "LOW"
        elif prob < 0.70:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        return {
            "prediction": pred,
            "failure_probability": round(prob, 4),
            "risk_level": risk
        }

    if isinstance(input_data, str):
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}")

    X     = _preprocess_maintenance_input(df, pipeline)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)

    df_out = df.copy()
    df_out["prediction"]          = preds
    df_out["failure_probability"] = np.round(probs, 4)
    return df_out


if __name__ == "__main__":
    sample = {
        "CPU_utilization": 85.5,
        "Memory_utilization": 90.0,
        "Disk_utilization": 45.0,
        "Load_1min": 22.4,
        "Load_5min": 18.2,
        "Load_15min": 14.5,
        "Duration_sec": 1200,
        "Server_Type": "ComputeNode"
    }
    print(predict_failure(sample))
