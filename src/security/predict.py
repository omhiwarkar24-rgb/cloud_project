"""
predict.py - Reusable Security Prediction API for single-record and batch CSV inference.
Applies the fitted preprocessing pipeline and serialized multiclass/binary model.
Handles both raw feature inputs and preprocessed feature vectors.
"""

import os
import joblib
import pandas as pd
import numpy as np

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PIPELINE_PATH = os.path.join(BASE_DIR, "models", "security", "preprocessing_pipeline.joblib")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "security", "xgboost_multiclass.joblib")


def load_security_pipeline(pipeline_path=DEFAULT_PIPELINE_PATH):
    """Loads and returns the serialized joblib preprocessing pipeline."""
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"[PREDICT ERROR]: Preprocessing pipeline not found at '{pipeline_path}'.")
    return joblib.load(pipeline_path)


def load_security_model(model_path=DEFAULT_MODEL_PATH):
    """Loads and returns the serialized joblib security classifier model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[PREDICT ERROR]: Security model file not found at '{model_path}'.")
    return joblib.load(model_path)


SECURITY_RISK_MAP = {
    'Normal': 'LOW',
    'Analysis': 'LOW',
    'Reconnaissance': 'MEDIUM',
    'Fuzzers': 'MEDIUM',
    'Generic': 'MEDIUM',
    'DoS': 'HIGH',
    'Exploits': 'CRITICAL',
    'Backdoor': 'CRITICAL',
    'Backdoors': 'CRITICAL',
    'Shellcode': 'CRITICAL',
    'Worms': 'CRITICAL',
}


def _get_risk_level(category):
    """Maps predicted attack category to risk level."""
    return SECURITY_RISK_MAP.get(str(category).strip(), 'MEDIUM')


def _preprocess_input(df_input, pipeline, strict=False):
    """Preprocesses input data, supporting both raw feature dicts and preprocessed DataFrames."""
    feature_names = pipeline['feature_names']
    
    # Check if input is already preprocessed (contains exact pipeline feature names)
    if all(col in df_input.columns for col in feature_names):
        return df_input[feature_names].values.astype(float)

    numeric_cols = pipeline['numeric_cols']
    categorical_cols = pipeline['categorical_cols']
    scaler = pipeline['scaler']
    ohe = pipeline['one_hot_encoder']

    if strict:
        missing = [c for c in numeric_cols + categorical_cols if c not in df_input.columns]
        if missing:
            raise KeyError(f"Missing required columns in input: {missing[:5]} (total {len(missing)} missing)")

    # Handle missing categorical / numeric features gracefully
    df_clean = df_input.copy()
    for col in categorical_cols:
        if col not in df_clean.columns:
            df_clean[col] = '-'
        df_clean[col] = df_clean[col].astype(str).fillna('-')

    for col in numeric_cols:
        if col not in df_clean.columns:
            df_clean[col] = 0.0
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col] = np.nan_to_num(df_clean[col], nan=0.0, posinf=1e6, neginf=-1e6)

    # Transform numeric and categorical features
    num_scaled = scaler.transform(df_clean[numeric_cols])
    cat_encoded = ohe.transform(df_clean[categorical_cols])

    X_processed = np.hstack([num_scaled, cat_encoded])
    return X_processed


def predict_attack(input_data, model_path=DEFAULT_MODEL_PATH, pipeline_path=DEFAULT_PIPELINE_PATH, strict=False):
    """
    Predicts network attack category and probability for input data.
    
    Supports:
    1. Single-record mode: input_data is a dict or single-row DataFrame.
       Returns: {"prediction": str, "probability": float, "risk_level": str}
    2. Batch mode: input_data is a CSV file path or multi-row DataFrame.
       Returns: DataFrame with 'predicted_attack', 'confidence_probability', and 'risk_level'.
    """
    pipeline = load_security_pipeline(pipeline_path)
    model = load_security_model(model_path)
    le_multiclass = pipeline['label_encoder_multiclass']
    attack_categories = pipeline['attack_categories']

    # --- Mode 1: Single Record Prediction ---
    if isinstance(input_data, dict):
        df_single = pd.DataFrame([input_data])
        X_proc = _preprocess_input(df_single, pipeline, strict=strict)
        
        pred_idx = model.predict(X_proc)[0]
        try:
            pred_label = le_multiclass.inverse_transform([pred_idx])[0]
        except Exception:
            pred_label = attack_categories[pred_idx] if pred_idx < len(attack_categories) else str(pred_idx)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_proc)[0]
            confidence = float(np.max(probs))
        else:
            confidence = 1.0

        return {
            "prediction": str(pred_label),
            "probability": round(confidence, 4),
            "risk_level": _get_risk_level(pred_label)
        }

    # --- Mode 2: Batch Prediction (CSV Path or Multi-Row DataFrame) ---
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"[PREDICT ERROR]: Batch input CSV file not found at '{input_data}'")
        df_batch = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df_batch = input_data.copy()
    else:
        raise TypeError(f"[PREDICT VALIDATION ERROR]: Unsupported input type '{type(input_data)}'.")

    if len(df_batch) == 1 and not isinstance(input_data, str):
        record_dict = df_batch.iloc[0].to_dict()
        return predict_attack(record_dict, model_path=model_path, pipeline_path=pipeline_path, strict=strict)

    X_proc = _preprocess_input(df_batch, pipeline, strict=strict)
    pred_indices = model.predict(X_proc)

    try:
        pred_labels = le_multiclass.inverse_transform(pred_indices)
    except Exception:
        pred_labels = [attack_categories[idx] if idx < len(attack_categories) else str(idx) for idx in pred_indices]

    if hasattr(model, "predict_proba"):
        probs_matrix = model.predict_proba(X_proc)
        confidences = np.max(probs_matrix, axis=1)
    else:
        confidences = np.ones(len(pred_indices))

    df_result = df_batch.copy()
    df_result['predicted_attack'] = pred_labels
    df_result['confidence_probability'] = np.round(confidences, 4)
    df_result['risk_level'] = [_get_risk_level(lbl) for lbl in pred_labels]

    return df_result


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=== Security Prediction API ===")
    test_csv = os.path.join(BASE_DIR, "data", "security", "UNSW_NB15_testing-set.csv")
    if os.path.exists(test_csv):
        df_sample = pd.read_csv(test_csv, nrows=5)
        
        # 1. Single-record demo
        sample_row = df_sample.iloc[0].to_dict()
        res = predict_attack(sample_row)
        print("\n[Demo 1] Single-Record Prediction:")
        print(f"  Prediction  : {res['prediction']}")
        print(f"  Probability : {res['probability']:.4f}")
        print(f"  Risk Level  : {res['risk_level']}")

        # 2. Batch demo
        df_res = predict_attack(df_sample)
        print("\n[Demo 2] Batch Prediction (5 records):")
        display_cols = [c for c in ['proto', 'service', 'state', 'dur', 'predicted_attack', 'confidence_probability', 'risk_level'] if c in df_res.columns]
        print(df_res[display_cols].to_string(index=False))
    else:
        sample = {
            "dur": 0.001,
            "proto": "udp",
            "service": "dns",
            "state": "INT",
            "spkts": 2,
            "dpkts": 0,
            "sbytes": 146,
            "dbytes": 0,
            "rate": 1000.0,
        }
        print("Sample prediction:", predict_attack(sample))
