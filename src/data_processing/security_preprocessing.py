"""
security_preprocessing.py - Leakage-free preprocessing module for UNSW-NB15 Cloud Security Dataset.
Fits all transformers strictly on the training set and transforms both training and testing sets.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TRAIN_PATH = os.path.join(BASE_DIR, "data", "security", "UNSW_NB15_training-set.csv")
DEFAULT_TEST_PATH = os.path.join(BASE_DIR, "data", "security", "UNSW_NB15_testing-set.csv")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "security", "processed")
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models", "security")


def preprocess_unsw_nb15(train_path=DEFAULT_TRAIN_PATH,
                         test_path=DEFAULT_TEST_PATH,
                         output_dir=DEFAULT_OUTPUT_DIR,
                         model_dir=DEFAULT_MODEL_DIR):
    """
    Performs leakage-free preprocessing on UNSW-NB15 dataset.
    
    Steps:
    1. Loads raw train and test CSVs.
    2. Drops non-informative identifier columns ('id').
    3. Handles missing values & infinite values cleanly.
    4. Fits OneHotEncoder on categorical features ('proto', 'service', 'state') using TRAIN data only.
    5. Fits StandardScaler on numeric features using TRAIN data only.
    6. Creates binary_label (0/1) and multiclass_label (0..N) targets.
    7. Preserves raw 'attack_cat' string labels for evaluation/display.
    8. Exports processed CSVs to output_dir and pipeline joblib to model_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"[SECURITY PREPROCESSING] Loading raw train: {train_path}")
    print(f"[SECURITY PREPROCESSING] Loading raw test: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 1. Drop irrelevant identifier columns ('id')
    cols_to_drop = ['id']
    for col in cols_to_drop:
        if col in df_train.columns:
            df_train = df_train.drop(columns=[col])
            print(f"[SECURITY PREPROCESSING] Dropped '{col}' column from Train dataset.")
        if col in df_test.columns:
            df_test = df_test.drop(columns=[col])
            print(f"[SECURITY PREPROCESSING] Dropped '{col}' column from Test dataset.")

    # 2. Extract targets and raw string labels
    y_train_binary = df_train['label'].values.astype(int)
    y_test_binary = df_test['label'].values.astype(int)

    attack_cat_train_raw = df_train['attack_cat'].astype(str).str.strip().values
    attack_cat_test_raw = df_test['attack_cat'].astype(str).str.strip().values

    # Encode multiclass target using LabelEncoder fit strictly on train target
    le_multiclass = LabelEncoder()
    # Ensure all test categories are present in fit classes
    all_cats = list(set(attack_cat_train_raw).union(set(attack_cat_test_raw)))
    le_multiclass.fit(all_cats)

    y_train_multiclass = le_multiclass.transform(attack_cat_train_raw)
    y_test_multiclass = le_multiclass.transform(attack_cat_test_raw)

    # Remove target columns from feature DataFrames
    target_cols = ['label', 'attack_cat']
    X_train_raw = df_train.drop(columns=[c for c in target_cols if c in df_train.columns])
    X_test_raw = df_test.drop(columns=[c for c in target_cols if c in df_test.columns])

    # 3. Identify categorical and numerical features
    categorical_cols = ['proto', 'service', 'state']
    categorical_cols = [c for c in categorical_cols if c in X_train_raw.columns]
    numeric_cols = [c for c in X_train_raw.columns if c not in categorical_cols]

    # Handle NaNs / Infs in numeric features
    for col in numeric_cols:
        X_train_raw[col] = np.nan_to_num(X_train_raw[col], nan=X_train_raw[col].median(), posinf=1e6, neginf=-1e6)
        X_test_raw[col] = np.nan_to_num(X_test_raw[col], nan=X_train_raw[col].median(), posinf=1e6, neginf=-1e6)

    # 4. One-Hot Encode categorical features (Fit ONLY on Train)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(X_train_raw[categorical_cols])

    train_cat_encoded = ohe.transform(X_train_raw[categorical_cols])
    test_cat_encoded = ohe.transform(X_test_raw[categorical_cols])

    encoded_cat_names = list(ohe.get_feature_names_out(categorical_cols))

    # 5. StandardScale numeric features (Fit ONLY on Train)
    scaler = StandardScaler()
    scaler.fit(X_train_raw[numeric_cols])

    train_num_scaled = scaler.transform(X_train_raw[numeric_cols])
    test_num_scaled = scaler.transform(X_test_raw[numeric_cols])

    # Combine scaled numeric + encoded categorical features
    X_train_processed = np.hstack([train_num_scaled, train_cat_encoded])
    X_test_processed = np.hstack([test_num_scaled, test_cat_encoded])

    feature_names = numeric_cols + encoded_cat_names

    # Convert back to clean DataFrames for export
    df_train_proc = pd.DataFrame(X_train_processed, columns=feature_names)
    df_train_proc['binary_label'] = y_train_binary
    df_train_proc['multiclass_label'] = y_train_multiclass
    df_train_proc['attack_cat_raw'] = attack_cat_train_raw

    df_test_proc = pd.DataFrame(X_test_processed, columns=feature_names)
    df_test_proc['binary_label'] = y_test_binary
    df_test_proc['multiclass_label'] = y_test_multiclass
    df_test_proc['attack_cat_raw'] = attack_cat_test_raw

    # Export Processed Datasets
    train_out_path = os.path.join(output_dir, "train_processed.csv")
    test_out_path = os.path.join(output_dir, "test_processed.csv")

    df_train_proc.to_csv(train_out_path, index=False)
    df_test_proc.to_csv(test_out_path, index=False)
    print(f"[SECURITY PREPROCESSING] Train processed exported to: {train_out_path}")
    print(f"[SECURITY PREPROCESSING] Test processed exported to: {test_out_path}")

    # Save Pipeline Object
    pipeline_obj = {
        'one_hot_encoder': ohe,
        'scaler': scaler,
        'label_encoder_multiclass': le_multiclass,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'feature_names': feature_names,
        'attack_categories': list(le_multiclass.classes_)
    }

    pipeline_path = os.path.join(model_dir, "preprocessing_pipeline.joblib")
    joblib.dump(pipeline_obj, pipeline_path)
    print(f"[SECURITY PREPROCESSING] Pipeline joblib saved to: {pipeline_path}")

    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train_binary': y_train_binary,
        'y_test_binary': y_test_binary,
        'y_train_multiclass': y_train_multiclass,
        'y_test_multiclass': y_test_multiclass,
        'attack_cat_train_raw': attack_cat_train_raw,
        'attack_cat_test_raw': attack_cat_test_raw,
        'feature_names': feature_names,
        'pipeline_path': pipeline_path,
        'multiclass_classes': list(le_multiclass.classes_)
    }


if __name__ == "__main__":
    res = preprocess_unsw_nb15()
    print("=" * 60)
    print("SUMMARY OF PREPROCESSED DATASET SHAPES:")
    print(f"X_train shape               : {res['X_train'].shape}")
    print(f"X_test shape                : {res['X_test'].shape}")
    print(f"y_train_binary shape        : {res['y_train_binary'].shape}")
    print(f"y_test_binary shape         : {res['y_test_binary'].shape}")
    print(f"y_train_multiclass shape    : {res['y_train_multiclass'].shape}")
    print(f"y_test_multiclass shape     : {res['y_test_multiclass'].shape}")
    print("=" * 60)
