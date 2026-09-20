"""
retrain.py - Environment B simulation, drift detection execution, adaptive retraining, and conditional deployment engine.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

try:
    from .drift_detection import detect_feature_drift_ks
except ImportError:
    from drift_detection import detect_feature_drift_ks

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "security", "processed", "train_processed.csv")
PROCESSED_TEST_PATH = os.path.join(BASE_DIR, "data", "security", "processed", "test_processed.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models", "security")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "adaptive")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def simulate_environment_b(df_source, random_state=42):
    """
    Simulates Environment B (Shifted Cloud Traffic Environment):
    1. Class Ratio Shift: Upsamples Attack instances (binary_label=1) to 75% to simulate a severe cyber-attack / DoS surge.
    2. Feature Distribution Shift: Adds mean shift (+1.5 std) and Gaussian noise to key traffic features ('dur', 'sbytes', 'rate', 'sttl').
    """
    np.random.seed(random_state)
    df_env_b = df_source.copy()

    # 1. Class Ratio Shift (Resample to 75% Attack, 25% Normal)
    df_attack = df_env_b[df_env_b['binary_label'] == 1]
    df_normal = df_env_b[df_env_b['binary_label'] == 0]

    # Oversample attack, undersample normal
    df_attack_upsampled = df_attack.sample(n=750, replace=True, random_state=random_state)
    df_normal_downsampled = df_normal.sample(n=250, replace=False, random_state=random_state)

    df_env_b = pd.concat([df_attack_upsampled, df_normal_downsampled], ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # 2. Feature Distribution Shift (Covariate Shift)
    shift_cols = ['dur', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'spkts', 'dpkts']
    shift_cols = [c for c in shift_cols if c in df_env_b.columns]

    for col in shift_cols:
        noise = np.random.normal(loc=1.5, scale=0.5, size=len(df_env_b))
        df_env_b[col] = df_env_b[col] + noise

    return df_env_b


def run_adaptive_pipeline():
    """
    Executes complete adaptive retraining flow:
    Environment A -> Shifted Environment B -> KS Drift Detection -> Adaptive Retraining -> Performance Validation -> Conditional Deployment
    """
    print("=" * 60)
    print("STARTING ADAPTIVE RETRAINING PIPELINE")
    print("=" * 60)

    # Load Environment A (Original Train)
    df_env_a = pd.read_csv(PROCESSED_TRAIN_PATH)
    feature_cols = [c for c in df_env_a.columns if c not in ['binary_label', 'multiclass_label', 'attack_cat_raw']]

    # Simulate Environment B
    df_env_b = simulate_environment_b(df_env_a)
    print(f"[ENV A] Shape: {df_env_a.shape} | Target Ratio: {df_env_a['binary_label'].mean():.2%}")
    print(f"[ENV B] Shape: {df_env_b.shape} | Target Ratio (Shifted): {df_env_b['binary_label'].mean():.2%}")

    # 1. Run Drift Detection (KS-Test)
    df_drift, drift_summary = detect_feature_drift_ks(df_env_a, df_env_b, feature_cols)
    print(f"\n--- DRIFT DETECTION SUMMARY ---")
    print(f"Total Features Evaluated : {drift_summary['total_features']}")
    print(f"Drifted Features Count   : {drift_summary['drifted_features_count']} ({drift_summary['pct_drifted']}%)")
    print(f"System Drift Flagged     : {drift_summary['system_drift_detected']}")

    # Plot & Save Drift Scores
    fig, ax = plt.subplots(figsize=(12, 6))
    top_drifted = df_drift.sort_values(by='ks_statistic', ascending=False).head(15)
    sns.barplot(data=top_drifted, x='ks_statistic', y='feature', hue='feature', palette='Reds_r', legend=False, ax=ax)
    ax.axvline(0.15, color='black', linestyle='--', label='Drift Threshold (0.15)')
    ax.set_title("Top 15 Feature Kolmogorov-Smirnov (KS) Drift Statistics", fontsize=12, fontweight='bold')
    ax.set_xlabel("KS Statistic (Distance)")
    ax.legend()
    drift_plot_path = os.path.join(RESULTS_DIR, "drift_scores_per_feature.png")
    plt.savefig(drift_plot_path, dpi=300)
    plt.close()
    print(f"[SAVE] Drift score plot saved to: {drift_plot_path}")

    # Split Environment B into Train & Validation
    X_b = df_env_b[feature_cols].values
    y_b = df_env_b['binary_label'].values
    X_b_train, X_b_val, y_b_train, y_b_val = train_test_split(X_b, y_b, test_size=0.3, random_state=42, stratify=y_b)

    # 2. Evaluate Old Model on Environment B Validation
    old_model_path = os.path.join(MODEL_DIR, "xgboost_binary.joblib")
    old_model = joblib.load(old_model_path)

    y_pred_old = old_model.predict(X_b_val)
    y_prob_old = old_model.predict_proba(X_b_val)[:, 1] if hasattr(old_model, "predict_proba") else y_pred_old

    acc_old = accuracy_score(y_b_val, y_pred_old)
    prec_old = precision_score(y_b_val, y_pred_old, zero_division=0)
    rec_old = recall_score(y_b_val, y_pred_old, zero_division=0)
    f1_old = f1_score(y_b_val, y_pred_old, zero_division=0)
    auc_old = roc_auc_score(y_b_val, y_prob_old)

    print(f"\n--- OLD MODEL PERFORMANCE ON SHIFTED ENV B ---")
    print(f"Accuracy: {acc_old:.4f} | F1: {f1_old:.4f} | ROC-AUC: {auc_old:.4f}")

    # 3. Adaptive Retraining (If Drift Detected)
    if drift_summary['system_drift_detected']:
        print("\n[ADAPTIVE RETRAINING] Drift detected! Training new model on combined data (Env A + Env B Train)...")
        X_a = df_env_a[feature_cols].values
        y_a = df_env_a['binary_label'].values

        X_combined = np.vstack([X_a, X_b_train])
        y_combined = np.concatenate([y_a, y_b_train])

        new_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        new_model.fit(X_combined, y_combined)

        y_pred_new = new_model.predict(X_b_val)
        y_prob_new = new_model.predict_proba(X_b_val)[:, 1] if hasattr(new_model, "predict_proba") else y_pred_new

        acc_new = accuracy_score(y_b_val, y_pred_new)
        prec_new = precision_score(y_b_val, y_pred_new, zero_division=0)
        rec_new = recall_score(y_b_val, y_pred_new, zero_division=0)
        f1_new = f1_score(y_b_val, y_pred_new, zero_division=0)
        auc_new = roc_auc_score(y_b_val, y_prob_new)

        print(f"\n--- NEW ADAPTIVE MODEL PERFORMANCE ON SHIFTED ENV B VALIDATION ---")
        print(f"Accuracy: {acc_new:.4f} | F1: {f1_new:.4f} | ROC-AUC: {auc_new:.4f}")

        # 4. Conditional Deployment Check
        is_deployed = bool(f1_new > f1_old or acc_new > acc_old)
        active_model_path = os.path.join(MODEL_DIR, "active_adaptive_model.joblib")

        if is_deployed:
            joblib.dump(new_model, active_model_path)
            deployment_status = "DEPLOYED (New Model Improved Validation F1/Accuracy)"
            print(f"\n[DEPLOYMENT SUCCESS]: New adaptive model outperforms old model! Saved to: {active_model_path}")
        else:
            joblib.dump(old_model, active_model_path)
            deployment_status = "RETAINED OLD MODEL (New Model Did Not Outperform)"
            print(f"\n[DEPLOYMENT SKIPPED]: Retained old model. Saved active checkpoint to: {active_model_path}")

    else:
        print("\n[ADAPTIVE RETRAINING] No system drift detected. Retaining old model.")
        acc_new, prec_new, rec_new, f1_new, auc_new = acc_old, prec_old, rec_old, f1_old, auc_old
        deployment_status = "RETAINED OLD MODEL (No Drift Flagged)"

    # Build & Export Comparison DataFrame
    df_comparison = pd.DataFrame([
        {"Model_Version": "Old Model (Pre-Drift)", "Accuracy": acc_old, "Precision": prec_old, "Recall": rec_old, "F1-Score": f1_old, "ROC-AUC": auc_old},
        {"Model_Version": "New Adaptive Model", "Accuracy": acc_new, "Precision": prec_new, "Recall": rec_new, "F1-Score": f1_new, "ROC-AUC": auc_new}
    ])

    csv_comp_path = os.path.join(RESULTS_DIR, "adaptive_retraining_comparison.csv")
    df_comparison.to_csv(csv_comp_path, index=False)

    # Plot & Save Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    df_plot_melt = pd.melt(df_comparison, id_vars=['Model_Version'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
    sns.barplot(data=df_plot_melt, x='variable', y='value', hue='Model_Version', palette=['#e74c3c', '#2ecc71'], ax=ax)
    ax.set_title("Adaptive Retraining: Old vs. New Model Performance on Shifted Env B", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Model Version")
    p_perf = os.path.join(RESULTS_DIR, "old_vs_new_model_performance.png")
    plt.savefig(p_perf, dpi=300)
    plt.close()

    # Save metrics to CSV
    df_metrics = pd.DataFrame([{
        "Metric": "Accuracy",
        "Before Adaptation (Old Model)": acc_old,
        "After Adaptation (New Model)": acc_new
    }, {
        "Metric": "F1-Score",
        "Before Adaptation (Old Model)": f1_old,
        "After Adaptation (New Model)": f1_new
    }, {
        "Metric": "ROC-AUC",
        "Before Adaptation (Old Model)": auc_old,
        "After Adaptation (New Model)": auc_new
    }])
    df_metrics.to_csv(os.path.join(RESULTS_DIR, "adaptive_learning_metrics.csv"), index=False)

    print(f"[SAVE] Performance comparison plot saved to: {p_perf}")

    return {
        "drift_summary": drift_summary,
        "df_drift": df_drift,
        "df_comparison": df_comparison,
        "deployment_status": deployment_status,
        "drift_plot_path": drift_plot_path,
        "perf_plot_path": p_perf
    }


if __name__ == "__main__":
    run_adaptive_pipeline()
