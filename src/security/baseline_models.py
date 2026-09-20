"""
baseline_models.py - Training, evaluation, and serialization engine for 5 baseline classifiers
on UNSW-NB15 Binary and Multiclass Security Tasks.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "security", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "security")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "security")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def calculate_rates(y_true, y_pred, is_multiclass=False):
    """Calculates False Positive Rate (FPR) and False Negative Rate (FNR)."""
    cm = confusion_matrix(y_true, y_pred)
    if not is_multiclass and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        # Multiclass overall macro/weighted FPR and FNR
        fp_list, fn_list, tn_list, tp_list = [], [], [], []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fp + fn)
            
            fp_list.append(fp)
            fn_list.append(fn)
            tn_list.append(tn)
            tp_list.append(tp)
            
        total_fp = np.sum(fp_list)
        total_tn = np.sum(tn_list)
        total_fn = np.sum(fn_list)
        total_tp = np.sum(tp_list)
        
        fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
        fnr = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0.0
        
    return fpr, fnr


def get_baseline_classifiers(random_state=42):
    """Instantiates the 5 specified baseline models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss'),
        "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state)
    }
    return models


def train_and_evaluate_security_baselines():
    """Trains 5 baseline classifiers on Binary and Multiclass security tasks."""
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "train_processed.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test_processed.csv"))

    feature_cols = [c for c in train_df.columns if c not in ['binary_label', 'multiclass_label', 'attack_cat_raw']]
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    y_tr_bin = train_df['binary_label'].values
    y_te_bin = test_df['binary_label'].values

    y_tr_multi = train_df['multiclass_label'].values
    y_te_multi = test_df['multiclass_label'].values

    results_list = []
    models_dict = get_baseline_classifiers()

    # --- 1. BINARY SECURITY TASK ---
    print("=" * 60)
    print("STARTING BINARY SECURITY BASELINE MODEL TRAINING")
    print("=" * 60)

    for name, model in models_dict.items():
        print(f"[BINARY] Training {name}...")
        model.fit(X_train, y_tr_bin)

        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_te_bin, y_prob)
        else:
            auc = np.nan

        acc = accuracy_score(y_te_bin, y_pred)
        prec = precision_score(y_te_bin, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_te_bin, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_te_bin, y_pred, average='binary', zero_division=0)
        fpr, fnr = calculate_rates(y_te_bin, y_pred, is_multiclass=False)

        # Plot & save confusion matrix
        cm = confusion_matrix(y_te_bin, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        ax.set_title(f"Binary Confusion Matrix: {name}", fontsize=12, fontweight='bold')
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        
        safe_name = name.lower().replace(" ", "_")
        cm_path = os.path.join(PLOT_DIR, f"cm_binary_{safe_name}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save feature importance if tree-based
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:] # top 10
            plt.figure(figsize=(8, 5))
            plt.title(f"Feature Importances (Binary): {name}")
            plt.barh(range(len(indices)), importances[indices], align="center")
            plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
            plt.xlabel("Relative Importance")
            fi_path = os.path.join(PLOT_DIR, f"fi_binary_{safe_name}.png")
            plt.savefig(fi_path, dpi=300, bbox_inches='tight')
            plt.close()

        # SHAP for the best model (e.g., Random Forest or XGBoost)
        if name in ["Random Forest", "XGBoost"]:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test[:500]) # subset to save time
                shap.summary_plot(shap_values, X_test[:500], feature_names=feature_cols, show=False)
                shap_path = os.path.join(PLOT_DIR, f"shap_binary_{safe_name}.png")
                plt.savefig(shap_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"SHAP failed for {name}: {e}")

        # Save trained model
        model_save_path = os.path.join(MODEL_DIR, f"{safe_name}_binary.joblib")
        joblib.dump(model, model_save_path)

        results_list.append({
            "Task": "Binary",
            "Model": name,
            "Accuracy": acc,
            "Precision (Binary/Weighted)": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "FPR": fpr,
            "FNR": fnr,
            "Model_Path": model_save_path,
            "CM_Path": cm_path
        })
        print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f} | FPR: {fpr:.4f}")

    # --- 2. MULTICLASS SECURITY TASK ---
    print("=" * 60)
    print("STARTING MULTICLASS SECURITY BASELINE MODEL TRAINING")
    print("=" * 60)

    multiclass_models = get_baseline_classifiers()

    for name, model in multiclass_models.items():
        print(f"[MULTICLASS] Training {name}...")
        model.fit(X_train, y_tr_multi)

        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
                auc = roc_auc_score(y_te_multi, y_prob, multi_class='ovr', average='weighted')
            except Exception:
                auc = np.nan
        else:
            auc = np.nan

        acc = accuracy_score(y_te_multi, y_pred)
        prec = precision_score(y_te_multi, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_te_multi, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_te_multi, y_pred, average='weighted', zero_division=0)
        fpr, fnr = calculate_rates(y_te_multi, y_pred, is_multiclass=True)

        # Plot & save confusion matrix
        cm = confusion_matrix(y_te_multi, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax)
        ax.set_title(f"Multiclass Confusion Matrix: {name}", fontsize=12, fontweight='bold')
        ax.set_ylabel("True Class Index")
        ax.set_xlabel("Predicted Class Index")
        
        safe_name = name.lower().replace(" ", "_")
        cm_path = os.path.join(PLOT_DIR, f"cm_multiclass_{safe_name}.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()

        # Save trained model
        model_save_path = os.path.join(MODEL_DIR, f"{safe_name}_multiclass.joblib")
        joblib.dump(model, model_save_path)

        results_list.append({
            "Task": "Multiclass",
            "Model": name,
            "Accuracy": acc,
            "Precision (Binary/Weighted)": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "FPR": fpr,
            "FNR": fnr,
            "Model_Path": model_save_path,
            "CM_Path": cm_path
        })
        print(f"  Accuracy: {acc:.4f} | F1 (Weighted): {f1:.4f} | ROC-AUC (OvR): {auc:.4f} | FPR: {fpr:.4f}")

    # Build & Export Comparison Table
    df_results = pd.DataFrame(results_list)
    csv_out_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_results.to_csv(csv_out_path, index=False)
    print("=" * 60)
    print(f"MODEL COMPARISON TABLE EXPORTED TO: {csv_out_path}")
    print("=" * 60)

    return df_results


if __name__ == "__main__":
    train_and_evaluate_security_baselines()
