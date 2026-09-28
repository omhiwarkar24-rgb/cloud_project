"""
baseline_models.py - Training, evaluation, and serialization engine for 5 baseline classifiers
on Alibaba Cluster Trace 2017 Predictive Maintenance failure detection.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "maintenance", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "maintenance")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "maintenance")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def get_maintenance_baseline_classifiers(imbalance_ratio=2.59, random_state=42):
    """Instantiates baseline models configured with class imbalance weights."""
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=random_state),
        "XGBoost": XGBClassifier(scale_pos_weight=imbalance_ratio, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=random_state, eval_metric='logloss'),
        "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state)
    }
    return models


def train_and_evaluate_maintenance_baselines():
    """Trains 5 baseline classifiers on Alibaba Cluster Trace 2017 server failure task."""
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "train_processed.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test_processed.csv"))

    feature_cols = [c for c in train_df.columns if c != 'Machine_failure']
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    y_train = train_df['Machine_failure'].values
    y_test = test_df['Machine_failure'].values

    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    imbalance_ratio = n_neg / n_pos if n_pos > 0 else 1.0

    print("=" * 60)
    print(f"STARTING ALIBABA CLUSTER TRACE 2017 MAINTENANCE BASELINE TRAINING")
    print(f"Training Class Ratio: {n_neg} Normal vs {n_pos} Failures ({imbalance_ratio:.2f}:1)")
    print("=" * 60)

    models_dict = get_maintenance_baseline_classifiers(imbalance_ratio=imbalance_ratio)
    results_list = []

    for name, model in models_dict.items():
        print(f"[ALIBABA MAINTENANCE] Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = np.nan

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax, xticklabels=['Normal (0)', 'Failure (1)'], yticklabels=['Normal (0)', 'Failure (1)'])
        ax.set_title(f"Alibaba Maintenance CM: {name}", fontsize=12, fontweight='bold')
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        
        safe_name = name.lower().replace(" ", "_")
        cm_path = os.path.join(PLOT_DIR, f"cm_{safe_name}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(8, 5))
            plt.title(f"Alibaba Feature Importances: {name}")
            plt.barh(range(len(indices)), importances[indices[::-1]], align="center")
            plt.yticks(range(len(indices)), [feature_cols[i] for i in indices[::-1]])
            plt.xlabel("Relative Importance")
            fi_path = os.path.join(PLOT_DIR, f"fi_{safe_name}.png")
            plt.savefig(fi_path, dpi=300, bbox_inches='tight')
            plt.close()

        # Model Save
        save_fname = f"{safe_name}_maintenance.joblib"
        model_save_path = os.path.join(MODEL_DIR, save_fname)
        joblib.dump(model, model_save_path)

        rel_model_path = os.path.relpath(model_save_path, BASE_DIR).replace("\\", "/")
        rel_cm_path = os.path.relpath(cm_path, BASE_DIR).replace("\\", "/")

        results_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "Model_Path": rel_model_path,
            "CM_Path": rel_cm_path
        })
        print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    df_results = pd.DataFrame(results_list)
    csv_out_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_results.to_csv(csv_out_path, index=False)
    print("=" * 60)
    print(f"ALIBABA MAINTENANCE MODEL COMPARISON EXPORTED TO: {csv_out_path}")
    print("=" * 60)

    return df_results


if __name__ == "__main__":
    train_and_evaluate_maintenance_baselines()
