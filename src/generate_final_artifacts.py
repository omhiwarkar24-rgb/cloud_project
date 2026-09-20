"""
generate_final_artifacts.py - Final Summary Plots, Tables, ROC Curves, PR Curves,
Feature Importance, SHAP, and consolidated CSVs for the final deliverable.
"""

import os
import sys
import shutil
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR   = os.path.join(BASE_DIR, "results", "final_summary")
PLOT_DIR  = os.path.join(OUT_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── Paths ────────────────────────────────────────
SEC_TRAIN = os.path.join(BASE_DIR, "data", "security", "processed", "train_processed.csv")
SEC_TEST  = os.path.join(BASE_DIR, "data", "security", "processed", "test_processed.csv")
SEC_PIPE  = os.path.join(BASE_DIR, "models", "security", "preprocessing_pipeline.joblib")
MNT_TRAIN = os.path.join(BASE_DIR, "data", "maintenance", "processed", "train_processed.csv")
MNT_TEST  = os.path.join(BASE_DIR, "data", "maintenance", "processed", "test_processed.csv")

SEC_MODEL_DIR = os.path.join(BASE_DIR, "models", "security")
MNT_MODEL_DIR = os.path.join(BASE_DIR, "models", "maintenance")

# ══════════════════════════════════════════════════
# 1. Load Data
# ══════════════════════════════════════════════════
print("Loading processed datasets...")
df_sec_train = pd.read_csv(SEC_TRAIN)
df_sec_test  = pd.read_csv(SEC_TEST)
df_mnt_train = pd.read_csv(MNT_TRAIN)
df_mnt_test  = pd.read_csv(MNT_TEST)

sec_feat_cols = [c for c in df_sec_test.columns if c not in ["binary_label","multiclass_label","attack_cat_raw"]]
X_sec_test = df_sec_test[sec_feat_cols].values
y_sec_test_bin = df_sec_test["binary_label"].values

mnt_feat_cols = [c for c in df_mnt_test.columns if c != "Machine_failure"]
X_mnt_test = df_mnt_test[mnt_feat_cols].values
y_mnt_test  = df_mnt_test["Machine_failure"].values
print(f"Security test: {X_sec_test.shape} | Maintenance test: {X_mnt_test.shape}")

# ══════════════════════════════════════════════════
# 2. Security ROC Curves (Binary — all 5 models)
# ══════════════════════════════════════════════════
print("\nGenerating Security ROC curves...")
sec_binary_models = {
    "Logistic Regression": "logistic_regression_binary.joblib",
    "Decision Tree":       "decision_tree_binary.joblib",
    "Random Forest":       "random_forest_binary.joblib",
    "XGBoost":             "xgboost_binary.joblib",
    "MLP Neural Network":  "mlp_neural_network_binary.joblib",
}

COLORS = ["#e63946","#2ecc71","#64d2ff","#f0b429","#a29bfe"]
fig, ax = plt.subplots(figsize=(8, 6))
for (name, fname), col in zip(sec_binary_models.items(), COLORS):
    mp = os.path.join(SEC_MODEL_DIR, fname)
    if not os.path.exists(mp): continue
    mdl = joblib.load(mp)
    probs = mdl.predict_proba(X_sec_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_sec_test_bin, probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

ax.plot([0,1],[0,1],"k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Security Models — ROC Curves (Binary Task)", fontweight="bold")
ax.legend(fontsize=9)
p = os.path.join(PLOT_DIR, "security_roc_curves.png")
plt.savefig(p, dpi=300); plt.close()
print(f"  Saved: {p}")

# ══════════════════════════════════════════════════
# 3. Security Precision-Recall Curves
# ══════════════════════════════════════════════════
print("Generating Security PR curves...")
fig, ax = plt.subplots(figsize=(8, 6))
for (name, fname), col in zip(sec_binary_models.items(), COLORS):
    mp = os.path.join(SEC_MODEL_DIR, fname)
    if not os.path.exists(mp): continue
    mdl = joblib.load(mp)
    probs = mdl.predict_proba(X_sec_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_sec_test_bin, probs)
    ap = average_precision_score(y_sec_test_bin, probs)
    ax.plot(rec, prec, color=col, lw=2, label=f"{name} (AP={ap:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Security Models — Precision-Recall Curves (Binary Task)", fontweight="bold")
ax.legend(fontsize=9)
p = os.path.join(PLOT_DIR, "security_pr_curves.png")
plt.savefig(p, dpi=300); plt.close()
print(f"  Saved: {p}")

# ══════════════════════════════════════════════════
# 4. Maintenance ROC & PR Curves
# ══════════════════════════════════════════════════
print("Generating Maintenance ROC and PR curves...")
mnt_models = {
    "Logistic Regression": "logistic_regression_maintenance.joblib",
    "Decision Tree":       "decision_tree_maintenance.joblib",
    "Random Forest":       "random_forest_maintenance.joblib",
    "XGBoost":             "xgboost_maintenance.joblib",
    "MLP Neural Network":  "mlp_neural_network_maintenance.joblib",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for (name, fname), col in zip(mnt_models.items(), COLORS):
    mp = os.path.join(MNT_MODEL_DIR, fname)
    if not os.path.exists(mp): continue
    mdl = joblib.load(mp)
    probs = mdl.predict_proba(X_mnt_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_mnt_test, probs)
    roc_auc_v = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_mnt_test, probs)
    ap = average_precision_score(y_mnt_test, probs)
    axes[0].plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={roc_auc_v:.3f})")
    axes[1].plot(rec, prec, color=col, lw=2, label=f"{name} (AP={ap:.3f})")

axes[0].plot([0,1],[0,1],"k--",lw=1)
axes[0].set(xlabel="FPR", ylabel="TPR", title="Maintenance ROC Curves")
axes[0].legend(fontsize=8)
axes[1].set(xlabel="Recall", ylabel="Precision", title="Maintenance PR Curves")
axes[1].legend(fontsize=8)
plt.suptitle("Maintenance Models — ROC & Precision-Recall Curves", fontweight="bold")
p = os.path.join(PLOT_DIR, "maintenance_roc_pr_curves.png")
plt.savefig(p, dpi=300); plt.close()
print(f"  Saved: {p}")

# ══════════════════════════════════════════════════
# 5. Feature Importance (XGBoost Security)
# ══════════════════════════════════════════════════
print("Generating feature importance plots...")
xgb_sec = joblib.load(os.path.join(SEC_MODEL_DIR, "xgboost_binary.joblib"))
importances = xgb_sec.feature_importances_
fi_df = pd.DataFrame({"Feature": sec_feat_cols, "Importance": importances})
fi_df = fi_df.sort_values("Importance", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(data=fi_df, x="Importance", y="Feature", hue="Feature", palette="Blues_r", legend=False, ax=ax)
ax.set_title("Top 20 Feature Importances — XGBoost Security Classifier (Binary)", fontweight="bold")
ax.set_xlabel("Feature Importance Score")
p = os.path.join(PLOT_DIR, "security_feature_importance.png")
plt.savefig(p, dpi=300); plt.close()
print(f"  Saved: {p}")

# Maintenance Feature Importance (Random Forest)
rf_mnt = joblib.load(os.path.join(MNT_MODEL_DIR, "random_forest_maintenance.joblib"))
fi_mnt = pd.DataFrame({"Feature": mnt_feat_cols, "Importance": rf_mnt.feature_importances_})
fi_mnt = fi_mnt.sort_values("Importance", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=fi_mnt, x="Importance", y="Feature", hue="Feature", palette="Purples_r", legend=False, ax=ax)
ax.set_title("Feature Importances — Random Forest Maintenance Classifier", fontweight="bold")
p = os.path.join(PLOT_DIR, "maintenance_feature_importance.png")
plt.savefig(p, dpi=300); plt.close()
print(f"  Saved: {p}")

# ══════════════════════════════════════════════════
# 6. SHAP Analysis (best models)
# ══════════════════════════════════════════════════
print("\nAttempting SHAP analysis...")
try:
    import shap
    explainer_sec = shap.TreeExplainer(xgb_sec)
    shap_vals_sec = explainer_sec.shap_values(X_sec_test[:200])
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals_sec, X_sec_test[:200], feature_names=sec_feat_cols, show=False, max_display=15)
    plt.title("SHAP Summary — XGBoost Security Classifier (Binary)", fontweight="bold")
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, "security_shap_summary.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  SHAP security saved: {p}")

    explainer_mnt = shap.TreeExplainer(rf_mnt)
    shap_vals_mnt = explainer_mnt.shap_values(X_mnt_test[:200])
    sv = shap_vals_mnt[1] if isinstance(shap_vals_mnt, list) else shap_vals_mnt
    plt.figure(figsize=(9, 5))
    shap.summary_plot(sv, X_mnt_test[:200], feature_names=mnt_feat_cols, show=False, max_display=10)
    plt.title("SHAP Summary — Random Forest Maintenance Classifier", fontweight="bold")
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, "maintenance_shap_summary.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  SHAP maintenance saved: {p}")

    SHAP_USED = True
except Exception as e:
    SHAP_USED = False
    print(f"  SHAP analysis skipped/failed ({e}).")

# ══════════════════════════════════════════════════
# 7. Copy existing plots into final_summary
# ══════════════════════════════════════════════════
print("\nCopying existing plots into final_summary...")
COPY_SOURCES = [
    (os.path.join(BASE_DIR, "results", "security", "plots"),   "security_cm"),
    (os.path.join(BASE_DIR, "results", "maintenance", "plots"), "maintenance_cm"),
    (os.path.join(BASE_DIR, "results", "adaptive"),             "adaptive"),
]
for src_dir, prefix in COPY_SOURCES:
    if os.path.isdir(src_dir):
        for f in os.listdir(src_dir):
            if f.endswith(".png"):
                shutil.copy2(os.path.join(src_dir, f), os.path.join(PLOT_DIR, f"{prefix}_{f}"))
print(f"  All plots consolidated to: {PLOT_DIR}")

# ══════════════════════════════════════════════════
# 8. Consolidated CSV & Markdown Tables
# ══════════════════════════════════════════════════
print("\nExporting consolidated summary CSVs and Markdown tables...")

df_sec = pd.read_csv(os.path.join(BASE_DIR, "results", "security", "model_comparison.csv"))
df_sec_clean = df_sec[["Task","Model","Accuracy","Precision (Binary/Weighted)","Recall","F1-Score","ROC-AUC"]].copy()
df_sec_clean.columns = ["Task","Model","Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
for c in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]:
    df_sec_clean[c] = df_sec_clean[c].round(4)
df_sec_clean.to_csv(os.path.join(OUT_DIR, "security_model_summary.csv"), index=False)

df_mnt = pd.read_csv(os.path.join(BASE_DIR, "results", "maintenance", "model_comparison.csv"))
df_mnt_clean = df_mnt[["Model","Accuracy","Precision","Recall","F1-Score","ROC-AUC"]].copy()
for c in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]:
    df_mnt_clean[c] = df_mnt_clean[c].round(4)
df_mnt_clean.to_csv(os.path.join(OUT_DIR, "maintenance_model_summary.csv"), index=False)

df_adapt = pd.read_csv(os.path.join(BASE_DIR, "results", "adaptive", "adaptive_retraining_comparison.csv"))
df_adapt_clean = df_adapt.copy()
for c in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]:
    if c in df_adapt_clean.columns:
        df_adapt_clean[c] = df_adapt_clean[c].round(4)
df_adapt_clean.to_csv(os.path.join(OUT_DIR, "adaptive_learning_summary.csv"), index=False)

def df_to_md(df):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"]*len(cols)) + " |"
    rows   = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join([header, sep] + rows)

md_path = os.path.join(OUT_DIR, "final_comparison_tables.md")
with open(md_path, "w") as f:
    f.write("# Final Model Comparison Tables\n\n")
    f.write("## 1. Security Detection Models (UNSW-NB15)\n\n")
    f.write("### Binary Classification Task\n\n")
    bin_df = df_sec_clean[df_sec_clean["Task"]=="Binary"].drop(columns=["Task"])
    f.write(df_to_md(bin_df) + "\n\n")
    f.write("### Multiclass Classification Task\n\n")
    mc_df = df_sec_clean[df_sec_clean["Task"]=="Multiclass"].drop(columns=["Task"])
    f.write(df_to_md(mc_df) + "\n\n")
    f.write("## 2. Predictive Maintenance Models (AI4I 2020)\n\n")
    f.write(df_to_md(df_mnt_clean) + "\n\n")
    f.write("## 3. Adaptive Learning — Before vs After Drift Adaptation\n\n")
    f.write(df_to_md(df_adapt_clean) + "\n\n")

print(f"\n  Markdown tables saved: {md_path}")
print("\n=== Final Artifact Generation COMPLETE ===")
