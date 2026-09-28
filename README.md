# Adaptive Learning for Cloud Security & Predictive Maintenance with Intelligent Failure Prevention

A comprehensive, end-to-end Machine Learning system integrating **Cloud Security Threat Detection**, **Cloud Server Predictive Maintenance**, **Adaptive Data Drift Learning**, and an **Intelligent Integrated Risk & Prevention Engine**.

---

## 🎯 Executive Summary

Modern cloud infrastructure faces dual operational hazards: **cybersecurity intrusions** and **server infrastructure hardware failures**. Traditional systems monitor these domains in silos using static models that fail when data distributions shift over time.

This project delivers an **Adaptive Learning Framework** that:
1. **Detects Cyber Threats** (UNSW-NB15 dataset) with up to **98.50% Accuracy** and **0.9991 ROC-AUC**.
2. **Predicts Cloud Server Failures** (**Alibaba Cluster Trace 2017 dataset**) with up to **91.33% Accuracy** and **0.9591 ROC-AUC**.
3. **Monitors Data Drift** dynamically using the **Kolmogorov-Smirnov (KS) 2-Sample Test** and automatically retrains models when environmental shifts occur.
4. **Fuses Risk Signals** into a unified score and triggers actionable recommendations via an interactive **Streamlit Dashboard**.

---

## 🏗️ System Architecture (4 Core Modules)

```
                       ┌───────────────────────────────────────────────┐
                       │           RAW DATA STREAMS / SENSORS          │
                       └───────────────────────┬───────────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │     DATA PREPROCESSING & FEATURE PIPELINES    │
                       └──────────────┬─────────────────┬──────────────┘
                                      │                 │
            ┌─────────────────────────┴──┐           ┌──┴─────────────────────────┐
            │ MODULE 1: SECURITY ENGINE  │           │ MODULE 2: MAINTENANCE      │
            │ UNSW-NB15 Threat Detection │           │ Alibaba Cluster Trace 2017 │
            │ (Random Forest / XGBoost)  │           │ (Random Forest / XGBoost)  │
            └─────────────┬──────────────┘           └──────────────┬─────────────┘
                          │ (Security Risk)                         │ (Failure Risk)
                          └───────────────────┬─────────────────────┘
                                              │
                               ┌──────────────┴──────────────┐
                               │ MODULE 3: ADAPTIVE LEARNING │
                               │ KS-Test Drift Detection &   │
                               │  Dynamic Model Retraining   │
                               └──────────────┬──────────────┘
                                              │
                               ┌──────────────┴──────────────┐
                               │ MODULE 4: PREVENTION ENGINE │
                               │  Weighted Risk Fusion &     │
                               │ Streamlit Control Dashboard │
                               └─────────────────────────────┘
```

---

## 📊 Core Performance Results

### 1. Cloud Security Threat Detection (UNSW-NB15 Dataset)

| Task | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Binary** | **Random Forest** | **98.50%** | **98.30%** | **98.50%** | **0.9830** | **0.9991** |
| Binary | XGBoost | 98.20% | 97.96% | 98.20% | 0.9796 | 0.9987 |
| Binary | MLP Neural Net | 97.90% | 97.63% | 97.90% | 0.9763 | 0.9972 |
| Binary | Decision Tree | 96.90% | 96.47% | 96.90% | 0.9647 | 0.9678 |
| Binary | Logistic Regression | 96.30% | 95.76% | 96.30% | 0.9576 | 0.9934 |
| **Multiclass** | **XGBoost** | **97.50%** | **97.40%** | **97.50%** | **0.9740** | **0.9987** |
| Multiclass | Random Forest | 97.20% | 97.08% | 97.20% | 0.9708 | 0.9988 |
| Multiclass | MLP Neural Net | 96.60% | 96.47% | 96.60% | 0.9647 | 0.9962 |

---

### 2. Cloud Server Predictive Maintenance (Alibaba Cluster Trace 2017 Dataset)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **91.33%** | **76.67%** | **99.04%** | **0.8643** | **0.9533** |
| **XGBoost** | **90.53%** | **75.94%** | **96.65%** | **0.8505** | **0.9581** |
| Logistic Regression | 90.00% | 75.00% | 96.17% | 0.8428 | **0.9591** |
| Decision Tree | 89.87% | 75.88% | 93.30% | 0.8369 | 0.9283 |
| MLP Neural Net | 89.60% | 78.11% | 87.08% | 0.8235 | **0.9591** |

---

### 3. Adaptive Learning & Data Drift Recovery

When operational environments shift (distribution drift), static models experience severe performance degradation:

- **Pre-Adaptation (Old Model on Shifted Stream)**: Accuracy dropped to **75.00%** (F1: 0.8571).
- **Drift Flagged**: KS-Test detected distribution shifts in **10 out of 51 features (19.61%)**.
- **Post-Adaptation (Retrained Model)**: Accuracy recovered to **100.00%** (F1: 1.0000).

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation
```bash
# Clone or navigate to project directory
cd .

# Install required packages
pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly streamlit joblib shap scipy
```

### 2. Run the Full ML Pipeline
```bash
# Trains all models, evaluates drift, and consolidates results
python scripts/run_all_models.py
```

### 3. Launch Interactive Streamlit Dashboard
```bash
python -m streamlit run app/dashboard.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Directory Structure

```
adaptive_cloud_project/
├── app/
│   └── dashboard.py               # Streamlit 4-tab interactive web interface
├── data/
│   ├── security/                  # UNSW-NB15 raw and processed datasets
│   └── maintenance/               # Alibaba Cluster Trace 2017 raw & processed datasets
├── models/
│   ├── security/                  # Trained security joblib models & pipelines
│   └── maintenance/               # Trained Alibaba maintenance joblib models
├── results/
│   ├── final_summary/             # Consolidated Markdown/CSV tables & ROC/PR plots
│   ├── security/                  # Security confusion matrices & SHAP plots
│   ├── maintenance/               # Alibaba maintenance plots & metrics
│   └── adaptive/                  # KS drift scores & retraining plots
├── scripts/
│   ├── run_all_models.py          # Master pipeline execution script
│   └── consolidate_results.py     # Results summary exporter
└── src/
    ├── data_processing/           # Dataset initializers & Alibaba preprocessor
    ├── security/                  # Security baseline model trainers & inference
    ├── maintenance/               # Alibaba Predictive maintenance trainers & inference
    ├── adaptive/                  # KS-test drift detector & retraining engine
    ├── risk_engine/               # Integrated weighted risk calculator
    └── prevention/                # Rule-based prevention recommendation engine
```

---

## 👨‍🎓 Project Credits & College HOD Defense Summary
- **Title**: Adaptive Learning for Cloud Security and Predictive Maintenance with Intelligent Failure Prevention
- **Core Technology Stack**: Python 3.14, Scikit-Learn, XGBoost, Streamlit, Plotly, SHAP, SciPy.
- **Key Highlight**: Integrated UNSW-NB15 cyber threat detection with real **Alibaba Cluster Trace 2017** cloud server failure prediction, augmented with automated distribution drift detection and real-time intervention recommendations.
