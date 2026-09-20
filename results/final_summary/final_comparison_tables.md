# Final Model Comparison Tables

## 1. Security Detection Models (UNSW-NB15)

### Binary Classification Task

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Decision Tree | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Random Forest | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| XGBoost | 0.998 | 0.9957 | 1.0 | 0.9979 | 1.0 |
| MLP Neural Network | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

### Multiclass Classification Task

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Decision Tree | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Random Forest | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| XGBoost | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| MLP Neural Network | 0.999 | 0.999 | 0.999 | 0.999 | 1.0 |

## 2. Predictive Maintenance Models (AI4I 2020)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9 | 0.75 | 0.9617 | 0.8428 | 0.9591 |
| Decision Tree | 0.8987 | 0.7588 | 0.933 | 0.8369 | 0.9283 |
| Random Forest | 0.9133 | 0.7667 | 0.9904 | 0.8643 | 0.9533 |
| XGBoost | 0.9053 | 0.7594 | 0.9665 | 0.8505 | 0.9581 |
| MLP Neural Network | 0.896 | 0.7811 | 0.8708 | 0.8235 | 0.9591 |

## 3. Adaptive Learning — Before vs After Drift Adaptation

| Model_Version | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Old Model (Pre-Drift) | 0.75 | 0.75 | 1.0 | 0.8571 | 0.786 |
| New Adaptive Model | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

