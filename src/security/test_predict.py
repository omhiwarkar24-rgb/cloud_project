"""
test_predict.py - Standalone validation runner for Security Prediction API.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from security.predict import predict_attack

TEST_CSV = os.path.join(BASE_DIR, "data", "security", "UNSW_NB15_testing-set.csv")
df_raw = pd.read_csv(TEST_CSV)

print("=== 1. SINGLE-RECORD PREDICTION: NORMAL SAMPLE ===")
normal_row = df_raw[df_raw['label'] == 0].iloc[0].to_dict()
res_normal = predict_attack(normal_row)
print("Normal Result:", res_normal)

print("\n=== 2. SINGLE-RECORD PREDICTION: ATTACK SAMPLE ===")
attack_row = df_raw[df_raw['label'] == 1].iloc[0].to_dict()
res_attack = predict_attack(attack_row)
print("Attack Result:", res_attack)

print("\n=== 3. BATCH PREDICTION (20 ROWS) ===")
df_batch = predict_attack(df_raw.head(20))
cols = [c for c in ['proto', 'service', 'state', 'dur', 'predicted_attack', 'confidence_probability', 'risk_level'] if c in df_batch.columns]
print(df_batch[cols].head(5))

print("\n=== 4. INPUT VALIDATION EDGE-CASE TEST ===")
# 4a. Graceful fallback on partial input (default imputation)
res_partial = predict_attack({"dur": 0.5, "proto": "tcp"})
print("Graceful fallback on partial input:", res_partial)

# 4b. Strict validation error handling
try:
    predict_attack({"dur": 0.5, "proto": "tcp"}, strict=True)
except KeyError as e:
    print("Successfully caught missing columns validation error (strict mode):", e)

# 4c. Unsupported input type handling
try:
    predict_attack([1, 2, 3])
except TypeError as e:
    print("Successfully caught unsupported input type error:", e)
