"""
test_risk_engine.py - Automated test script for Part 9 Risk Assessment Engine.
"""

import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src", "risk_engine"))

from integrated_risk import evaluate_integrated_risk

scenarios = [
    {"case": "Case A (Normal + Healthy)", "attack": "Normal", "prob": 0.10},
    {"case": "Case B (DoS Attack + Healthy)", "attack": "DoS", "prob": 0.15},
    {"case": "Case C (Normal + High Failure Risk)", "attack": "Normal", "prob": 0.85},
    {"case": "Case D (Exploits Attack + High Failure Risk)", "attack": "Exploits", "prob": 0.90}
]

print("=" * 70)
print("PHASE 9 RISK ASSESSMENT ENGINE TEST RESULTS")
print("=" * 70)

results = []
for s in scenarios:
    res = evaluate_integrated_risk(s["attack"], s["prob"])
    results.append({
        "Scenario": s["case"],
        "Attack_Cat": s["attack"],
        "Security_Level": res["security_risk_level"],
        "Security_Score": res["security_risk_score"],
        "Failure_Prob": res["failure_probability"],
        "Failure_Level": res["failure_risk_level"],
        "Overall_Score": res["overall_risk_score"],
        "Overall_Level": res["overall_risk_level"]
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))
print("=" * 70)
