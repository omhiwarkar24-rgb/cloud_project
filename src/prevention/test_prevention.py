"""
test_prevention.py - Automated test runner for Part 10: Prevention Engine full pipeline.
Prints full rule table + 6 end-to-end scenario outputs.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))
sys.path.append(os.path.join(BASE_DIR, "src", "risk_engine"))

from risk_engine.integrated_risk import evaluate_integrated_risk
from prevention.prevention_engine import generate_recommendation, RECOMMENDATION_RULE_MATRIX

# ===========================================================
# PART 1: Print Full Rule Combination Table
# ===========================================================
print("=" * 80)
print("FULL RECOMMENDATION RULE MATRIX (All Combinations)")
print("=" * 80)
print(f"{'Security Level':<16} {'Failure Level':<16} {'Priority':<12} {'Recommendation'}")
print("-" * 80)

for (sec, fail), rec in RECOMMENDATION_RULE_MATRIX.items():
    result = generate_recommendation(sec, fail)
    print(f"{sec:<16} {fail:<16} {result['action_priority']:<12} {rec}")

print()

# ===========================================================
# PART 2: End-to-End Pipeline: 6 Scenarios
# ===========================================================
scenarios = [
    {"case": "Scenario 1: Normal Traffic + Healthy Machine",     "attack": "Normal",          "prob": 0.10},
    {"case": "Scenario 2: Reconnaissance + Moderate Wear",       "attack": "Reconnaissance",  "prob": 0.45},
    {"case": "Scenario 3: DoS Attack + Healthy Machine",         "attack": "DoS",             "prob": 0.15},
    {"case": "Scenario 4: Normal Traffic + High Failure Risk",   "attack": "Normal",          "prob": 0.85},
    {"case": "Scenario 5: Exploits Attack + High Failure Risk",  "attack": "Exploits",        "prob": 0.90},
    {"case": "Scenario 6: Worms Infection + Critical Failure",   "attack": "Worms",           "prob": 0.95},
]

print("=" * 80)
print("END-TO-END PIPELINE: Prediction -> Risk -> Recommendation")
print("=" * 80)

for s in scenarios:
    risk = evaluate_integrated_risk(s["attack"], s["prob"])
    rec  = generate_recommendation(risk["security_risk_level"], risk["failure_risk_level"], s["prob"])

    print(f"\n{s['case']}")
    print(f"  Attack Category  : {s['attack']}")
    print(f"  Failure Prob     : {s['prob']:.0%}")
    print(f"  Security Risk    : {risk['security_risk_level']} (score={risk['security_risk_score']})")
    print(f"  Failure Risk     : {risk['failure_risk_level']}  (prob={risk['failure_probability']})")
    print(f"  Overall Risk     : {risk['overall_risk_level']} (score={risk['overall_risk_score']})")
    print(f"  Action Priority  : {rec['action_priority']}")
    print(f"  Recommendation   : {rec['recommendation']}")
