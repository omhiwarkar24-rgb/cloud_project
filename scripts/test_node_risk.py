"""Validate independent component fusion for per-node risk."""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from risk_engine.node_risk import combine_node_risk


def run_tests():
    result = combine_node_risk(
        "NODE_03",
        {
            "security_risk_score": 0.65,
            "security_risk": "HIGH",
            "timestamp": "2026-09-20T10:00:00+00:00",
        },
        {
            "anomaly_probability": 0.72,
            "health_status": "ANOMALOUS",
            "timestamp": "2026-09-20T10:01:00+00:00",
            "feature_vector": {"cpu_utilization": 82.0},
        },
    )
    assert result["security_risk"] == 0.65
    assert result["maintenance_risk"] == 0.72
    assert result["overall_risk"] == 0.685
    assert result["risk_level"] == "HIGH"
    assert "Elevated security risk" in result["contributing_factors"]
    assert "High resource utilization" in result["contributing_factors"]
    assert "Increased health/failure probability" in result["contributing_factors"]

    missing_security = combine_node_risk(
        "NODE_NO_FLOW",
        None,
        {"anomaly_probability": 0.4, "timestamp": "2026-09-20T10:00:00+00:00"},
    )
    assert missing_security["security_risk"] is None
    assert missing_security["overall_risk"] == 0.4
    assert "No network-flow security result available" in missing_security["contributing_factors"]

    print("[SUCCESS] NODE RISK TESTS PASSED")


if __name__ == "__main__":
    run_tests()
