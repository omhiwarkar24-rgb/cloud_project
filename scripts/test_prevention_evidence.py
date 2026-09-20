"""Validate evidence-based node recommendations."""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from prevention.prevention_engine import generate_node_recommendations


def run_tests():
    recommendations = generate_node_recommendations(
        {
            "node_id": "NODE_03",
            "risk_level": "HIGH",
            "overall_risk": 0.69,
            "security_risk": 0.65,
            "maintenance_risk": 0.81,
            "timestamp": "2026-09-20T12:00:00+00:00",
        },
        {
            "feature_vector": {
                "cpu_utilization": 94.0,
                "ram_utilization": 91.0,
                "disk_utilization": 40.0,
            }
        },
        {"attack_type": "DoS"},
    )
    problems = {item["detected_problem"] for item in recommendations}
    assert "High CPU utilization" in problems
    assert "High memory utilization" in problems
    assert "Increasing health/failure risk" in problems
    assert "Security anomaly (DoS)" in problems
    for item in recommendations:
        assert item["node_id"] == "NODE_03"
        assert item["evidence"]
        assert item["recommended_action"]
        assert item["why_generated"]
        assert item["advisory_only"] is True
        assert "automatic" in item["disclaimer"].lower()

    assert generate_node_recommendations({"node_id": "LOW", "risk_level": "LOW"}) == []
    print("[SUCCESS] PREVENTION EVIDENCE TESTS PASSED")


if __name__ == "__main__":
    run_tests()
