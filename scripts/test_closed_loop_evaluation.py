"""Test the controlled closed-loop workload experiment."""

import os
import sys
import tempfile
from datetime import datetime, timezone

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from central_server.database import init_db, register_node, record_telemetry, save_node_security_prediction
from experiments.closed_loop_evaluation import run_closed_loop_evaluation, get_latest_closed_loop_evaluation


def run_tests():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        os.environ["TELEMETRY_DB_PATH"] = os.path.join(directory, "experiment.db")
        init_db()
        node_id = "closed_loop_test_node"
        register_node(node_id, "closed-loop-test")
        record_telemetry({
            "node_id": node_id,
            "timestamp": datetime(2026, 9, 20, tzinfo=timezone.utc).isoformat(),
            "metrics": {
                "cpu_utilization": 92.0,
                "ram_utilization": 90.0,
                "disk_utilization": 40.0,
                "process_count": 500,
                "network_sent_mb": 100.0,
                "network_recv_mb": 100.0,
            },
        })
        save_node_security_prediction({
            "node_id": node_id,
            "classification": "ATTACK",
            "attack_type": "DoS",
            "security_risk": "HIGH",
            "confidence": 0.95,
            "timestamp": datetime(2026, 9, 20, tzinfo=timezone.utc).isoformat(),
            "model_name": "test",
            "model_version": "test",
            "prediction_source": "DEMO_SAMPLE",
            "is_simulated": True,
            "source_label": "controlled test",
            "security_risk_score": 0.8,
        })

        result = run_closed_loop_evaluation(node_id)
        assert result["label"] == "Controlled Experimental Evaluation"
        assert result["before"]["cpu"] == 92.0
        assert result["after"]["cpu"] == 46.0
        assert result["before"]["security_risk"] == result["after"]["security_risk"] == 0.8
        assert result["after"]["maintenance_risk"] < result["before"]["maintenance_risk"]
        assert result["after"]["overall_risk"] < result["before"]["overall_risk"]
        assert result["measured_change"]["overall_risk_reduction"] > 0
        assert result["recommendation"]
        assert "No host workload was changed" in result["intervention"]
        assert get_latest_closed_loop_evaluation(node_id)["experiment_id"] == result["experiment_id"]

    print("[SUCCESS] CLOSED-LOOP EVALUATION TESTS PASSED")


if __name__ == "__main__":
    run_tests()
