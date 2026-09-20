"""Test retraining gates, deployment comparison, and duplicate-event protection."""

import json
import os
import sys
import tempfile

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from central_server.database import get_connection, init_db, register_node
from adaptive.retraining_controller import process_drift_event


def runner():
    return {
        "deployment_status": "DEPLOYED (New Model Improved Validation F1/Accuracy)",
        "df_comparison": pd.DataFrame([
            {"Model_Version": "Old Model", "F1-Score": 0.70, "Accuracy": 0.72},
            {"Model_Version": "New Model", "F1-Score": 0.82, "Accuracy": 0.80},
        ]),
    }


def run_tests():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        os.environ["TELEMETRY_DB_PATH"] = os.path.join(directory, "retraining.db")
        init_db()
        register_node("retrain_test_node", "retrain-test")
        event = {
            "node_id": "retrain_test_node",
            "status": "DRIFT DETECTED",
            "affected_features": ["dur"],
            "current_window": "controlled window (n=30)",
            "reference_window": "reference window (n=30)",
            "event_id": "controlled-event-1",
        }

        blocked = process_drift_event(event, compatible_training_data=False)
        assert blocked["status"] == "BLOCKED"
        assert blocked["deployment_decision"] == "REJECTED_INCOMPATIBLE_DATA"

        approved = process_drift_event(
            {**event, "event_id": "controlled-event-2"},
            compatible_training_data=True,
            pipeline_runner=runner,
            force=True,
        )
        assert approved["status"] == "COMPLETED"
        assert approved["deployment_decision"] == "APPROVED"
        assert approved["new_model_version"] == "V2"
        assert approved["validation_metrics"]["comparison"]

        duplicate = process_drift_event(
            {**event, "event_id": "controlled-event-2"},
            compatible_training_data=True,
            pipeline_runner=runner,
        )
        assert duplicate["status"] == "SKIPPED"
        assert "already" in duplicate["reason"]

        cooldown = process_drift_event(
            {**event, "event_id": "controlled-event-3"},
            compatible_training_data=True,
            pipeline_runner=runner,
        )
        assert cooldown["status"] == "SKIPPED"
        assert "cooldown" in cooldown["reason"]

    print("[SUCCESS] RETRAINING CONTROLLER TESTS PASSED")


if __name__ == "__main__":
    run_tests()
