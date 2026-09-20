"""Test live KS drift detection with a normal window and a controlled shift."""

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from central_server.database import init_db, record_telemetry, register_node
from adaptive.live_drift import detect_live_drift


def payload(node_id, timestamp, cpu, ram, disk, process_count, network):
    return {
        "node_id": node_id,
        "hostname": "drift-test-node",
        "operating_system": "test",
        "timestamp": timestamp.isoformat(),
        "metrics": {
            "cpu_utilization": cpu,
            "ram_utilization": ram,
            "disk_utilization": disk,
            "process_count": process_count,
            "network_sent_mb": network,
            "network_recv_mb": 0.0,
        },
    }


def run_tests():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        os.environ["TELEMETRY_DB_PATH"] = os.path.join(directory, "drift.db")
        init_db()
        node_id = "drift_test_node"
        register_node(node_id, "drift-test-node")
        start = datetime(2026, 9, 20, tzinfo=timezone.utc)

        for index in range(30):
            record_telemetry(payload(node_id, start + timedelta(seconds=index), 20, 35, 30, 100, index))
        for index in range(30, 50):
            record_telemetry(payload(node_id, start + timedelta(seconds=index), 20, 35, 30, 100, index))
        normal = detect_live_drift(node_id)
        assert normal["status"] == "NO DRIFT"
        assert not normal["affected_features"]

        for index in range(50, 80):
            record_telemetry(payload(node_id, start + timedelta(seconds=index), 95, 95, 90, 500, 1000 + index * 10))
        shifted = detect_live_drift(node_id)
        assert shifted["status"] == "DRIFT DETECTED"
        assert {"cpu_utilization", "ram_utilization", "disk_utilization", "process_count"}.issubset(
            set(shifted["affected_features"])
        )
        assert all("reference_window" in result and "current_window" in result for result in shifted["features"])

    print("[SUCCESS] LIVE DRIFT TESTS PASSED")


if __name__ == "__main__":
    run_tests()
