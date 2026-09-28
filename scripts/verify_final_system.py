"""
verify_final_system.py - End-to-end verification of the finalized Adaptive Cloud Security
and Predictive Maintenance system with real live nodes.
"""

import os
import sys
import json
import socket
from datetime import datetime, timezone

# Ensure src/ and submodules are on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "src", "security"))
sys.path.insert(0, os.path.join(BASE_DIR, "src", "maintenance"))
sys.path.insert(0, os.path.join(BASE_DIR, "src", "risk_engine"))
sys.path.insert(0, os.path.join(BASE_DIR, "src", "prevention"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Use an isolated test database so production telemetry.db is not modified
TEST_DB = os.path.join(BASE_DIR, "data", "test_verification.db")
os.environ["TELEMETRY_DB_PATH"] = TEST_DB

from central_server.database import (
    init_db,
    record_telemetry,
    get_all_nodes,
    get_latest_telemetry,
)
from node_agent.metrics import collect_node_telemetry
from maintenance.predict import predict_failure
from risk_engine.integrated_risk import evaluate_integrated_risk
from prevention.prevention_engine import generate_recommendation
from adaptive.live_drift import detect_all_live_drift
from adaptive.retraining_controller import get_latest_retraining_status


def main():
    print("=" * 65)
    print("VERIFYING FINAL SYSTEM INTEGRATION (ISOLATED TEST DB)")
    print("=" * 65)

    try:
        print("\n[1] Testing Live Infrastructure & Real Telemetry Collection...")
        init_db()

        # Collect and record real live telemetry for this machine
        real_host = socket.gethostname()
        live_node_id = f"node_{real_host.lower()}_verify"
        real_payload = collect_node_telemetry(live_node_id)
        record_telemetry(real_payload)

        nodes = get_all_nodes()
        print(f"  -> Total registered nodes found: {len(nodes)}")
        assert len(nodes) > 0, "Expected at least 1 registered node"
        
        target_node = next((n for n in nodes if n["node_id"] == live_node_id), nodes[0])
        print(f"  -> Verified live node: {target_node['node_id']} (Status: {target_node['status']})")
        assert target_node["status"] == "ONLINE", f"Expected ONLINE, got {target_node['status']}"

        latest = get_latest_telemetry(target_node["node_id"])
        assert latest is not None, "Failed to retrieve latest telemetry"
        m = latest.get("metrics", {})
        cpu = latest.get("cpu_utilization")
        ram = latest.get("ram_utilization")
        disk = latest.get("disk_utilization")
        print(f"  -> Live Metrics from host: CPU={cpu}%, RAM={ram}%, Disk={disk}%")
        print(f"  -> Load Averages: 1m={m.get('load_1min')}, 5m={m.get('load_5min')}, 15m={m.get('load_15min')}")
        print(f"  -> Host Execution Uptime: {m.get('duration_sec')}s | Role: {m.get('server_type')}")

        print("\n[2] Testing Live Node Failure Prediction (Scientific Features)...")
        input_data = {
            "CPU_utilization": float(cpu or 0.0),
            "Memory_utilization": float(ram or 0.0),
            "Disk_utilization": float(disk or 0.0),
            "Load_1min": float(m.get("load_1min", 0.0)),
            "Load_5min": float(m.get("load_5min", 0.0)),
            "Load_15min": float(m.get("load_15min", 0.0)),
            "Duration_sec": int(m.get("duration_sec", 0)),
            "Server_Type": str(m.get("server_type", "ComputeNode"))
        }
        pred_res = predict_failure(input_data)
        print(f"  -> Model Prediction: {pred_res['prediction']}")
        print(f"  -> Failure Probability: {pred_res['failure_probability'] * 100:.2f}%")
        print(f"  -> Risk Level: {pred_res['risk_level']}")
        assert "failure_probability" in pred_res
        assert "risk_level" in pred_res

        print("\n[3] Testing Predictive Maintenance Model Sensitivity Under Elevated Load...")
        stress_data = {
            "CPU_utilization": 95.0,
            "Memory_utilization": 96.0,
            "Disk_utilization": 89.0,
            "Load_1min": 28.5,
            "Load_5min": 22.0,
            "Load_15min": 18.0,
            "Duration_sec": 3600,
            "Server_Type": "ComputeNode"
        }
        stress_pred = predict_failure(stress_data)
        print(f"  -> Elevated Load Failure Probability: {stress_pred['failure_probability'] * 100:.2f}%")
        print(f"  -> Elevated Load Risk Level: {stress_pred['risk_level']}")
        assert stress_pred["failure_probability"] > 0.85, "High stress should yield high failure probability"
        assert stress_pred["risk_level"] == "HIGH", "High stress should yield HIGH risk level"

        print("\n[4] Testing Integrated Risk Fusion Engine...")
        risk_out = evaluate_integrated_risk("DoS", stress_pred["failure_probability"], w_sec=0.5, w_fail=0.5)
        print(f"  -> Security Risk: {risk_out['security_risk_level']} (Score: {risk_out['security_risk_score']})")
        print(f"  -> Failure Risk: {risk_out['failure_risk_level']} (Score: {risk_out['failure_risk_score']})")
        print(f"  -> Overall Integrated Risk: {risk_out['overall_risk_level']} (Score: {risk_out['overall_risk_score']})")
        assert risk_out["overall_risk_level"] in ["HIGH", "CRITICAL"]

        print("\n[5] Testing Intelligent Failure Prevention Recommendation Engine...")
        rec_out = generate_recommendation(risk_out["security_risk_level"], risk_out["failure_risk_level"], stress_pred["failure_probability"])
        print(f"  -> Action Priority: {rec_out['action_priority']}")
        print(f"  -> Recommendation: {rec_out['recommendation']}")
        assert rec_out["action_priority"] == "URGENT"

        print("\n[6] Testing Live Drift & Adaptive Retraining Controller...")
        drift_res = detect_all_live_drift()
        print(f"  -> Live drift evaluated for {len(drift_res)} registered nodes")
        retrain_status = get_latest_retraining_status()
        print(f"  -> Latest retraining controller decision: {retrain_status.get('deployment_decision') if retrain_status else 'Initialized'}")

        print("\n" + "=" * 65)
        print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
        print("=" * 65)
    finally:
        # Ensure test database is removed so no test artifacts linger
        try:
            if os.path.exists(TEST_DB):
                os.remove(TEST_DB)
        except Exception:
            pass


if __name__ == "__main__":
    main()
