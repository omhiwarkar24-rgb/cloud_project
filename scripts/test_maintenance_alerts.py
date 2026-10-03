"""
test_maintenance_alerts.py - Automated test suite for the Predictive Maintenance Alert System.
Tests alert triggering, cooldown/deduplication, database storage, auto-recovery, and REST endpoints.
"""

import os
import sys
import json
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from central_server.database import (
    init_db,
    save_maintenance_alert,
    get_latest_active_alert_for_node,
    get_active_maintenance_alerts,
    get_all_maintenance_alerts,
    acknowledge_maintenance_alert,
    resolve_maintenance_alert,
)
from maintenance.alerts import (
    evaluate_node_for_alert,
    fetch_active_alerts,
    fetch_alert_history,
    acknowledge_alert_by_id,
    resolve_alert_by_id,
    generate_preventive_action,
)
from fastapi.testclient import TestClient
from central_server.api import app

def run_tests():
    print("=" * 65)
    print("TESTING PREDICTIVE MAINTENANCE ALERT SYSTEM")
    print("=" * 65)

    init_db()

    # 1. Verification of preventive action generation
    print("\n[1] Testing Preventive Action Generator...")
    rec = generate_preventive_action(
        risk_level="HIGH",
        failure_probability=0.88,
        telemetry={"cpu_utilization": 85.0, "ram_utilization": 92.0, "disk_utilization": 50.0, "cpu_temperature_c": 81.0},
        contributing_factors=["High RAM utilization (92.0%)", "Critical compound resource stress"]
    )
    print("  -> Generated Action:", rec)
    assert "chassis cooling" in rec.lower() or "throttle" in rec.lower()
    assert "inspect the node and reduce workload" in rec.lower()

    # 2. Triggering real alert with HIGH failure risk
    print("\n[2] Testing Real Alert Triggering on High Risk...")
    node_test = "test_alert_node_42"
    high_risk_pred = {
        "prediction": 1,
        "failure_probability": 0.895,
        "risk_level": "HIGH",
        "main_contributing_factors": [
            "Heavy CPU processor load (88.0%)",
            "High RAM utilization (92.0%)",
            "Critical compound resource stress (81.0)"
        ],
        "features_used": {"CPU_utilization": 88.0, "Memory_utilization": 92.0, "Disk_utilization": 60.0}
    }
    high_risk_telem = {
        "cpu_utilization": 88.0,
        "ram_utilization": 92.0,
        "disk_utilization": 60.0,
        "cpu_temperature_c": 81.0,
        "process_count": 215,
        "duration_sec": 4500
    }

    alert = evaluate_node_for_alert(
        node_id=node_test,
        telemetry=high_risk_telem,
        maint_result=high_risk_pred,
        cooldown_seconds=300
    )
    assert alert is not None
    assert alert["node_id"] == node_test
    assert alert["risk_level"] == "HIGH"
    assert alert["failure_probability"] == 0.895
    assert alert["status"] == "ACTIVE"
    assert alert["is_new"] is True
    assert alert["repeat_count"] == 1
    print(f"  -> Generated Alert ID: {alert['alert_id']}")
    print(f"  -> Reason: {alert['reason']}")
    print(f"  -> Recommended Action: {alert['recommended_action']}")

    # 3. Deduplication & Cooldown protection
    print("\n[3] Testing Deduplication & Cooldown Protection (within 300s window)...")
    for cycle in range(2, 5):
        repeat_alert = evaluate_node_for_alert(
            node_id=node_test,
            telemetry=high_risk_telem,
            maint_result=high_risk_pred,
            cooldown_seconds=300
        )
        assert repeat_alert is not None
        assert repeat_alert["alert_id"] == alert["alert_id"], "Alert ID changed during cooldown!"
        assert repeat_alert["repeat_count"] == cycle, f"Expected repeat_count={cycle}, got {repeat_alert['repeat_count']}"
        assert repeat_alert["deduplicated"] is True
    print(f"  -> Successfully deduplicated 3 consecutive cycles into Alert ID {alert['alert_id']} (repeat_count={repeat_alert['repeat_count']})")

    # 4. Auto-recovery when node returns to normal
    print("\n[4] Testing Auto-Recovery when Node Health Returns to Low Risk...")
    low_risk_pred = {
        "prediction": 0,
        "failure_probability": 0.12,
        "risk_level": "LOW",
        "main_contributing_factors": []
    }
    low_risk_telem = {
        "cpu_utilization": 22.0,
        "ram_utilization": 35.0,
        "disk_utilization": 50.0,
        "cpu_temperature_c": 45.0
    }
    rec_result = evaluate_node_for_alert(
        node_id=node_test,
        telemetry=low_risk_telem,
        maint_result=low_risk_pred,
        cooldown_seconds=300
    )
    assert rec_result is None
    # Verify active alerts no longer contains this node
    active_now = [a for a in fetch_active_alerts() if a["node_id"] == node_test]
    assert len(active_now) == 0, "Node alert was not resolved upon return to healthy state!"
    
    # Verify alert history preserved the incident as RESOLVED
    hist_entries = [h for h in fetch_alert_history(limit=20) if h["alert_id"] == alert["alert_id"]]
    assert len(hist_entries) == 1
    assert hist_entries[0]["status"] == "RESOLVED"
    assert hist_entries[0]["resolved_at"] is not None
    print(f"  -> Alert {alert['alert_id']} correctly marked RESOLVED at {hist_entries[0]['resolved_at']}")

    # 5. REST API Endpoints testing
    print("\n[5] Testing Central Server REST API Endpoints...")
    client = TestClient(app)

    # Trigger fresh alert for API testing
    alert_api = evaluate_node_for_alert(
        node_id="test_api_node_99",
        telemetry=high_risk_telem,
        maint_result=high_risk_pred,
        cooldown_seconds=300
    )
    api_alert_id = alert_api["alert_id"]

    # GET /alerts/active
    r_act = client.get("/alerts/active")
    assert r_act.status_code == 200
    active_ids = [a["alert_id"] for a in r_act.json().get("alerts", [])]
    assert api_alert_id in active_ids
    print(f"  -> GET /alerts/active returned {len(active_ids)} active alert(s)")

    # POST /alerts/{id}/acknowledge
    r_ack = client.post(f"/alerts/{api_alert_id}/acknowledge")
    assert r_ack.status_code == 200
    assert r_ack.json()["state"] == "ACKNOWLEDGED"
    print(f"  -> POST /alerts/{api_alert_id}/acknowledge returned state: ACKNOWLEDGED")

    # POST /alerts/{id}/resolve
    r_res = client.post(f"/alerts/{api_alert_id}/resolve")
    assert r_res.status_code == 200
    assert r_res.json()["state"] == "RESOLVED"
    print(f"  -> POST /alerts/{api_alert_id}/resolve returned state: RESOLVED")

    # GET /alerts (history)
    r_all = client.get("/alerts?limit=10")
    assert r_all.status_code == 200
    all_alerts = r_all.json().get("alerts", [])
    assert any(a["alert_id"] == api_alert_id for a in all_alerts)
    print(f"  -> GET /alerts returned {len(all_alerts)} historical alert incident(s)")

    print("\n" + "=" * 65)
    print("ALL PREDICTIVE MAINTENANCE ALERT TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
