"""
test_end_to_end_predictive_maintenance.py

Complete End-to-End Verification of the Predictive Maintenance System:
Tests the exact 15-step workflow and 6 core scenarios against the live
Central Server, SQLite Database, ML Model, Alert Engine, and Node Agent.
"""

import os
import sys
import time
import json
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from maintenance.predict import predict_failure
from maintenance.alerts import (
    evaluate_node_for_alert,
    fetch_active_alerts,
    fetch_alert_history,
    acknowledge_alert_by_id,
    resolve_alert_by_id,
)
from central_server.database import (
    get_connection,
    register_node,
    record_telemetry,
    heartbeat_node,
    update_node_statuses,
    get_all_nodes,
    get_node_by_id,
)
from node_agent.notifier import NodeAlertNotifier

SERVER_URL = "http://localhost:8000"
DB_PATH = os.path.join(BASE_DIR, "data", "telemetry.db")


def http_get(endpoint: str):
    url = f"{SERVER_URL}{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "E2ETester/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())


def http_post(endpoint: str, data: dict):
    url = f"{SERVER_URL}{endpoint}"
    payload_bytes = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={"Content-Type": "application/json", "User-Agent": "E2ETester/1.0"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())


def run_e2e_tests():
    print("=" * 70)
    print("STARTING COMPLETE PREDICTIVE-MAINTENANCE END-TO-END TEST")
    print("=" * 70)

    test_results = {}
    created_test_nodes = []
    created_test_alerts = []

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Verify Central Server
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 1] Checking Central Server Status...")
    try:
        status, data = http_get("/nodes")
        assert status == 200, f"Expected 200 OK, got {status}"
        print(f"  [PASS] Central Server is ONLINE at {SERVER_URL} (Total registered nodes: {data.get('total_nodes')})")
        test_results["Step 1 - Central Server Online"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Step 1 FAILED: Central Server unreachable: {e}")
        test_results["Step 1 - Central Server Online"] = f"FAIL: {e}"
        return test_results

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Verify Real Node Agent
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 2] Verifying Real Node Agent on live machine...")
    try:
        online_nodes = [n for n in data.get("nodes", []) if n.get("status") == "ONLINE"]
        assert len(online_nodes) > 0, "No online node agents found in registry"
        real_node = online_nodes[0]
        real_nid = real_node["node_id"]
        print(f"  [PASS] Real Node Agent is active: Node ID='{real_nid}', Hostname='{real_node.get('hostname')}', OS='{real_node.get('operating_system')}'")
        test_results["Step 2 - Real Node Agent Online"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Step 2 FAILED: {e}")
        test_results["Step 2 - Real Node Agent Online"] = f"FAIL: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Confirm Telemetry Reaches Central Server
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 3] Confirming telemetry reaches Central Server...")
    try:
        status, node_details = http_get(f"/nodes/{real_nid}?limit=5")
        assert status == 200, f"Expected 200 OK, got {status}"
        recent_telemetry = node_details.get("recent_telemetry", [])
        assert len(recent_telemetry) > 0, f"No telemetry records found for node {real_nid}"
        latest_tel = recent_telemetry[0]
        tel_time = latest_tel.get("timestamp")
        print(f"  [PASS] Telemetry successfully received by Central Server. Latest timestamp: {tel_time}")
        print(f"    Metrics: CPU={latest_tel.get('cpu_utilization')}%, RAM={latest_tel.get('ram_utilization')}%, Disk={latest_tel.get('disk_utilization')}%")
        test_results["Step 3 - Telemetry Reaches Server"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Step 3 FAILED: {e}")
        test_results["Step 3 - Telemetry Reaches Server"] = f"FAIL: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Confirm Telemetry is Stored in SQLite
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 4] Confirming telemetry persistence in SQLite database...")
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, node_id, timestamp, cpu_utilization, ram_utilization, disk_utilization, process_count
                FROM telemetry
                WHERE node_id = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (real_nid,))
            row = cursor.fetchone()
            assert row is not None, f"No row found in SQLite telemetry for {real_nid}"
            print(f"  [PASS] Direct SQLite query confirmed: snapshot id={row['id']}, node={row['node_id']}, CPU={row['cpu_utilization']}%, RAM={row['ram_utilization']}%")
        test_results["Step 4 - SQLite Persistence"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Step 4 FAILED: {e}")
        test_results["Step 4 - SQLite Persistence"] = f"FAIL: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Confirm ML Model Receives Correctly Formatted Features
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 5] Confirming ML Model Feature Extraction & Formatting...")
    try:
        sample_raw = {
            "cpu_utilization": row["cpu_utilization"],
            "ram_utilization": row["ram_utilization"],
            "disk_utilization": row["disk_utilization"],
            "process_count": row["process_count"],
            "cpu_temperature_c": 52.0,
            "network_sent_mb": 45.2,
            "network_recv_mb": 110.8
        }
        pred_res = predict_failure(sample_raw)
        feats_used = pred_res.get("features_used", {})
        expected_features = ["CPU_utilization", "Memory_utilization", "Disk_utilization", "Resource_pressure"]
        for ef in expected_features:
            assert ef in feats_used, f"Missing expected training feature '{ef}'"
        
        # Verify Resource Pressure mathematical compound: (CPU * RAM) / 100
        expected_rp = round((float(sample_raw["cpu_utilization"]) * float(sample_raw["ram_utilization"])) / 100.0, 4)
        actual_rp = round(float(feats_used["Resource_pressure"]), 4)
        assert abs(actual_rp - expected_rp) < 1e-3, f"Resource pressure mismatch: got {actual_rp}, expected {expected_rp}"
        print(f"  [PASS] Features correctly formatted and verified: {feats_used}")
        test_results["Step 5 - Feature Formatting"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Step 5 FAILED: {e}")
        test_results["Step 5 - Feature Formatting"] = f"FAIL: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6, 7 & 8: Real Prediction, Probability, and Risk Level
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEPS 6-8] Verifying Real Prediction, Calibrated Probability, and Risk Level...")
    try:
        prob = pred_res["failure_probability"]
        risk = pred_res["risk_level"]
        pred_class = pred_res["prediction"]
        assert 0.0 <= prob <= 1.0, f"Probability out of bounds: {prob}"
        assert risk in ("LOW", "MEDIUM", "HIGH"), f"Invalid risk level: {risk}"
        assert pred_class in (0, 1), f"Invalid prediction class: {pred_class}"

        if prob < 0.30:
            assert risk == "LOW", f"Expected LOW for prob {prob}"
        elif prob < 0.70:
            assert risk == "MEDIUM", f"Expected MEDIUM for prob {prob}"
        else:
            assert risk == "HIGH", f"Expected HIGH for prob {prob}"

        print(f"  [PASS] ML Prediction: Class={pred_class}, Failure Probability={prob*100:.2f}%, Risk Level={risk}")
        test_results["Steps 6-8 - Prediction & Risk Level"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Steps 6-8 FAILED: {e}")
        test_results["Steps 6-8 - Prediction & Risk Level"] = f"FAIL: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9: Trigger HIGH/CRITICAL Alert when Appropriate
    # ─────────────────────────────────────────────────────────────────────────
    test_node_id = f"node_e2e_stress_{int(time.time())}"
    created_test_nodes.append(test_node_id)
    print(f"\n[STEP 9] Triggering Real HIGH-RISK Alert for test node '{test_node_id}'...")
    try:
        # Register test node
        register_node(
            node_id=test_node_id,
            hostname="E2E-TEST-NODE",
            ip_address="127.0.0.1",
            operating_system="Windows Server 2022"
        )
        
        # High stress telemetry: CPU 89%, RAM 94%, Temp 84 deg C
        stress_payload = {
            "node_id": test_node_id,
            "hostname": "E2E-TEST-NODE",
            "operating_system": "Windows Server 2022",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "cpu_utilization": 89.0,
                "ram_utilization": 94.0,
                "disk_utilization": 75.0,
                "cpu_temperature_c": 84.0,
                "network_sent_mb": 180.0,
                "network_recv_mb": 350.0,
                "process_count": 350,
                "duration_sec": 7200
            }
        }
        status, resp_data = http_post("/telemetry", stress_payload)
        assert status == 200, f"Expected 200 OK from /telemetry, got {status}"
        alert_data = resp_data.get("alert")
        assert alert_data is not None, "Expected alert to be generated in /telemetry response, but was None"
        assert alert_data["risk_level"] in ("HIGH", "CRITICAL"), f"Expected HIGH risk, got {alert_data['risk_level']}"
        assert alert_data["failure_probability"] >= 0.70, f"Expected prob >= 0.70, got {alert_data['failure_probability']}"
        alert_id = alert_data["alert_id"]
        created_test_alerts.append(alert_id)

        print(f"  [PASS] Alert successfully triggered: Alert ID='{alert_id}', Risk='{alert_data['risk_level']}', Prob={alert_data['failure_probability']*100:.1f}%")
        print(f"    Reason: {alert_data['reason']}")
        print(f"    Action: {alert_data['recommended_action']}")
        test_results["Step 9 - Alert Triggering"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Step 9 FAILED: {e}")
        test_results["Step 9 - Alert Triggering"] = f"FAIL: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 10: Show Alert on Central Server Dashboard
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 10] Confirming Alert Visibility on Central Server Dashboard...")
    try:
        active_alerts = fetch_active_alerts()
        alert_match = next((a for a in active_alerts if a["alert_id"] == alert_id), None)
        assert alert_match is not None, f"Alert {alert_id} not found in fetch_active_alerts()"
        
        status, api_active = http_get("/alerts/active")
        assert status == 200, f"Expected 200 OK from /alerts/active, got {status}"
        api_alerts_list = api_active.get("active_alerts") or api_active.get("alerts") or []
        api_match = next((a for a in api_alerts_list if a.get("alert_id") == alert_id), None)
        assert api_match is not None, f"Alert {alert_id} not found in REST endpoint /alerts/active"
        
        print(f"  [PASS] Alert {alert_id} confirmed active on Dashboard and Central REST API.")
        test_results["Step 10 - Alert Dashboard Visibility"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Step 10 FAILED: {e}")
        test_results["Step 10 - Alert Dashboard Visibility"] = f"FAIL: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 11: Send Alert to Affected Node Agent
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 11] Verifying Alert Transmission to Affected Node Agent...")
    try:
        status, node_alert_resp = http_get(f"/nodes/{test_node_id}/alerts/active")
        assert status == 200, f"Expected 200 OK from /nodes/{test_node_id}/alerts/active, got {status}"
        assert node_alert_resp["has_active_alert"] is True, "Expected has_active_alert=True"
        assert node_alert_resp["alert"]["alert_id"] == alert_id, "Alert ID mismatch in node endpoint"
        print(f"  [PASS] Node endpoint verified: GET /nodes/{test_node_id}/alerts/active dispatched alert {alert_id} to node.")
        test_results["Step 11 - Alert Transmission to Node"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Step 11 FAILED: {e}")
        test_results["Step 11 - Alert Transmission to Node"] = f"FAIL: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 12: Show Node-Side Notification
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 12] Testing Node-Side Notification Execution...")
    try:
        notifier = NodeAlertNotifier(test_node_id)
        displayed = notifier.process_server_alert(alert_data)
        assert displayed is True, "First alert notification was not displayed by notifier"
        print(f"  [PASS] Node-side notifier processed alert and rendered notification cleanly.")
        test_results["Step 12 - Node Notification Execution"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Step 12 FAILED: {e}")
        test_results["Step 12 - Node Notification Execution"] = f"FAIL: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 13: Verify Alert Deduplication & Cooldown
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 13] Verifying Alert Deduplication & Cooldown Protection...")
    try:
        # Send 3 subsequent ticks of identical high strain within cooldown window
        for i in range(3):
            stress_payload["timestamp"] = datetime.now(timezone.utc).isoformat()
            status, rep_resp = http_post("/telemetry", stress_payload)
            assert status == 200, f"Expected 200 OK, got {status}"
            assert rep_resp.get("alert") is not None
            assert rep_resp["alert"]["alert_id"] == alert_id, f"Duplicate alert created! Expected {alert_id}, got {rep_resp['alert']['alert_id']}"
            
            # Test notifier deduplication
            suppressed = notifier.process_server_alert(rep_resp["alert"])
            assert suppressed is False, f"Duplicate alert popup was not suppressed on tick {i+1}"

        # Verify repeat count in database
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT repeat_count, status FROM maintenance_alerts WHERE alert_id = ?", (alert_id,))
            a_row = cursor.fetchone()
            assert a_row is not None
            assert a_row["repeat_count"] == 4, f"Expected repeat_count=4, got {a_row['repeat_count']}"
            print(f"  [PASS] Deduplication confirmed: 4 consecutive cycles consolidated into single Alert {alert_id} (repeat_count={a_row['repeat_count']}). Popups suppressed.")
        test_results["Step 13 - Deduplication & Cooldown"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Step 13 FAILED: {e}")
        test_results["Step 13 - Deduplication & Cooldown"] = f"FAIL: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 14: Verify Telemetry Collection Continues
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 14] Verifying Telemetry Collection Continues Uninterrupted...")
    try:
        cont_payload = {
            "node_id": test_node_id,
            "hostname": "E2E-TEST-NODE",
            "operating_system": "Windows Server 2022",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "cpu_utilization": 85.0,
                "ram_utilization": 91.0,
                "disk_utilization": 75.0,
                "cpu_temperature_c": 80.0,
                "network_sent_mb": 190.0,
                "network_recv_mb": 360.0,
                "process_count": 352,
                "duration_sec": 7260
            }
        }
        status, cont_resp = http_post("/telemetry", cont_payload)
        assert status == 200, f"Expected 200 OK, got {status}"
        
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as c FROM telemetry WHERE node_id = ?", (test_node_id,))
            c_count = cursor.fetchone()["c"]
            assert c_count >= 4, f"Expected at least 4 snapshots, got {c_count}"
            print(f"  [PASS] Telemetry stream continued without interruption (Total snapshots recorded: {c_count}).")
        test_results["Step 14 - Continuous Telemetry Collection"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Step 14 FAILED: {e}")
        test_results["Step 14 - Continuous Telemetry Collection"] = f"FAIL: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 15: Verify Multiple Nodes Do Not Interfere with Each Other
    # ─────────────────────────────────────────────────────────────────────────
    node_b_id = f"node_e2e_healthy_{int(time.time())}"
    created_test_nodes.append(node_b_id)
    print(f"\n[STEP 15] Verifying Multi-Node Isolation between '{test_node_id}' and '{node_b_id}'...")
    try:
        register_node(
            node_id=node_b_id,
            hostname="E2E-HEALTHY-NODE",
            ip_address="127.0.0.1",
            operating_system="Linux Ubuntu 22.04"
        )
        healthy_payload = {
            "node_id": node_b_id,
            "hostname": "E2E-HEALTHY-NODE",
            "operating_system": "Linux Ubuntu 22.04",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "cpu_utilization": 18.0,
                "ram_utilization": 38.0,
                "disk_utilization": 42.0,
                "cpu_temperature_c": 44.0,
                "network_sent_mb": 12.0,
                "network_recv_mb": 25.0,
                "process_count": 140,
                "duration_sec": 12000
            }
        }
        status, b_resp = http_post("/telemetry", healthy_payload)
        assert status == 200
        assert b_resp.get("alert") is None, f"Healthy Node B received unexpected alert: {b_resp.get('alert')}"
        assert b_resp["predictive_maintenance"]["risk_level"] == "LOW"
        
        status, b_alert_api = http_get(f"/nodes/{node_b_id}/alerts/active")
        assert b_alert_api["has_active_alert"] is False
        assert b_alert_api["alert"] is None
        
        # Verify Node A alert is still active and isolated to Node A
        status, a_alert_api = http_get(f"/nodes/{test_node_id}/alerts/active")
        assert a_alert_api["has_active_alert"] is True
        assert a_alert_api["alert"]["node_id"] == test_node_id
        
        print(f"  [PASS] Multi-Node isolation strictly confirmed:")
        print(f"    - Node A ({test_node_id}): Alert ACTIVE (HIGH risk)")
        print(f"    - Node B ({node_b_id}): Alert NONE (LOW risk)")
        test_results["Step 15 - Multi-Node Isolation"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Step 15 FAILED: {e}")
        test_results["Step 15 - Multi-Node Isolation"] = f"FAIL: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # SCENARIO TESTS (6 Scenarios)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RUNNING OPERATIONAL SCENARIO TESTS")
    print("=" * 70)

    # Scenario 1: Normal Node
    print("\n[SCENARIO 1] Normal Node Operating Baseline...")
    try:
        res_norm = predict_failure({"cpu_utilization": 22.0, "ram_utilization": 35.0, "disk_utilization": 40.0})
        assert res_norm["risk_level"] == "LOW"
        assert res_norm["failure_probability"] < 0.10
        print(f"  [PASS] Scenario 1: Prob={res_norm['failure_probability']*100:.2f}%, Risk={res_norm['risk_level']}")
        test_results["Scenario 1 - Normal Node"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Scenario 1: {e}")
        test_results["Scenario 1 - Normal Node"] = f"FAIL: {e}"

    # Scenario 2: Increasing Resource Usage
    print("\n[SCENARIO 2] Increasing Resource Usage Progression (Low -> Medium -> High)...")
    try:
        res_low = predict_failure({"cpu_utilization": 25.0, "ram_utilization": 30.0, "disk_utilization": 30.0})
        res_med = predict_failure({"cpu_utilization": 68.0, "ram_utilization": 72.0, "disk_utilization": 60.0})
        res_high = predict_failure({"cpu_utilization": 88.0, "ram_utilization": 93.0, "disk_utilization": 80.0})
        
        assert res_low["failure_probability"] < res_med["failure_probability"] < res_high["failure_probability"], \
            f"Expected monotonic probability climb: {res_low['failure_probability']} < {res_med['failure_probability']} < {res_high['failure_probability']}"
        assert res_low["risk_level"] == "LOW"
        assert res_high["risk_level"] == "HIGH"
        print(f"  [PASS] Scenario 2: Monotonic Risk Progression Verified:")
        print(f"    - Low Strain   : CPU 25%, RAM 30% -> Prob={res_low['failure_probability']*100:.2f}% [{res_low['risk_level']}]")
        print(f"    - Med Strain   : CPU 68%, RAM 72% -> Prob={res_med['failure_probability']*100:.2f}% [{res_med['risk_level']}]")
        print(f"    - High Strain  : CPU 88%, RAM 93% -> Prob={res_high['failure_probability']*100:.2f}% [{res_high['risk_level']}]")
        test_results["Scenario 2 - Increasing Resource Usage"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Scenario 2: {e}")
        test_results["Scenario 2 - Increasing Resource Usage"] = f"FAIL: {e}"

    # Scenario 3: High-Risk Condition
    print("\n[SCENARIO 3] High-Risk Condition & Automated Alert Creation...")
    try:
        res_crit = predict_failure({"cpu_utilization": 92.0, "ram_utilization": 96.0, "disk_utilization": 88.0})
        assert res_crit["risk_level"] == "HIGH"
        assert res_crit["failure_probability"] > 0.95
        assert len(res_crit["main_contributing_factors"]) > 0
        print(f"  [PASS] Scenario 3: High-Risk detected with Prob={res_crit['failure_probability']*100:.2f}% [{res_crit['risk_level']}]")
        print(f"    Dominant factors: {res_crit['main_contributing_factors']}")
        test_results["Scenario 3 - High-Risk Condition"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Scenario 3: {e}")
        test_results["Scenario 3 - High-Risk Condition"] = f"FAIL: {e}"

    # Scenario 4: Node Disconnect
    print("\n[SCENARIO 4] Node Disconnect & Offline Detection...")
    disco_node_id = f"node_e2e_disconnect_{int(time.time())}"
    created_test_nodes.append(disco_node_id)
    try:
        register_node(disco_node_id, "DISCONNECT-NODE", "127.0.0.1", "Linux")
        past_iso = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        with get_connection() as conn:
            conn.execute("UPDATE nodes SET last_seen = ? WHERE node_id = ?", (past_iso, disco_node_id))
            conn.commit()
        
        update_node_statuses(offline_threshold_sec=60)
        
        status, d_info = http_get(f"/nodes/{disco_node_id}")
        assert d_info["status"] == "OFFLINE", f"Expected OFFLINE status, got {d_info['status']}"
        print(f"  [PASS] Scenario 4: Node '{disco_node_id}' successfully transitioned to OFFLINE after missed heartbeat window.")
        test_results["Scenario 4 - Node Disconnect"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Scenario 4: {e}")
        test_results["Scenario 4 - Node Disconnect"] = f"FAIL: {e}"

    # Scenario 5: Node Reconnect
    print("\n[SCENARIO 5] Node Reconnect & Online Recovery...")
    try:
        status, hb_res = http_post("/heartbeat", {"node_id": disco_node_id})
        assert status == 200
        
        status, rec_info = http_get(f"/nodes/{disco_node_id}")
        assert rec_info["status"] == "ONLINE", f"Expected ONLINE status after reconnect, got {rec_info['status']}"
        print(f"  [PASS] Scenario 5: Node '{disco_node_id}' reconnected and returned to ONLINE state.")
        test_results["Scenario 5 - Node Reconnect"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Scenario 5: {e}")
        test_results["Scenario 5 - Node Reconnect"] = f"FAIL: {e}"

    # Scenario 6: Multiple Nodes
    print("\n[SCENARIO 6] Multiple Concurrent Nodes Verification...")
    try:
        status, all_nodes = http_get("/nodes")
        assert len(all_nodes["nodes"]) >= 3, f"Expected at least 3 nodes, got {len(all_nodes['nodes'])}"
        node_ids = [n["node_id"] for n in all_nodes["nodes"]]
        print(f"  [PASS] Scenario 6: Concurrent multi-node registry verified. Registered nodes: {len(node_ids)}")
        test_results["Scenario 6 - Multiple Nodes"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Scenario 6: {e}")
        test_results["Scenario 6 - Multiple Nodes"] = f"FAIL: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # CLEANUP TEST ARTIFACTS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[CLEANUP] Cleaning up test-generated rows from SQLite...")
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            for tn in created_test_nodes:
                cursor.execute("DELETE FROM maintenance_alerts WHERE node_id = ?", (tn,))
                cursor.execute("DELETE FROM telemetry WHERE node_id = ?", (tn,))
                cursor.execute("DELETE FROM live_health_predictions WHERE node_id = ?", (tn,))
                cursor.execute("DELETE FROM node_security_predictions WHERE node_id = ?", (tn,))
                cursor.execute("DELETE FROM nodes WHERE node_id = ?", (tn,))
            conn.commit()
        print(f"  [PASS] Cleaned up {len(created_test_nodes)} test nodes and associated alerts/snapshots.")
    except Exception as clean_err:
        print(f"  [!] Cleanup note: {clean_err}")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY REPORT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("END-TO-END TEST EXECUTION SUMMARY REPORT")
    print("=" * 70)
    all_passed = True
    for test_name, result in test_results.items():
        status_symbol = "[PASS]" if result == "PASS" else "[FAIL]"
        print(f"  {status_symbol:6} | {test_name}")
        if result != "PASS":
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL 15 WORKFLOW STEPS AND 6 SCENARIOS PASSED WITH 100% SUCCESS!")
    else:
        print("SOME TESTS FAILED. CHECK LOGS ABOVE.")
    print("=" * 70)
    return test_results


if __name__ == "__main__":
    results = run_e2e_tests()
    all_ok = all(v == "PASS" for v in results.values())
    sys.exit(0 if all_ok else 1)
