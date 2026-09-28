"""
test_live_http_scenarios.py - HTTP Integration Test verifying live communication between
Node Agents, Central Server (port 8000), SQLite telemetry.db, and Dashboard.
"""

import os
import sys
import time
import json
import socket
import urllib.request
import urllib.error
import sqlite3

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

SERVER_URL = "http://localhost:8000"
DB_PATH = os.path.join(BASE_DIR, "data", "telemetry.db")


def http_get(path: str):
    url = f"{SERVER_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def http_post(path: str, data: dict):
    url = f"{SERVER_URL}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def main():
    print("=" * 70)
    print("RUNNING LIVE HTTP SCENARIOS AGAINST CENTRAL SERVER (PORT 8000)")
    print("=" * 70)

    # TEST 1: Check root and initial fleet
    print("\n[TEST 1] Querying Central Server with zero agents...")
    root = http_get("/")
    assert root["status"] == "online"
    print(f"  -> Central API status: {root['status']}")

    # Clean DB for fresh test
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM telemetry")
    conn.execute("DELETE FROM nodes")
    conn.commit()
    conn.close()

    resp = http_get("/nodes")
    node_list = resp.get("nodes", [])
    print(f"  -> Nodes returned: {node_list}")
    assert len(node_list) == 0, f"Expected 0 nodes, got {len(node_list)}"
    print("  [PASS] Test 1: Exactly 0 live nodes exist. No dummy nodes.")

    # TEST 2: Start 1 real agent via node_agent metrics & API
    print("\n[TEST 2] Connecting ONE real Node Agent...")
    from node_agent.metrics import collect_node_telemetry
    hostname = socket.gethostname()
    node_id_1 = f"live_node_{hostname.lower()}_alpha"

    # Agent registers
    reg_resp = http_post("/register", {
        "node_id": node_id_1,
        "hostname": hostname,
        "operating_system": "Windows",
        "hardware_info": {"cpu_cores": 8, "ram_gb": 16}
    })
    print(f"  -> Registration response: {reg_resp}")

    # Agent sends real telemetry
    telemetry_1 = collect_node_telemetry(node_id_1)
    tel_resp = http_post("/telemetry", telemetry_1)
    print(f"  -> Telemetry response: {tel_resp}")

    # Query /nodes
    resp = http_get("/nodes")
    node_list = resp.get("nodes", [])
    assert len(node_list) == 1, f"Expected 1 node, got {len(node_list)}"
    assert node_list[0]["node_id"] == node_id_1
    assert node_list[0]["status"] == "ONLINE"
    print(f"  [PASS] Test 2: Exactly 1 real node ({node_id_1}) is registered and ONLINE.")

    # TEST 3: Start a second real agent
    print("\n[TEST 3] Connecting SECOND real Node Agent...")
    node_id_2 = f"live_node_{hostname.lower()}_beta"
    reg_resp_2 = http_post("/register", {
        "node_id": node_id_2,
        "hostname": f"{hostname}-worker2",
        "operating_system": "Windows",
        "hardware_info": {"cpu_cores": 4, "ram_gb": 8}
    })
    telemetry_2 = collect_node_telemetry(node_id_2)
    http_post("/telemetry", telemetry_2)

    resp = http_get("/nodes")
    node_list = resp.get("nodes", [])
    assert len(node_list) == 2, f"Expected 2 nodes, got {len(node_list)}"
    n_ids = {n["node_id"] for n in node_list}
    assert node_id_1 in n_ids and node_id_2 in n_ids
    print(f"  [PASS] Test 3: Both nodes dynamically active: {n_ids}")

    # TEST 4: Stop agent 1 (simulate heartbeat expiry)
    print("\n[TEST 4] Stopping Node Agent 1 (Simulate Heartbeat Timeout)...")
    from datetime import datetime, timezone, timedelta
    stale_time = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE nodes SET last_seen = ? WHERE node_id = ?", (stale_time, node_id_1))
    conn.commit()
    conn.close()

    resp = http_get("/nodes")
    node_list = resp.get("nodes", [])
    node_map = {n["node_id"]: n for n in node_list}
    assert node_map[node_id_1]["status"] == "OFFLINE", f"Expected OFFLINE for {node_id_1}, got {node_map[node_id_1]['status']}"
    assert node_map[node_id_2]["status"] == "ONLINE", f"Expected ONLINE for {node_id_2}, got {node_map[node_id_2]['status']}"
    print(f"  [PASS] Test 4: Stopped node {node_id_1} transitioned to OFFLINE. Active node {node_id_2} is ONLINE.")

    # TEST 5: Restart agent 1
    print("\n[TEST 5] Restarting Node Agent 1 (Sending fresh heartbeat/telemetry)...")
    hb_resp = http_post("/heartbeat", {"node_id": node_id_1})
    print(f"  -> Heartbeat response: {hb_resp}")
    telemetry_1_fresh = collect_node_telemetry(node_id_1)
    http_post("/telemetry", telemetry_1_fresh)

    resp = http_get("/nodes")
    node_list = resp.get("nodes", [])
    node_map = {n["node_id"]: n for n in node_list}
    assert node_map[node_id_1]["status"] == "ONLINE", f"Expected ONLINE for {node_id_1}, got {node_map[node_id_1]['status']}"
    assert node_map[node_id_2]["status"] == "ONLINE", f"Expected ONLINE for {node_id_2}, got {node_map[node_id_2]['status']}"
    print(f"  [PASS] Test 5: Node {node_id_1} recovered to ONLINE automatically.")

    print("\n" + "=" * 70)
    print("ALL 5 LIVE HTTP SCENARIOS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
