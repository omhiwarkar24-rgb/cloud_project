"""
test_live_security.py - Validate live-node security adapter.

Checks:
  1. Host telemetry (CPU/RAM/Disk) is rejected, not mapped to UNSW-NB15.
  2. A genuine UNSW-NB15 flow record scores without feature-shape errors.
  3. A demo/sample event is labeled as simulated.
  4. REST endpoints /network-flow, /security/demo-event, /security/nodes.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd
from fastapi.testclient import TestClient

from central_server.api import app
from central_server.database import init_db, register_node
from security.live_network_security import (
    InvalidNetworkFlowError,
    ingest_demo_event,
    ingest_genuine_flow,
    list_demo_attack_categories,
    validate_network_flow,
    CORE_FLOW_FIELDS,
    UNSW_RAW_FEATURES,
)


def _sample_flow(attack_cat: str = "Normal"):
    path = os.path.join(PROJECT_DIR, "data", "security", "UNSW_NB15_testing-set.csv")
    df = pd.read_csv(path)
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    row = df[df["attack_cat"].str.lower() == attack_cat.lower()].iloc[0].to_dict()
    return {k: row[k] for k in UNSW_RAW_FEATURES if k in row}


def run_tests():
    print("=" * 65)
    print("[TEST] LIVE NODE SECURITY ADAPTER")
    print("=" * 65)
    print(f"  Required UNSW raw features : {len(UNSW_RAW_FEATURES)}")
    print(f"  Core flow fields           : {CORE_FLOW_FIELDS}")

    init_db()
    register_node(
        node_id="sec_test_node",
        hostname="sec-test-host",
        operating_system="Test OS",
    )

    print("\n[TEST 1] Reject CPU/RAM/Disk host telemetry ...")
    host_payload = {
        "cpu_utilization": 88.0,
        "ram_utilization": 70.0,
        "disk_utilization": 40.0,
        "process_count": 200,
    }
    rejected = False
    try:
        validate_network_flow(host_payload)
    except InvalidNetworkFlowError as e:
        rejected = True
        print(f"  Rejected as expected: {e}")
    assert rejected, "Host telemetry must not be accepted as a network flow"
    print("  [OK] Host telemetry rejected")

    print("\n[TEST 2] Genuine UNSW-NB15 flow inference ...")
    flow = _sample_flow("Normal")
    genuine = ingest_genuine_flow("sec_test_node", flow, persist=True)
    print(f"  classification : {genuine['classification']}")
    print(f"  attack_type    : {genuine['attack_type']}")
    print(f"  security_risk  : {genuine['security_risk']}")
    print(f"  confidence     : {genuine['confidence']}")
    print(f"  model_version  : {genuine['model_version']}")
    print(f"  source         : {genuine['prediction_source']} simulated={genuine['is_simulated']}")
    assert genuine["node_id"] == "sec_test_node"
    assert genuine["classification"] in ("NORMAL", "ATTACK")
    assert genuine["prediction_source"] == "GENUINE_NETWORK_FLOW"
    assert genuine["is_simulated"] is False
    assert genuine["confidence"] is not None
    assert genuine["model_version"]
    print("  [OK] Genuine flow scored without feature-shape errors")

    print("\n[TEST 3] Demo/sample event is labeled simulated ...")
    cats = list_demo_attack_categories()
    assert "Normal" in cats or any(c.lower() == "normal" for c in cats)
    demo_cat = "DoS" if "DoS" in cats else cats[0]
    demo = ingest_demo_event("sec_test_node", demo_cat, persist=True)
    print(f"  demo category  : {demo_cat}")
    print(f"  classification : {demo['classification']}")
    print(f"  attack_type    : {demo['attack_type']}")
    print(f"  source_label   : {demo['source_label']}")
    assert demo["is_simulated"] is True
    assert demo["prediction_source"] == "DEMO_SAMPLE"
    assert "Simulated" in demo["source_label"] or "demo" in demo["source_label"].lower()
    print("  [OK] Demo event clearly labeled")

    print("\n[TEST 4] REST API endpoints ...")
    client = TestClient(app)

    bad = client.post("/network-flow", json={
        "node_id": "sec_test_node",
        "flow": host_payload,
    })
    print(f"  POST /network-flow (host metrics) -> {bad.status_code}")
    assert bad.status_code == 422

    ok = client.post("/network-flow", json={
        "node_id": "sec_test_node",
        "flow": _sample_flow("Exploits") if "Exploits" in cats else flow,
    })
    print(f"  POST /network-flow (valid) -> {ok.status_code} {ok.json().get('classification')}")
    assert ok.status_code == 200
    body = ok.json()
    assert body["prediction_source"] == "GENUINE_NETWORK_FLOW"
    assert body["is_simulated"] is False

    demo_api = client.post("/security/demo-event", json={
        "node_id": "sec_test_node",
        "attack_category": demo_cat,
    })
    print(f"  POST /security/demo-event -> {demo_api.status_code} simulated={demo_api.json().get('is_simulated')}")
    assert demo_api.status_code == 200
    assert demo_api.json()["is_simulated"] is True

    listing = client.get("/security/nodes")
    print(f"  GET /security/nodes -> {listing.status_code} total={listing.json().get('total_nodes')}")
    assert listing.status_code == 200
    ids = [n["node_id"] for n in listing.json()["nodes"]]
    assert "sec_test_node" in ids

    one = client.get("/security/nodes/sec_test_node")
    print(f"  GET /security/nodes/sec_test_node -> {one.status_code} {one.json().get('classification')}")
    assert one.status_code == 200
    for key in ("node_id", "classification", "security_risk", "confidence", "timestamp", "model_version"):
        assert key in one.json(), f"missing {key}"

    missing_node = client.post("/security/demo-event", json={
        "node_id": "does_not_exist",
        "attack_category": "Normal",
    })
    assert missing_node.status_code == 404

    print("\n" + "=" * 65)
    print("[SUCCESS] LIVE NODE SECURITY TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
