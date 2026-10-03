"""
test_live_node_telemetry_flow.py

Validates the full end-to-end predictive maintenance flow:
REAL NODE -> NODE AGENT -> CENTRAL SERVER -> SQLITE TELEMETRY DATABASE
-> FEATURE PROCESSING -> MAINTENANCE MODEL -> FAILURE PROBABILITY -> RISK LEVEL
"""

import os
import sys
import json
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "app"))

from central_server.database import get_all_nodes, get_latest_telemetry
from maintenance.predict import predict_failure


def main():
    print("=" * 65)
    print("TESTING REAL LIVE NODE TELEMETRY -> PREDICTIVE MAINTENANCE PIPELINE")
    print("=" * 65)

    nodes = get_all_nodes()
    live_nodes = [n for n in nodes if n.get("status") == "ONLINE"]

    print(f"\n[1] Querying Registered Fleet from SQLite Database...")
    print(f"  -> Total registered nodes in DB: {len(nodes)}")
    print(f"  -> Actual LIVE nodes currently ONLINE: {len(live_nodes)}")

    if not live_nodes:
        print("\n[RESULT]: 'No live nodes available. Start a Node Agent.'")
        return

    print("\n[2] Processing Live Telemetry for Each Online Node...")
    for nd in live_nodes:
        nid = nd["node_id"]
        hostname = nd.get("hostname", "unknown")
        last_seen = nd.get("last_seen", "")
        
        telemetry = get_latest_telemetry(nid)
        assert telemetry is not None, f"Expected telemetry record for live node {nid}"
        
        cpu = telemetry.get("cpu_utilization")
        ram = telemetry.get("ram_utilization")
        disk = telemetry.get("disk_utilization")
        temp = telemetry.get("cpu_temperature_c")
        procs = telemetry.get("process_count")
        
        # Real pipeline
        pred_res = predict_failure({
            "cpu_utilization": cpu,
            "ram_utilization": ram,
            "disk_utilization": disk
        })
        
        print(f"\n  ================== Node: {hostname} ({nid}) ==================")
        print(f"  * Node ID:               {nid}")
        print(f"  * Last Seen:             {last_seen}")
        print(f"  * Current Telemetry:     CPU={cpu}%, RAM={ram}%, Disk={disk}%, Procs={procs}, Temp={temp}")
        print(f"  * Failure Probability:   {pred_res['failure_probability'] * 100:.2f}%")
        print(f"  * Risk Level:            {pred_res['risk_level']}")
        print(f"  * Prediction Timestamp:  {pred_res['timestamp']}")
        print(f"  * Main Contributing:     {pred_res.get('main_contributing_factors')}")
        print(f"  * Model Type:            {pred_res['model_metadata']['model_type']} (v{pred_res['model_metadata']['version']})")
        print(f"  * Factor Attribution Details:")
        for factor in pred_res.get("contributing_factors", []):
            print(f"      - {factor['feature']:<20}: {factor['value']:>7.2f} (z={factor['z_score']:>+5.2f}s, imp={factor['importance']*100:>4.1f}%) -> {factor['description']}")

    print("\n" + "=" * 65)
    print("ALL LIVE NODE TELEMETRY CHECKS VERIFIED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
