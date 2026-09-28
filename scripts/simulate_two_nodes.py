"""
simulate_two_nodes.py - Simulates independent node agents running concurrently
on the same machine, each with their own unique node_id and reporting to the
Central Telemetry API.

Usage:
    python scripts/simulate_two_nodes.py [--server-url URL] [--interval N] [--duration N]

Arguments:
    --server-url   Central server base URL (default: http://localhost:8000)
    --interval     Telemetry collection interval in seconds (default: 5)
    --duration     How long to run in seconds; 0 = run forever (default: 30)

Each simulated node:
  - Gets a unique persistent node_id via --node-suffix
  - Registers with the central server
  - Sends telemetry every --interval seconds
  - Sends heartbeats every 15 seconds
  - Reconnects automatically if the server goes offline
"""

import sys
import os
import time
import threading
import argparse

# ── Path bootstrap ────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR     = os.path.join(PROJECT_DIR, "src")
for p in (SRC_DIR, PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
# ─────────────────────────────────────────────────────────────────────────────

from node_agent.collector import run_agent
from node_agent.config    import SERVER_BASE_URL, TELEMETRY_INTERVAL_SEC


def _agent_thread(name: str, suffix: str, server_url: str, interval: int, duration: int):
    """Runs one agent in its own thread for up to *duration* seconds (0 = forever)."""
    print(f"\n[SIM] Starting simulated node '{name}' (suffix='{suffix}') ...")

    def _run():
        run_agent(
            server_base_url=server_url,
            interval_sec=interval,
            heartbeat_sec=15,
            enable_post=True,
            node_suffix=suffix,
        )

    if duration <= 0:
        _run()
        return

    t = threading.Thread(target=_run, daemon=True, name=name)
    t.start()
    t.join(timeout=duration)
    print(f"\n[SIM] Simulated node '{name}' finished after {duration}s.")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate multiple node agents reporting to Central Telemetry API"
    )
    parser.add_argument("--server-url", type=str, default=SERVER_BASE_URL,
                        help=f"Central server base URL (default: {SERVER_BASE_URL})")
    parser.add_argument("--interval",   type=int, default=TELEMETRY_INTERVAL_SEC,
                        help=f"Telemetry interval in seconds (default: {TELEMETRY_INTERVAL_SEC})")
    parser.add_argument("--duration",   type=int, default=30,
                        help="Run duration in seconds; 0 = run forever (default: 30)")
    args = parser.parse_args()

    print("=" * 65)
    print("[SIMULATION] MULTI-NODE AGENT SIMULATION")
    print(f"  Server URL : {args.server_url}")
    print(f"  Interval   : {args.interval}s")
    print(f"  Duration   : {args.duration}s (0 = forever)")
    print("=" * 65)

    # ── Define the two simulated nodes ────────────────────────────────────────
    nodes = [
        {"name": "SimNode-Alpha", "suffix": "sim_alpha"},
        {"name": "SimNode-Beta",  "suffix": "sim_beta"},
        {"name": "SimNode-Gamma", "suffix": "sim_gamma"},
    ]

    threads = []
    for node in nodes:
        t = threading.Thread(
            target=_agent_thread,
            kwargs={
                "name":       node["name"],
                "suffix":     node["suffix"],
                "server_url": args.server_url,
                "interval":   args.interval,
                "duration":   args.duration,
            },
            daemon=True,
            name=node["name"],
        )
        threads.append(t)

    # Start both threads with a small stagger to avoid simultaneous registration
    for i, t in enumerate(threads):
        t.start()
        if i < len(threads) - 1:
            time.sleep(1)   # 1 s stagger

    print(f"\n[SIMULATION] Both nodes running. Press Ctrl+C to stop early.\n")

    try:
        if args.duration > 0:
            time.sleep(args.duration + 5)     # +5 s grace period
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SIMULATION] Stopped by user.")

    print("\n[SIMULATION] Verifying node registry on server ...")
    _print_registry(args.server_url)
    print("[SIMULATION] Done.")


def _print_registry(server_url: str):
    """Quick GET /nodes check after the simulation run."""
    import urllib.request, json
    try:
        url = f"{server_url}/nodes"
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        print(f"\n  Total nodes in registry: {data['total_nodes']}")
        for n in data["nodes"]:
            print(f"    [{n['status']:7}] {n['node_id']:<40}  host={n['hostname']}")
    except Exception as e:
        print(f"  [WARN] Could not fetch node registry: {e}")


if __name__ == "__main__":
    main()
