"""
server_client.py - HTTP client for the Central Telemetry REST API.

Handles:
  - Node registration  (POST /register)
  - Telemetry ingestion (POST /telemetry)
  - Heartbeat          (POST /heartbeat)
  - Exponential back-off retry on any connection failure
  - Transparent reconnection — callers never need to handle URLError
"""

from __future__ import annotations

import json
import time
import socket
import os
import sys
import platform
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, Tuple

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

try:
    from node_agent.config import (
        REGISTER_URL,
        TELEMETRY_URL,
        HEARTBEAT_URL,
        HTTP_TIMEOUT_SEC,
        RETRY_BASE_WAIT_SEC,
        MAX_RETRY_WAIT_SEC,
    )
except ImportError:
    from config import (
        REGISTER_URL,
        TELEMETRY_URL,
        HEARTBEAT_URL,
        HTTP_TIMEOUT_SEC,
        RETRY_BASE_WAIT_SEC,
        MAX_RETRY_WAIT_SEC,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Low-level HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

def _post_json(url: str, payload: dict, timeout: float = HTTP_TIMEOUT_SEC):
    """
    POST a JSON payload to *url*.
    Returns (True, response_dict) on success.
    Returns (False, None) on any network / HTTP error (caller decides retry logic).
    """
    try:
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent":   "LiveNodeAgent/2.0",
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return True, body
    except urllib.error.HTTPError as e:
        print(f"[CLIENT HTTP ERROR] {url} -> HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        print(f"[CLIENT CONN WARN] Cannot reach server at {url}: {e.reason}")
    except Exception as e:
        print(f"[CLIENT ERROR] Unexpected error posting to {url}: {e}")
    return False, None


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

def _build_register_payload(node_id: str) -> dict:
    hostname = socket.gethostname()
    import psutil
    vm   = psutil.virtual_memory()
    disk = psutil.disk_usage("C:\\" if platform.system() == "Windows" else "/")
    return {
        "node_id":          node_id,
        "hostname":         hostname,
        "operating_system": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
        "hardware_info": {
            "cpu_count_logical":  psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_mb":       round(vm.total / (1024 ** 2), 2),
            "disk_total_gb":      round(disk.total / (1024 ** 3), 2),
        },
    }


def register_with_retry(node_id: str) -> bool:
    """
    Attempts to register the node with the central server.
    Retries indefinitely with exponential back-off until the server accepts the
    registration.  Returns True once registered successfully.
    """
    payload  = _build_register_payload(node_id)
    wait_sec = RETRY_BASE_WAIT_SEC
    attempt  = 0

    while True:
        attempt += 1
        ok, _ = _post_json(REGISTER_URL, payload)
        if ok:
            print(f"[REGISTER] Node '{node_id}' registered with central server. (attempt {attempt})")
            return True

        if attempt == 2 and ("localhost" in REGISTER_URL or "127.0.0.1" in REGISTER_URL):
            print("\n" + "!" * 68)
            print("[CONNECTION TIP] The node agent is attempting to connect to 'localhost:8000'.")
            print("If this node is running on a SEPARATE LAPTOP / MACHINE, 'localhost' will NOT")
            print("reach the Central Server! Run collector with your Central Server's IP:")
            print("    python collector.py --server-url http://<CENTRAL_SERVER_IP>:8000")
            print("Or create a 'server_url.txt' file in this folder containing that address.")
            print("!" * 68 + "\n")

        print(f"[REGISTER] Server unavailable at {REGISTER_URL}. Retrying in {wait_sec}s (attempt {attempt}) ...")
        time.sleep(wait_sec)
        wait_sec = min(wait_sec * 2, MAX_RETRY_WAIT_SEC)


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry
# ─────────────────────────────────────────────────────────────────────────────

def send_telemetry_with_response(payload: dict) -> tuple[bool, Optional[dict]]:
    """
    Sends one telemetry snapshot.
    Returns (ok, response_body) where response_body includes real ML predictions
    and any active predictive maintenance alert for this node.
    """
    return _post_json(TELEMETRY_URL, payload)


def send_telemetry(payload: dict) -> bool:
    """
    Sends one telemetry snapshot.  Returns True on success, False otherwise.
    Does NOT retry — the calling loop will simply try again on the next tick.
    """
    ok, _ = _post_json(TELEMETRY_URL, payload)
    return ok


def fetch_node_alert(node_id: str) -> Optional[dict]:
    """
    Queries Central Server directly for active alerts on this node.
    """
    try:
        from node_agent.config import SERVER_BASE_URL, HTTP_TIMEOUT_SEC
    except ImportError:
        from config import SERVER_BASE_URL, HTTP_TIMEOUT_SEC
    url = f"{SERVER_BASE_URL}/nodes/{node_id}/alerts"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LiveNodeAgent/2.0"}, method="GET")
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("has_active_alert"):
                return data.get("alert")
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Heartbeat
# ─────────────────────────────────────────────────────────────────────────────

def send_heartbeat(node_id: str) -> bool:
    """
    Sends a lightweight heartbeat.  Returns True on success.
    Does NOT retry — the heartbeat thread will retry on the next scheduled beat.
    """
    ok, _ = _post_json(HEARTBEAT_URL, {"node_id": node_id})
    return ok
