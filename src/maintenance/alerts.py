"""
alerts.py - Predictive Maintenance Alert System for Central Server & Dashboard.

Monitors real ML predictive-maintenance outputs, generating deduplicated alerts
with cooldown protection when failure risk is HIGH or CRITICAL.
"""

from __future__ import annotations

import os
import sys
import json
import importlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Ensure project root and src/ are in sys.path regardless of execution entrypoint
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_DIR = os.path.join(_BASE_DIR, "src")
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Dynamically import database module and ensure fresh attributes across Streamlit reloads
try:
    import central_server.database as _db
except ImportError:
    import src.central_server.database as _db

if not hasattr(_db, "save_maintenance_alert"):
    _db = importlib.reload(_db)

save_maintenance_alert = _db.save_maintenance_alert
update_maintenance_alert_repeat = _db.update_maintenance_alert_repeat
get_latest_active_alert_for_node = _db.get_latest_active_alert_for_node
get_active_maintenance_alerts = _db.get_active_maintenance_alerts
get_all_maintenance_alerts = _db.get_all_maintenance_alerts
acknowledge_maintenance_alert = _db.acknowledge_maintenance_alert
resolve_maintenance_alert = _db.resolve_maintenance_alert

DEFAULT_COOLDOWN_SECONDS = int(os.getenv("MAINTENANCE_ALERT_COOLDOWN_SEC", "300"))  # 5 minutes default


def generate_preventive_action(
    risk_level: str,
    failure_probability: float,
    telemetry: Dict[str, Any],
    contributing_factors: List[str]
) -> str:
    """
    Generates plain-English actionable recommendations based on root-cause factor attribution.
    """
    cpu = float(telemetry.get("cpu_utilization") or 0.0)
    ram = float(telemetry.get("ram_utilization") or 0.0)
    disk = float(telemetry.get("disk_utilization") or 0.0)
    temp = telemetry.get("cpu_temperature_c")

    actions = []

    # Check for specific stress drivers
    if temp is not None and float(temp) >= 75.0:
        actions.append("Check chassis cooling/fans and throttle heavy processes")

    if ram >= 85.0 and cpu >= 60.0:
        actions.append("Inspect the node and reduce workload (high CPU & RAM compound stress)")
    elif ram >= 85.0:
        actions.append("Isolate memory-intensive tasks, clear cached memory buffers, and reduce workload")
    elif cpu >= 75.0:
        actions.append("Inspect active threads, re-distribute CPU-intensive batch jobs, and reduce workload")
    elif disk >= 85.0:
        actions.append("Clean up temporary files/logs and expand volume capacity to prevent disk exhaustion")
    else:
        actions.append("Inspect the node and reduce workload")

    if failure_probability >= 0.90:
        actions.append("Prepare automated task failover to backup node")

    return "; ".join(actions) + "."


def evaluate_node_for_alert(
    node_id: str,
    telemetry: Dict[str, Any],
    maint_result: Dict[str, Any],
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS
) -> Optional[Dict[str, Any]]:
    """
    Evaluates ML failure prediction for alert generation with cooldown and deduplication.

    Requirements enforced:
      1. Triggered strictly by real ML prediction (risk_level in HIGH/CRITICAL or prob >= 0.70).
      2. No fake alerts.
      3. Deduplicated within cooldown window to prevent flooding.
      4. Stored persistently in SQLite.
      5. Automatically resolves alerts when node returns to healthy status.
    """
    if not maint_result or not isinstance(maint_result, dict):
        return None

    prob = float(maint_result.get("failure_probability", 0.0))
    risk = maint_result.get("risk_level", "LOW")
    now_utc = datetime.now(timezone.utc)
    now_iso = now_utc.isoformat()

    is_high_risk = (risk in ("HIGH", "CRITICAL") or prob >= 0.70)
    existing_alert = get_latest_active_alert_for_node(node_id)

    # ── CASE A: Node is healthy (LOW/MEDIUM) ──────────────────────────────────
    if not is_high_risk:
        if existing_alert:
            # Auto-resolve previous active alert as node has recovered
            resolve_maintenance_alert(existing_alert["alert_id"], resolved_at=now_iso)
        return None

    # ── CASE B: Node is in HIGH or CRITICAL risk ─────────────────────────────
    # Extract important telemetry summary
    telemetry_summary = {
        "cpu_utilization": telemetry.get("cpu_utilization"),
        "ram_utilization": telemetry.get("ram_utilization"),
        "disk_utilization": telemetry.get("disk_utilization"),
        "cpu_temperature_c": telemetry.get("cpu_temperature_c"),
        "process_count": telemetry.get("process_count"),
        "network_sent_mb": telemetry.get("network_sent_mb"),
        "network_recv_mb": telemetry.get("network_recv_mb"),
        "duration_sec": telemetry.get("duration_sec"),
    }

    # Factor attribution / reason
    factors = maint_result.get("main_contributing_factors") or []
    if factors:
        reason = "; ".join(factors)
    else:
        reason = f"High failure probability ({prob * 100:.1f}%) and sustained resource utilization."

    rec_action = generate_preventive_action(risk, prob, telemetry, factors)

    # ── DEDUPLICATION & COOLDOWN CHECK ───────────────────────────────────────
    if existing_alert:
        try:
            alert_ts = datetime.fromisoformat(existing_alert["timestamp"])
            elapsed_sec = (now_utc - alert_ts).total_seconds()
        except Exception:
            elapsed_sec = 999999.0

        if elapsed_sec < cooldown_seconds:
            # DEDUPLICATE: Update existing alert repeat_count and last_seen instead of inserting duplicate row
            update_maintenance_alert_repeat(
                alert_id=existing_alert["alert_id"],
                new_prob=prob,
                new_risk=risk,
                new_telemetry=telemetry_summary,
                last_seen_at=now_iso
            )
            existing_alert["repeat_count"] = existing_alert.get("repeat_count", 1) + 1
            existing_alert["last_seen_at"] = now_iso
            existing_alert["failure_probability"] = prob
            existing_alert["risk_level"] = risk
            existing_alert["telemetry_summary"] = telemetry_summary
            existing_alert["is_new"] = False
            existing_alert["deduplicated"] = True
            return existing_alert
        else:
            # Cooldown expired while still in failure risk: resolve old and create new tracking incident
            resolve_maintenance_alert(existing_alert["alert_id"], resolved_at=now_iso)

    # ── CREATE NEW ALERT ─────────────────────────────────────────────────────
    clean_nid = node_id.replace("-", "_").replace(" ", "_")
    alert_id = f"ALERT_{clean_nid}_{int(now_utc.timestamp())}"

    new_alert = {
        "alert_id": alert_id,
        "node_id": node_id,
        "timestamp": now_iso,
        "last_seen_at": now_iso,
        "failure_probability": prob,
        "risk_level": risk,
        "telemetry_summary": telemetry_summary,
        "reason": reason,
        "recommended_action": rec_action,
        "status": "ACTIVE",
        "repeat_count": 1,
        "resolved_at": None,
        "is_new": True,
        "deduplicated": False
    }

    save_maintenance_alert(new_alert)
    return new_alert


def fetch_active_alerts() -> List[Dict[str, Any]]:
    """Return all active alerts from the database."""
    return get_active_maintenance_alerts()


def fetch_alert_history(limit: int = 50, node_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return alert history log."""
    return get_all_maintenance_alerts(limit=limit, node_id=node_id)


def acknowledge_alert_by_id(alert_id: str) -> bool:
    """Acknowledge an alert."""
    return acknowledge_maintenance_alert(alert_id)


def resolve_alert_by_id(alert_id: str) -> bool:
    """Resolve an alert."""
    return resolve_maintenance_alert(alert_id)
