"""Controlled, isolated before/after evaluation of an advisory intervention."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from central_server.database import (
    get_connection,
    get_latest_node_security,
    get_latest_telemetry,
)
from maintenance.live_infrastructure_health import predict_live_health
from prevention.prevention_engine import generate_node_recommendations
from risk_engine.node_risk import combine_node_risk


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config", "closed_loop_experiment.json")


def load_experiment_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _score_text(value: Optional[float]) -> Optional[str]:
    return None if value is None else f"{float(value) * 100:.1f}%"


def _risk_snapshot(node_id: str, metrics: Dict[str, Any], security: Optional[Dict[str, Any]], timestamp: str):
    health = predict_live_health(metrics, node_id=node_id, timestamp=timestamp, node_status="EXPERIMENT")
    risk = combine_node_risk(node_id, security, health, timestamp=timestamp)
    return health, risk


def _controlled_workload_change(metrics: Dict[str, Any], scale_factor: float) -> Dict[str, Any]:
    changed = dict(metrics)
    for key in ("cpu_utilization", "ram_utilization", "disk_utilization", "process_count"):
        if changed.get(key) is not None:
            changed[key] = round(float(changed[key]) * scale_factor, 4)
    return changed


def _recommendation_for_experiment(risk: Dict[str, Any], health: Dict[str, Any], security: Optional[Dict[str, Any]]):
    recommendations = generate_node_recommendations(risk, health, security)
    if recommendations:
        return recommendations
    return [{
        "node_id": risk["node_id"],
        "detected_problem": "No high-risk factor detected at baseline",
        "evidence": f"Overall risk = {_score_text(risk.get('overall_risk')) or 'N/A'}",
        "recommended_action": "No preventive action recommended; workload reduction is applied only inside this controlled experiment.",
        "priority": "ROUTINE",
        "timestamp": risk["timestamp"],
        "why_generated": "The baseline did not meet a high-risk prevention threshold.",
        "advisory_only": True,
        "disclaimer": "Controlled Experimental Evaluation. No real infrastructure action was performed.",
    }]


def _store_experiment(result: Dict[str, Any]) -> None:
    connection = get_connection()
    try:
        connection.execute(
            """
            INSERT INTO closed_loop_experiments (
                experiment_id, node_id, timestamp, baseline_json, after_json,
                recommendation_json, intervention, measured_change_json, label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["experiment_id"], result["node_id"], result["timestamp"],
                json.dumps(result["before"]), json.dumps(result["after"]),
                json.dumps(result["recommendation"]), result["intervention"],
                json.dumps(result["measured_change"]), result["label"],
            ),
        )
        connection.commit()
    finally:
        connection.close()


def run_closed_loop_evaluation(
    node_id: str,
    scale_factor: Optional[float] = None,
    *,
    persist: bool = True,
) -> Dict[str, Any]:
    """Evaluate a safe workload reduction on a copied test payload only."""
    config = load_experiment_config()
    factor = float(scale_factor if scale_factor is not None else config["default_workload_scale_factor"])
    if not config["minimum_scale_factor"] <= factor <= config["maximum_scale_factor"]:
        raise ValueError("scale_factor is outside the configured safe experimental range")

    telemetry = get_latest_telemetry(node_id)
    if not telemetry or not telemetry.get("metrics"):
        raise ValueError(f"No telemetry is available for node '{node_id}'")

    timestamp = _now_iso()
    security = get_latest_node_security(node_id)
    baseline_metrics = dict(telemetry["metrics"])
    baseline_health, baseline_risk = _risk_snapshot(node_id, baseline_metrics, security, telemetry.get("timestamp") or timestamp)
    recommendation = _recommendation_for_experiment(baseline_risk, baseline_health, security)

    after_metrics = _controlled_workload_change(baseline_metrics, factor)
    after_timestamp = _now_iso()
    after_health, after_risk = _risk_snapshot(node_id, after_metrics, security, after_timestamp)

    before = {
        "cpu": baseline_metrics.get("cpu_utilization"),
        "ram": baseline_metrics.get("ram_utilization"),
        "security_risk": baseline_risk.get("security_risk"),
        "maintenance_risk": baseline_risk.get("maintenance_risk"),
        "overall_risk": baseline_risk.get("overall_risk"),
        "risk_level": baseline_risk.get("risk_level"),
        "timestamp": baseline_risk.get("timestamp"),
    }
    after = {
        "cpu": after_metrics.get("cpu_utilization"),
        "ram": after_metrics.get("ram_utilization"),
        "security_risk": after_risk.get("security_risk"),
        "maintenance_risk": after_risk.get("maintenance_risk"),
        "overall_risk": after_risk.get("overall_risk"),
        "risk_level": after_risk.get("risk_level"),
        "timestamp": after_risk.get("timestamp"),
    }
    measured_change = {
        "cpu_percentage_points": round(float(after["cpu"] or 0) - float(before["cpu"] or 0), 4),
        "ram_percentage_points": round(float(after["ram"] or 0) - float(before["ram"] or 0), 4),
        "security_risk_change": round(float(after["security_risk"] or 0) - float(before["security_risk"] or 0), 4) if before["security_risk"] is not None and after["security_risk"] is not None else None,
        "maintenance_risk_change": round(float(after["maintenance_risk"] or 0) - float(before["maintenance_risk"] or 0), 4) if before["maintenance_risk"] is not None and after["maintenance_risk"] is not None else None,
        "overall_risk_change": round(float(after["overall_risk"] or 0) - float(before["overall_risk"] or 0), 4) if before["overall_risk"] is not None and after["overall_risk"] is not None else None,
        "overall_risk_reduction": round(float(before["overall_risk"] or 0) - float(after["overall_risk"] or 0), 4) if before["overall_risk"] is not None and after["overall_risk"] is not None else None,
    }
    result = {
        "experiment_id": f"exp_{uuid.uuid4().hex[:12]}",
        "node_id": node_id,
        "label": config["label"],
        "before": before,
        "after": after,
        "recommendation": recommendation,
        "intervention": f"Safe simulated workload reduction in an isolated test payload: CPU/RAM/disk/process metrics scaled by {factor:.0%}. No host workload was changed.",
        "measured_change": measured_change,
        "timestamp": timestamp,
        "limitations": "This evaluates measured risk change in the test environment and does not prove real-world cloud failure prevention.",
    }
    if persist:
        _store_experiment(result)
    return result


def get_latest_closed_loop_evaluation(node_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    connection = get_connection()
    try:
        if node_id:
            row = connection.execute(
                "SELECT * FROM closed_loop_experiments WHERE node_id = ? ORDER BY id DESC LIMIT 1",
                (node_id,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM closed_loop_experiments ORDER BY id DESC LIMIT 1"
            ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    result = dict(row)
    for field, source in (("before", "baseline_json"), ("after", "after_json"), ("recommendation", "recommendation_json"), ("measured_change", "measured_change_json")):
        result[field] = json.loads(result.pop(source))
    result["label"] = result.pop("label")
    result["intervention"] = result["intervention"]
    return result
