"""Guarded bridge from live drift events to the existing retraining pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import pandas as pd

from central_server.database import get_connection


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config", "adaptive_retraining.json")
DEFAULT_MODEL_VERSION = "V1"


def load_retraining_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_id(event: Dict[str, Any]) -> str:
    payload = {
        "node_id": event.get("node_id"),
        "status": event.get("status"),
        "affected_features": sorted(event.get("affected_features", [])),
        "reference_window": event.get("reference_window"),
        "current_window": event.get("current_window"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _new_data_samples(event: Dict[str, Any]) -> int:
    value = event.get("new_data_samples")
    if value is not None:
        return int(value)
    label = str(event.get("current_window", ""))
    marker = "n="
    if marker not in label:
        return 0
    try:
        return int(label.split(marker, 1)[1].split(")", 1)[0])
    except (TypeError, ValueError):
        return 0


def _latest_run(node_id: str, target_model: str) -> Optional[Dict[str, Any]]:
    connection = get_connection()
    try:
        row = connection.execute(
            """
            SELECT * FROM adaptive_retraining_runs
            WHERE node_id = ? AND target_model = ?
            ORDER BY id DESC LIMIT 1
            """,
            (node_id, target_model),
        ).fetchone()
    finally:
        connection.close()
    return dict(row) if row else None


def _next_model_version(current: str) -> str:
    try:
        return f"V{int(str(current).lstrip('V')) + 1}"
    except (TypeError, ValueError):
        return "V2"


def _current_model_version(target_model: str) -> str:
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT current_version FROM adaptive_model_state WHERE target_model = ?",
            (target_model,),
        ).fetchone()
    finally:
        connection.close()
    return row["current_version"] if row else DEFAULT_MODEL_VERSION


def _store_run(result: Dict[str, Any]) -> None:
    connection = get_connection()
    try:
        connection.execute(
            """
            INSERT INTO adaptive_retraining_runs (
                event_id, node_id, target_model, previous_model_version,
                new_model_version, drift_event, affected_features,
                new_data_samples, training_time_seconds, validation_metrics,
                deployment_decision, status, reason, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["event_id"], result["node_id"], result["target_model"],
                result["previous_model_version"], result["new_model_version"],
                result["drift_event"], json.dumps(result["affected_features"]),
                result["new_data_samples"], result["training_time_seconds"],
                json.dumps(result["validation_metrics"]), result["deployment_decision"],
                result["status"], result["reason"], result["timestamp"],
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _result(
    event: Dict[str, Any],
    config: Dict[str, Any],
    *,
    event_id: str,
    previous_version: str,
    new_version: str,
    status: str,
    reason: str,
    deployment_decision: str,
    training_time: float = 0.0,
    validation_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "node_id": event.get("node_id"),
        "target_model": config["target_model"],
        "previous_model_version": previous_version,
        "new_model_version": new_version,
        "drift_event": event.get("status", "UNKNOWN"),
        "affected_features": event.get("affected_features", []),
        "new_data_samples": _new_data_samples(event),
        "training_time_seconds": round(training_time, 4),
        "validation_metrics": validation_metrics or {},
        "deployment_decision": deployment_decision,
        "status": status,
        "reason": reason,
        "timestamp": _now_iso(),
    }


def process_drift_event(
    event: Dict[str, Any],
    *,
    compatible_training_data: bool = False,
    pipeline_runner: Optional[Callable[[], Dict[str, Any]]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Process one event with gates before invoking the unchanged retrainer.

    Live host telemetry is deliberately incompatible with the UNSW-NB15 model.
    Tests or a future network-flow training adapter may set compatible_training_data
    only when the supplied data matches the model's processed feature contract.
    """
    config = load_retraining_config()
    node_id = event.get("node_id", "unknown")
    target_model = config["target_model"]
    event_id = event.get("event_id") or _event_id(event)
    previous_version = _current_model_version(target_model)
    new_version = _next_model_version(previous_version)

    if event.get("status") != config["required_drift_status"]:
        result = _result(event, config, event_id=event_id, previous_version=previous_version,
                         new_version=new_version, status="SKIPPED", reason="Drift is not significant.",
                         deployment_decision="NOT_ATTEMPTED")
    elif not force and _latest_run(node_id, target_model) and _latest_run(node_id, target_model)["event_id"] == event_id:
        result = _result(event, config, event_id=event_id, previous_version=previous_version,
                         new_version=new_version, status="SKIPPED", reason="Same drift event was already processed.",
                         deployment_decision="NOT_ATTEMPTED")
    elif not force and _cooldown_active(node_id, target_model, int(config["cooldown_seconds"])):
        result = _result(event, config, event_id=event_id, previous_version=previous_version,
                         new_version=new_version, status="SKIPPED", reason="Retraining cooldown is active.",
                         deployment_decision="NOT_ATTEMPTED")
    elif _new_data_samples(event) < int(config["minimum_new_samples"]):
        result = _result(event, config, event_id=event_id, previous_version=previous_version,
                         new_version=new_version, status="SKIPPED", reason="Insufficient new data for retraining.",
                         deployment_decision="NOT_ATTEMPTED")
    elif not compatible_training_data:
        result = _result(event, config, event_id=event_id, previous_version=previous_version,
                         new_version=new_version, status="BLOCKED", reason=(
                             "Live telemetry features are not valid UNSW-NB15 training inputs; "
                             "no model retraining was attempted."
                         ), deployment_decision="REJECTED_INCOMPATIBLE_DATA")
    else:
        runner = pipeline_runner
        if runner is None:
            from adaptive.retrain import run_adaptive_pipeline
            runner = run_adaptive_pipeline
        started = time.perf_counter()
        pipeline_result = runner()
        elapsed = time.perf_counter() - started
        comparison = pipeline_result.get("df_comparison", pd.DataFrame())
        metrics = comparison.to_dict("records") if isinstance(comparison, pd.DataFrame) else comparison
        deployment_status = str(pipeline_result.get("deployment_status", ""))
        approved = "DEPLOYED" in deployment_status and "RETAINED" not in deployment_status
        result = _result(
            event, config, event_id=event_id, previous_version=previous_version,
            new_version=new_version, status="COMPLETED", reason=deployment_status,
            deployment_decision="APPROVED" if approved else "REJECTED",
            training_time=elapsed, validation_metrics={"comparison": metrics},
        )
        if approved:
            _set_current_model_version(target_model, new_version)

    _store_run(result)
    return result


def _cooldown_active(node_id: str, target_model: str, cooldown_seconds: int) -> bool:
    connection = get_connection()
    try:
        row = connection.execute(
            """
            SELECT * FROM adaptive_retraining_runs
            WHERE node_id = ? AND target_model = ?
              AND deployment_decision != 'NOT_ATTEMPTED'
            ORDER BY id DESC LIMIT 1
            """,
            (node_id, target_model),
        ).fetchone()
    finally:
        connection.close()
    latest = dict(row) if row else None
    if not latest:
        return False
    try:
        occurred = datetime.fromisoformat(latest["timestamp"])
        return (datetime.now(timezone.utc) - occurred).total_seconds() < cooldown_seconds
    except (TypeError, ValueError):
        return False


def _set_current_model_version(target_model: str, version: str) -> None:
    connection = get_connection()
    try:
        connection.execute(
            """
            INSERT INTO adaptive_model_state (target_model, current_version, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(target_model) DO UPDATE SET current_version=excluded.current_version, updated_at=excluded.updated_at
            """,
            (target_model, version, _now_iso()),
        )
        connection.commit()
    finally:
        connection.close()


def get_latest_retraining_status() -> Optional[Dict[str, Any]]:
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT * FROM adaptive_retraining_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    result = dict(row)
    result["affected_features"] = json.loads(result["affected_features"] or "[]")
    result["validation_metrics"] = json.loads(result["validation_metrics"] or "{}")
    return result
