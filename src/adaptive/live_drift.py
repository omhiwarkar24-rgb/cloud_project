"""Live telemetry adapter for the existing KS drift detector."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from adaptive.drift_detection import detect_feature_drift_ks
from central_server.database import get_connection


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config", "live_drift.json")


def load_live_drift_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _telemetry_frame(node_id: str) -> pd.DataFrame:
    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT timestamp, cpu_utilization, ram_utilization, disk_utilization,
                   process_count, network_sent_mb, network_recv_mb
            FROM telemetry
            WHERE node_id = ?
            ORDER BY timestamp, id
            """,
            (node_id,),
        ).fetchall()
    finally:
        connection.close()

    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    sent = pd.to_numeric(frame["network_sent_mb"], errors="coerce").fillna(0.0)
    received = pd.to_numeric(frame["network_recv_mb"], errors="coerce").fillna(0.0)
    cumulative_network = sent + received
    frame["network_activity_delta_mb"] = cumulative_network.diff().clip(lower=0).fillna(0.0)
    for feature in ("cpu_utilization", "ram_utilization", "disk_utilization", "process_count"):
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    return frame


def _reference_rows(node_id: str) -> pd.DataFrame:
    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT timestamp, feature, value
            FROM drift_reference_samples
            WHERE node_id = ?
            ORDER BY timestamp, feature
            """,
            (node_id,),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(row) for row in rows])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["timestamp"])


def _store_reference(node_id: str, frame: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    selected = frame[["timestamp"] + features].dropna(subset=features)
    records = []
    for row in selected.to_dict("records"):
        for feature in features:
            records.append((node_id, row["timestamp"].isoformat(), feature, float(row[feature])))

    connection = get_connection()
    try:
        connection.executemany(
            """
            INSERT INTO drift_reference_samples (node_id, timestamp, feature, value)
            VALUES (?, ?, ?, ?)
            """,
            records,
        )
        connection.commit()
    finally:
        connection.close()
    return selected


def _reference_frame(node_id: str, telemetry: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    stored = _reference_rows(node_id)
    if not stored.empty:
        return stored.pivot(index="timestamp", columns="feature", values="value").reset_index()

    if telemetry.empty:
        return pd.DataFrame()

    features = config["features"]
    reference_size = int(config["reference_window_size"])
    candidate = telemetry[["timestamp"] + features].dropna(subset=features).head(reference_size)
    minimum = int(config["minimum_samples"])
    if len(candidate) < minimum:
        return pd.DataFrame()
    return _store_reference(node_id, candidate, features)


def _window_label(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "empty"
    return f"{frame['timestamp'].min().isoformat()} to {frame['timestamp'].max().isoformat()} (n={len(frame)})"


def _store_results(node_id: str, timestamp: str, results: List[Dict[str, Any]]) -> None:
    connection = get_connection()
    try:
        connection.execute(
            "DELETE FROM live_drift_results WHERE node_id = ?",
            (node_id,),
        )
        connection.executemany(
            """
            INSERT INTO live_drift_results (
                node_id, feature, drift_statistic, p_value, drift_status,
                timestamp, reference_window, current_window
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    node_id,
                    result["feature"],
                    result["drift_statistic"],
                    result["p_value"],
                    result["drift_status"],
                    timestamp,
                    result["reference_window"],
                    result["current_window"],
                )
                for result in results
            ],
        )
        connection.commit()
    finally:
        connection.close()


def detect_live_drift(node_id: str, persist: bool = True) -> Dict[str, Any]:
    """Compare a node's persisted reference window with its recent telemetry."""
    config = load_live_drift_config()
    features = list(config["features"])
    timestamp = _now_iso()
    telemetry = _telemetry_frame(node_id)
    reference = _reference_frame(node_id, telemetry, config)
    minimum = int(config["minimum_samples"])

    if reference.empty:
        return {
            "node_id": node_id,
            "data_source": "LIVE_HOST_TELEMETRY",
            "status": "COLLECTING DATA",
            "timestamp": timestamp,
            "new_data_samples": 0,
            "affected_features": [],
            "features": [],
            "reference_window": "collecting reference samples",
            "current_window": "collecting current samples",
        }

    reference_end = reference["timestamp"].max()
    current = telemetry[telemetry["timestamp"] > reference_end].tail(int(config["current_window_size"]))
    if len(reference) < minimum or len(current) < minimum:
        return {
            "node_id": node_id,
            "data_source": "LIVE_HOST_TELEMETRY",
            "status": "COLLECTING DATA",
            "timestamp": timestamp,
            "new_data_samples": len(current),
            "affected_features": [],
            "features": [],
            "reference_window": _window_label(reference),
            "current_window": _window_label(current),
        }

    current = current[["timestamp"] + features].dropna(subset=features)
    reference_values = reference[features].dropna()
    current_values = current[features].dropna()
    if len(current_values) < minimum or len(reference_values) < minimum:
        status = "COLLECTING DATA"
        results = []
    else:
        drift_frame, summary = detect_feature_drift_ks(
            reference_values,
            current_values,
            features,
            p_value_threshold=float(config["p_value_threshold"]),
            ks_stat_threshold=float(config["ks_stat_threshold"]),
        )
        results = []
        for row in drift_frame.to_dict("records"):
            results.append({
                "feature": row["feature"],
                "drift_statistic": round(float(row["ks_statistic"]), 4),
                "p_value": round(float(row["p_value"]), 6),
                "drift_status": "DRIFT DETECTED" if row["is_drifted"] else "NO DRIFT",
                "timestamp": timestamp,
                "reference_window": _window_label(reference),
                "current_window": _window_label(current),
            })
        status = "DRIFT DETECTED" if summary["system_drift_detected"] else "NO DRIFT"

    if persist and results:
        _store_results(node_id, timestamp, results)

    return {
        "node_id": node_id,
        "data_source": "LIVE_HOST_TELEMETRY",
        "status": status,
        "timestamp": timestamp,
        "new_data_samples": len(current),
        "affected_features": [row["feature"] for row in results if row["drift_status"] == "DRIFT DETECTED"],
        "features": results,
        "reference_window": _window_label(reference),
        "current_window": _window_label(current),
        "configuration": {
            "p_value_threshold": config["p_value_threshold"],
            "ks_stat_threshold": config["ks_stat_threshold"],
            "minimum_samples": minimum,
        },
    }


def detect_all_live_drift(persist: bool = True) -> List[Dict[str, Any]]:
    connection = get_connection()
    try:
        node_ids = [row["node_id"] for row in connection.execute("SELECT node_id FROM nodes ORDER BY node_id")]
    finally:
        connection.close()
    return [detect_live_drift(node_id, persist=persist) for node_id in node_ids]


def get_stored_live_drift(node_id: str) -> List[Dict[str, Any]]:
    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT feature, drift_statistic, p_value, drift_status,
                   timestamp, reference_window, current_window
            FROM live_drift_results
            WHERE node_id = ?
            ORDER BY feature
            """,
            (node_id,),
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]
