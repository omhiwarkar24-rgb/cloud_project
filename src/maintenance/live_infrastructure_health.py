"""
live_infrastructure_health.py

LIVE INFRASTRUCTURE HEALTH / ANOMALY LAYER
==========================================
This module is intentionally SEPARATE from the historical Alibaba Cluster Trace
predictive-maintenance model (`maintenance.predict.predict_failure`).

Why a separate model
--------------------
The historical Random Forest was trained on:

    CPU_utilization, Memory_utilization, Disk_utilization,
    Load_1min, Load_5min, Load_15min, Duration_sec, Resource_pressure,
    Server_Type (ComputeNode / ContainerServer / BatchTask)

Live node-agent telemetry provides:

    cpu_utilization, ram_utilization, disk_utilization,
    process_count, cpu_temperature_c, network/disk counters, RAM/disk sizes

Compatible overlap is only the three utilisation percentages (and a derived
resource-pressure term). Load averages, Alibaba job Duration_sec, and
Server_Type are NOT present on the live agent. Filling those with zeros or
hard-coded categories would fabricate features and bias the historical model.

This layer therefore:
  * leaves the Alibaba training data and joblib models untouched
  * scores each node with an Isolation Forest fitted on a documented healthy
    operational envelope in the LIVE feature space
  * never calls predict_failure() on live telemetry
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIVE_MODEL_DIR = os.path.join(BASE_DIR, "models", "live_infrastructure")
LIVE_MODEL_PATH = os.path.join(LIVE_MODEL_DIR, "isolation_forest_health.joblib")

MODEL_NAME = "LiveInfrastructureIsolationForest"
MODEL_VERSION = "1.0.0"

# Features that actually exist on the node agent (plus one derived quantity).
LIVE_FEATURE_NAMES = [
    "cpu_utilization",
    "ram_utilization",
    "disk_utilization",
    "resource_pressure",
    "process_count",
    "cpu_temperature_c",
]

# Used only when the host does not report temperature, so missing temp does
# not look like an anomaly (matches the healthy-prior mean).
TEMP_IMPUTE_C = 48.0

HEALTHY_MAX = 0.30
WATCH_MAX = 0.70

_PIPELINE_CACHE: Optional[dict] = None


def _generate_healthy_prior(n: int = 2500, seed: int = 42) -> np.ndarray:
    """
    Synthetic healthy cloud-node envelope in the LIVE feature space.

    This is NOT Alibaba Cluster Trace data and does NOT reuse Machine_failure
    labels. It encodes typical idle-to-moderate utilisation for a compute node.
    """
    rng = np.random.default_rng(seed)
    cpu = np.clip(rng.normal(22.0, 12.0, n), 1.0, 65.0)
    ram = np.clip(rng.normal(40.0, 14.0, n), 10.0, 72.0)
    disk = np.clip(rng.normal(35.0, 15.0, n), 8.0, 78.0)
    procs = np.clip(rng.normal(160.0, 45.0, n), 40.0, 320.0)
    temp = np.clip(rng.normal(TEMP_IMPUTE_C, 7.0, n), 32.0, 68.0)
    pressure = (cpu * ram) / 100.0
    return np.column_stack([cpu, ram, disk, pressure, procs, temp])


def _fit_pipeline() -> dict:
    X = _generate_healthy_prior()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_scaled)
    pipeline = {
        "scaler": scaler,
        "model": model,
        "feature_names": list(LIVE_FEATURE_NAMES),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "n_features": len(LIVE_FEATURE_NAMES),
        "trained_on": "healthy_operational_envelope_v1",
    }
    os.makedirs(LIVE_MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, LIVE_MODEL_PATH)
    return pipeline


def load_live_health_pipeline(force_reload: bool = False) -> dict:
    """Load the live Isolation Forest pipeline, fitting once if missing."""
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is not None and not force_reload:
        return _PIPELINE_CACHE

    if os.path.exists(LIVE_MODEL_PATH) and not force_reload:
        pipeline = joblib.load(LIVE_MODEL_PATH)
        names = pipeline.get("feature_names", [])
        if names != LIVE_FEATURE_NAMES or pipeline.get("model_version") != MODEL_VERSION:
            pipeline = _fit_pipeline()
    else:
        pipeline = _fit_pipeline()

    _PIPELINE_CACHE = pipeline
    return pipeline


def metrics_to_live_feature_vector(metrics: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    Map a node-agent metrics dict onto the live feature vector.

    Shape is always (1, n_features). Missing numeric fields become 0.0 except
    temperature, which is imputed to TEMP_IMPUTE_C so absence is not an outlier.
    """
    m = metrics or {}
    cpu = float(m.get("cpu_utilization") or 0.0)
    ram = float(m.get("ram_utilization") or 0.0)
    disk = float(m.get("disk_utilization") or 0.0)
    procs = float(m.get("process_count") or 0.0)
    temp_raw = m.get("cpu_temperature_c")
    temp = float(temp_raw) if temp_raw is not None else TEMP_IMPUTE_C
    pressure = (cpu * ram) / 100.0
    row = np.array([[cpu, ram, disk, pressure, procs, temp]], dtype=float)
    expected = len(LIVE_FEATURE_NAMES)
    if row.shape != (1, expected):
        raise ValueError(
            f"Live health feature shape {row.shape} != (1, {expected})"
        )
    return row


def _anomaly_probability(decision_value: float) -> float:
    """
    Map IsolationForest.decision_function (higher = more normal, 0 ≈ threshold)
    to an anomaly probability in [0, 1].
    """
    # Negative decision => outlier. Logistic keeps the mapping smooth and bounded.
    return float(1.0 / (1.0 + np.exp(4.0 * decision_value)))


def _status_from_probability(prob: float) -> str:
    if prob < HEALTHY_MAX:
        return "HEALTHY"
    if prob < WATCH_MAX:
        return "WATCH"
    return "ANOMALOUS"


def _contributing_features(raw_row: np.ndarray, scaler: StandardScaler, top_k: int = 4) -> List[Dict[str, Any]]:
    z = (raw_row.reshape(-1) - scaler.mean_) / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    order = np.argsort(np.abs(z))[::-1][:top_k]
    out = []
    for i in order:
        out.append({
            "feature": LIVE_FEATURE_NAMES[int(i)],
            "value": round(float(raw_row.reshape(-1)[int(i)]), 4),
            "z_score": round(float(z[int(i)]), 4),
        })
    return out


def predict_live_health(
    metrics: Optional[Dict[str, Any]],
    node_id: str = "unknown",
    timestamp: Optional[str] = None,
    node_status: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Score one telemetry snapshot with the live Isolation Forest.

    Returns a dict with node_id, health_status, anomaly_probability,
    model_name, model_version, timestamp, contributing_features.
    """
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    if not metrics:
        return {
            "node_id": node_id,
            "health_status": "NO_TELEMETRY",
            "anomaly_probability": None,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "timestamp": ts,
            "contributing_features": [],
            "node_status": node_status,
            "disclaimer": (
                "Live infrastructure anomaly layer. Not the historical "
                "Alibaba predictive-maintenance model."
            ),
        }

    pipeline = load_live_health_pipeline()
    scaler = pipeline["scaler"]
    model = pipeline["model"]

    X = metrics_to_live_feature_vector(metrics)
    n_expected = int(getattr(model, "n_features_in_", X.shape[1]))
    if X.shape[1] != n_expected:
        raise ValueError(
            f"Feature-shape error: live vector has {X.shape[1]} columns, "
            f"model expects {n_expected}"
        )

    X_scaled = scaler.transform(X)
    if X_scaled.shape[1] != n_expected:
        raise ValueError(
            f"Feature-shape error after scaling: {X_scaled.shape} vs {n_expected}"
        )

    decision = float(model.decision_function(X_scaled)[0])
    prob = round(_anomaly_probability(decision), 4)
    status = _status_from_probability(prob)

    return {
        "node_id": node_id,
        "health_status": status,
        "anomaly_probability": prob,
        "model_name": pipeline.get("model_name", MODEL_NAME),
        "model_version": pipeline.get("model_version", MODEL_VERSION),
        "timestamp": ts,
        "contributing_features": _contributing_features(X, scaler),
        "node_status": node_status,
        "feature_vector": {
            name: round(float(X[0, i]), 4)
            for i, name in enumerate(LIVE_FEATURE_NAMES)
        },
        "disclaimer": (
            "Live infrastructure anomaly layer using node-agent features only. "
            "Not the historical Alibaba Cluster Trace maintenance model."
        ),
    }


def evaluate_and_store_live_health(payload: Dict[str, Any], node_status: Optional[str] = None) -> Dict[str, Any]:
    """Predict from a telemetry POST payload and persist the result."""
    from central_server.database import save_live_health_prediction

    node_id = payload.get("node_id", "unknown")
    ts = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
    metrics = payload.get("metrics") or {}
    result = predict_live_health(metrics, node_id=node_id, timestamp=ts, node_status=node_status)
    save_live_health_prediction(result)
    return result


def predict_all_registered_nodes() -> List[Dict[str, Any]]:
    """Score every registered node from its latest stored telemetry snapshot."""
    from central_server.database import (
        get_all_nodes,
        get_latest_telemetry,
        save_live_health_prediction,
    )

    results = []
    for node in get_all_nodes():
        nid = node["node_id"]
        latest = get_latest_telemetry(nid)
        metrics = None
        ts = node.get("last_seen")
        if latest:
            metrics = latest.get("metrics") or {
                "cpu_utilization": latest.get("cpu_utilization"),
                "ram_utilization": latest.get("ram_utilization"),
                "disk_utilization": latest.get("disk_utilization"),
                "process_count": latest.get("process_count"),
                "cpu_temperature_c": latest.get("cpu_temperature_c"),
            }
            ts = latest.get("timestamp") or ts
        result = predict_live_health(
            metrics,
            node_id=nid,
            timestamp=ts,
            node_status=node.get("status"),
        )
        save_live_health_prediction(result)
        results.append(result)
    return results
