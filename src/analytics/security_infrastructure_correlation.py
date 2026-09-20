"""Time-aligned analysis of security activity and infrastructure behavior."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from central_server.database import get_connection, get_all_nodes
from risk_engine.node_risk import combine_node_risk


CORRELATION_METRICS = {
    "maintenance_risk": "maintenance/health risk",
    "cpu_utilization": "CPU utilization",
    "ram_utilization": "RAM utilization",
    "network_activity_mb": "network activity",
}
MIN_CORRELATION_SAMPLES = 3


def _query_rows(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    with get_connection() as connection:
        return [dict(row) for row in connection.execute(sql, params).fetchall()]


def _timestamp_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _load_telemetry(node_id: str) -> pd.DataFrame:
    rows = _query_rows(
        """
        SELECT timestamp, cpu_utilization, ram_utilization, disk_utilization,
               network_sent_mb, network_recv_mb
        FROM telemetry
        WHERE node_id = ?
        ORDER BY timestamp
        """,
        (node_id,),
    )
    frame = _timestamp_frame(rows)
    if frame.empty:
        return frame
    frame["network_activity_mb"] = (
        pd.to_numeric(frame["network_sent_mb"], errors="coerce").fillna(0.0)
        + pd.to_numeric(frame["network_recv_mb"], errors="coerce").fillna(0.0)
    )
    return frame[[
        "timestamp", "cpu_utilization", "ram_utilization", "disk_utilization",
        "network_sent_mb", "network_recv_mb", "network_activity_mb",
    ]]


def _load_health(node_id: str) -> pd.DataFrame:
    rows = _query_rows(
        """
        SELECT timestamp, anomaly_probability, health_status
        FROM live_health_predictions
        WHERE node_id = ?
        ORDER BY timestamp
        """,
        (node_id,),
    )
    return _timestamp_frame(rows)


def _load_security(node_id: str) -> pd.DataFrame:
    rows = _query_rows(
        """
        SELECT timestamp, classification, attack_type, security_risk,
               confidence, prediction_source, is_simulated, extra_json
        FROM node_security_predictions
        WHERE node_id = ?
        ORDER BY timestamp
        """,
        (node_id,),
    )
    frame = _timestamp_frame(rows)
    if frame.empty:
        return frame

    def extra_value(raw: Any, key: str) -> Any:
        try:
            return (json.loads(raw) if raw else {}).get(key)
        except (TypeError, json.JSONDecodeError):
            return None

    frame["security_risk_score"] = frame["extra_json"].map(
        lambda value: extra_value(value, "security_risk_score")
    )
    frame["event_label"] = frame.apply(
        lambda row: row.get("attack_type") or row.get("classification") or "Security event",
        axis=1,
    )
    frame["is_simulated"] = frame["is_simulated"].astype(bool)
    return frame[[
        "timestamp", "security_risk_score", "classification", "attack_type",
        "confidence", "prediction_source", "is_simulated", "event_label",
    ]]


def _asof_join(timeline: pd.DataFrame, source: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if source.empty:
        for column in columns:
            timeline[column] = np.nan
        return timeline
    right = source[["timestamp"] + columns].sort_values("timestamp")
    return pd.merge_asof(
        timeline.sort_values("timestamp"),
        right,
        on="timestamp",
        direction="backward",
    )


def _build_timeline(
    telemetry: pd.DataFrame,
    health: pd.DataFrame,
    security: pd.DataFrame,
) -> pd.DataFrame:
    frames = [frame[["timestamp"]] for frame in (telemetry, health, security) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["timestamp"])

    timeline = pd.concat(frames, ignore_index=True).drop_duplicates().sort_values("timestamp")
    timeline = timeline.reset_index(drop=True)
    timeline = _asof_join(timeline, telemetry, [
        "cpu_utilization", "ram_utilization", "disk_utilization",
        "network_sent_mb", "network_recv_mb", "network_activity_mb",
    ])
    timeline = _asof_join(timeline, health, ["anomaly_probability", "health_status"])
    timeline = _asof_join(timeline, security, ["security_risk_score"])

    exact_events = security.set_index("timestamp") if not security.empty else pd.DataFrame()
    timeline["security_event"] = timeline["timestamp"].isin(exact_events.index) if not security.empty else False
    timeline["security_event_label"] = timeline["timestamp"].map(
        exact_events["event_label"] if not security.empty else pd.Series(dtype=object)
    )
    timeline["security_event_source"] = timeline["timestamp"].map(
        exact_events["prediction_source"] if not security.empty else pd.Series(dtype=object)
    )
    timeline["security_event_simulated"] = timeline["timestamp"].map(
        exact_events["is_simulated"] if not security.empty else pd.Series(dtype=object)
    ).fillna(False).astype(bool)
    return timeline


def _add_overall_risk(node_id: str, timeline: pd.DataFrame) -> pd.DataFrame:
    if timeline.empty:
        timeline["overall_risk"] = pd.Series(dtype=float)
        return timeline

    overall = []
    for row in timeline.to_dict("records"):
        result = combine_node_risk(
            node_id,
            {"security_risk_score": row.get("security_risk_score"), "timestamp": row["timestamp"].isoformat()},
            {"anomaly_probability": row.get("anomaly_probability"), "timestamp": row["timestamp"].isoformat()},
            timestamp=row["timestamp"].isoformat(),
        )
        overall.append(result["overall_risk"])
    timeline["overall_risk"] = overall
    return timeline


def _correlation_result(frame: pd.DataFrame, metric: str, label: str) -> Dict[str, Any]:
    paired = frame[["security_risk", metric]].dropna()
    sample_size = len(paired)
    result = {
        "metric": metric,
        "label": label,
        "correlation": None,
        "p_value": None,
        "sample_size": sample_size,
        "statistically_meaningful": False,
        "status": "insufficient_samples" if sample_size < MIN_CORRELATION_SAMPLES else "not_testable",
        "explanation": "Not enough paired observations to assess an observed relationship.",
    }
    if sample_size < MIN_CORRELATION_SAMPLES or paired["security_risk"].nunique() < 2 or paired[metric].nunique() < 2:
        return result

    correlation, p_value = pearsonr(paired["security_risk"], paired[metric])
    result.update({
        "correlation": round(float(correlation), 4),
        "p_value": round(float(p_value), 4),
        "statistically_meaningful": bool(p_value < 0.05),
        "status": "meaningful" if p_value < 0.05 else "not_statistically_meaningful",
        "explanation": (
            f"Security risk was {'positively' if correlation > 0 else 'negatively'} correlated with {label}."
            if p_value < 0.05 else
            f"No statistically meaningful observed relationship with {label} was detected."
        ),
    })
    return result


def _event_windows(timeline: pd.DataFrame, security: pd.DataFrame, window_minutes: int) -> List[Dict[str, Any]]:
    if timeline.empty or security.empty:
        return []

    observations = []
    window = timedelta(minutes=window_minutes)
    for event in security.to_dict("records"):
        event_time = event["timestamp"]
        before = timeline[(timeline["timestamp"] < event_time) & (timeline["timestamp"] >= event_time - window)]
        after = timeline[(timeline["timestamp"] > event_time) & (timeline["timestamp"] <= event_time + window)]
        summary = {
            "timestamp": event_time.isoformat(),
            "event": event.get("event_label"),
            "classification": event.get("classification"),
            "attack_type": event.get("attack_type"),
            "prediction_source": event.get("prediction_source"),
            "is_simulated": bool(event.get("is_simulated")),
            "security_risk": event.get("security_risk_score"),
            "before_samples": len(before),
            "after_samples": len(after),
        }
        for metric in ("cpu_utilization", "ram_utilization", "network_activity_mb", "anomaly_probability", "overall_risk"):
            before_mean = before[metric].dropna().mean() if metric in before else np.nan
            after_mean = after[metric].dropna().mean() if metric in after else np.nan
            summary[f"before_{metric}"] = None if pd.isna(before_mean) else round(float(before_mean), 4)
            summary[f"after_{metric}"] = None if pd.isna(after_mean) else round(float(after_mean), 4)
            summary[f"delta_{metric}"] = (
                None if pd.isna(before_mean) or pd.isna(after_mean)
                else round(float(after_mean - before_mean), 4)
            )
        summary["observed_relationship"] = (
            "Resource observations available before and after this security event."
            if summary["before_samples"] and summary["after_samples"]
            else "Insufficient telemetry around this event for a before/after comparison."
        )
        observations.append(summary)
    return observations


def analyze_node(node_id: str, window_minutes: int = 5) -> Dict[str, Any]:
    """Return aligned time series, correlations, event timeline, and explanation."""
    telemetry = _load_telemetry(node_id)
    health = _load_health(node_id)
    security = _load_security(node_id)
    timeline = _add_overall_risk(node_id, _build_timeline(telemetry, health, security))
    if not timeline.empty:
        timeline["security_risk"] = timeline["security_risk_score"]
        timeline["maintenance_risk"] = timeline["anomaly_probability"]

    correlations = [
        _correlation_result(timeline, metric, label)
        for metric, label in CORRELATION_METRICS.items()
        if not timeline.empty
    ]
    return {
        "node_id": node_id,
        "aligned_data": timeline,
        "correlations": correlations,
        "event_timeline": _event_windows(timeline, security, window_minutes),
        "security_event_count": len(security),
        "explanation": (
            "These are observational associations between independently produced signals. "
            "Correlation does not establish that security activity causes infrastructure changes."
        ),
    }


def analyze_all_nodes(window_minutes: int = 5) -> List[Dict[str, Any]]:
    return [analyze_node(node["node_id"], window_minutes) for node in get_all_nodes()]
