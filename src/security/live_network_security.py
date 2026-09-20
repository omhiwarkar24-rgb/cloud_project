"""
live_network_security.py

LIVE NODE SECURITY ADAPTER
==========================
Connects the existing UNSW-NB15 security classifier to registered nodes
ONLY when a valid network-flow record is supplied.

This module does NOT:
  * map CPU / RAM / Disk (or other host telemetry) into UNSW-NB15 features
  * modify security.predict or the trained xgboost_multiclass model
  * invent packet/flow statistics from node-agent counters

The live node agent currently reports host metrics (utilisation, process
count, cumulative byte counters). Those are not UNSW-NB15 flow features
(dur, proto, spkts, sttl, ct_* , ...). Genuine model inference therefore
requires an explicit network-flow payload.

Demo path: controlled samples copied from the UNSW-NB15 test set, clearly
labeled as simulated/demo events.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from security.predict import DEFAULT_MODEL_PATH, predict_attack
from risk_engine.security_risk import evaluate_security_risk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNSW_TEST_CSV = os.path.join(BASE_DIR, "data", "security", "UNSW_NB15_testing-set.csv")

MODEL_NAME = "xgboost_multiclass"
MODEL_VERSION = "1.0.0"

# Raw UNSW-NB15 columns required by the existing preprocessing pipeline
# (everything except identifiers and labels).
UNSW_RAW_FEATURES = [
    "dur", "proto", "service", "state",
    "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean",
    "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]

CORE_FLOW_FIELDS = [
    "dur", "proto", "service", "state",
    "spkts", "dpkts", "sbytes", "dbytes", "rate",
]

HOST_TELEMETRY_KEYS = {
    "cpu_utilization", "ram_utilization", "disk_utilization",
    "cpu_count_logical", "cpu_count_physical", "cpu_temperature_c",
    "ram_used_mb", "ram_total_mb", "disk_used_gb", "disk_total_gb",
    "network_sent_bytes", "network_recv_bytes", "network_sent_mb", "network_recv_mb",
    "disk_read_bytes", "disk_write_bytes", "disk_read_mb", "disk_write_mb",
    "process_count",
    "CPU_utilization", "Memory_utilization", "Disk_utilization",
    "Load_1min", "Load_5min", "Load_15min", "Duration_sec", "Server_Type",
}

SOURCE_GENUINE = "GENUINE_NETWORK_FLOW"
SOURCE_DEMO = "DEMO_SAMPLE"
SOURCE_NONE = "NONE"

LABEL_GENUINE = "Genuine network-flow prediction (UNSW-NB15 features)"
LABEL_DEMO = "Simulated/demo security event (UNSW-NB15 test-set sample)"
LABEL_NONE = "No network-flow record. Host telemetry is not used for security inference."

_DEMO_CACHE: Optional[pd.DataFrame] = None


class InvalidNetworkFlowError(ValueError):
    """Raised when a payload is not a valid UNSW-NB15 network-flow record."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_host_telemetry_payload(payload: Dict[str, Any]) -> bool:
    """True when the dict looks like node-agent host metrics, not a flow record."""
    if not isinstance(payload, dict) or not payload:
        return False
    keys = set(payload.keys())
    host_hits = len(keys & HOST_TELEMETRY_KEYS)
    has_core_flow = "proto" in keys and "dur" in keys
    return host_hits >= 2 and not has_core_flow


def extract_unsw_flow_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only UNSW-NB15 raw feature columns. Host metrics are dropped, never mapped.
    """
    flow = {}
    for col in UNSW_RAW_FEATURES:
        if col in raw and raw[col] is not None:
            flow[col] = raw[col]
    return flow


def validate_network_flow(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept only explicit network-flow dictionaries.

    Rejects host telemetry. Requires CORE_FLOW_FIELDS. Extra UNSW columns are
    forwarded; missing non-core columns are left absent (the existing predictor
    may default them — they are never filled from CPU/RAM/Disk).
    """
    if not isinstance(raw, dict):
        raise InvalidNetworkFlowError("Network-flow payload must be a JSON object.")

    # Nested metrics from a telemetry POST must not be treated as a flow.
    candidate = raw.get("flow") if isinstance(raw.get("flow"), dict) else raw
    if "metrics" in candidate and is_host_telemetry_payload(candidate.get("metrics") or {}):
        raise InvalidNetworkFlowError(
            "Host telemetry (CPU/RAM/Disk) cannot be converted into UNSW-NB15 "
            "network-flow features. Submit a real flow record via POST /network-flow."
        )

    if is_host_telemetry_payload(candidate):
        raise InvalidNetworkFlowError(
            "Rejected host telemetry payload. The security model requires UNSW-NB15 "
            "network-flow fields (dur, proto, spkts, sttl, ct_*, ...), which the "
            "node agent does not collect."
        )

    missing_core = [c for c in CORE_FLOW_FIELDS if c not in candidate or candidate[c] in (None, "")]
    if missing_core:
        raise InvalidNetworkFlowError(
            f"Invalid network-flow record; missing required fields: {missing_core}. "
            "Do not substitute host metrics."
        )

    flow = extract_unsw_flow_features(candidate)
    if "proto" not in flow or "dur" not in flow:
        raise InvalidNetworkFlowError("Network-flow record must include at least proto and dur.")
    return flow


def _classification_from_attack_cat(attack_cat: str) -> Tuple[str, Optional[str]]:
    label = str(attack_cat).strip()
    if label.lower() in ("normal", "benign"):
        return "NORMAL", None
    return "ATTACK", label


def _empty_result(node_id: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
    return {
        "node_id": node_id,
        "classification": "NO_FLOW_DATA",
        "attack_type": None,
        "security_risk": "N/A",
        "confidence": None,
        "timestamp": timestamp or _now_iso(),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_path": os.path.basename(DEFAULT_MODEL_PATH),
        "prediction_source": SOURCE_NONE,
        "is_simulated": False,
        "source_label": LABEL_NONE,
    }


def _build_result(
    node_id: str,
    pred: Dict[str, Any],
    *,
    timestamp: str,
    prediction_source: str,
    is_simulated: bool,
    source_label: str,
    sample_attack_cat: Optional[str] = None,
) -> Dict[str, Any]:
    attack_cat = str(pred.get("prediction", "Normal"))
    classification, attack_type = _classification_from_attack_cat(attack_cat)
    risk = evaluate_security_risk(attack_cat)
    return {
        "node_id": node_id,
        "classification": classification,
        "attack_type": attack_type,
        "security_risk": risk["security_risk_level"],
        "confidence": round(float(pred.get("probability", 0.0)), 4),
        "timestamp": timestamp,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_path": os.path.basename(DEFAULT_MODEL_PATH),
        "prediction_source": prediction_source,
        "is_simulated": is_simulated,
        "source_label": source_label,
        "predicted_attack_cat": attack_cat,
        "sample_attack_cat": sample_attack_cat,
        "security_risk_score": risk["security_risk_score"],
        "security_risk_reason": risk["reason"],
    }


def score_network_flow(
    node_id: str,
    flow: Dict[str, Any],
    *,
    timestamp: Optional[str] = None,
    prediction_source: str = SOURCE_GENUINE,
    is_simulated: bool = False,
    source_label: str = LABEL_GENUINE,
    sample_attack_cat: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the unchanged UNSW-NB15 model on a validated flow record."""
    validated = validate_network_flow(flow)
    pred = predict_attack(validated)
    return _build_result(
        node_id,
        pred,
        timestamp=timestamp or _now_iso(),
        prediction_source=prediction_source,
        is_simulated=is_simulated,
        source_label=source_label,
        sample_attack_cat=sample_attack_cat,
    )


def _load_demo_table() -> pd.DataFrame:
    global _DEMO_CACHE
    if _DEMO_CACHE is not None:
        return _DEMO_CACHE
    if not os.path.exists(UNSW_TEST_CSV):
        raise FileNotFoundError(f"UNSW-NB15 test set not found at {UNSW_TEST_CSV}")
    df = pd.read_csv(UNSW_TEST_CSV)
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    _DEMO_CACHE = df
    return df


def list_demo_attack_categories() -> List[str]:
    df = _load_demo_table()
    return sorted(df["attack_cat"].unique().tolist())


def load_demo_flow(attack_category: Optional[str] = "Normal") -> Tuple[Dict[str, Any], str]:
    """Return one complete UNSW-NB15 test-set row (features only) plus its catalog label."""
    df = _load_demo_table()
    wanted = (attack_category or "Normal").strip()
    subset = df[df["attack_cat"].str.lower() == wanted.lower()]
    if subset.empty:
        available = ", ".join(list_demo_attack_categories())
        raise InvalidNetworkFlowError(
            f"Unknown demo attack category '{wanted}'. Available: {available}"
        )
    row = subset.iloc[0].to_dict()
    catalog_label = str(row.get("attack_cat", wanted))
    flow = extract_unsw_flow_features(row)
    return flow, catalog_label


def ingest_genuine_flow(
    node_id: str,
    flow: Dict[str, Any],
    timestamp: Optional[str] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    result = score_network_flow(
        node_id,
        flow,
        timestamp=timestamp,
        prediction_source=SOURCE_GENUINE,
        is_simulated=False,
        source_label=LABEL_GENUINE,
    )
    if persist:
        from central_server.database import save_node_security_prediction
        save_node_security_prediction(result)
    return result


def ingest_demo_event(
    node_id: str,
    attack_category: Optional[str] = "Normal",
    persist: bool = True,
) -> Dict[str, Any]:
    """Attach a labeled UNSW-NB15 test sample to a node (demo only)."""
    flow, catalog = load_demo_flow(attack_category)
    result = score_network_flow(
        node_id,
        flow,
        prediction_source=SOURCE_DEMO,
        is_simulated=True,
        source_label=LABEL_DEMO,
        sample_attack_cat=catalog,
    )
    if persist:
        from central_server.database import save_node_security_prediction
        save_node_security_prediction(result)
    return result


def security_status_for_node(node_id: str) -> Dict[str, Any]:
    from central_server.database import get_latest_node_security
    stored = get_latest_node_security(node_id)
    if stored:
        return stored
    return _empty_result(node_id)


def security_status_for_all_nodes() -> List[Dict[str, Any]]:
    from central_server.database import get_all_nodes, get_all_latest_node_security
    stored = {s["node_id"]: s for s in get_all_latest_node_security()}
    out = []
    for node in get_all_nodes():
        nid = node["node_id"]
        out.append(stored.get(nid) or _empty_result(nid, timestamp=node.get("last_seen")))
    return out
