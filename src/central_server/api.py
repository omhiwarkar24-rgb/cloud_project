"""
api.py - REST API endpoints for the Central Telemetry Server using FastAPI.
Exposes endpoints for node registration, telemetry ingestion, heartbeats, and node listing.
"""

from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from central_server.database import (
    init_db,
    register_node,
    record_telemetry,
    heartbeat_node,
    get_all_nodes,
    get_node_by_id,
    get_latest_live_health,
    get_latest_telemetry,
    get_latest_node_security,
)

app = FastAPI(
    title="Central Telemetry API",
    description="Lightweight control server REST API for real-time node monitoring and telemetry collection.",
    version="1.0.0"
)


# Pydantic Request Models
class NodeRegisterRequest(BaseModel):
    node_id: str = Field(..., description="Unique persistent identifier for the node")
    hostname: str = Field(..., description="Hostname of the node")
    ip_address: Optional[str] = Field(None, description="IP address of the node")
    operating_system: Optional[str] = Field(None, description="OS details")
    hardware_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Hardware specs")


class DemoSecurityEventRequest(BaseModel):
    node_id: str = Field(..., description="Registered node to attach the demo event to")
    attack_category: Optional[str] = Field(
        "Normal",
        description="UNSW-NB15 attack_cat sample to load (e.g. Normal, DoS, Exploits)",
    )


class NetworkFlowRequest(BaseModel):
    node_id: str = Field(..., description="Registered node that observed this flow")
    timestamp: Optional[str] = Field(None, description="Flow timestamp (ISO-8601)")
    flow: Dict[str, Any] = Field(..., description="UNSW-NB15 raw network-flow feature object")


class NodeHeartbeatRequest(BaseModel):
    node_id: str = Field(..., description="Unique persistent identifier for the node")


class ClosedLoopExperimentRequest(BaseModel):
    node_id: str = Field(..., description="Registered node used for the isolated test evaluation")
    scale_factor: Optional[float] = Field(None, ge=0.1, le=0.9, description="Safe simulated workload scale factor")


@app.on_event("startup")
def startup_event():
    """Ensure database schema is created on server start."""
    init_db()


@app.get("/")
def root():
    return {
        "service": "Central Telemetry API",
        "status": "online",
        "endpoints": [
            "/register", "/telemetry", "/nodes", "/nodes/{node_id}",
            "/heartbeat", "/live-health", "/live-health/{node_id}",
            "/network-flow", "/security/demo-event",
            "/security/nodes", "/security/nodes/{node_id}",
            "/risk/nodes", "/risk/nodes/{node_id}",
            "/live-drift", "/live-drift/{node_id}",
            "/adaptive/retraining",
            "/prevention/nodes",
            "/experiments/closed-loop", "/experiments/closed-loop/{node_id}",
            "/alerts", "/alerts/active", "/alerts/{alert_id}/acknowledge",
        ]
    }


@app.post("/register", summary="Register or update a live node")
def register(req: NodeRegisterRequest, request: Request):
    """
    Registers a new node or updates an existing node's hardware & OS metadata.
    """
    client_ip = req.ip_address or request.client.host if request.client else "127.0.0.1"
    result = register_node(
        node_id=req.node_id,
        hostname=req.hostname,
        ip_address=client_ip,
        operating_system=req.operating_system,
        hardware_info=req.hardware_info
    )
    return JSONResponse(status_code=200, content=result)


@app.post("/telemetry", summary="Ingest telemetry metric payload from a node")
@app.post("/api/telemetry", summary="Ingest telemetry metric payload (alias)")
async def ingest_telemetry(request: Request):
    """
    Ingests timestamped system telemetry (CPU, RAM, Disk, Network, Disk I/O, Temp, etc.).
    Automatically registers new nodes upon first telemetry reception.
    """
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")
        
    if not isinstance(payload, dict) or "node_id" not in payload:
        raise HTTPException(status_code=422, detail="Payload must be a JSON object containing 'node_id'")
        
    result = record_telemetry(payload)
    maint_input = payload.get("metrics") if (payload.get("metrics") and isinstance(payload.get("metrics"), dict)) else payload

    # Real ML Predictive-Maintenance Evaluation & Alerts
    try:
        from maintenance.predict import predict_failure
        from maintenance.alerts import evaluate_node_for_alert

        maint_pred = predict_failure(maint_input)
        result["predictive_maintenance"] = {
            "prediction": maint_pred.get("prediction"),
            "failure_probability": maint_pred.get("failure_probability"),
            "risk_level": maint_pred.get("risk_level"),
            "timestamp": maint_pred.get("timestamp"),
            "main_contributing_factors": maint_pred.get("main_contributing_factors", []),
        }

        # Real ML predictive-maintenance alert with deduplication & cooldown
        alert = evaluate_node_for_alert(
            node_id=payload["node_id"],
            telemetry=maint_input,
            maint_result=maint_pred,
        )
        if alert:
            result["alert"] = alert
    except Exception as maint_err:
        result["maintenance_error"] = str(maint_err)

    # Live Infrastructure Health (Isolation Forest baseline preserved)
    try:
        from maintenance.live_infrastructure_health import evaluate_and_store_live_health
        health = evaluate_and_store_live_health(payload, node_status="ONLINE")
        result["live_health"] = {
            "health_status": health.get("health_status"),
            "anomaly_probability": health.get("anomaly_probability"),
            "failure_probability": health.get("failure_probability"),
            "model_name": health.get("model_name"),
            "model_version": health.get("model_version"),
        }
    except Exception as e:
        result["live_health_error"] = str(e)
    try:
        from adaptive.live_drift import detect_live_drift
        drift = detect_live_drift(payload["node_id"])
        result["live_drift"] = {
            "status": drift.get("status"),
            "affected_features": drift.get("affected_features", []),
            "timestamp": drift.get("timestamp"),
        }
        if drift.get("status") == "DRIFT DETECTED":
            from adaptive.retraining_controller import process_drift_event
            retraining = process_drift_event(drift, compatible_training_data=False)
            result["retraining"] = {
                "status": retraining["status"],
                "deployment_decision": retraining["deployment_decision"],
                "reason": retraining["reason"],
            }
    except Exception as e:
        result["live_drift_error"] = str(e)
    return JSONResponse(status_code=200, content=result)


@app.post("/heartbeat", summary="Process node heartbeat check-in")
def heartbeat(req: NodeHeartbeatRequest):
    """
    Updates a node's last_seen timestamp and status to ONLINE.
    """
    result = heartbeat_node(req.node_id)
    return JSONResponse(status_code=200, content=result)


@app.get("/nodes", summary="List all registered nodes and their status")
def list_nodes():
    """
    Returns list of all nodes in the registry, their metadata, last_seen, and status (ONLINE/OFFLINE).
    """
    nodes = get_all_nodes()
    return JSONResponse(status_code=200, content={"total_nodes": len(nodes), "nodes": nodes})


@app.get("/nodes/{node_id}", summary="Get details and telemetry history for a node")
def get_node_details(node_id: str, limit: int = Query(default=50, ge=1, le=500)):
    """
    Retrieves metadata, online status, and recent telemetry history for a specific node_id.
    """
    node_data = get_node_by_id(node_id=node_id, limit_telemetry=limit)
    if not node_data:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found in registry.")
    return JSONResponse(status_code=200, content=node_data)


@app.get("/live-health", summary="Live infrastructure health for all registered nodes")
def list_live_health():
    """
    Scores every registered node with the LIVE infrastructure Isolation Forest.
    This is not the historical Alibaba predictive-maintenance model.
    """
    from maintenance.live_infrastructure_health import (
        MODEL_NAME,
        MODEL_VERSION,
        predict_all_registered_nodes,
    )
    nodes = predict_all_registered_nodes()
    return JSONResponse(status_code=200, content={
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "total_nodes": len(nodes),
        "nodes": nodes,
    })


@app.get("/live-health/{node_id}", summary="Live infrastructure health for one node")
def get_node_live_health(node_id: str):
    """Return stored live-health score, recomputing from latest telemetry if needed."""
    node_data = get_node_by_id(node_id=node_id, limit_telemetry=1)
    if not node_data:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found in registry.")

    from maintenance.live_infrastructure_health import evaluate_and_store_live_health, predict_live_health

    latest = get_latest_telemetry(node_id)
    if latest:
        payload = {
            "node_id": node_id,
            "timestamp": latest.get("timestamp"),
            "metrics": latest.get("metrics") or {},
        }
        result = evaluate_and_store_live_health(payload, node_status=node_data.get("status"))
        return JSONResponse(status_code=200, content=result)

    stored = get_latest_live_health(node_id)
    if stored:
        return JSONResponse(status_code=200, content=stored)

    result = predict_live_health(
        None,
        node_id=node_id,
        timestamp=node_data.get("last_seen"),
        node_status=node_data.get("status"),
    )
    return JSONResponse(status_code=200, content=result)


@app.post("/network-flow", summary="Score a genuine UNSW-NB15 network-flow record for a node")
def ingest_network_flow(req: NetworkFlowRequest):
    """
    Accepts a real network-flow feature object (UNSW-NB15 schema).
    Host telemetry (CPU/RAM/Disk) is rejected — it is not a valid mapping.
    """
    node = get_node_by_id(req.node_id, limit_telemetry=1)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{req.node_id}' is not registered.")

    from security.live_network_security import InvalidNetworkFlowError, ingest_genuine_flow
    try:
        result = ingest_genuine_flow(req.node_id, req.flow, timestamp=req.timestamp, persist=True)
    except InvalidNetworkFlowError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Security inference failed: {e}")
    return JSONResponse(status_code=200, content=result)


@app.post("/security/demo-event", summary="Attach a labeled UNSW-NB15 demo sample to a node")
def ingest_security_demo(req: DemoSecurityEventRequest):
    """
    Controlled/demo path only. Loads a complete test-set flow for the requested
    attack category and scores it with the existing security model.
    The stored result is labeled is_simulated=True.
    """
    node = get_node_by_id(req.node_id, limit_telemetry=1)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{req.node_id}' is not registered.")

    from security.live_network_security import InvalidNetworkFlowError, ingest_demo_event
    try:
        result = ingest_demo_event(req.node_id, req.attack_category, persist=True)
    except InvalidNetworkFlowError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Demo security event failed: {e}")
    return JSONResponse(status_code=200, content=result)


@app.get("/security/nodes", summary="Latest security result for every registered node")
def list_node_security():
    from security.live_network_security import MODEL_NAME, MODEL_VERSION, security_status_for_all_nodes
    nodes = security_status_for_all_nodes()
    return JSONResponse(status_code=200, content={
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "total_nodes": len(nodes),
        "nodes": nodes,
        "limitation": (
            "The node agent does not collect UNSW-NB15 network-flow features. "
            "Results are NO_FLOW_DATA unless a genuine /network-flow payload or "
            "a labeled /security/demo-event was submitted."
        ),
    })


@app.get("/security/nodes/{node_id}", summary="Latest security result for one node")
def get_node_security(node_id: str):
    node = get_node_by_id(node_id, limit_telemetry=1)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' is not registered.")
    from security.live_network_security import security_status_for_node
    stored = get_latest_node_security(node_id)
    return JSONResponse(status_code=200, content=stored or security_status_for_node(node_id))


@app.get("/risk/nodes", summary="Unified risk result for every registered node")
def list_node_risk():
    from risk_engine.node_risk import get_all_node_risks, load_node_risk_config

    nodes = get_all_node_risks()
    return JSONResponse(status_code=200, content={
        "total_nodes": len(nodes),
        "config": load_node_risk_config(),
        "nodes": nodes,
        "causation_note": (
            "Security risk and maintenance risk are independent signals. "
            "This score does not claim that security activity causes infrastructure failure."
        ),
    })


@app.get("/risk/nodes/{node_id}", summary="Unified risk result for one registered node")
def get_node_risk(node_id: str):
    node = get_node_by_id(node_id, limit_telemetry=1)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found in registry.")
    from risk_engine.node_risk import get_node_risk as build_node_risk
    return JSONResponse(status_code=200, content=build_node_risk(node_id))


@app.get("/live-drift", summary="Current KS drift status for all registered nodes")
def list_live_drift():
    from adaptive.live_drift import detect_all_live_drift
    results = detect_all_live_drift()
    return JSONResponse(status_code=200, content={
        "total_nodes": len(results),
        "nodes": results,
        "retraining": "guarded",
    })


@app.get("/live-drift/{node_id}", summary="Current KS drift status for one node")
def get_node_live_drift(node_id: str):
    node = get_node_by_id(node_id, limit_telemetry=1)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found in registry.")
    from adaptive.live_drift import detect_live_drift
    return JSONResponse(status_code=200, content=detect_live_drift(node_id))


@app.get("/adaptive/retraining", summary="Latest guarded adaptive retraining status")
def adaptive_retraining_status():
    from adaptive.retraining_controller import get_latest_retraining_status
    return JSONResponse(status_code=200, content={
        "retraining": get_latest_retraining_status(),
        "automatic_retraining": "guarded",
    })


@app.get("/prevention/nodes", summary="Evidence-based advisory recommendations for high-risk nodes")
def list_node_prevention_recommendations():
    from prevention.prevention_engine import get_all_node_recommendations
    return JSONResponse(status_code=200, content={
        "recommendations": get_all_node_recommendations(),
        "automatic_actions": "disabled",
        "disclaimer": "Recommendations are advisory only; no arbitrary infrastructure repair or isolation is performed.",
    })


@app.post("/experiments/closed-loop", summary="Run a controlled closed-loop risk evaluation")
def run_closed_loop_experiment(req: ClosedLoopExperimentRequest):
    node = get_node_by_id(req.node_id, limit_telemetry=1)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{req.node_id}' not found in registry.")
    from experiments.closed_loop_evaluation import run_closed_loop_evaluation
    try:
        result = run_closed_loop_evaluation(req.node_id, req.scale_factor)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return JSONResponse(status_code=200, content=result)


@app.get("/experiments/closed-loop/{node_id}", summary="Get the latest closed-loop risk evaluation")
def get_closed_loop_experiment(node_id: str):
    node = get_node_by_id(node_id, limit_telemetry=1)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found in registry.")
    from experiments.closed_loop_evaluation import get_latest_closed_loop_evaluation
    result = get_latest_closed_loop_evaluation(node_id)
    return JSONResponse(status_code=200, content=result or {"node_id": node_id, "result": None})


# ─────────────────────────────────────────────
# Predictive Maintenance Alert Endpoints
# ─────────────────────────────────────────────
@app.get("/alerts", summary="List predictive maintenance alerts (active and historical)")
def list_alerts(
    node_id: Optional[str] = Query(None, description="Filter by node ID"),
    limit: int = Query(50, ge=1, le=200, description="Max alerts to return")
):
    from maintenance.alerts import fetch_alert_history, fetch_active_alerts
    active = fetch_active_alerts()
    history = fetch_alert_history(limit=limit, node_id=node_id)
    return JSONResponse(status_code=200, content={
        "active_count": len(active),
        "alerts": history,
        "active_alerts": active,
        "history": history
    })


@app.get("/alerts/active", summary="List all currently active predictive maintenance alerts")
def list_active_alerts(node_id: Optional[str] = Query(None, description="Filter by node ID")):
    from maintenance.alerts import fetch_active_alerts
    active = fetch_active_alerts()
    if node_id:
        active = [a for a in active if a.get("node_id") == node_id]
    return JSONResponse(status_code=200, content={
        "active_count": len(active),
        "alerts": active,
        "active_alerts": active
    })


@app.get("/nodes/{node_id}/alerts", summary="Get active predictive maintenance alert for a specific node")
@app.get("/nodes/{node_id}/alerts/active", summary="Get active predictive maintenance alert for a specific node (alias)")
def get_node_alerts(node_id: str):
    from central_server.database import get_latest_active_alert_for_node
    alert = get_latest_active_alert_for_node(node_id)
    return JSONResponse(status_code=200, content={
        "node_id": node_id,
        "has_active_alert": alert is not None,
        "alert": alert
    })


@app.post("/alerts/{alert_id}/acknowledge", summary="Acknowledge an active alert")
def acknowledge_alert(alert_id: str):
    from maintenance.alerts import acknowledge_alert_by_id
    success = acknowledge_alert_by_id(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found or already acknowledged.")
    return JSONResponse(status_code=200, content={"status": "success", "alert_id": alert_id, "state": "ACKNOWLEDGED"})


@app.post("/alerts/{alert_id}/resolve", summary="Resolve an alert")
def resolve_alert(alert_id: str):
    from maintenance.alerts import resolve_alert_by_id
    success = resolve_alert_by_id(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return JSONResponse(status_code=200, content={"status": "success", "alert_id": alert_id, "state": "RESOLVED"})

