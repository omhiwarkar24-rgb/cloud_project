"""
database.py - SQLite Database management for Central Telemetry Server.
Handles database schema initialization, node registration, telemetry storage, and node status tracking.
"""

import os
import json
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta

# Default database location inside data/ directory
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "telemetry.db"
)
DEFAULT_OFFLINE_THRESHOLD_SEC = 30


def get_db_path() -> str:
    db_path = os.getenv("TELEMETRY_DB_PATH", DEFAULT_DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return db_path


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def get_offline_threshold_sec() -> int:
    return int(os.getenv("NODE_OFFLINE_THRESHOLD_SEC", str(DEFAULT_OFFLINE_THRESHOLD_SEC)))


def init_db():
    """
    Initializes SQLite tables for registered nodes and time-series telemetry.
    """
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Nodes Registry Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                hostname TEXT NOT NULL,
                ip_address TEXT,
                operating_system TEXT,
                hardware_info TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ONLINE'
            )
        """)
        
        # Telemetry Time-Series Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                cpu_utilization REAL,
                ram_utilization REAL,
                disk_utilization REAL,
                network_sent_mb REAL,
                network_recv_mb REAL,
                disk_read_mb REAL,
                disk_write_mb REAL,
                cpu_temperature_c REAL,
                process_count INTEGER,
                raw_metrics TEXT,
                FOREIGN KEY (node_id) REFERENCES nodes(node_id)
            )
        """)
        
        # Index on (node_id, timestamp) for fast telemetry lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_telemetry_node_time 
            ON telemetry(node_id, timestamp)
        """)

        # Live infrastructure health predictions (separate from Alibaba maintenance model)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_health_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                health_status TEXT NOT NULL,
                anomaly_probability REAL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                contributing_features TEXT,
                feature_vector TEXT,
                node_status TEXT,
                FOREIGN KEY (node_id) REFERENCES nodes(node_id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_live_health_node_time
            ON live_health_predictions(node_id, timestamp)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS node_security_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                classification TEXT NOT NULL,
                attack_type TEXT,
                security_risk TEXT,
                confidence REAL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                prediction_source TEXT NOT NULL,
                is_simulated INTEGER NOT NULL DEFAULT 0,
                source_label TEXT,
                extra_json TEXT,
                FOREIGN KEY (node_id) REFERENCES nodes(node_id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_node_security_node_time
            ON node_security_predictions(node_id, timestamp)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_reference_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                feature TEXT NOT NULL,
                value REAL NOT NULL,
                FOREIGN KEY (node_id) REFERENCES nodes(node_id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_drift_reference_node_time
            ON drift_reference_samples(node_id, timestamp)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_drift_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                feature TEXT NOT NULL,
                drift_statistic REAL,
                p_value REAL,
                drift_status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                reference_window TEXT NOT NULL,
                current_window TEXT NOT NULL,
                FOREIGN KEY (node_id) REFERENCES nodes(node_id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_live_drift_node_feature
            ON live_drift_results(node_id, feature, timestamp)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS adaptive_model_state (
                target_model TEXT PRIMARY KEY,
                current_version TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS adaptive_retraining_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                target_model TEXT NOT NULL,
                previous_model_version TEXT NOT NULL,
                new_model_version TEXT NOT NULL,
                drift_event TEXT NOT NULL,
                affected_features TEXT NOT NULL,
                new_data_samples INTEGER NOT NULL,
                training_time_seconds REAL NOT NULL,
                validation_metrics TEXT NOT NULL,
                deployment_decision TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_adaptive_retraining_node_time
            ON adaptive_retraining_runs(node_id, target_model, timestamp)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS closed_loop_experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL UNIQUE,
                node_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                baseline_json TEXT NOT NULL,
                after_json TEXT NOT NULL,
                recommendation_json TEXT NOT NULL,
                intervention TEXT NOT NULL,
                measured_change_json TEXT NOT NULL,
                label TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_closed_loop_node_time
            ON closed_loop_experiments(node_id, timestamp)
        """)
        
        conn.commit()
    print(f"[CENTRAL DB]: Initialized SQLite database at '{db_path}'")


def update_node_statuses(offline_threshold_sec: Optional[int] = None):
    """
    Marks nodes as OFFLINE if their last_seen timestamp is older than offline_threshold_sec.
    """
    if offline_threshold_sec is None:
        offline_threshold_sec = get_offline_threshold_sec()
    now_utc = datetime.now(timezone.utc)
    threshold_time = (now_utc - timedelta(seconds=offline_threshold_sec)).isoformat()
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE nodes 
            SET status = 'OFFLINE'
            WHERE last_seen < ? AND status != 'OFFLINE'
        """, (threshold_time,))
        conn.commit()


def register_node(
    node_id: str,
    hostname: str,
    ip_address: Optional[str] = "127.0.0.1",
    operating_system: Optional[str] = "Unknown OS",
    hardware_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Registers a node or updates its metadata if already registered.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    hw_str = json.dumps(hardware_info or {})
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT node_id, first_seen FROM nodes WHERE node_id = ?", (node_id,))
        row = cursor.fetchone()
        
        if row:
            # Update existing node
            cursor.execute("""
                UPDATE nodes 
                SET hostname = ?, ip_address = ?, operating_system = ?, 
                    hardware_info = ?, last_seen = ?, status = 'ONLINE'
                WHERE node_id = ?
            """, (hostname, ip_address, operating_system, hw_str, now_iso, node_id))
            first_seen = row["first_seen"]
        else:
            # Insert new node
            first_seen = now_iso
            cursor.execute("""
                INSERT INTO nodes (node_id, hostname, ip_address, operating_system, hardware_info, first_seen, last_seen, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'ONLINE')
            """, (node_id, hostname, ip_address, operating_system, hw_str, first_seen, now_iso))
        
        conn.commit()
        
    return {
        "node_id": node_id,
        "hostname": hostname,
        "ip_address": ip_address,
        "operating_system": operating_system,
        "hardware_info": hardware_info or {},
        "first_seen": first_seen,
        "last_seen": now_iso,
        "status": "ONLINE"
    }


def record_telemetry(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stores telemetry payload and updates node last_seen timestamp & status.
    Auto-registers node if not seen before.
    """
    node_id = payload.get("node_id")
    if not node_id:
        raise ValueError("Payload missing required 'node_id'")
        
    hostname = payload.get("hostname", "Unknown-Host")
    operating_system = payload.get("operating_system", "Unknown OS")
    timestamp = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
    metrics = payload.get("metrics", {})
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Ensure node is registered
        cursor.execute("SELECT node_id FROM nodes WHERE node_id = ?", (node_id,))
        if not cursor.fetchone():
            register_node(
                node_id=node_id,
                hostname=hostname,
                operating_system=operating_system,
                hardware_info={
                    "cpu_count_logical": metrics.get("cpu_count_logical"),
                    "cpu_count_physical": metrics.get("cpu_count_physical"),
                    "ram_total_mb": metrics.get("ram_total_mb"),
                    "disk_total_gb": metrics.get("disk_total_gb")
                }
            )
        else:
            # Update last_seen and status
            cursor.execute("""
                UPDATE nodes 
                SET last_seen = ?, status = 'ONLINE'
                WHERE node_id = ?
            """, (timestamp, node_id))
            
        # Insert telemetry data
        cursor.execute("""
            INSERT INTO telemetry (
                node_id, timestamp, cpu_utilization, ram_utilization, disk_utilization,
                network_sent_mb, network_recv_mb, disk_read_mb, disk_write_mb,
                cpu_temperature_c, process_count, raw_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node_id,
            timestamp,
            metrics.get("cpu_utilization"),
            metrics.get("ram_utilization"),
            metrics.get("disk_utilization"),
            metrics.get("network_sent_mb"),
            metrics.get("network_recv_mb"),
            metrics.get("disk_read_mb"),
            metrics.get("disk_write_mb"),
            metrics.get("cpu_temperature_c"),
            metrics.get("process_count"),
            json.dumps(metrics)
        ))
        
        conn.commit()
        
    return {"status": "success", "node_id": node_id, "timestamp": timestamp}


def heartbeat_node(node_id: str) -> Dict[str, Any]:
    """
    Updates node's last_seen and status to ONLINE.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT node_id FROM nodes WHERE node_id = ?", (node_id,))
        if not cursor.fetchone():
            register_node(node_id=node_id, hostname=f"Host-{node_id[:8]}")
            
        cursor.execute("""
            UPDATE nodes 
            SET last_seen = ?, status = 'ONLINE'
            WHERE node_id = ?
        """, (now_iso, node_id))
        conn.commit()
        
    return {"status": "acknowledged", "node_id": node_id, "last_seen": now_iso}


def get_all_nodes() -> List[Dict[str, Any]]:
    """
    Returns list of all registered nodes with metadata and updated status.
    """
    update_node_statuses()
    health_by_node = {h["node_id"]: h for h in get_all_latest_live_health()}
    security_by_node = {s["node_id"]: s for s in get_all_latest_node_security()}
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM nodes ORDER BY last_seen DESC")
        rows = cursor.fetchall()
        
        nodes = []
        for r in rows:
            nid = r["node_id"]
            nodes.append({
                "node_id": nid,
                "hostname": r["hostname"],
                "ip_address": r["ip_address"],
                "operating_system": r["operating_system"],
                "hardware_info": json.loads(r["hardware_info"]) if r["hardware_info"] else {},
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
                "status": r["status"],
                "latest_live_health": health_by_node.get(nid),
                "latest_security": security_by_node.get(nid),
            })
        return nodes


def get_node_by_id(node_id: str, limit_telemetry: int = 50) -> Optional[Dict[str, Any]]:
    """
    Returns single node metadata, status, and recent telemetry history.
    """
    update_node_statuses()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,))
        row = cursor.fetchone()
        if not row:
            return None
            
        node_data = {
            "node_id": row["node_id"],
            "hostname": row["hostname"],
            "ip_address": row["ip_address"],
            "operating_system": row["operating_system"],
            "hardware_info": json.loads(row["hardware_info"]) if row["hardware_info"] else {},
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "status": row["status"],
            "recent_telemetry": []
        }
        
        cursor.execute("""
            SELECT * FROM telemetry 
            WHERE node_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (node_id, limit_telemetry))
        t_rows = cursor.fetchall()
        
        telemetry_list = []
        for t in t_rows:
            telemetry_list.append({
                "id": t["id"],
                "timestamp": t["timestamp"],
                "cpu_utilization": t["cpu_utilization"],
                "ram_utilization": t["ram_utilization"],
                "disk_utilization": t["disk_utilization"],
                "network_sent_mb": t["network_sent_mb"],
                "network_recv_mb": t["network_recv_mb"],
                "disk_read_mb": t["disk_read_mb"],
                "disk_write_mb": t["disk_write_mb"],
                "cpu_temperature_c": t["cpu_temperature_c"],
                "process_count": t["process_count"],
                "metrics": json.loads(t["raw_metrics"]) if t["raw_metrics"] else {}
            })
            
        node_data["recent_telemetry"] = telemetry_list
        node_data["latest_live_health"] = get_latest_live_health(node_id)
        node_data["latest_security"] = get_latest_node_security(node_id)
        return node_data


def get_latest_telemetry(node_id: str) -> Optional[Dict[str, Any]]:
    """Return the most recent telemetry row for a node, or None."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM telemetry
            WHERE node_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
        """, (node_id,))
        t = cursor.fetchone()
        if not t:
            return None
        return {
            "id": t["id"],
            "timestamp": t["timestamp"],
            "cpu_utilization": t["cpu_utilization"],
            "ram_utilization": t["ram_utilization"],
            "disk_utilization": t["disk_utilization"],
            "network_sent_mb": t["network_sent_mb"],
            "network_recv_mb": t["network_recv_mb"],
            "disk_read_mb": t["disk_read_mb"],
            "disk_write_mb": t["disk_write_mb"],
            "cpu_temperature_c": t["cpu_temperature_c"],
            "process_count": t["process_count"],
            "metrics": json.loads(t["raw_metrics"]) if t["raw_metrics"] else {}
        }


def save_live_health_prediction(result: Dict[str, Any]) -> None:
    """Persist one live-infrastructure health prediction."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO live_health_predictions (
                node_id, timestamp, health_status, anomaly_probability,
                model_name, model_version, contributing_features, feature_vector, node_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.get("node_id"),
            result.get("timestamp"),
            result.get("health_status"),
            result.get("anomaly_probability"),
            result.get("model_name"),
            result.get("model_version"),
            json.dumps(result.get("contributing_features") or []),
            json.dumps(result.get("feature_vector") or {}),
            result.get("node_status"),
        ))
        conn.commit()


def get_latest_live_health(node_id: str) -> Optional[Dict[str, Any]]:
    """Return the most recent live-health prediction for a node."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM live_health_predictions
            WHERE node_id = ?
            ORDER BY id DESC
            LIMIT 1
        """, (node_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return _row_to_live_health(row)


def get_all_latest_live_health() -> List[Dict[str, Any]]:
    """Return the latest live-health prediction for every node that has one."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.* FROM live_health_predictions p
            INNER JOIN (
                SELECT node_id, MAX(id) AS max_id
                FROM live_health_predictions
                GROUP BY node_id
            ) latest ON p.id = latest.max_id
        """)
        return [_row_to_live_health(r) for r in cursor.fetchall()]


def save_node_security_prediction(result: Dict[str, Any]) -> None:
    """Persist one per-node security result (genuine flow or demo sample)."""
    extra = {
        k: result.get(k)
        for k in (
            "predicted_attack_cat", "sample_attack_cat",
            "security_risk_score", "security_risk_reason", "model_path",
        )
    }
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO node_security_predictions (
                node_id, timestamp, classification, attack_type, security_risk,
                confidence, model_name, model_version, prediction_source,
                is_simulated, source_label, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.get("node_id"),
            result.get("timestamp"),
            result.get("classification"),
            result.get("attack_type"),
            result.get("security_risk"),
            result.get("confidence"),
            result.get("model_name"),
            result.get("model_version"),
            result.get("prediction_source"),
            1 if result.get("is_simulated") else 0,
            result.get("source_label"),
            json.dumps(extra),
        ))
        conn.commit()


def get_latest_node_security(node_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT * FROM node_security_predictions
                WHERE node_id = ?
                ORDER BY id DESC
                LIMIT 1
            """, (node_id,))
        except sqlite3.OperationalError:
            return None
        row = cursor.fetchone()
        if not row:
            return None
        return _row_to_node_security(row)


def get_all_latest_node_security() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT p.* FROM node_security_predictions p
                INNER JOIN (
                    SELECT node_id, MAX(id) AS max_id
                    FROM node_security_predictions
                    GROUP BY node_id
                ) latest ON p.id = latest.max_id
            """)
        except sqlite3.OperationalError:
            return []
        return [_row_to_node_security(r) for r in cursor.fetchall()]


def _row_to_node_security(row: sqlite3.Row) -> Dict[str, Any]:
    extra = json.loads(row["extra_json"]) if row["extra_json"] else {}
    result = {
        "node_id": row["node_id"],
        "classification": row["classification"],
        "attack_type": row["attack_type"],
        "security_risk": row["security_risk"],
        "confidence": row["confidence"],
        "timestamp": row["timestamp"],
        "model_name": row["model_name"],
        "model_version": row["model_version"],
        "prediction_source": row["prediction_source"],
        "is_simulated": bool(row["is_simulated"]),
        "source_label": row["source_label"],
    }
    result.update(extra)
    return result


def _row_to_live_health(row: sqlite3.Row) -> Dict[str, Any]:
    contrib = json.loads(row["contributing_features"]) if row["contributing_features"] else []
    fvec = json.loads(row["feature_vector"]) if row["feature_vector"] else {}
    return {
        "node_id": row["node_id"],
        "health_status": row["health_status"],
        "anomaly_probability": row["anomaly_probability"],
        "model_name": row["model_name"],
        "model_version": row["model_version"],
        "timestamp": row["timestamp"],
        "contributing_features": contrib,
        "feature_vector": fvec,
        "node_status": row["node_status"],
    }
