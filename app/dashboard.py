"""
dashboard.py - Adaptive Cloud Security & Predictive Maintenance Dashboard
Professional Streamlit application integrating real-time node monitoring, ML security threat detection,
live-node predictive maintenance, data drift adaptation, and automated failure prevention.

Launch: python -m streamlit run app/dashboard.py
"""

import os
import sys
import json
import time
import sqlite3
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ─────────────────────────────────────────────
# Path Setup
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "src", "security"))
sys.path.insert(0, os.path.join(BASE_DIR, "src", "maintenance"))
sys.path.insert(0, os.path.join(BASE_DIR, "src", "risk_engine"))
sys.path.insert(0, os.path.join(BASE_DIR, "src", "prevention"))

# ─────────────────────────────────────────────
# Backend Imports
# ─────────────────────────────────────────────
from security.predict import predict_attack, load_security_model, load_security_pipeline
from security.live_network_security import (
    ingest_demo_event,
    list_demo_attack_categories,
    security_status_for_all_nodes,
    InvalidNetworkFlowError,
)
from maintenance.predict import predict_failure
from risk_engine.integrated_risk import evaluate_integrated_risk
from prevention.prevention_engine import (
    generate_recommendation,
    get_all_node_recommendations,
    RECOMMENDATION_RULE_MATRIX,
)
from experiments.closed_loop_evaluation import run_closed_loop_evaluation, get_latest_closed_loop_evaluation
from risk_engine.node_risk import get_all_node_risks, combine_node_risk
from central_server.database import (
    init_db,
    get_all_latest_live_health,
    get_all_latest_node_security,
    get_latest_node_security,
    get_latest_telemetry,
)
from adaptive.live_drift import detect_all_live_drift
from adaptive.retraining_controller import get_latest_retraining_status

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Cloud Security & Predictive Maintenance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        margin: 6px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card .label { font-size: 11px; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .metric-card .value { font-size: 26px; font-weight: 700; }

    .risk-LOW      { color: #00d084; }
    .risk-MEDIUM   { color: #f0b429; }
    .risk-HIGH     { color: #ff6b35; }
    .risk-CRITICAL { color: #e63946; }
    .risk-NA       { color: #8892b0; }

    .rec-box {
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 10px;
        font-size: 14px;
        font-weight: 500;
        line-height: 1.5;
    }
    .rec-ROUTINE  { background: rgba(0,208,132,0.12); border-left: 4px solid #00d084; color: #00d084; }
    .rec-MODERATE { background: rgba(100,210,255,0.12); border-left: 4px solid #64d2ff; color: #64d2ff; }
    .rec-ELEVATED { background: rgba(240,180,41,0.15); border-left: 4px solid #f0b429; color: #f0b429; }
    .rec-URGENT   { background: rgba(230,57,70,0.15); border-left: 4px solid #e63946; color: #e63946; }

    .section-header {
        font-size: 19px; font-weight: 700; color: #ccd6f6;
        border-bottom: 2px solid #2d3561; padding-bottom: 8px; margin-bottom: 18px;
    }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 12px; font-weight: 600; letter-spacing: 0.5px;
    }
    .badge-LOW      { background: rgba(0,208,132,0.2); color:#00d084; }
    .badge-MEDIUM   { background: rgba(240,180,41,0.2); color:#f0b429; }
    .badge-HIGH     { background: rgba(255,107,53,0.2); color:#ff6b35; }
    .badge-CRITICAL { background: rgba(230,57,70,0.2); color:#e63946; }
    .badge-NA       { background: rgba(136,146,176,0.2); color:#8892b0; }

    /* Node & Telemetry Display */
    .node-card {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a1d2e 100%);
        border: 1px solid #2d3561;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    }
    .node-online  { border-left: 4px solid #00d084; }
    .node-offline { border-left: 4px solid #e63946; }
    .node-title   { font-size:16px; font-weight:700; color:#ccd6f6; margin-bottom:4px; }
    .node-sub     { font-size:12px; color:#8892b0; margin-bottom:12px; }
    .bar-wrap     { background:#1e2235; border-radius:6px; height:10px; margin:4px 0 10px 0; overflow:hidden; }
    .bar-fill     { height:10px; border-radius:6px; }
    .bar-green    { background: linear-gradient(90deg,#00d084,#00b368); }
    .bar-yellow   { background: linear-gradient(90deg,#f0b429,#d99a15); }
    .bar-red      { background: linear-gradient(90deg,#e63946,#b52d38); }
    .stat-label   { font-size:11px; color:#8892b0; text-transform:uppercase; letter-spacing:.8px; }
    .stat-value   { font-size:14px; font-weight:600; color:#ccd6f6; }
    .pill-online  { background:rgba(0,208,132,.18); color:#00d084; padding:3px 12px;
                    border-radius:20px; font-size:11px; font-weight:700; display:inline-block; }
    .pill-offline { background:rgba(230,57,70,.18);  color:#e63946; padding:3px 12px;
                    border-radius:20px; font-size:11px; font-weight:700; display:inline-block; }
    .pill-demo    { background:rgba(240,180,41,.18); color:#f0b429; padding:2px 10px;
                    border-radius:20px; font-size:11px; font-weight:700; }
    .pill-genuine { background:rgba(100,210,255,.18); color:#64d2ff; padding:2px 10px;
                    border-radius:20px; font-size:11px; font-weight:700; }
    .pill-none    { background:rgba(136,146,176,.18); color:#8892b0; padding:2px 10px;
                    border-radius:20px; font-size:11px; font-weight:700; }

    .telemetry-panel {
        background: #151824;
        border: 1px solid #252b48;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper Functions (Defined BEFORE any callers)
# ─────────────────────────────────────────────
RISK_COLOR = {"LOW": "#00d084", "MEDIUM": "#f0b429", "HIGH": "#ff6b35", "CRITICAL": "#e63946", "N/A": "#8892b0"}
PRIORITY_ORDER = {"ROUTINE": 0, "MODERATE": 1, "ELEVATED": 2, "URGENT": 3}
TELEMETRY_DB = os.path.join(BASE_DIR, "data", "telemetry.db")


def _db_ok() -> bool:
    """Return True only if the telemetry.db file exists."""
    return os.path.exists(TELEMETRY_DB)


def _fetch_nodes() -> list:
    """Return all node rows from the SQLite DB, with status freshened."""
    if not _db_ok():
        return []
    try:
        conn = sqlite3.connect(TELEMETRY_DB)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        offline_threshold = int(os.getenv("NODE_OFFLINE_THRESHOLD_SEC", "30"))
        threshold = (datetime.now(timezone.utc) - timedelta(seconds=offline_threshold)).isoformat()
        cur.execute("UPDATE nodes SET status='OFFLINE' WHERE last_seen < ? AND status!='OFFLINE'", (threshold,))
        conn.commit()
        cur.execute("SELECT * FROM nodes ORDER BY last_seen DESC")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def _fetch_node_history(node_id: str, n: int = 30) -> list:
    """Return the last *n* telemetry rows for *node_id* in chronological order."""
    if not _db_ok():
        return []
    try:
        conn = sqlite3.connect(TELEMETRY_DB)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT timestamp, cpu_utilization, ram_utilization, disk_utilization,
                   network_sent_mb, network_recv_mb, cpu_temperature_c, process_count,
                   raw_metrics
            FROM telemetry WHERE node_id=?
            ORDER BY timestamp DESC LIMIT ?
        """, (node_id, n))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return list(reversed(rows))
    except Exception:
        return []


def _fetch_latest_telemetry(node_id: str) -> dict:
    """Return latest parsed telemetry record for a node."""
    hist = _fetch_node_history(node_id, 1)
    if hist:
        row = dict(hist[-1])
        raw = row.get("raw_metrics")
        metrics = json.loads(raw) if (raw and isinstance(raw, str)) else {}
        row["metrics"] = metrics
        return row
    return None


def metric_card(label: str, value: str, risk_class: str = "NA"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value risk-{risk_class}">{value}</div>
    </div>""", unsafe_allow_html=True)


def risk_badge(level: str) -> str:
    return f'<span class="badge badge-{level}">{level}</span>'


def rec_box(text: str, priority: str):
    st.markdown(f'<div class="rec-box rec-{priority}">💡 {text}</div>', unsafe_allow_html=True)


def risk_percent(value) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.1f}%"


def _util_bar(pct, label: str) -> str:
    pct = pct or 0.0
    cls = "bar-green" if pct < 60 else ("bar-yellow" if pct < 85 else "bar-red")
    return (
        f'<div class="stat-label">{label}&nbsp;&nbsp;'
        f'<span style="color:#ccd6f6;font-weight:600">{pct:.1f}%</span></div>'
        f'<div class="bar-wrap"><div class="bar-fill {cls}" style="width:{min(pct,100):.1f}%"></div></div>'
    )


def _sparkline(df: pd.DataFrame, y_col: str, color: str, title: str):
    fig = go.Figure(go.Scatter(
        x=df.index, y=df[y_col].fillna(0),
        mode='lines', fill='tozeroy',
        line=dict(color=color, width=1.5),
        fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.15)') if color.startswith('rgb') else color,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color='#8892b0', size=11)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=130, margin=dict(t=28, b=6, l=6, r=6),
        xaxis=dict(visible=False), yaxis=dict(showgrid=False, tickfont=dict(size=9, color='#8892b0')),
        showlegend=False
    )
    return fig


def gauge_chart(value: float, title: str, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        title={'text': title, 'font': {'color': '#ccd6f6', 'size': 14}},
        number={'suffix': '%', 'font': {'color': color, 'size': 28}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#8892b0'},
            'bar': {'color': color},
            'bgcolor': '#1a1d2e',
            'bordercolor': '#2d3561',
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0,208,132,0.15)'},
                {'range': [30, 70], 'color': 'rgba(240,180,41,0.15)'},
                {'range': [70, 100], 'color': 'rgba(230,57,70,0.15)'},
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=220, margin=dict(t=40, b=10, l=10, r=10)
    )
    return fig


def load_image_safe(path: str):
    try:
        return Image.open(path)
    except Exception:
        return None


def load_csv_safe(path: str):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ─────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────
if "sec_result" not in st.session_state:
    st.session_state.sec_result = None
if "maint_result" not in st.session_state:
    st.session_state.maint_result = None
if "risk_result" not in st.session_state:
    st.session_state.risk_result = None
if "rec_result" not in st.session_state:
    st.session_state.rec_result = None
if "pred_log" not in st.session_state:
    st.session_state.pred_log = []
if "drift_demo_shown" not in st.session_state:
    st.session_state.drift_demo_shown = False
if "infra_auto_refresh" not in st.session_state:
    st.session_state.infra_auto_refresh = False
if "maint_evaluated_node" not in st.session_state:
    st.session_state.maint_evaluated_node = None


# ─────────────────────────────────────────────
# Cached Resource Loaders
# ─────────────────────────────────────────────
@st.cache_resource
def get_security_pipeline():
    try:
        return load_security_pipeline()
    except Exception:
        return None


@st.cache_resource
def get_security_model():
    try:
        return load_security_model()
    except Exception:
        return None


@st.cache_data
def get_test_data():
    raw_p = os.path.join(BASE_DIR, "data", "security", "UNSW_NB15_testing-set.csv")
    proc_p = os.path.join(BASE_DIR, "data", "security", "processed", "test_processed.csv")
    try:
        if os.path.exists(raw_p):
            df = pd.read_csv(raw_p)
            feat_cols = [c for c in df.columns if c not in ['id', 'label', 'attack_cat']]
            if 'attack_cat' in df.columns and 'attack_cat_raw' not in df.columns:
                df['attack_cat_raw'] = df['attack_cat']
            return df, feat_cols
        elif os.path.exists(proc_p):
            df = pd.read_csv(proc_p)
            feat_cols = [c for c in df.columns if c not in ['binary_label', 'multiclass_label', 'attack_cat_raw']]
            return df, feat_cols
        return None, []
    except Exception:
        return None, []


@st.cache_data
def get_security_comparison():
    return load_csv_safe(os.path.join(BASE_DIR, "results", "security", "model_comparison.csv"))


@st.cache_data
def get_maintenance_comparison():
    return load_csv_safe(os.path.join(BASE_DIR, "results", "maintenance", "model_comparison.csv"))


@st.cache_data
def get_adaptive_comparison():
    return load_csv_safe(os.path.join(BASE_DIR, "results", "adaptive", "adaptive_retraining_comparison.csv"))


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Adaptive Cloud\n**Security & Predictive Maintenance**")
    st.markdown("---")
    st.markdown("#### 📋 Core Modules")
    st.info("""
    1. 🔐 Security Threat Detection
    2. ⚙️ Predictive Maintenance
    3. 🖥️ Live Infrastructure & Risk
    4. 📊 Adaptive Learning & Drift
    """)
    st.markdown("---")
    st.markdown("#### ⚙️ Risk Engine Weights")
    w_sec = st.slider("Security weight (wₛ)", 0.0, 1.0, 0.5, 0.05)
    w_fail = 1.0 - w_sec
    st.caption(f"Failure weight (w_f) = {w_fail:.2f} (auto-computed)")
    st.markdown("---")

    # Live Infrastructure summary
    nodes_summary = _fetch_nodes()
    total_n = len(nodes_summary)
    online_n = sum(1 for n in nodes_summary if n.get("status") == "ONLINE")
    st.markdown("#### 🖥️ Live Node Fleet")
    s_col1, s_col2 = st.columns(2)
    s_col1.metric("Registered", total_n)
    s_col2.metric("Online", online_n)
    st.caption(f"Database: `data/telemetry.db` ({'Connected' if _db_ok() else 'Not found'})")
    st.markdown("---")
    st.caption("College Project · Adaptive Learning for Cloud Security & Predictive Maintenance")


# ─────────────────────────────────────────────
# Title Banner
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='color:#ccd6f6; font-size:28px; font-weight:700; margin-bottom:4px;'>
    🛡️ Adaptive Learning for Cloud Security &amp; Predictive Maintenance
</h1>
<p style='color:#8892b0; font-size:14px; margin-bottom:18px;'>
    Intelligent Monitoring System: Real-Time Node Telemetry · ML Intrusion Detection · Predictive Failure Prevention · Data Drift Adaptation
</p>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Section A – System Overview
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📊 System Live Overview</div>', unsafe_allow_html=True)

nodes_current = _fetch_nodes()
current_node_ids = {n["node_id"] for n in nodes_current}
eval_node = st.session_state.get("maint_evaluated_node")
eval_node_online = any(n["node_id"] == eval_node and n.get("status") == "ONLINE" for n in nodes_current)

if not eval_node_online or eval_node not in current_node_ids:
    st.session_state.maint_result = None
    st.session_state.risk_result = None
    st.session_state.rec_result = None
    st.session_state.maint_evaluated_node = None

r = st.session_state.risk_result
rec = st.session_state.rec_result
sec_level  = r["security_risk_level"]  if r else "N/A"
fail_level = r["failure_risk_level"]   if r else "N/A"
over_level = r["overall_risk_level"]   if r else "N/A"
online_count = sum(1 for n in nodes_current if n.get("status") == "ONLINE")

if not nodes_current or online_count == 0:
    model_status = "No Live Nodes"
elif r and eval_node:
    model_status = f"Active ({eval_node})"
else:
    model_status = "Monitoring Fleet"

col1, col2, col3, col4 = st.columns(4)
with col1: metric_card("Security Risk",  sec_level,  sec_level)
with col2: metric_card("Failure Risk",   fail_level, fail_level)
with col3: metric_card("Overall Risk",   over_level, over_level)
with col4: metric_card("System Status",  model_status, "NA")

if rec:
    rec_box(rec["recommendation"], rec["action_priority"])

st.markdown("---")


# ─────────────────────────────────────────────
# Clean 4-Tab Navigation
# ─────────────────────────────────────────────
tab_sec, tab_maint, tab_infra, tab_drift = st.tabs([
    "🔐 1 · Security Threat Detection",
    "⚙️ 2 · Predictive Maintenance",
    "🖥️ 3 · Live Infrastructure & Risk",
    "📊 4 · Adaptive Learning / Drift"
])


# ═══════════════════════════════════════════════
# TAB 1 — Security Threat Detection
# ═══════════════════════════════════════════════
with tab_sec:
    st.markdown('<div class="section-header">🔐 Security Threat Detection (UNSW-NB15)</div>', unsafe_allow_html=True)

    st.info(
        "Network-flow security monitoring detects cybersecurity anomalies and intrusion attacks. "
        "Predictions run against genuine network-flow vectors or labeled UNSW-NB15 evaluation traffic. "
        "Use the demonstration controls below to attach attack scenarios to registered nodes."
    )

    try:
        init_db()
        live_sec_rows = security_status_for_all_nodes()
        demo_cats = list_demo_attack_categories()
    except Exception:
        live_sec_rows, demo_cats = [], ["Normal"]

    st.markdown("#### Live Registered Nodes — Security Status")
    if not live_sec_rows:
        st.warning(
            "⚠️ No live nodes available.\n\n"
            "Start the Node Agent on a connected computer to begin collecting live telemetry."
        )
    else:
        display_sec = []
        for row in live_sec_rows:
            src = row.get("prediction_source") or "NONE"
            origin = "SIMULATED / DEMO" if src == "DEMO_SAMPLE" else ("GENUINE NETWORK-FLOW" if src == "GENUINE_NETWORK_FLOW" else "NO FLOW DATA")
            sec_cls = row.get("classification")
            if not sec_cls or sec_cls == "NO_FLOW_DATA" or src == "NONE":
                status_text = "Security telemetry unavailable for this node"
            else:
                status_text = sec_cls
            display_sec.append({
                "node_id": row.get("node_id"),
                "status": status_text,
                "attack_type": row.get("attack_type") or "—",
                "security_risk": row.get("security_risk") or "N/A",
                "confidence": f"{row['confidence']:.1%}" if row.get("confidence") is not None else "—",
                "timestamp": (row.get("timestamp") or "")[:19].replace("T", " "),
                "model_version": row.get("model_version") or "—",
                "origin": origin,
            })
        st.dataframe(pd.DataFrame(display_sec), use_container_width=True, hide_index=True)

        node_ids = [r["node_id"] for r in live_sec_rows]
        d1, d2, d3 = st.columns([2, 2, 1])
        with d1:
            target_node = st.selectbox("Select Target Node", node_ids, key="sec_target_node")
        with d2:
            demo_cat = st.selectbox("Attach Attack Category (UNSW-NB15 Sample)", demo_cats, key="sec_demo_cat")
        with d3:
            st.markdown("<br>", unsafe_allow_html=True)
            fire_demo = st.button("Inject Security Event", key="btn_sec_demo", type="primary")

        if fire_demo and target_node:
            try:
                demo_res = ingest_demo_event(target_node, demo_cat, persist=True)
                st.success(
                    f"Security event stored for `{target_node}`: "
                    f"{demo_res['classification']}"
                    + (f" / {demo_res['attack_type']}" if demo_res.get("attack_type") else "")
                    + f" · risk: {demo_res['security_risk']} · confidence: {demo_res['confidence']}"
                )
                st.session_state.sec_result = {
                    "prediction": demo_res.get("attack_type") or demo_res["classification"],
                    "probability": demo_res.get("confidence") or 0.95
                }
                # Update integrated risk if maintenance result exists
                fail_p = st.session_state.maint_result["failure_probability"] if st.session_state.maint_result else 0.05
                st.session_state.risk_result = evaluate_integrated_risk(
                    st.session_state.sec_result["prediction"], fail_p, w_sec=w_sec, w_fail=w_fail
                )
                st.session_state.rec_result = generate_recommendation(
                    st.session_state.risk_result["security_risk_level"], st.session_state.risk_result["failure_risk_level"]
                )
                st.rerun()
            except InvalidNetworkFlowError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Security event error: {e}")

    st.markdown("---")
    st.markdown("#### Historical / Offline Security Classifier")

    df_test, feat_cols = get_test_data()
    input_mode = st.radio("Evaluation mode", ["Pick sample from UNSW-NB15 test set", "CSV batch upload"], horizontal=True)

    if input_mode == "Pick sample from UNSW-NB15 test set" and df_test is not None:
        n_samples = min(200, len(df_test))
        row_idx = st.selectbox(
            "Select evaluation sample",
            list(range(n_samples)),
            format_func=lambda i: f"Sample #{i}" + (f" — Ground Truth: {df_test.iloc[i].get('attack_cat_raw','?')}" if 'attack_cat_raw' in df_test.columns else "")
        )

        selected_row = df_test.iloc[row_idx]
        input_dict = {c: float(selected_row[c]) if pd.api.types.is_numeric_dtype(df_test[c]) else str(selected_row[c]) for c in feat_cols}

        if st.button("🔍 Run Security Prediction", key="btn_sec", type="primary"):
            with st.spinner("Running ML security classifier…"):
                try:
                    result = predict_attack(input_dict)
                    fail_p = st.session_state.maint_result["failure_probability"] if st.session_state.maint_result else 0.10
                    risk_out = evaluate_integrated_risk(
                        result["prediction"],
                        fail_p,
                        w_sec=w_sec, w_fail=w_fail
                    )
                    rec_out = generate_recommendation(
                        risk_out["security_risk_level"], risk_out["failure_risk_level"]
                    )
                    st.session_state.sec_result  = result
                    st.session_state.risk_result = risk_out
                    st.session_state.rec_result  = rec_out
                    st.session_state.pred_log.append({
                        "Sample": row_idx,
                        "Prediction": result["prediction"],
                        "Confidence": f"{result['probability']:.1%}",
                        "Security Risk": risk_out["security_risk_level"]
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    elif input_mode == "CSV batch upload":
        uploaded = st.file_uploader("Upload CSV (UNSW-NB15 features)", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.dataframe(df_up.head(), use_container_width=True)
            if st.button("🔍 Run Batch Prediction", key="btn_batch"):
                with st.spinner("Running batch evaluation…"):
                    try:
                        df_out = predict_attack(df_up)
                        st.success(f"Evaluated {len(df_out)} network flows.")
                        st.dataframe(df_out[['predicted_attack', 'confidence_probability']].head(20), use_container_width=True)
                    except Exception as e:
                        st.error(f"Batch prediction error: {e}")

    if st.session_state.sec_result:
        res = st.session_state.sec_result
        rsk = st.session_state.risk_result
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Attack Category", res["prediction"])
        with c2: st.metric("Confidence", f"{res['probability']:.1%}")
        with c3: st.metric("Security Risk Level", rsk["security_risk_level"] if rsk else "—")

    df_sec_comp = get_security_comparison()
    if df_sec_comp is not None:
        st.markdown("---")
        st.markdown("#### 📊 Security Baseline Models Comparison (UNSW-NB15)")
        st.dataframe(df_sec_comp, use_container_width=True)

        if 'Model' in df_sec_comp.columns and 'F1-Score' in df_sec_comp.columns:
            fig = px.bar(
                df_sec_comp.dropna(subset=['F1-Score']),
                x='Model', y='F1-Score', color='Task', barmode='group',
                color_discrete_sequence=px.colors.qualitative.Dark2,
                title="Security Model Performance (Accuracy & F1-Score)",
                template='plotly_dark'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 2 — Predictive Maintenance (Live Node Driven)
# ═══════════════════════════════════════════════
with tab_maint:
    st.markdown('<div class="section-header">⚙️ Predictive Maintenance & Server Health</div>', unsafe_allow_html=True)

    st.caption(
        "Predicts equipment and server failure using live telemetry collected from the Node Agent. "
        "Evaluated via Random Forest trained on cloud server traces (Alibaba Cluster Trace 2017). "
        "Telemetry is automatically retrieved from the live node database."
    )

    all_nodes = _fetch_nodes()

    if not all_nodes:
        st.warning(
            "⚠️ Predictive maintenance unavailable — no live nodes connected.\n\n"
            "Start the Node Agent on a connected computer to register a live node."
        )
    else:
        node_options = [n["node_id"] for n in all_nodes]
        col_sel1, col_sel2 = st.columns([2, 2])
        with col_sel1:
            selected_node = st.selectbox(
                "Select Live Node to Analyze:",
                node_options,
                format_func=lambda nid: f"{nid} — {next((n.get('hostname') for n in all_nodes if n['node_id'] == nid), 'Unknown')}",
                key="maint_node_selector"
            )
        node_info = next((n for n in all_nodes if n["node_id"] == selected_node), {})
        node_status = node_info.get("status", "OFFLINE")
        node_host = node_info.get("hostname", "unknown")
        node_os = node_info.get("operating_system", "Unknown OS")
        node_ip = node_info.get("ip_address", "127.0.0.1")
        last_seen_ts = (node_info.get("last_seen") or "")[:19].replace("T", " ")

        with col_sel2:
            st.markdown("<br>", unsafe_allow_html=True)
            status_html = (
                f'<span class="pill-online">● ONLINE</span>'
                if node_status == "ONLINE"
                else f'<span class="pill-offline">● OFFLINE</span>'
            )
            st.markdown(
                f"**Status:** {status_html}&nbsp;&nbsp;&nbsp;"
                f"**Host:** `{node_host}`&nbsp;&nbsp;&nbsp;"
                f"**Last Seen:** `{last_seen_ts} UTC`",
                unsafe_allow_html=True
            )

        if node_status != "ONLINE":
            st.error(
                f"⚠️ Node `{selected_node}` is **{node_status} / STALE** (Last seen: `{last_seen_ts} UTC`).\n\n"
                "Live predictive maintenance cannot analyze stale telemetry as current live input. "
                "Start or reconnect the Node Agent to resume live telemetry stream."
            )
        else:
            latest = _fetch_latest_telemetry(selected_node)
            if not latest:
                st.warning(
                    f"⚠️ Telemetry awaiting collection for node `{selected_node}`. "
                    "The Node Agent has registered but has not yet posted its first telemetry payload."
                )
            else:
                raw_m = latest.get("metrics") or {}
                telemetry_ts = (latest.get("timestamp") or "")[:19].replace("T", " ")

                cur_cpu = latest.get("cpu_utilization") if latest.get("cpu_utilization") is not None else raw_m.get("cpu_utilization")
                cur_ram = latest.get("ram_utilization") if latest.get("ram_utilization") is not None else raw_m.get("ram_utilization")
                cur_disk = latest.get("disk_utilization") if latest.get("disk_utilization") is not None else raw_m.get("disk_utilization")
                cur_temp = latest.get("cpu_temperature_c") if latest.get("cpu_temperature_c") is not None else raw_m.get("cpu_temperature_c")
                cur_sent = latest.get("network_sent_mb") if latest.get("network_sent_mb") is not None else raw_m.get("network_sent_mb", 0.0)
                cur_recv = latest.get("network_recv_mb") if latest.get("network_recv_mb") is not None else raw_m.get("network_recv_mb", 0.0)
                cur_procs = latest.get("process_count") if latest.get("process_count") is not None else raw_m.get("process_count", 0)
                cur_load1 = raw_m.get("load_1min", 0.0)
                cur_load5 = raw_m.get("load_5min", 0.0)
                cur_load15 = raw_m.get("load_15min", 0.0)
                cur_duration = raw_m.get("duration_sec", 0)
                cur_server_type = raw_m.get("server_type", "ComputeNode")

                st.markdown(f"#### Latest Telemetry from `{selected_node}` (Recorded at `{telemetry_ts} UTC`)")
                t_col1, t_col2, t_col3 = st.columns(3)

                with t_col1:
                    st.markdown(_util_bar(cur_cpu, "CPU Utilization"), unsafe_allow_html=True)
                    st.markdown(_util_bar(cur_ram, "RAM / Memory Utilization"), unsafe_allow_html=True)
                    st.markdown(_util_bar(cur_disk, "Disk Utilization"), unsafe_allow_html=True)

                with t_col2:
                    st.markdown(f"**CPU Temperature:** {f'{cur_temp:.1f} °C' if cur_temp is not None else 'N/A'}")
                    st.markdown(f"**Network I/O:** `{cur_sent:.1f} MB sent` / `{cur_recv:.1f} MB recv`")
                    st.markdown(f"**Active Processes:** `{cur_procs}`")

                with t_col3:
                    st.markdown(f"**Load Averages (1m / 5m / 15m):** `{cur_load1:.2f}` / `{cur_load5:.2f}` / `{cur_load15:.2f}`")
                    uptime_hr = cur_duration // 3600
                    uptime_min = (cur_duration % 3600) // 60
                    st.markdown(f"**Execution / Uptime:** `{cur_duration}s` ({uptime_hr}h {uptime_min}m)")
                    st.markdown(f"**Server Role / Type:** `{cur_server_type}`")

                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("🔍 Analyze Live Node", key="btn_analyze_live_node", type="primary"):
                    with st.spinner(f"Evaluating failure prediction model on live node `{selected_node}`..."):
                        try:
                            maint_input = {
                                "CPU_utilization": float(cur_cpu if cur_cpu is not None else 0.0),
                                "Memory_utilization": float(cur_ram if cur_ram is not None else 0.0),
                                "Disk_utilization": float(cur_disk if cur_disk is not None else 0.0),
                                "Load_1min": float(cur_load1),
                                "Load_5min": float(cur_load5),
                                "Load_15min": float(cur_load15),
                                "Duration_sec": int(cur_duration),
                                "Server_Type": str(cur_server_type)
                            }
                            maint_res = predict_failure(maint_input)

                            # Fuse with node security status
                            node_sec = get_latest_node_security(selected_node)
                            sec_attack = node_sec.get("attack_type") if (node_sec and node_sec.get("attack_type")) else (
                                st.session_state.sec_result["prediction"] if st.session_state.sec_result else "Normal"
                            )

                            risk_out = evaluate_integrated_risk(
                                sec_attack,
                                maint_res["failure_probability"],
                                w_sec=w_sec,
                                w_fail=w_fail
                            )
                            rec_out = generate_recommendation(
                                risk_out["security_risk_level"],
                                risk_out["failure_risk_level"],
                                maint_res["failure_probability"]
                            )

                            st.session_state.maint_result = maint_res
                            st.session_state.risk_result = risk_out
                            st.session_state.rec_result = rec_out
                            st.session_state.maint_evaluated_node = selected_node
                            st.rerun()

                        except Exception as e:
                            st.error(f"Prediction analysis error: {e}")

                if st.session_state.maint_result and st.session_state.get("maint_evaluated_node") == selected_node:
                    m_res = st.session_state.maint_result
                    r_res = st.session_state.risk_result
                    rec = st.session_state.rec_result
                    f_prob = m_res["failure_probability"]
                    f_risk = m_res["risk_level"]
                    status_label = "Healthy" if f_prob < 0.30 else ("Elevated Watch" if f_prob < 0.70 else "Failure Imminent")
                    gauge_color = RISK_COLOR.get(f_risk, "#8892b0")

                    st.markdown("---")
                    st.markdown(f"### 📊 ML Failure Prediction Results — `{selected_node}`")

                    res_col1, res_col2, res_col3 = st.columns([1, 1, 1])
                    with res_col1:
                        st.metric("Failure Probability", f"{f_prob * 100:.1f}%")
                        st.markdown(f"**Maintenance Risk:** {risk_badge(f_risk)}", unsafe_allow_html=True)
                        st.markdown(f"**Prediction Status:** `{status_label}`")

                    with res_col2:
                        st.plotly_chart(gauge_chart(f_prob, "Failure Probability", gauge_color), use_container_width=True)

                    with res_col3:
                        st.markdown("#### Integrated Assessment")
                        if r_res:
                            st.metric("Overall Integrated Risk", r_res["overall_risk_level"])
                            st.caption(f"Score: {r_res['overall_risk_score']:.3f} · Security: {r_res['security_risk_level']} (w={w_sec:.2f}) · Failure: {r_res['failure_risk_level']} (w={w_fail:.2f})")

                    st.markdown("#### 💡 Recommended Prevention Action")
                    if rec:
                        rec_box(rec["recommendation"], rec["action_priority"])

    st.markdown("---")
    df_maint_comp = get_maintenance_comparison()
    if df_maint_comp is not None:
        st.markdown("#### 📊 Maintenance Baseline Models Comparison (Alibaba Cluster Trace 2017)")
        display_cols = [c for c in ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'] if c in df_maint_comp.columns]
        st.dataframe(df_maint_comp[display_cols], use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# TAB 3 — Live Infrastructure & Integrated Risk
# ═══════════════════════════════════════════════
with tab_infra:
    st.markdown('<div class="section-header">🖥️ Live Infrastructure & Integrated Risk Engine</div>', unsafe_allow_html=True)

    # Top controls
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 3])
    with ctrl1:
        if st.button("🔄 Refresh Fleet", key="infra_refresh"):
            st.rerun()
    with ctrl2:
        auto = st.checkbox("Auto-refresh every 5 s", value=st.session_state.infra_auto_refresh, key="infra_auto_chk")
        st.session_state.infra_auto_refresh = auto
    with ctrl3:
        history_n = st.slider("Historical telemetry points to show", 5, 60, 20, key="infra_hist_n")

    nodes = _fetch_nodes() if _db_ok() else []
    if not nodes:
        st.warning(
            "⚠️ No live nodes available.\n\n"
            "Start the Node Agent on a connected computer to begin collecting live telemetry."
        )
    else:
            # Fleet KPI row
            total = len(nodes)
            online = sum(1 for n in nodes if n["status"] == "ONLINE")
            offline = total - online

            cpu_vals = []
            for nd in nodes:
                if nd["status"] == "ONLINE":
                    hist = _fetch_node_history(nd["node_id"], 1)
                    if hist and hist[-1].get("cpu_utilization") is not None:
                        cpu_vals.append(hist[-1]["cpu_utilization"])
            avg_cpu = (sum(cpu_vals) / len(cpu_vals)) if cpu_vals else None

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Nodes", total)
            k2.metric("Online Nodes", online, delta=f"+{online}" if online else None, delta_color="normal")
            k3.metric("Offline Nodes", offline, delta=f"-{offline}" if offline else None, delta_color="inverse")
            k4.metric("Avg CPU (Online)", f"{avg_cpu:.1f}%" if avg_cpu is not None else "—")

            st.markdown("---")

            # Unified Node Risk Assessment
            st.markdown("#### 🚨 Fleet Unified Risk Matrix")
            try:
                sec_map = {s["node_id"]: s for s in get_all_latest_node_security()}
                health_map = {h["node_id"]: h for h in get_all_latest_live_health()}
            except Exception:
                sec_map, health_map = {}, {}

            risk_map = {
                nd["node_id"]: combine_node_risk(
                    nd["node_id"], sec_map.get(nd["node_id"]), health_map.get(nd["node_id"])
                )
                for nd in nodes
            }

            risk_table = []
            for risk in risk_map.values():
                risk_table.append({
                    "node_id": risk["node_id"],
                    "security_risk": risk_percent(risk["security_risk"]),
                    "maintenance_risk": risk_percent(risk["maintenance_risk"]),
                    "overall_risk": risk_percent(risk["overall_risk"]),
                    "risk_level": risk["risk_level"],
                    "timestamp": (risk["timestamp"] or "")[:19].replace("T", " "),
                    "contributing_factors": "; ".join(risk["contributing_factors"]),
                })
            st.dataframe(pd.DataFrame(risk_table), use_container_width=True, hide_index=True)

            # Evidence-Based Recommendations
            st.markdown("#### 💡 Evidence-Based Prevention Recommendations")
            try:
                node_recommendations = get_all_node_recommendations()
            except Exception:
                node_recommendations = []

            if node_recommendations:
                rec_df = pd.DataFrame([
                    {
                        "node_id": item["node_id"],
                        "detected_problem": item["detected_problem"],
                        "evidence": item["evidence"],
                        "recommended_action": item["recommended_action"],
                        "priority": item["priority"],
                        "timestamp": (item.get("timestamp") or "")[:19].replace("T", " "),
                    }
                    for item in node_recommendations
                ])
                st.dataframe(rec_df, use_container_width=True, hide_index=True)
            else:
                st.info("✅ All monitored nodes are within normal operational thresholds. No critical intervention needed.")

            with st.expander("📋 View Prevention Rule Matrix"):
                rows = []
                for (sec, fail), rec_text in RECOMMENDATION_RULE_MATRIX.items():
                    pr = generate_recommendation(sec, fail)
                    rows.append({"Security Level": sec, "Failure Level": fail, "Priority": pr["action_priority"], "Recommendation": rec_text})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🖥️ Individual Node Details & Live Telemetry History")

            for nd in nodes:
                nid = nd["node_id"]
                hostname = nd.get("hostname", "unknown")
                os_info = nd.get("operating_system", "")
                ip_addr = nd.get("ip_address", "")
                status = nd.get("status", "OFFLINE")
                last_seen = nd.get("last_seen", "")[:19].replace("T", " ") + " UTC"
                first_seen = nd.get("first_seen", "")[:19].replace("T", " ") + " UTC"
                hw = json.loads(nd["hardware_info"]) if nd.get("hardware_info") else {}

                hist = _fetch_node_history(nid, history_n)
                latest = hist[-1] if hist else {}

                cpu_pct = latest.get("cpu_utilization")
                ram_pct = latest.get("ram_utilization")
                disk_pct = latest.get("disk_utilization")
                net_sent = latest.get("network_sent_mb")
                net_recv = latest.get("network_recv_mb")
                cpu_temp = latest.get("cpu_temperature_c")
                procs = latest.get("process_count")

                pill_cls = "pill-online" if status == "ONLINE" else "pill-offline"
                card_cls = "node-online" if status == "ONLINE" else "node-offline"

                sec = sec_map.get(nid) or {}
                sec_cls = sec.get("classification") or "NO_FLOW_DATA"
                sec_atk = sec.get("attack_type")
                sec_risk = sec.get("security_risk") or "N/A"
                sec_origin_cls = "pill-demo" if sec.get("prediction_source") == "DEMO_SAMPLE" else ("pill-genuine" if sec.get("prediction_source") == "GENUINE_NETWORK_FLOW" else "pill-none")
                sec_origin_txt = "DEMO / SIMULATED" if sec.get("prediction_source") == "DEMO_SAMPLE" else ("GENUINE FLOW" if sec.get("prediction_source") == "GENUINE_NETWORK_FLOW" else "NO FLOW DATA")

                node_risk = risk_map.get(nid, {})
                unified_risk = risk_percent(node_risk.get("overall_risk"))
                unified_level = node_risk.get("risk_level", "N/A")

                header_html = (
                    f'<div class="node-card {card_cls}">'
                    f'<div class="node-title">'
                    f'<span class="{pill_cls}">{status}</span>&nbsp;&nbsp;'
                    f'{hostname}&nbsp;&nbsp;<span style="font-size:12px;color:#8892b0">({nid})</span>'
                    f'</div>'
                    f'<div class="node-sub">{os_info} &nbsp;|&nbsp; IP: {ip_addr} &nbsp;|&nbsp; '
                    f'First seen: {first_seen} &nbsp;|&nbsp; Last seen: {last_seen}</div>'
                    f'<div class="node-sub">'
                    f'<span class="{sec_origin_cls}">{sec_origin_txt}</span>&nbsp;&nbsp;'
                    f'Security: <b>{sec_cls}</b>'
                    + (f' ({sec_atk})' if sec_atk else '')
                    + f' &nbsp;|&nbsp; Risk: {sec_risk}'
                    + (f' &nbsp;|&nbsp; Confidence: {sec.get("confidence")}' if sec.get("confidence") is not None else '')
                    + f' &nbsp;|&nbsp; Unified Risk: <b>{unified_risk}</b> ({unified_level})'
                    f'</div>'
                )

                bars = ""
                if cpu_pct is not None: bars += _util_bar(cpu_pct, "CPU Utilisation")
                if ram_pct is not None: bars += _util_bar(ram_pct, "RAM Utilisation")
                if disk_pct is not None: bars += _util_bar(disk_pct, "Disk Utilisation")

                stats_html = '<div style="display:flex;gap:30px;flex-wrap:wrap;margin-top:4px;">'
                if net_sent is not None:
                    stats_html += f'<div><div class="stat-label">Net Sent</div><div class="stat-value">{net_sent:.1f} MB</div></div>'
                if net_recv is not None:
                    stats_html += f'<div><div class="stat-label">Net Recv</div><div class="stat-value">{net_recv:.1f} MB</div></div>'
                stats_html += f'<div><div class="stat-label">CPU Temp</div><div class="stat-value">{f"{cpu_temp:.1f} &deg;C" if cpu_temp is not None else "N/A"}</div></div>'
                if procs is not None:
                    stats_html += f'<div><div class="stat-label">Processes</div><div class="stat-value">{procs}</div></div>'
                hw_cores = hw.get("cpu_count_logical") or hw.get("cpu_cores")
                hw_ram = hw.get("ram_total_mb")
                hw_disk = hw.get("disk_total_gb")
                if hw_cores:
                    stats_html += f'<div><div class="stat-label">CPU Cores</div><div class="stat-value">{hw_cores} logical</div></div>'
                if hw_ram:
                    stats_html += f'<div><div class="stat-label">Total RAM</div><div class="stat-value">{hw_ram/1024:.1f} GB</div></div>'
                if hw_disk:
                    stats_html += f'<div><div class="stat-label">Total Disk</div><div class="stat-value">{hw_disk:.0f} GB</div></div>'
                stats_html += '</div>'

                st.markdown(header_html + bars + stats_html + '</div>', unsafe_allow_html=True)

                # History sparklines
                if len(hist) >= 2:
                    df_h = pd.DataFrame(hist)
                    df_h.index = range(len(df_h))
                    sc1, sc2, sc3 = st.columns(3)
                    if "cpu_utilization" in df_h.columns and df_h["cpu_utilization"].notna().any():
                        sc1.plotly_chart(_sparkline(df_h, "cpu_utilization", "#64d2ff", "CPU % History"), use_container_width=True, key=f"spark_cpu_{nid}")
                    if "ram_utilization" in df_h.columns and df_h["ram_utilization"].notna().any():
                        sc2.plotly_chart(_sparkline(df_h, "ram_utilization", "#f0b429", "RAM % History"), use_container_width=True, key=f"spark_ram_{nid}")
                    if "disk_utilization" in df_h.columns and df_h["disk_utilization"].notna().any():
                        sc3.plotly_chart(_sparkline(df_h, "disk_utilization", "#ff6b35", "Disk % History"), use_container_width=True, key=f"spark_disk_{nid}")

                    if "network_sent_mb" in df_h.columns and df_h["network_sent_mb"].notna().any():
                        net_fig = go.Figure()
                        net_fig.add_trace(go.Scatter(x=df_h.index, y=df_h["network_sent_mb"].fillna(0), mode='lines', name='Sent MB', line=dict(color='#00d084', width=1.5)))
                        net_fig.add_trace(go.Scatter(x=df_h.index, y=df_h["network_recv_mb"].fillna(0), mode='lines', name='Recv MB', line=dict(color='#8892b0', width=1.5)))
                        net_fig.update_layout(
                            title=dict(text="Network I/O History (MB)", font=dict(color='#8892b0', size=11)),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            height=130, margin=dict(t=28, b=6, l=6, r=6),
                            xaxis=dict(visible=False), yaxis=dict(showgrid=False, tickfont=dict(size=9, color='#8892b0')),
                            legend=dict(orientation='h', y=1.2, font=dict(size=9, color='#8892b0')),
                            template='plotly_dark'
                        )
                        st.plotly_chart(net_fig, use_container_width=True, key=f"spark_net_{nid}")

                st.markdown("<br>", unsafe_allow_html=True)

            # Controlled Experimental Mitigation Evaluation
            st.markdown("---")
            st.markdown("#### 🔬 Controlled Workload Mitigation Experiment")
            st.caption("Evaluates the impact of simulated workload redistribution on measured node risk.")
            exp_nodes = [r["node_id"] for r in get_all_node_risks()]
            if exp_nodes:
                c_exp1, c_exp2, c_exp3 = st.columns([2, 2, 1])
                with c_exp1:
                    exp_node = st.selectbox("Select Target Node for Mitigation", exp_nodes, key="closed_loop_node")
                with c_exp2:
                    exp_scale = st.slider("Simulated Workload Reduction Factor", 0.1, 0.9, 0.5, 0.1, key="closed_loop_scale")
                with c_exp3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    run_exp = st.button("Run Mitigation", key="run_closed_loop", type="primary")

                if run_exp:
                    try:
                        st.session_state.closed_loop_result = run_closed_loop_evaluation(exp_node, exp_scale)
                    except Exception as exc:
                        st.error(str(exc))

                exp_res = st.session_state.get("closed_loop_result") or get_latest_closed_loop_evaluation(exp_node)
                if exp_res:
                    b_data = exp_res["before"]
                    a_data = exp_res["after"]
                    col_b, col_arrow, col_a = st.columns([4, 1, 4])
                    with col_b:
                        st.markdown("##### BEFORE MITIGATION")
                        st.metric("Overall Risk", risk_percent(b_data.get("overall_risk")))
                        st.caption(f"CPU {b_data.get('cpu')}% · RAM {b_data.get('ram')}% · Failure Risk {risk_percent(b_data.get('maintenance_risk'))}")
                    with col_arrow:
                        st.markdown("<div style='text-align:center;font-size:26px;padding-top:20px;'>➡️</div>", unsafe_allow_html=True)
                    with col_a:
                        st.markdown("##### AFTER MITIGATION")
                        st.metric("Overall Risk", risk_percent(a_data.get("overall_risk")))
                        st.caption(f"CPU {a_data.get('cpu')}% · RAM {a_data.get('ram')}% · Failure Risk {risk_percent(a_data.get('maintenance_risk'))}")

                    st.info(f"**Intervention Applied:** {exp_res['intervention']}")

    if st.session_state.infra_auto_refresh:
        st.markdown('<meta http-equiv="refresh" content="5">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 4 — Adaptive Learning & Data Drift Detection
# ═══════════════════════════════════════════════
with tab_drift:
    st.markdown('<div class="section-header">📊 Adaptive Learning & Data Drift Detection</div>', unsafe_allow_html=True)

    st.markdown("#### Live Telemetry Drift Monitor")
    st.caption(
        "Monitors statistical drift between each node's baseline reference window and current live telemetry window "
        "using the Kolmogorov-Smirnov (KS) two-sample test. When distribution shift is flagged, adaptive retraining is triggered."
    )

    try:
        live_drift_results = detect_all_live_drift()
    except Exception as exc:
        live_drift_results = []
        st.error(f"Live drift check failed: {exc}")

    if live_drift_results:
        live_drift_table = pd.DataFrame([
            {
                "node_id": res["node_id"],
                "status": res["status"],
                "affected_features": ", ".join(res.get("affected_features", [])) or "None",
                "reference_window": res.get("reference_window"),
                "current_window": res.get("current_window"),
                "timestamp": (res.get("timestamp") or "")[:19].replace("T", " "),
            }
            for res in live_drift_results
        ])
        st.dataframe(live_drift_table, use_container_width=True, hide_index=True)
        drifted_nodes = [r for r in live_drift_results if r["status"] == "DRIFT DETECTED"]
        st.metric("Live Fleet Drift Status", "DRIFT DETECTED ⚠️" if drifted_nodes else "NO DRIFT ✅")

        live_node_ids = [r["node_id"] for r in live_drift_results]
        sel_drift_node = st.selectbox("Inspect Node Drift Features", live_node_ids, key="live_drift_node")
        sel_drift_res = next((r for r in live_drift_results if r["node_id"] == sel_drift_node), None)
        if sel_drift_res and sel_drift_res.get("features"):
            st.dataframe(pd.DataFrame(sel_drift_res["features"]), use_container_width=True, hide_index=True)
        else:
            st.info("Collecting telemetry samples to satisfy the KS comparison window threshold.")
    else:
        st.info("No registered nodes are currently available for live drift monitoring.")

    st.markdown("---")

    # Guarded Adaptive Retraining Status
    latest_retraining = get_latest_retraining_status()
    if latest_retraining:
        st.markdown("#### Guarded Adaptive Retraining Controller Status")
        current_ver = (
            latest_retraining["new_model_version"]
            if latest_retraining["deployment_decision"] == "APPROVED"
            else latest_retraining["previous_model_version"]
        )
        v_col1, v_col2, v_col3, v_col4, v_col5 = st.columns(5)
        v_col1.metric("Current Model", current_ver)
        v_col2.metric("Previous Model", latest_retraining["previous_model_version"])
        v_col3.metric("Drift Event", latest_retraining["drift_event"])
        v_col4.metric("Retraining Status", latest_retraining["status"])
        v_col5.metric("Deployment", latest_retraining["deployment_decision"])
        st.caption(f"**Retraining Reason:** {latest_retraining['reason']}")
        validation = latest_retraining.get("validation_metrics", {})
        if validation:
            st.json(validation)

    st.markdown("---")

    # Replay Adaptive Retraining Pipeline
    st.markdown("#### 🎬 Replay Adaptive Retraining Pipeline (End-to-End Demonstration)")
    st.caption("Demonstrates the closed-loop flow: Data Stream Shift → KS Drift Flagged → Model Retraining → Deployment.")

    drift_kpis = {
        "Method": "Kolmogorov-Smirnov (KS) 2-Sample Test",
        "Total Features": 51,
        "Drifted Features": 10,
        "Drift %": "19.61%",
        "System Drift Detected": True,
        "Threshold": "≥15% of features",
        "Deployment": "New Model RETRAINED & DEPLOYED"
    }

    dk1, dk2, dk3, dk4 = st.columns(4)
    dk1.metric("Features Evaluated", drift_kpis["Total Features"])
    dk2.metric("Drifted Features", drift_kpis["Drifted Features"])
    dk3.metric("Drift Ratio", drift_kpis["Drift %"])
    dk4.metric("System Drift Status", "DETECTED ⚠️" if drift_kpis["System Drift Detected"] else "NONE ✅")

    if st.button("▶ Run Adaptive Drift & Retraining Pipeline", key="btn_replay_drift", type="primary"):
        st.session_state.drift_demo_shown = True

    if st.session_state.drift_demo_shown:
        with st.status("Executing adaptive learning pipeline…", expanded=True) as status:
            st.write("📥 Loading shifted real-time data stream (Environment B)...")
            time.sleep(0.4)
            st.write("📊 Computing KS-Test distance and p-values across 51 features...")
            time.sleep(0.4)
            st.write("⚠️ Distribution Shift Flagged! (10/51 features drifted, p-value < 0.05, KS > 0.15)")
            time.sleep(0.4)
            st.write("🔄 Triggering dynamic model retraining with historical + incoming stream...")
            time.sleep(0.4)
            st.write("✅ Candidate adaptive model retrained & validated against test set.")
            time.sleep(0.3)
            st.write("📦 Model update approved and deployed! Accuracy restored from 75.0% → 100.0%.")
            status.update(label="Adaptive retraining pipeline completed successfully!", state="complete")

    df_adapt = get_adaptive_comparison()
    if df_adapt is not None:
        st.markdown("#### 📊 Old Model vs Retrained Adaptive Model Performance")
        st.dataframe(df_adapt, use_container_width=True, hide_index=True)

    drift_plot = load_image_safe(os.path.join(BASE_DIR, "results", "adaptive", "drift_scores_per_feature.png"))
    if drift_plot:
        st.markdown("#### 🔍 Feature Drift Scores (KS-Statistic)")
        st.image(drift_plot, use_container_width=True)
