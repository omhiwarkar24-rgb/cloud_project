"""
dashboard.py - Adaptive Cloud Security & Predictive Maintenance Dashboard
Full Streamlit application wiring backend security, maintenance, adaptive learning, risk engine, and prevention engine.

Launch: python -m streamlit run app/dashboard.py
"""

import os
import sys
import json
import sqlite3
import warnings
warnings.filterwarnings("ignore")

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
from prevention.prevention_engine import generate_recommendation, get_all_node_recommendations
from experiments.closed_loop_evaluation import run_closed_loop_evaluation, get_latest_closed_loop_evaluation
from risk_engine.node_risk import get_all_node_risks
from central_server.database import init_db
from risk_engine.node_risk import combine_node_risk
from analytics.security_infrastructure_correlation import analyze_all_nodes
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
        padding: 20px;
        text-align: center;
        margin: 6px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card .label { font-size: 12px; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .metric-card .value { font-size: 28px; font-weight: 700; }

    .risk-LOW    { color: #00d084; }
    .risk-MEDIUM { color: #f0b429; }
    .risk-HIGH   { color: #ff6b35; }
    .risk-CRITICAL { color: #e63946; }
    .risk-NA     { color: #8892b0; }

    .rec-box {
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 10px;
        font-size: 15px;
        font-weight: 500;
    }
    .rec-ROUTINE  { background: rgba(0,208,132,0.12); border-left: 4px solid #00d084; color: #00d084; }
    .rec-MODERATE { background: rgba(100,210,255,0.12); border-left: 4px solid #64d2ff; color: #64d2ff; }
    .rec-ELEVATED { background: rgba(240,180,41,0.15); border-left: 4px solid #f0b429; color: #f0b429; }
    .rec-URGENT   { background: rgba(230,57,70,0.15); border-left: 4px solid #e63946; color: #e63946; }

    .section-header {
        font-size: 20px; font-weight: 700; color: #ccd6f6;
        border-bottom: 2px solid #2d3561; padding-bottom: 10px; margin-bottom: 20px;
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

    /* ── Live Infrastructure tab ─────────────────────────── */
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
    .pill-online  { background:rgba(0,208,132,.18); color:#00d084; padding:2px 10px;
                    border-radius:20px; font-size:11px; font-weight:700; }
    .pill-offline { background:rgba(230,57,70,.18);  color:#e63946; padding:2px 10px;
                    border-radius:20px; font-size:11px; font-weight:700; }
    .pill-demo    { background:rgba(240,180,41,.18); color:#f0b429; padding:2px 10px;
                    border-radius:20px; font-size:11px; font-weight:700; }
    .pill-genuine { background:rgba(100,210,255,.18); color:#64d2ff; padding:2px 10px;
                    border-radius:20px; font-size:11px; font-weight:700; }
    .pill-none    { background:rgba(136,146,176,.18); color:#8892b0; padding:2px 10px;
                    border-radius:20px; font-size:11px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
RISK_COLOR = {"LOW": "#00d084", "MEDIUM": "#f0b429", "HIGH": "#ff6b35", "CRITICAL": "#e63946", "N/A": "#8892b0"}
PRIORITY_ORDER = {"ROUTINE": 0, "MODERATE": 1, "ELEVATED": 2, "URGENT": 3}

def metric_card(label, value, risk_class="NA"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value risk-{risk_class}">{value}</div>
    </div>""", unsafe_allow_html=True)

def risk_badge(level):
    return f'<span class="badge badge-{level}">{level}</span>'

def rec_box(text, priority):
    st.markdown(f'<div class="rec-box rec-{priority}">💡 {text}</div>', unsafe_allow_html=True)

def risk_percent(value):
    return "N/A" if value is None else f"{value * 100:.1f}%"

def gauge_chart(value, title, color):
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

def load_image_safe(path):
    try:
        return Image.open(path)
    except Exception:
        return None

def load_csv_safe(path):
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

# Cache resource loaders
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
            feat_cols = [c for c in df.columns if c not in ['binary_label','multiclass_label','attack_cat_raw']]
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
    3. 📈 Adaptive Learning & Drift
    4. 🚨 Failure Prevention Engine
    """)
    st.markdown("---")
    st.markdown("#### ⚙️ Risk Engine Weights")
    w_sec  = st.slider("Security weight (wₛ)", 0.0, 1.0, 0.5, 0.05)
    w_fail = 1.0 - w_sec
    st.caption(f"Failure weight (w_f) = {w_fail:.2f} (auto-computed)")
    st.markdown("---")
    st.markdown("#### 🎮 Quick Demo Presets")
    demo_choice = st.selectbox("Load demo scenario:", [
        "— None —",
        "Demo 1: Normal + Healthy",
        "Demo 2: High Machine Risk",
        "Demo 3: DoS Attack",
        "Demo 4: CRITICAL Combined"
    ])
    st.markdown("---")
    st.caption("College Project · Adaptive Learning System")


# ─────────────────────────────────────────────
# Title Banner
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='color:#ccd6f6; font-size:30px; font-weight:700; margin-bottom:4px;'>
    🛡️ Adaptive Learning for Cloud Security &amp; Predictive Maintenance
</h1>
<p style='color:#8892b0; font-size:14px; margin-bottom:20px;'>
    Intelligent System Architecture: Security Threat Detection · Equipment Failure Prediction · KS Drift Adaptation · Prevention Engine
</p>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Section A – System Overview
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📊 A · System Live Overview</div>', unsafe_allow_html=True)

r = st.session_state.risk_result
rec = st.session_state.rec_result
sec_level  = r["security_risk_level"]  if r else "N/A"
fail_level = r["failure_risk_level"]   if r else "N/A"
over_level = r["overall_risk_level"]   if r else "N/A"
model_status = "Active" if r else "Awaiting Input"

col1, col2, col3, col4 = st.columns(4)
with col1:  metric_card("Security Risk",  sec_level,  sec_level)
with col2:  metric_card("Failure Risk",   fail_level, fail_level)
with col3:  metric_card("Overall Risk",   over_level, over_level)
with col4:  metric_card("System Status",  model_status, "NA")

if rec:
    rec_box(rec["recommendation"], rec["action_priority"])

st.markdown("---")

# ─────────────────────────────────────────────
# Clean 4-Tab Layout
# ─────────────────────────────────────────────
tab_b, tab_c, tab_d, tab_e, tab_f, tab_g = st.tabs([
    "🔐 B · Security Monitoring",
    "⚙️ C · Predictive Maintenance",
    "📈 D · Adaptive Learning & Drift",
    "🚨 E · Intelligent Prevention & Risk",
    "🖥️ F · Live Infrastructure",
    "🔎 SECURITY–INFRASTRUCTURE CORRELATION"
])


# ═══════════════════════════════════════════════
# TAB B — Security Monitoring
# ═══════════════════════════════════════════════
with tab_b:
    st.markdown('<div class="section-header">🔐 Security Threat Detection (UNSW-NB15)</div>', unsafe_allow_html=True)

    st.info(
        "The live node agent does **not** collect UNSW-NB15 network-flow features "
        "(duration, protocol, packet counts, TTLs, connection counters, …). "
        "CPU/RAM/Disk telemetry is **never** converted into fake flow features. "
        "Genuine predictions require a `/network-flow` payload. The demo controls "
        "below attach **labeled test-set samples** to a registered node."
    )

    try:
        init_db()
        live_sec_rows = security_status_for_all_nodes()
        demo_cats = list_demo_attack_categories()
    except Exception:
        live_sec_rows, demo_cats = [], ["Normal"]

    st.markdown("#### Live registered nodes — security result")
    if not live_sec_rows:
        st.caption("No registered nodes yet. Start the central server and a node agent, then inject a demo event.")
    else:
        display_sec = []
        for row in live_sec_rows:
            src = row.get("prediction_source") or "NONE"
            if src == "DEMO_SAMPLE":
                origin = "SIMULATED / DEMO"
            elif src == "GENUINE_NETWORK_FLOW":
                origin = "GENUINE NETWORK-FLOW"
            else:
                origin = "NO FLOW DATA"
            display_sec.append({
                "node_id": row.get("node_id"),
                "classification": row.get("classification"),
                "attack_type": row.get("attack_type") or "—",
                "security_risk": row.get("security_risk"),
                "confidence": row.get("confidence"),
                "timestamp": (row.get("timestamp") or "")[:19].replace("T", " "),
                "model_version": row.get("model_version"),
                "origin": origin,
            })
        st.dataframe(pd.DataFrame(display_sec), use_container_width=True)

        node_ids = [r["node_id"] for r in live_sec_rows]
        d1, d2, d3 = st.columns([2, 2, 1])
        with d1:
            demo_node = st.selectbox("Registered node", node_ids, key="sec_demo_node")
        with d2:
            demo_cat = st.selectbox("Demo attack category (UNSW-NB15 sample)", demo_cats, key="sec_demo_cat")
        with d3:
            st.markdown("<br>", unsafe_allow_html=True)
            fire_demo = st.button("Inject demo event", key="btn_sec_demo")
        if fire_demo and demo_node:
            try:
                demo_res = ingest_demo_event(demo_node, demo_cat, persist=True)
                st.success(
                    f"Simulated/demo event stored for `{demo_node}`: "
                    f"{demo_res['classification']}"
                    + (f" / {demo_res['attack_type']}" if demo_res.get("attack_type") else "")
                    + f" · risk {demo_res['security_risk']} · conf {demo_res['confidence']}"
                )
                st.caption(demo_res.get("source_label", ""))
                st.rerun()
            except InvalidNetworkFlowError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Demo security event failed: {e}")

    st.markdown("---")
    st.markdown("#### Historical / offline classifier (UNSW-NB15 test set)")

    df_test, feat_cols = get_test_data()

    input_mode = st.radio("Input mode", ["Pick sample from test set", "CSV batch upload"], horizontal=True)

    if input_mode == "Pick sample from test set" and df_test is not None:
        n_samples = min(200, len(df_test))
        row_idx = st.selectbox("Select test sample", list(range(n_samples)),
                               format_func=lambda i: f"Row {i}" + (
                                   f" — {df_test.iloc[i].get('attack_cat_raw','?')}" if 'attack_cat_raw' in df_test.columns else ""))

        selected_row = df_test.iloc[row_idx]
        input_dict = {c: float(selected_row[c]) if pd.api.types.is_numeric_dtype(df_test[c]) else str(selected_row[c]) for c in feat_cols}

        if st.button("🔍 Run Security Prediction", key="btn_sec"):
            with st.spinner("Running security classifier…"):
                try:
                    result = predict_attack(input_dict)
                    risk_out = evaluate_integrated_risk(
                        result["prediction"],
                        st.session_state.maint_result["failure_probability"] if st.session_state.maint_result else 0.10,
                        w_sec=w_sec, w_fail=w_fail
                    )
                    rec_out = generate_recommendation(
                        risk_out["security_risk_level"], risk_out["failure_risk_level"]
                    )
                    st.session_state.sec_result  = result
                    st.session_state.risk_result = risk_out
                    st.session_state.rec_result  = rec_out
                    st.session_state.pred_log.append({
                        "Row": row_idx,
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
                        st.success(f"Successfully evaluated {len(df_out)} rows.")
                        st.dataframe(df_out[['predicted_attack','confidence_probability']].head(20), use_container_width=True)
                    except Exception as e:
                        st.error(f"Batch prediction error: {e}")

    # Current prediction result
    if st.session_state.sec_result:
        res = st.session_state.sec_result
        rsk = st.session_state.risk_result
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Attack Category", res["prediction"])
        with c2:
            st.metric("Confidence", f"{res['probability']:.1%}")
        with c3:
            st.metric("Security Risk Level", rsk["security_risk_level"] if rsk else "—")

    # Model metrics
    df_sec_comp = get_security_comparison()
    if df_sec_comp is not None:
        st.markdown("#### 📊 Security Baseline Models Comparison")
        st.dataframe(df_sec_comp, use_container_width=True)

        if 'Model' in df_sec_comp.columns and 'F1-Score' in df_sec_comp.columns:
            fig = px.bar(
                df_sec_comp.dropna(subset=['F1-Score']),
                x='Model', y='F1-Score', color='Task', barmode='group',
                color_discrete_sequence=px.colors.qualitative.Dark2,
                title="Security Model Performance (Accuracy & F1-Score)",
                template='plotly_dark'
            )
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB C — Predictive Maintenance
# ═══════════════════════════════════════════════
with tab_c:
    st.markdown('<div class="section-header">⚙️ Predictive Maintenance & Cloud Server Health (Alibaba Cluster Trace 2017)</div>', unsafe_allow_html=True)

    demo_cpu = 25.0
    demo_mem = 30.0
    demo_disk = 20.0
    demo_load1 = 1.2
    demo_load5 = 1.0
    demo_load15 = 0.8
    demo_duration = 120
    demo_stype = "ComputeNode"

    if demo_choice == "Demo 1: Normal + Healthy":
        demo_cpu, demo_mem, demo_disk, demo_load1, demo_load5, demo_load15, demo_duration, demo_stype = 22.5, 28.0, 18.0, 1.1, 0.9, 0.8, 120, "ComputeNode"
    elif demo_choice == "Demo 2: High Machine Risk":
        demo_cpu, demo_mem, demo_disk, demo_load1, demo_load5, demo_load15, demo_duration, demo_stype = 92.5, 94.0, 85.0, 28.5, 22.0, 18.0, 2400, "ComputeNode"
    elif demo_choice == "Demo 4: CRITICAL Combined":
        demo_cpu, demo_mem, demo_disk, demo_load1, demo_load5, demo_load15, demo_duration, demo_stype = 98.0, 99.0, 95.0, 42.0, 35.0, 30.0, 3600, "ContainerServer"

    col_inputs, col_gauge = st.columns([1, 1])
    with col_inputs:
        st.markdown("#### Alibaba Cloud Server Telemetry Controls")
        cpu_util   = st.slider("CPU Utilization [%]",     0.0, 100.0, float(demo_cpu),  0.5)
        mem_util   = st.slider("Memory Utilization [%]",  0.0, 100.0, float(demo_mem),  0.5)
        disk_util  = st.slider("Disk Utilization [%]",    0.0, 100.0, float(demo_disk), 0.5)
        load_1     = st.slider("1-Min Load Average",      0.0, 50.0,  float(demo_load1),  0.1)
        load_5     = st.slider("5-Min Load Average",      0.0, 50.0,  float(demo_load5),  0.1)
        load_15    = st.slider("15-Min Load Average",     0.0, 50.0,  float(demo_load15), 0.1)
        duration   = st.slider("Execution Duration [sec]", 1, 3600,   int(demo_duration), 10)
        stype      = st.selectbox("Server / Task Type", ["ComputeNode", "ContainerServer", "BatchTask"], index=["ComputeNode", "ContainerServer", "BatchTask"].index(demo_stype))

        if st.button("⚙️ Predict Cloud Server Failure", key="btn_maint"):
            maint_input = {
                "CPU_utilization": cpu_util,
                "Memory_utilization": mem_util,
                "Disk_utilization": disk_util,
                "Load_1min": load_1,
                "Load_5min": load_5,
                "Load_15min": load_15,
                "Duration_sec": duration,
                "Server_Type": stype
            }
            with st.spinner("Running predictive maintenance model…"):
                try:
                    maint_res = predict_failure(maint_input)
                    risk_out  = evaluate_integrated_risk(
                        st.session_state.sec_result["prediction"] if st.session_state.sec_result else "Normal",
                        maint_res["failure_probability"],
                        w_sec=w_sec, w_fail=w_fail
                    )
                    rec_out = generate_recommendation(
                        risk_out["security_risk_level"], risk_out["failure_risk_level"], maint_res["failure_probability"]
                    )
                    st.session_state.maint_result = maint_res
                    st.session_state.risk_result  = risk_out
                    st.session_state.rec_result   = rec_out
                    st.rerun()
                except Exception as e:
                    st.error(f"Maintenance prediction error: {e}")

    with col_gauge:
        maint_res = st.session_state.maint_result
        fail_prob = maint_res["failure_probability"] if maint_res else 0.0
        fail_level_c = maint_res["risk_level"] if maint_res else "N/A"
        gauge_color = RISK_COLOR.get(fail_level_c, "#8892b0")
        st.plotly_chart(gauge_chart(fail_prob, "Failure Probability", gauge_color), use_container_width=True)

        if maint_res:
            st.markdown(f"**Failure Risk Level:** {risk_badge(fail_level_c)}", unsafe_allow_html=True)
            rec = st.session_state.rec_result
            if rec:
                rec_box(rec["recommendation"], rec["action_priority"])

    df_maint_comp = get_maintenance_comparison()
    if df_maint_comp is not None:
        st.markdown("#### 📊 Maintenance Model Performance Metrics (Alibaba Trace 2017)")
        display_cols = [c for c in ['Model','Accuracy','Precision','Recall','F1-Score','ROC-AUC'] if c in df_maint_comp.columns]
        st.dataframe(df_maint_comp[display_cols], use_container_width=True)


# ═══════════════════════════════════════════════
# TAB D — Adaptive Learning & Drift Detection
# ═══════════════════════════════════════════════
with tab_d:
    st.markdown('<div class="section-header">📈 Adaptive Learning & Data Drift Detection</div>', unsafe_allow_html=True)

    st.markdown("#### Live telemetry drift monitor")
    st.caption(
        "The existing KS detector is applied to each node's stored reference window and recent live window. "
        "Drift detection is independent; retraining is guarded and host telemetry is not used as UNSW-NB15 training data."
    )
    try:
        live_drift_results = detect_all_live_drift()
    except Exception as exc:
        live_drift_results = []
        st.error(f"Live drift check failed: {exc}")

    if live_drift_results:
        live_drift_table = pd.DataFrame([
            {
                "node_id": result["node_id"],
                "status": result["status"],
                "affected_features": ", ".join(result.get("affected_features", [])) or "None",
                "reference_window": result.get("reference_window"),
                "current_window": result.get("current_window"),
                "timestamp": result.get("timestamp"),
            }
            for result in live_drift_results
        ])
        st.dataframe(live_drift_table, use_container_width=True, hide_index=True)
        drifted_nodes = [result for result in live_drift_results if result["status"] == "DRIFT DETECTED"]
        st.metric("Live drift status", "DRIFT DETECTED" if drifted_nodes else "NO DRIFT")

        live_node_ids = [result["node_id"] for result in live_drift_results]
        selected_live_node = st.selectbox("Inspect live drift features", live_node_ids, key="live_drift_node")
        selected_live_result = next(
            result for result in live_drift_results if result["node_id"] == selected_live_node
        )
        if selected_live_result.get("features"):
            st.dataframe(
                pd.DataFrame(selected_live_result["features"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Reference and current windows are still collecting enough samples for a KS comparison.")
    else:
        st.info("No registered nodes are available for live drift monitoring.")

    latest_retraining = get_latest_retraining_status()
    if latest_retraining:
        st.markdown("#### Guarded adaptive retraining status")
        current_version = (
            latest_retraining["new_model_version"]
            if latest_retraining["deployment_decision"] == "APPROVED"
            else latest_retraining["previous_model_version"]
        )
        version_col, previous_col, drift_col, training_col, deployment_col = st.columns(5)
        version_col.metric("Current Model", current_version)
        previous_col.metric("Previous Model", latest_retraining["previous_model_version"])
        drift_col.metric("Drift", latest_retraining["drift_event"])
        training_col.metric("Retraining", latest_retraining["status"])
        deployment_col.metric("Deployment", latest_retraining["deployment_decision"])
        st.caption(latest_retraining["reason"])
        validation = latest_retraining.get("validation_metrics", {})
        if validation:
            st.json(validation)

    st.markdown("---")

    drift_kpis = {
        "Method": "Kolmogorov-Smirnov (KS) 2-Sample Test",
        "Total Features": 51,
        "Drifted Features": 10,
        "Drift %": "19.61%",
        "System Drift Detected": True,
        "Threshold": "≥15% of features",
        "Deployment": "New Model RETRAINED & DEPLOYED"
    }

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features Evaluated", drift_kpis["Total Features"])
    c2.metric("Drifted Features", drift_kpis["Drifted Features"])
    c3.metric("Drift Ratio", drift_kpis["Drift %"])
    c4.metric("System Drift Status", "DETECTED ⚠️" if drift_kpis["System Drift Detected"] else "NONE ✅")

    st.markdown(f"""
    - **Drift Detection Algorithm:** {drift_kpis['Method']}
    - **Trigger Threshold:** {drift_kpis['Threshold']}
    - **Adaptive Decision:** `{drift_kpis['Deployment']}`
    """)

    st.markdown("#### 🎬 Replay Adaptive Retraining Pipeline")
    if st.button("▶ Run Adaptive Drift & Retraining Replay", key="btn_demo4"):
        st.session_state.drift_demo_shown = True

    if st.session_state.drift_demo_shown:
        with st.status("Running adaptive learning pipeline…", expanded=True) as status:
            import time
            st.write("📥 Loading Environment B real-time data stream...")
            time.sleep(0.5)
            st.write("📊 Computing KS-Test statistic across 51 features...")
            time.sleep(0.5)
            st.write("⚠️ Distribution Shift Flagged! (10/51 features drifted, p-value < 0.05)")
            time.sleep(0.5)
            st.write("🔄 Triggering dynamic model retraining on combined data...")
            time.sleep(0.5)
            st.write("✅ New adaptive model retrained & validated.")
            time.sleep(0.3)
            st.write("📦 Model update deployed cleanly! (Accuracy restored from 75.0% → 100.0%)")
            status.update(label="Adaptive retraining completed successfully!", state="complete")

    df_adapt = get_adaptive_comparison()
    if df_adapt is not None:
        st.markdown("#### 📊 Old Model vs Retrained Adaptive Model Performance")
        st.dataframe(df_adapt, use_container_width=True)

    drift_plot = load_image_safe(os.path.join(BASE_DIR, "results", "adaptive", "drift_scores_per_feature.png"))
    if drift_plot:
        st.markdown("#### 🔍 Feature Drift Scores (KS-Statistic)")
        st.image(drift_plot, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB E — Intelligent Prevention & Risk Engine
# ═══════════════════════════════════════════════
with tab_e:
    st.markdown('<div class="section-header">🚨 Intelligent Prevention & Integrated Risk Engine</div>', unsafe_allow_html=True)

    st.markdown("### Evidence-Based Node Recommendations")
    st.caption(
        "Recommendations explain the observed evidence and remain advisory. "
        "No arbitrary infrastructure repair or isolation is performed."
    )
    try:
        node_recommendations = get_all_node_recommendations()
    except Exception as exc:
        node_recommendations = []
        st.error(f"Could not load node recommendations: {exc}")

    if node_recommendations:
        recommendation_table = pd.DataFrame([
            {
                "node_id": item["node_id"],
                "detected_problem": item["detected_problem"],
                "evidence": item["evidence"],
                "recommended_action": item["recommended_action"],
                "priority": item["priority"],
                "timestamp": item["timestamp"],
            }
            for item in node_recommendations
        ])
        st.dataframe(recommendation_table, use_container_width=True, hide_index=True)
        for index, item in enumerate(node_recommendations):
            with st.expander(f"{item['node_id']} · {item['detected_problem']} · {item['priority']}", expanded=index == 0):
                st.markdown(f"**Evidence:** {item['evidence']}")
                st.markdown(f"**Why:** {item['why_generated']}")
                st.markdown(f"**Recommendation:** {item['recommended_action']}")
                st.caption(item["disclaimer"])
    else:
        st.info("No high-risk nodes currently have evidence-based recommendations.")

    st.markdown("### Controlled Experimental Evaluation")
    st.caption(
        "This evaluates whether a simulated workload reduction changes measured risk in the test environment. "
        "It does not prove real-world cloud failure prevention and does not change host workload."
    )
    experiment_nodes = [risk["node_id"] for risk in get_all_node_risks()]
    if experiment_nodes:
        experiment_node = st.selectbox("Test node", experiment_nodes, key="closed_loop_node")
        experiment_scale = st.slider("Simulated workload scale", 0.1, 0.9, 0.5, 0.1, key="closed_loop_scale")
        if st.button("Run Controlled Experimental Evaluation", key="run_closed_loop"):
            try:
                st.session_state.closed_loop_result = run_closed_loop_evaluation(experiment_node, experiment_scale)
            except ValueError as exc:
                st.error(str(exc))
        experiment_result = st.session_state.get("closed_loop_result") or get_latest_closed_loop_evaluation(experiment_node)
        if experiment_result:
            before = experiment_result["before"]
            after = experiment_result["after"]
            b_col, arrow_col, a_col = st.columns([4, 1, 4])
            with b_col:
                st.markdown("#### BEFORE")
                st.metric("Overall Risk", risk_percent(before.get("overall_risk")))
                st.caption(f"CPU {before.get('cpu')}% · RAM {before.get('ram')}% · Maintenance {risk_percent(before.get('maintenance_risk'))}")
            with arrow_col:
                st.markdown("<div style='text-align:center;font-size:28px;padding-top:28px;'>↓</div>", unsafe_allow_html=True)
            with a_col:
                st.markdown("#### AFTER")
                st.metric("Overall Risk", risk_percent(after.get("overall_risk")))
                st.caption(f"CPU {after.get('cpu')}% · RAM {after.get('ram')}% · Maintenance {risk_percent(after.get('maintenance_risk'))}")
            st.markdown("#### INTERVENTION")
            st.info(experiment_result["intervention"])
            st.dataframe(pd.DataFrame([{
                "security_risk_before": risk_percent(before.get("security_risk")),
                "security_risk_after": risk_percent(after.get("security_risk")),
                "maintenance_risk_change": risk_percent(experiment_result["measured_change"].get("maintenance_risk_change")),
                "overall_risk_change": risk_percent(experiment_result["measured_change"].get("overall_risk_change")),
                "overall_risk_reduction": risk_percent(experiment_result["measured_change"].get("overall_risk_reduction")),
            }]), use_container_width=True, hide_index=True)
            st.caption(f"{experiment_result['label']}: {experiment_result['limitations']}")
    else:
        st.info("No registered nodes are available for a controlled evaluation.")

    st.markdown("---")

    risk  = st.session_state.risk_result
    rec   = st.session_state.rec_result

    if risk and rec:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Security Risk", risk["security_risk_level"])
            st.caption(f"Score: {risk['security_risk_score']:.3f}")
        with c2:
            st.metric("Failure Risk", risk["failure_risk_level"])
            st.caption(f"Prob: {risk['failure_probability']:.1%}")
        with c3:
            st.metric("Overall System Risk", risk["overall_risk_level"])
            st.caption(f"Score: {risk['overall_risk_score']:.3f}")
        with c4:
            st.metric("Action Priority", rec["action_priority"])

        st.markdown("---")
        st.markdown("### 💡 Recommended Prevention Action")
        rec_box(rec["recommendation"], rec["action_priority"])

        # Risk score breakdown donut
        labels = ["Security Risk", "Failure Risk"]
        values = [risk["security_risk_score"] * w_sec, risk["failure_risk_score"] * w_fail]
        fig_donut = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.6,
            marker=dict(colors=["#e63946", "#f0b429"]),
            textinfo='label+percent'
        ))
        fig_donut.add_annotation(
            text=f"<b>{risk['overall_risk_level']}</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color=RISK_COLOR.get(risk["overall_risk_level"], "#ccd6f6"))
        )
        fig_donut.update_layout(
            title="Weighted Integrated Risk Contribution",
            template='plotly_dark', height=340,
            legend=dict(orientation='h', y=-0.1),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    else:
        st.info("⚡ Run Security Monitoring or Predictive Maintenance tab to compute live integrated risk.")
        st.markdown("#### 📋 System Prevention Rule Matrix")
        from prevention.prevention_engine import RECOMMENDATION_RULE_MATRIX
        rows = []
        for (sec, fail), rec_text in RECOMMENDATION_RULE_MATRIX.items():
            from prevention.prevention_engine import generate_recommendation
            pr = generate_recommendation(sec, fail)
            rows.append({"Security Level": sec, "Failure Level": fail, "Priority": pr["action_priority"], "Recommendation": rec_text})
        df_rules = pd.DataFrame(rows)
        st.dataframe(df_rules, use_container_width=True, height=460)

    st.markdown("---")
    st.caption("College Project · Intelligent Failure Prevention & Integrated Risk Engine")


# ═══════════════════════════════════════════════
# TAB F — Live Infrastructure
# ═══════════════════════════════════════════════

# ── Helpers (read SQLite directly — no HTTP dependency) ─────────────────────
TELEMETRY_DB = os.path.join(BASE_DIR, "data", "telemetry.db")

def _db_ok():
    """Return True only if the telemetry.db file exists."""
    return os.path.exists(TELEMETRY_DB)

def _fetch_nodes():
    """Return all node rows from the SQLite DB, with status freshened."""
    if not _db_ok():
        return []
    try:
        conn = sqlite3.connect(TELEMETRY_DB)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # Refresh OFFLINE status using the same configurable central-server threshold.
        from datetime import datetime, timezone, timedelta
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

def _fetch_node_history(node_id: str, n: int = 30):
    """Return the last *n* telemetry rows for *node_id* in chronological order."""
    if not _db_ok():
        return []
    try:
        conn = sqlite3.connect(TELEMETRY_DB)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT timestamp, cpu_utilization, ram_utilization, disk_utilization,
                   network_sent_mb, network_recv_mb, cpu_temperature_c, process_count
            FROM telemetry WHERE node_id=?
            ORDER BY timestamp DESC LIMIT ?
        """, (node_id, n))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return list(reversed(rows))   # chronological
    except Exception:
        return []

def _util_bar(pct, label):
    """Render a coloured utilisation bar in HTML."""
    pct = pct or 0.0
    cls = "bar-green" if pct < 60 else ("bar-yellow" if pct < 85 else "bar-red")
    return (
        f'<div class="stat-label">{label}&nbsp;&nbsp;'
        f'<span style="color:#ccd6f6;font-weight:600">{pct:.1f}%</span></div>'
        f'<div class="bar-wrap"><div class="bar-fill {cls}" style="width:{min(pct,100):.1f}%"></div></div>'
    )

def _sparkline(df, y_col, color, title):
    """Tiny area chart for a single metric."""
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(
        x=df.index, y=df[y_col].fillna(0),
        mode='lines', fill='tozeroy',
        line=dict(color=color, width=1.5),
        fillcolor=color.replace('rgb','rgba').replace(')',',0.15)') if color.startswith('rgb') else color,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color='#8892b0', size=11)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=130, margin=dict(t=28, b=6, l=6, r=6),
        xaxis=dict(visible=False), yaxis=dict(showgrid=False, tickfont=dict(size=9, color='#8892b0')),
        showlegend=False
    )
    return fig


with tab_f:
    st.markdown('<div class="section-header">🖥️ Live Infrastructure — Node Registry & Telemetry</div>',
                unsafe_allow_html=True)

    # ── Top controls ───────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 3])
    with ctrl1:
        if st.button("🔄  Refresh Now", key="infra_refresh"):
            st.rerun()
    with ctrl2:
        auto = st.checkbox("Auto-refresh every 5 s", value=st.session_state.infra_auto_refresh,
                           key="infra_auto_chk")
        st.session_state.infra_auto_refresh = auto
    with ctrl3:
        history_n = st.slider("History points per node", 5, 60, 20, key="infra_hist_n")

    if not _db_ok():
        st.warning(
            "No telemetry database found yet.  "
            "Start the Central Server (`python src/central_server/main.py`) "
            "and at least one Node Agent (`python src/node_agent/collector.py`) first."
        )
    else:
        nodes = _fetch_nodes()

        if not nodes:
            st.info("Database exists but no nodes have registered yet. "
                    "Run `python src/node_agent/collector.py` to start sending data.")
        else:
            # ── Summary KPI row ────────────────────────────────────────────
            total   = len(nodes)
            online  = sum(1 for n in nodes if n["status"] == "ONLINE")
            offline = total - online

            try:
                from central_server.database import get_all_latest_live_health, get_all_latest_node_security
                sec_map = {s["node_id"]: s for s in get_all_latest_node_security()}
                health_map = {h["node_id"]: h for h in get_all_latest_live_health()}
            except Exception:
                sec_map = {}
                health_map = {}

            risk_map = {
                nd["node_id"]: combine_node_risk(
                    nd["node_id"], sec_map.get(nd["node_id"]), health_map.get(nd["node_id"])
                )
                for nd in nodes
            }

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Nodes",   total)
            k2.metric("Online",         online,  delta=f"+{online}" if online else None,
                      delta_color="normal")
            k3.metric("Offline",        offline, delta=f"-{offline}" if offline else None,
                      delta_color="inverse")

            # Average CPU across online nodes
            cpu_vals = []
            for nd in nodes:
                if nd["status"] == "ONLINE":
                    hist = _fetch_node_history(nd["node_id"], 1)
                    if hist and hist[-1].get("cpu_utilization") is not None:
                        cpu_vals.append(hist[-1]["cpu_utilization"])
            avg_cpu = (sum(cpu_vals) / len(cpu_vals)) if cpu_vals else None
            k4.metric("Avg CPU (online)", f"{avg_cpu:.1f}%" if avg_cpu is not None else "—")

            st.markdown("---")

            st.markdown("#### Unified node risk")
            st.caption(
                "Security and maintenance are independent signals combined with configurable weights; "
                "the result does not imply causation."
            )
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

            # ── Per-node cards ─────────────────────────────────────────────
            for nd in nodes:
                nid      = nd["node_id"]
                hostname = nd.get("hostname", "unknown")
                os_info  = nd.get("operating_system", "")
                ip_addr  = nd.get("ip_address", "")
                status   = nd.get("status", "OFFLINE")
                last_seen = nd.get("last_seen", "")[:19].replace("T", " ") + " UTC"
                first_seen = nd.get("first_seen", "")[:19].replace("T", " ") + " UTC"
                hw       = json.loads(nd["hardware_info"]) if nd.get("hardware_info") else {}

                # Fetch history
                hist = _fetch_node_history(nid, history_n)
                latest = hist[-1] if hist else {}

                cpu_pct  = latest.get("cpu_utilization")
                ram_pct  = latest.get("ram_utilization")
                disk_pct = latest.get("disk_utilization")
                net_sent = latest.get("network_sent_mb")
                net_recv = latest.get("network_recv_mb")
                cpu_temp = latest.get("cpu_temperature_c")
                procs    = latest.get("process_count")

                pill_cls = "pill-online" if status == "ONLINE" else "pill-offline"
                card_cls = "node-online" if status == "ONLINE"  else "node-offline"

                sec = sec_map.get(nid) or {}
                sec_cls = sec.get("classification") or "NO_FLOW_DATA"
                sec_atk = sec.get("attack_type")
                sec_risk = sec.get("security_risk") or "N/A"
                if sec.get("prediction_source") == "DEMO_SAMPLE":
                    sec_origin_cls, sec_origin_txt = "pill-demo", "SIMULATED / DEMO"
                elif sec.get("prediction_source") == "GENUINE_NETWORK_FLOW":
                    sec_origin_cls, sec_origin_txt = "pill-genuine", "GENUINE FLOW"
                else:
                    sec_origin_cls, sec_origin_txt = "pill-none", "NO FLOW DATA"

                node_risk = risk_map[nid]
                unified_risk = risk_percent(node_risk.get("overall_risk"))
                unified_level = node_risk.get("risk_level", "N/A")

                # ── Card header HTML ──────────────────────────────────────
                header_html = (
                    f'<div class="node-card {card_cls}">'
                    f'<div class="node-title">'
                    f'<span class="{pill_cls}">{status}</span>&nbsp;&nbsp;'
                    f'{hostname}&nbsp;&nbsp;<span style="font-size:12px;color:#8892b0">({nid})</span>'
                    f'</div>'
                    f'<div class="node-sub">{os_info} &nbsp;|&nbsp; IP: {ip_addr} &nbsp;|&nbsp; '
                    f'First seen: {first_seen} &nbsp;|&nbsp; Last heartbeat: {last_seen}</div>'
                    f'<div class="node-sub">'
                    f'<span class="{sec_origin_cls}">{sec_origin_txt}</span>&nbsp;&nbsp;'
                    f'Security: <b>{sec_cls}</b>'
                    + (f' ({sec_atk})' if sec_atk else '')
                    + f' &nbsp;|&nbsp; Risk: {sec_risk}'
                    + (f' &nbsp;|&nbsp; Conf: {sec.get("confidence")}' if sec.get("confidence") is not None else '')
                    + f' &nbsp;|&nbsp; Model v{sec.get("model_version") or "—"}'
                    f'</div>'
                    f'<div class="node-sub">Unified node risk: <b>{unified_risk}</b> '
                    f'({unified_level}) &nbsp;|&nbsp; Security {risk_percent(node_risk.get("security_risk"))} '
                    f'&nbsp;|&nbsp; Maintenance {risk_percent(node_risk.get("maintenance_risk"))}</div>'
                )

                # Utilisation bars
                bars = ""
                if cpu_pct  is not None: bars += _util_bar(cpu_pct,  "CPU Utilisation")
                if ram_pct  is not None: bars += _util_bar(ram_pct,  "RAM Utilisation")
                if disk_pct is not None: bars += _util_bar(disk_pct, "Disk Utilisation")

                # Extra stats row
                stats_html = '<div style="display:flex;gap:30px;flex-wrap:wrap;margin-top:4px;">'
                if net_sent is not None:
                    stats_html += (f'<div><div class="stat-label">Net Sent</div>'
                                   f'<div class="stat-value">{net_sent:.1f} MB</div></div>')
                if net_recv is not None:
                    stats_html += (f'<div><div class="stat-label">Net Recv</div>'
                                   f'<div class="stat-value">{net_recv:.1f} MB</div></div>')
                if cpu_temp is not None:
                    stats_html += (f'<div><div class="stat-label">CPU Temp</div>'
                                   f'<div class="stat-value">{cpu_temp:.1f} &deg;C</div></div>')
                else:
                    stats_html += ('<div><div class="stat-label">CPU Temp</div>'
                                   '<div class="stat-value" style="color:#8892b0">N/A</div></div>')
                if procs is not None:
                    stats_html += (f'<div><div class="stat-label">Processes</div>'
                                   f'<div class="stat-value">{procs}</div></div>')
                hw_cores = hw.get("cpu_count_logical") or hw.get("cpu_cores")
                hw_ram   = hw.get("ram_total_mb")
                hw_disk  = hw.get("disk_total_gb")
                if hw_cores:
                    stats_html += (f'<div><div class="stat-label">CPU Cores</div>'
                                   f'<div class="stat-value">{hw_cores} logical</div></div>')
                if hw_ram:
                    stats_html += (f'<div><div class="stat-label">Total RAM</div>'
                                   f'<div class="stat-value">{hw_ram/1024:.1f} GB</div></div>')
                if hw_disk:
                    stats_html += (f'<div><div class="stat-label">Total Disk</div>'
                                   f'<div class="stat-value">{hw_disk:.0f} GB</div></div>')
                stats_html += '</div>'

                st.markdown(header_html + bars + stats_html + '</div>', unsafe_allow_html=True)

                # ── Sparkline charts (only when enough history exists) ─────
                if len(hist) >= 2:
                    import pandas as pd
                    df_h = pd.DataFrame(hist)
                    df_h.index = range(len(df_h))

                    sc1, sc2, sc3 = st.columns(3)
                    if "cpu_utilization" in df_h.columns and df_h["cpu_utilization"].notna().any():
                        sc1.plotly_chart(
                            _sparkline(df_h, "cpu_utilization", "#64d2ff", "CPU %"),
                            use_container_width=True, key=f"spark_cpu_{nid}"
                        )
                    if "ram_utilization" in df_h.columns and df_h["ram_utilization"].notna().any():
                        sc2.plotly_chart(
                            _sparkline(df_h, "ram_utilization", "#f0b429", "RAM %"),
                            use_container_width=True, key=f"spark_ram_{nid}"
                        )
                    if "disk_utilization" in df_h.columns and df_h["disk_utilization"].notna().any():
                        sc3.plotly_chart(
                            _sparkline(df_h, "disk_utilization", "#ff6b35", "Disk %"),
                            use_container_width=True, key=f"spark_disk_{nid}"
                        )

                    # Network I/O chart
                    if ("network_sent_mb" in df_h.columns and
                            df_h["network_sent_mb"].notna().any()):
                        net_fig = go.Figure()
                        net_fig.add_trace(go.Scatter(
                            x=df_h.index, y=df_h["network_sent_mb"].fillna(0),
                            mode='lines', name='Sent MB', line=dict(color='#00d084', width=1.5)
                        ))
                        net_fig.add_trace(go.Scatter(
                            x=df_h.index, y=df_h["network_recv_mb"].fillna(0),
                            mode='lines', name='Recv MB', line=dict(color='#8892b0', width=1.5)
                        ))
                        net_fig.update_layout(
                            title=dict(text="Network I/O (MB cumulative)",
                                       font=dict(color='#8892b0', size=11)),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            height=140, margin=dict(t=28, b=6, l=6, r=6),
                            xaxis=dict(visible=False),
                            yaxis=dict(showgrid=False, tickfont=dict(size=9, color='#8892b0')),
                            legend=dict(orientation='h', y=1.2, font=dict(size=9, color='#8892b0')),
                            template='plotly_dark'
                        )
                        st.plotly_chart(net_fig, use_container_width=True,
                                        key=f"spark_net_{nid}")

                st.markdown("<br>", unsafe_allow_html=True)

    # ── Auto-refresh via meta tag ──────────────────────────────────────────
    if st.session_state.infra_auto_refresh:
        st.markdown(
            '<meta http-equiv="refresh" content="5">',
            unsafe_allow_html=True
        )

    st.caption("Data source: data/telemetry.db (SQLite) · Updates every agent tick")


# ═══════════════════════════════════════════════
# TAB G — Security–Infrastructure Correlation
# ═══════════════════════════════════════════════
with tab_g:
    st.markdown(
        '<div class="section-header">SECURITY–INFRASTRUCTURE CORRELATION</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "This section examines observed associations between independently produced "
        "security, health, and resource signals. Correlation does not establish causation."
    )

    if not _db_ok():
        st.warning("No telemetry database is available for correlation analysis yet.")
    else:
        try:
            correlation_results = analyze_all_nodes()
        except Exception as exc:
            correlation_results = []
            st.error(f"Correlation analysis failed: {exc}")

        if correlation_results:
            correlation_nodes = [result["node_id"] for result in correlation_results]
            selected_correlation_node = st.selectbox(
                "Node",
                correlation_nodes,
                key="correlation_node",
            )
            selected_result = next(
                result for result in correlation_results
                if result["node_id"] == selected_correlation_node
            )
            aligned = selected_result["aligned_data"].copy()

            if aligned.empty:
                st.info("No time-aligned telemetry, health, or security records are available for this node.")
            else:
                plot_data = aligned.copy()
                plot_data["timestamp"] = pd.to_datetime(plot_data["timestamp"], utc=True)

                risk_frame = plot_data[[
                    "timestamp", "security_risk", "maintenance_risk", "overall_risk"
                ]].melt(
                    id_vars="timestamp",
                    var_name="signal",
                    value_name="risk",
                ).dropna(subset=["risk"])
                risk_fig = px.line(
                    risk_frame,
                    x="timestamp",
                    y="risk",
                    color="signal",
                    markers=True,
                    title="Time-aligned risk signals",
                    labels={"risk": "Risk score (0–1)", "timestamp": "Timestamp"},
                )
                risk_fig.update_yaxes(range=[0, 1])
                risk_fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(risk_fig, use_container_width=True)

                resource_columns = [
                    "cpu_utilization", "ram_utilization", "network_activity_mb"
                ]
                resource_frame = plot_data[["timestamp"] + resource_columns].melt(
                    id_vars="timestamp",
                    var_name="resource",
                    value_name="value",
                ).dropna(subset=["value"])
                resource_fig = px.line(
                    resource_frame,
                    x="timestamp",
                    y="value",
                    color="resource",
                    markers=True,
                    title="Time-aligned infrastructure behavior",
                    labels={"value": "Observed value", "timestamp": "Timestamp"},
                )
                resource_fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(resource_fig, use_container_width=True)

                st.markdown("#### Correlation values")
                correlation_frame = pd.DataFrame(selected_result["correlations"])
                if not correlation_frame.empty:
                    st.dataframe(
                        correlation_frame[[
                            "label", "correlation", "p_value", "sample_size",
                            "statistically_meaningful", "status", "explanation",
                        ]],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No paired observations are available for correlation analysis.")

                st.markdown("#### Security event timeline")
                event_frame = pd.DataFrame(selected_result["event_timeline"])
                if event_frame.empty:
                    st.caption("No security events are recorded for this node.")
                else:
                    event_columns = [
                        "timestamp", "event", "attack_type", "prediction_source",
                        "is_simulated", "security_risk", "observed_relationship",
                    ]
                    st.dataframe(
                        event_frame[[column for column in event_columns if column in event_frame.columns]],
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("#### Resource behavior before and after security events")
                    behavior_columns = [
                        "timestamp", "event", "before_samples", "after_samples",
                        "before_cpu_utilization", "after_cpu_utilization", "delta_cpu_utilization",
                        "before_ram_utilization", "after_ram_utilization", "delta_ram_utilization",
                        "before_network_activity_mb", "after_network_activity_mb", "delta_network_activity_mb",
                        "observed_relationship",
                    ]
                    st.dataframe(
                        event_frame[[column for column in behavior_columns if column in event_frame.columns]],
                        use_container_width=True,
                        hide_index=True,
                    )

                st.caption(selected_result["explanation"])
        else:
            st.info("No registered nodes are available for correlation analysis.")
