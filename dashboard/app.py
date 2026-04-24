"""
NetSentinel-RL — Real-time Monitoring Dashboard
Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os
from pathlib import Path

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="NetSentinel-RL",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar config ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ NetSentinel-RL")
    st.caption("Autonomous Network Threat Response")
    st.divider()
    simulation_speed = st.slider("Simulation speed (flows/sec)", 1, 50, 10)
    show_shap = st.toggle("Show SHAP explanations", True)
    show_marl = st.toggle("Show MARL agent panel", True)
    st.divider()
    st.subheader("System Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.metric("Detector", "✅ Live")
        st.metric("Classifier", "✅ Live")
    with status_col2:
        st.metric("Responder", "✅ Live")
        st.metric("LLM Orch.", "✅ Live")

# ── Synthetic real-time data generator ───────────────────────────────────────
ATTACK_TYPES  = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "Botnet", "WebAttack"]
ACTIONS       = ["no_action", "alert_soc", "rate_limit", "block_ip", "deep_inspect"]
SEVERITY_COLOURS = {"low": "#4CAF50", "medium": "#FF9800", "high": "#F44336", "critical": "#9C27B0"}


@st.cache_data(ttl=0)
def generate_flow_batch(n: int = 20, seed: int = None) -> pd.DataFrame:
    """Generate a synthetic batch of network flows for demo."""
    if seed:
        np.random.seed(seed)
    labels = np.random.choice(
        ATTACK_TYPES,
        size=n,
        p=[0.70, 0.07, 0.07, 0.06, 0.04, 0.03, 0.03],
    )
    confidences = np.where(
        labels != "Benign",
        np.random.uniform(0.65, 0.99, n),
        np.random.uniform(0.80, 0.99, n),
    )
    actions = []
    for label, conf in zip(labels, confidences):
        if label == "Benign":
            actions.append("no_action")
        elif conf > 0.90:
            actions.append(np.random.choice(["block_ip", "deep_inspect"]))
        elif conf > 0.70:
            actions.append(np.random.choice(["rate_limit", "alert_soc"]))
        else:
            actions.append("alert_soc")

    severities = []
    for conf, label in zip(confidences, labels):
        if label == "Benign":
            severities.append("low")
        elif conf > 0.90:
            severities.append("critical" if label in ["DDoS", "DoS"] else "high")
        elif conf > 0.70:
            severities.append("high")
        else:
            severities.append("medium")

    return pd.DataFrame({
        "timestamp":    pd.date_range("now", periods=n, freq="100ms"),
        "src_ip":       [f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n)],
        "threat_type":  labels,
        "confidence":   np.round(confidences, 3),
        "action":       actions,
        "severity":     severities,
        "packets_per_s": np.random.exponential(100, n).astype(int),
        "bytes_rate":    np.random.exponential(50000, n).astype(int),
        "syn_flags":     np.random.binomial(1, 0.1, n),
        "latency_ms":    np.round(np.random.exponential(45, n), 1),
        "shap_top_feat": np.random.choice(
            ["SYN_Flag_Count", "Flow_Packets/s", "Flow_IAT_Mean", "Down/Up_Ratio", "Avg_Packet_Size"],
            size=n,
        ),
    })


# ── Session state for accumulating history ────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()
if "tick" not in st.session_state:
    st.session_state.tick = 0

# ── Main dashboard ────────────────────────────────────────────────────────────
st.title("🛡️ NetSentinel-RL — Live Monitoring Dashboard")

# Auto-refresh
placeholder = st.empty()

with placeholder.container():

    # Generate new batch
    new_flows = generate_flow_batch(n=simulation_speed, seed=st.session_state.tick)
    st.session_state.history = pd.concat(
        [st.session_state.history, new_flows], ignore_index=True
    ).tail(500)
    st.session_state.tick += 1

    history = st.session_state.history
    recent  = history.tail(100)

    # ── Row 1: KPI metrics ────────────────────────────────────────────────
    kpi_cols = st.columns(6)
    total        = len(history)
    threats      = (history["threat_type"] != "Benign").sum()
    threat_rate  = threats / max(1, total)
    mean_lat     = history["latency_ms"].mean()
    blocked      = (history["action"] == "block_ip").sum()
    critical_n   = (history["severity"] == "critical").sum()

    kpi_cols[0].metric("Flows analysed",  f"{total:,}")
    kpi_cols[1].metric("Threats detected", f"{threats:,}", delta=f"{threat_rate:.1%}")
    kpi_cols[2].metric("Blocked",          f"{blocked:,}")
    kpi_cols[3].metric("Critical alerts",  f"{critical_n:,}")
    kpi_cols[4].metric("Mean latency",     f"{mean_lat:.1f}ms")
    kpi_cols[5].metric("False positive est.", "~1.1%")

    st.divider()

    # ── Row 2: Threat timeline + distribution ─────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Threat timeline")
        timeline_data = history.copy()
        timeline_data["is_threat"] = (timeline_data["threat_type"] != "Benign").astype(int)
        rolling = timeline_data["is_threat"].rolling(20).mean().fillna(0)
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=list(range(len(rolling))),
            y=rolling,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(244,67,54,0.15)",
            line=dict(color="#F44336", width=2),
            name="Threat rate",
        ))
        fig_timeline.update_layout(
            height=200, margin=dict(l=0, r=0, t=10, b=10),
            xaxis_title="Flow sequence", yaxis_title="Rate",
            yaxis=dict(range=[0, 1]),
            showlegend=False,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    with col_right:
        st.subheader("Attack type breakdown")
        threat_only = history[history["threat_type"] != "Benign"]
        if len(threat_only) > 0:
            type_counts = threat_only["threat_type"].value_counts()
            fig_pie = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set2,
                height=200,
            )
            fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=10), showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No threats detected yet.")

    # ── Row 3: MARL agent panel ────────────────────────────────────────────
    if show_marl:
        st.divider()
        st.subheader("Multi-Agent RL — Response Distribution")
        agent_cols = st.columns(3)

        with agent_cols[0]:
            st.markdown("**Detector Agent**")
            det_benign = (history["threat_type"] == "Benign").sum()
            det_threat = (history["threat_type"] != "Benign").sum()
            fig_det = go.Figure(go.Bar(
                x=["Benign", "Threat"],
                y=[det_benign, det_threat],
                marker_color=["#4CAF50", "#F44336"],
            ))
            fig_det.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_det, use_container_width=True)

        with agent_cols[1]:
            st.markdown("**Classifier Agent — Confidence Distribution**")
            fig_conf = px.histogram(
                recent, x="confidence", nbins=20,
                color_discrete_sequence=["#2196F3"],
            )
            fig_conf.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_conf, use_container_width=True)

        with agent_cols[2]:
            st.markdown("**Responder Agent — Action Distribution**")
            action_counts = history["action"].value_counts()
            fig_act = go.Figure(go.Bar(
                x=action_counts.index.tolist(),
                y=action_counts.values.tolist(),
                marker_color=["#607D8B", "#FF9800", "#2196F3", "#F44336", "#9C27B0"],
            ))
            fig_act.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_act, use_container_width=True)

    # ── Row 4: Live alert feed ────────────────────────────────────────────
    st.divider()
    st.subheader("Live Alert Feed (most recent 10 threats)")

    threats_df = history[history["threat_type"] != "Benign"].tail(10).iloc[::-1]
    if len(threats_df) > 0:
        for _, row in threats_df.iterrows():
            sev_colour = SEVERITY_COLOURS.get(row["severity"], "#607D8B")
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 2, 3])
                c1.markdown(f"**{row['src_ip']}**")
                c2.markdown(f"🔴 `{row['threat_type']}`")
                c3.markdown(f"{row['confidence']:.0%}")
                c4.markdown(f"`{row['action']}`")
                if show_shap:
                    c5.markdown(f"Top feature: `{row['shap_top_feat']}`")
    else:
        st.success("No threats detected in recent window. All traffic nominal.")

    # ── Row 5: Latency chart ──────────────────────────────────────────────
    st.divider()
    st.subheader("System Latency (rolling P50 / P95)")
    lat_rolling_p50 = history["latency_ms"].rolling(50).median().bfill()
    lat_rolling_p95 = (
    history["latency_ms"]
    .rolling(50)
    .quantile(0.95)
    .bfill()
)

    fig_lat = go.Figure()
    fig_lat.add_trace(go.Scatter(y=lat_rolling_p50, name="P50", line=dict(color="#4CAF50")))
    fig_lat.add_trace(go.Scatter(y=lat_rolling_p95, name="P95", line=dict(color="#FF9800")))
    fig_lat.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="100ms SLA")
    fig_lat.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=10), yaxis_title="ms")
    st.plotly_chart(fig_lat, use_container_width=True)

# Auto-refresh
time.sleep(1.0 / max(1, simulation_speed // 5))
st.rerun()
