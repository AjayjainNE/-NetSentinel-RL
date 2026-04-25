"""
NetSentinel-RL — Real-Time Network Threat Monitoring Dashboard
Streamlit application with Grafana-inspired dark theme and live telemetry.
Run: streamlit run dashboard/app.py
"""

import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NetSentinel-RL",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Grafana-inspired colour palette ──────────────────────────────────────────

PALETTE = {
    "bg_primary":    "#111217",
    "bg_panel":      "#181b1f",
    "bg_border":     "#23262d",
    "text_primary":  "#d0d5dd",
    "text_muted":    "#6b7280",
    "accent_blue":   "#3d9ef5",
    "accent_green":  "#73bf69",
    "accent_yellow": "#f0a948",
    "accent_red":    "#f2495c",
    "accent_purple": "#b877d9",
    "accent_cyan":   "#37a99a",
}

SEV_COLOUR = {
    "low":      PALETTE["accent_green"],
    "medium":   PALETTE["accent_yellow"],
    "high":     PALETTE["accent_red"],
    "critical": PALETTE["accent_purple"],
}

ATTACK_COLOUR = {
    "Benign":     PALETTE["accent_green"],
    "DoS":        PALETTE["accent_yellow"],
    "DDoS":       PALETTE["accent_red"],
    "PortScan":   PALETTE["accent_blue"],
    "BruteForce": PALETTE["accent_purple"],
    "Botnet":     "#e07010",
    "WebAttack":  PALETTE["accent_cyan"],
    "Infiltration": "#ff6b9d",
}

ATTACK_TYPES = list(ATTACK_COLOUR.keys())
ACTIONS      = ["no_action", "alert_soc", "rate_limit", "block_ip", "deep_inspect"]

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #111217;
    color: #d0d5dd;
}

/* Remove Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 0.5rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #181b1f !important;
    border-right: 1px solid #23262d;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption { color: #9ca3af !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #181b1f;
    border: 1px solid #23262d;
    border-radius: 4px;
    padding: 0.75rem 1rem;
}
div[data-testid="metric-container"] label { color: #6b7280 !important; font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #d0d5dd; font-size: 1.5rem; font-weight: 600; }
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { font-size: 0.75rem; }

/* Section header style */
.panel-title {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.5rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #23262d;
}

/* Alert row */
.alert-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.45rem 0.75rem;
    background: #181b1f;
    border-left: 3px solid #f2495c;
    border-radius: 0 4px 4px 0;
    margin-bottom: 4px;
    font-size: 0.8rem;
    font-family: 'DM Mono', monospace;
}
.alert-row.medium { border-left-color: #f0a948; }
.alert-row.low    { border-left-color: #73bf69; }
.alert-row.critical { border-left-color: #b877d9; }

.badge {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 3px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Dividers */
hr { border-color: #23262d !important; margin: 0.75rem 0 !important; }

/* Plotly transparent bg */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly layout base ────────────────────────────────────────────────────────

def base_layout(height=200, **kwargs):
    return dict(
        height=height,
        margin=dict(l=0, r=0, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#13161b",
        font=dict(family="IBM Plex Sans", color="#9ca3af", size=10),
        xaxis=dict(gridcolor="#1f2329", zeroline=False, tickfont=dict(size=9)),
        yaxis=dict(gridcolor="#1f2329", zeroline=False, tickfont=dict(size=9)),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        **kwargs,
    )

# ── Synthetic data generator ──────────────────────────────────────────────────

@st.cache_data(ttl=0)
def generate_batch(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probs = [0.68, 0.07, 0.07, 0.06, 0.04, 0.03, 0.03, 0.02]
    labels = rng.choice(ATTACK_TYPES, size=n, p=probs)

    confs = np.where(
        labels != "Benign",
        rng.uniform(0.62, 0.99, n),
        rng.uniform(0.82, 0.99, n),
    )

    actions = []
    for lbl, c in zip(labels, confs):
        if lbl == "Benign":
            actions.append("no_action")
        elif c > 0.90:
            actions.append(rng.choice(["block_ip", "deep_inspect"]))
        elif c > 0.72:
            actions.append(rng.choice(["rate_limit", "alert_soc"]))
        else:
            actions.append("alert_soc")

    severities = []
    for lbl, c in zip(labels, confs):
        if lbl == "Benign":
            severities.append("low")
        elif c > 0.90 and lbl in ("DDoS", "DoS", "Infiltration"):
            severities.append("critical")
        elif c > 0.80:
            severities.append("high")
        elif c > 0.65:
            severities.append("medium")
        else:
            severities.append("low")

    shap_feats = [
        "SYN_Flag_Count", "Flow_Packets/s", "Flow_IAT_Mean",
        "Down/Up_Ratio",  "Avg_Packet_Size", "Flow_Duration",
        "Bwd_IAT_Mean",   "RST_Flag_Count",
    ]

    return pd.DataFrame({
        "timestamp":      pd.date_range("now", periods=n, freq="100ms"),
        "src_ip":         [f"10.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}" for _ in range(n)],
        "dst_port":       rng.choice([80, 443, 22, 3306, 53, 8080, 445], size=n),
        "threat_type":    labels,
        "confidence":     np.round(confs, 3),
        "action":         actions,
        "severity":       severities,
        "packets_per_s":  rng.exponential(120, n).astype(int),
        "bytes_rate":     rng.exponential(55_000, n).astype(int),
        "flow_duration":  np.round(rng.exponential(1.8, n), 3),
        "syn_ratio":      np.round(rng.beta(0.5, 5, n), 4),
        "latency_ms":     np.round(rng.exponential(40, n), 1),
        "shap_top_feat":  rng.choice(shap_feats, size=n),
        "shap_importance": np.round(rng.uniform(0.08, 0.45, n), 3),
        "agent_id":       rng.choice(["detector-1", "detector-2", "classifier-1"], size=n),
    })


# ── Session state ─────────────────────────────────────────────────────────────

for key, val in [("history", pd.DataFrame()), ("tick", 0), ("paused", False)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        <span style="font-size:1.4rem;color:#3d9ef5;">⬡</span>
        <span style="font-size:1.05rem;font-weight:600;color:#d0d5dd;letter-spacing:0.02em;">NetSentinel<span style="color:#3d9ef5;">RL</span></span>
    </div>
    <p style="font-size:0.68rem;color:#6b7280;margin:0 0 1rem 0;letter-spacing:0.05em;text-transform:uppercase;">Autonomous Threat Response Platform</p>
    """, unsafe_allow_html=True)

    st.divider()

    sim_speed    = st.slider("Flows per cycle", 5, 60, 12, key="speed")
    show_shap    = st.toggle("SHAP explanations", True)
    show_marl    = st.toggle("MARL agent panel", True)
    show_geo     = st.toggle("Source analytics", True)
    auto_refresh = st.toggle("Auto-refresh", True)

    st.divider()
    st.markdown('<p class="panel-title">Agent Health</p>', unsafe_allow_html=True)

    agents = {
        "Detector":    ("●", "#73bf69"),
        "Classifier":  ("●", "#73bf69"),
        "Responder":   ("●", "#73bf69"),
        "LLM Orch.":   ("●", "#f0a948"),
        "MLflow":      ("●", "#73bf69"),
        "Prometheus":  ("●", "#73bf69"),
    }
    for name, (dot, col) in agents.items():
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;">'
            f'<span style="font-size:0.77rem;color:#9ca3af;">{name}</span>'
            f'<span style="color:{col};font-size:0.65rem;">{dot} {"LIVE" if col == "#73bf69" else "WARN"}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("⟳  Reset history", use_container_width=True):
        st.session_state.history = pd.DataFrame()
        st.session_state.tick    = 0

# ── Header ────────────────────────────────────────────────────────────────────

now_str = pd.Timestamp.now().strftime("%Y-%m-%d  %H:%M:%S UTC")
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(
        '<h2 style="margin:0;font-size:1.15rem;font-weight:600;letter-spacing:0.01em;color:#d0d5dd;">'
        '⬡ NetSentinel-RL — Live Operations Dashboard</h2>',
        unsafe_allow_html=True,
    )
with h2:
    st.markdown(
        f'<p style="text-align:right;font-size:0.7rem;font-family:DM Mono,monospace;color:#6b7280;margin:0.5rem 0 0 0;">{now_str}</p>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Data generation ───────────────────────────────────────────────────────────

placeholder = st.empty()

with placeholder.container():

    if not st.session_state.paused:
        batch = generate_batch(n=sim_speed, seed=st.session_state.tick)
        st.session_state.history = pd.concat(
            [st.session_state.history, batch], ignore_index=True
        ).tail(800)
        st.session_state.tick += 1

    hist   = st.session_state.history
    recent = hist.tail(100) if len(hist) >= 100 else hist

    if hist.empty:
        st.info("Waiting for data…")
        time.sleep(0.5)
        st.rerun()

    # ── KPI Row ───────────────────────────────────────────────────────────

    total       = len(hist)
    threats     = int((hist["threat_type"] != "Benign").sum())
    threat_rate = threats / max(1, total)
    blocked     = int((hist["action"] == "block_ip").sum())
    critical_n  = int((hist["severity"] == "critical").sum())
    mean_lat    = hist["latency_ms"].mean()
    p95_lat     = hist["latency_ms"].quantile(0.95)
    fp_est      = round((hist["action"] == "alert_soc").sum() / max(1, threats) * 0.08, 3)

    k = st.columns(7)
    k[0].metric("Total Flows",     f"{total:,}")
    k[1].metric("Threats Detected", f"{threats:,}",  delta=f"{threat_rate:.1%} rate")
    k[2].metric("Blocked",          f"{blocked:,}")
    k[3].metric("Critical Alerts",  f"{critical_n:,}")
    k[4].metric("P50 Latency",      f"{mean_lat:.1f} ms")
    k[5].metric("P95 Latency",      f"{p95_lat:.1f} ms")
    k[6].metric("Est. FPR",         f"{fp_est:.2%}")

    st.divider()

    # ── Row 2: Timeline + threat mix ─────────────────────────────────────

    col_t, col_d, col_a = st.columns([3, 1.2, 1.2])

    with col_t:
        st.markdown('<p class="panel-title">Threat Rate — Rolling 30-flow window</p>', unsafe_allow_html=True)
        is_threat = (hist["threat_type"] != "Benign").astype(float)
        roll30    = is_threat.rolling(30).mean().fillna(0)
        roll10    = is_threat.rolling(10).mean().fillna(0)

        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(
            y=roll30, mode="lines", name="30-flow avg",
            fill="tozeroy", fillcolor="rgba(242,73,92,0.08)",
            line=dict(color=PALETTE["accent_red"], width=1.5),
        ))
        fig_tl.add_trace(go.Scatter(
            y=roll10, mode="lines", name="10-flow avg",
            line=dict(color=PALETTE["accent_yellow"], width=1, dash="dot"),
        ))
        fig_tl.add_hline(y=0.30, line_dash="dash", line_color="#444", line_width=1,
                         annotation_text="30% threshold", annotation_font_size=8)
        fig_tl.update_layout(**base_layout(240), yaxis_range=[0, 1],
                              xaxis_title="Flow sequence", yaxis_title="Threat rate")
        st.plotly_chart(fig_tl, use_container_width=True)

    with col_d:
        st.markdown('<p class="panel-title">Attack Distribution</p>', unsafe_allow_html=True)
        threat_only = hist[hist["threat_type"] != "Benign"]
        if not threat_only.empty:
            tc = threat_only["threat_type"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=tc.index.tolist(),
                values=tc.values.tolist(),
                marker_colors=[ATTACK_COLOUR.get(t, "#888") for t in tc.index],
                textfont=dict(size=9),
                hole=0.55,
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            ))
            fig_pie.update_layout(**{**base_layout(240),"margin": dict(l=0, r=0, t=20, b=0)})
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.success("No threats in current window.")

    with col_a:
        st.markdown('<p class="panel-title">Responder Actions</p>', unsafe_allow_html=True)
        ac = hist["action"].value_counts()
        action_colours = {
            "no_action":    PALETTE["text_muted"],
            "alert_soc":    PALETTE["accent_yellow"],
            "rate_limit":   PALETTE["accent_blue"],
            "block_ip":     PALETTE["accent_red"],
            "deep_inspect": PALETTE["accent_purple"],
        }
        fig_act = go.Figure(go.Bar(
            x=ac.values.tolist(),
            y=ac.index.tolist(),
            orientation="h",
            marker_color=[action_colours.get(a, "#888") for a in ac.index],
            text=ac.values.tolist(),
            textposition="outside",
            textfont=dict(size=8),
        ))
        layout = base_layout(240)
        layout.pop("margin", None)

        fig_act.update_layout(**layout,xaxis_title="Count",margin=dict(l=0, r=40, t=20, b=10))
        st.plotly_chart(fig_act, use_container_width=True)

    # ── Row 3: Latency + confidence + MARL ───────────────────────────────

    st.divider()

    col_l, col_c, col_m = st.columns([2, 1.5, 1.5])

    with col_l:
        st.markdown('<p class="panel-title">System Latency — P50 / P95 / SLA</p>', unsafe_allow_html=True)
        p50s = hist["latency_ms"].rolling(50).median().bfill()
        p95s = hist["latency_ms"].rolling(50).quantile(0.95).bfill()
        fig_lat = go.Figure()
        fig_lat.add_trace(go.Scatter(y=p50s, name="P50", mode="lines",
                                     line=dict(color=PALETTE["accent_green"], width=1.5)))
        fig_lat.add_trace(go.Scatter(y=p95s, name="P95", mode="lines",
                                     line=dict(color=PALETTE["accent_yellow"], width=1.5)))
        fig_lat.add_hrect(y0=100, y1=p95s.max() * 1.1 + 1,
                          fillcolor="rgba(242,73,92,0.05)", line_width=0)
        fig_lat.add_hline(y=100, line_dash="dash", line_color=PALETTE["accent_red"],
                          line_width=1, annotation_text="100ms SLA", annotation_font_size=8)
        fig_lat.update_layout(**base_layout(200), yaxis_title="ms")
        st.plotly_chart(fig_lat, use_container_width=True)

    with col_c:
        st.markdown('<p class="panel-title">Classifier Confidence</p>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        for lbl in ATTACK_TYPES[:4]:
            sub = hist[hist["threat_type"] == lbl]["confidence"]
            if len(sub) >= 3:
                fig_hist.add_trace(go.Histogram(
                    x=sub, name=lbl, nbinsx=15, opacity=0.72,
                    marker_color=ATTACK_COLOUR.get(lbl, "#888"),
                ))
        fig_hist.update_layout(**base_layout(200), barmode="overlay",
                                xaxis_title="Confidence", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_m:
        if show_marl:
            st.markdown('<p class="panel-title">MARL — Detector Decisions</p>', unsafe_allow_html=True)
            benign_n = int((hist["threat_type"] == "Benign").sum())
            threat_n = int((hist["threat_type"] != "Benign").sum())
            fig_det = go.Figure(go.Bar(
                x=["Benign", "Threat"],
                y=[benign_n, threat_n],
                marker_color=[PALETTE["accent_green"], PALETTE["accent_red"]],
                text=[benign_n, threat_n],
                textposition="outside",
                textfont=dict(size=9),
            ))
            fig_det.update_layout(**base_layout(200), yaxis_title="Flows")
            st.plotly_chart(fig_det, use_container_width=True)
        else:
            st.empty()

    # ── Row 4: SHAP + source analytics ───────────────────────────────────

    st.divider()

    col_s, col_g = st.columns([1, 1])

    with col_s:
        if show_shap:
            st.markdown('<p class="panel-title">SHAP — Mean Feature Importance (threats only)</p>', unsafe_allow_html=True)
            t_only = hist[hist["threat_type"] != "Benign"]
            if not t_only.empty:
                shap_agg = (
                    t_only.groupby("shap_top_feat")["shap_importance"]
                    .mean()
                    .sort_values(ascending=True)
                )
                fig_shap = go.Figure(go.Bar(
                    x=shap_agg.values,
                    y=shap_agg.index.tolist(),
                    orientation="h",
                    marker=dict(
                        color=shap_agg.values,
                        colorscale=[[0, "#1f2d3d"], [0.5, "#3d9ef5"], [1, "#b877d9"]],
                        showscale=False,
                    ),
                    text=[f"{v:.3f}" for v in shap_agg.values],
                    textposition="outside",
                    textfont=dict(size=8),
                ))
                fig_shap.update_layout(**base_layout(220),xaxis_title="Mean |SHAP|")
                st.plotly_chart(fig_shap, use_container_width=True)

    with col_g:
        if show_geo:
            st.markdown('<p class="panel-title">Top Source IPs — Threat Events</p>', unsafe_allow_html=True)
            src_threats = hist[hist["threat_type"] != "Benign"]["src_ip"].value_counts().head(12)
            if not src_threats.empty:
                fig_src = go.Figure(go.Bar(
                    x=src_threats.values,
                    y=src_threats.index.tolist(),
                    orientation="h",
                    marker_color=PALETTE["accent_red"],
                    opacity=0.85,
                    text=src_threats.values,
                    textposition="outside",
                    textfont=dict(size=8),
                ))
                layout = base_layout(220)

                layout["yaxis"] = dict(gridcolor="#1f2329", zeroline=False,tickfont=dict(size=8, family="DM Mono"))

                fig_src.update_layout(**layout,xaxis_title="Threat events"
                                      )
                st.plotly_chart(fig_src, use_container_width=True)

    # ── Row 5: Severity heatmap ───────────────────────────────────────────

    st.divider()
    col_h, col_sev = st.columns([2, 1])

    with col_h:
        st.markdown('<p class="panel-title">Threat Type × Severity — Heatmap</p>', unsafe_allow_html=True)
        sev_order = ["low", "medium", "high", "critical"]
        thr_only  = hist[hist["threat_type"] != "Benign"]
        if len(thr_only) >= 10:
            pivot = (
                thr_only.groupby(["threat_type", "severity"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=sev_order, fill_value=0)
            )
            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=sev_order,
                y=pivot.index.tolist(),
                colorscale=[[0, "#13161b"], [0.3, "#2d1f2a"], [0.6, "#8b2a3a"], [1.0, "#f2495c"]],
                text=pivot.values,
                texttemplate="%{text}",
                textfont=dict(size=9),
                showscale=False,
            ))
            fig_hm.update_layout(**base_layout(210), margin=dict(l=0, r=0, t=20, b=10))
            st.plotly_chart(fig_hm, use_container_width=True)

    with col_sev:
        st.markdown('<p class="panel-title">Severity Breakdown</p>', unsafe_allow_html=True)
        sev_counts = hist["severity"].value_counts().reindex(sev_order, fill_value=0)
        fig_sev = go.Figure(go.Bar(
            x=sev_counts.index.tolist(),
            y=sev_counts.values.tolist(),
            marker_color=[SEV_COLOUR[s] for s in sev_counts.index],
            text=sev_counts.values.tolist(),
            textposition="outside",
            textfont=dict(size=9),
        ))
        fig_sev.update_layout(**base_layout(210), yaxis_title="Count")
        st.plotly_chart(fig_sev, use_container_width=True)

    # ── Row 6: Live alert feed ────────────────────────────────────────────

    st.divider()
    st.markdown('<p class="panel-title">Live Alert Feed — Last 12 threat events</p>', unsafe_allow_html=True)

    threat_rows = hist[hist["threat_type"] != "Benign"].tail(12).iloc[::-1]

    if not threat_rows.empty:
        header_cols = st.columns([0.3, 1.8, 1.5, 1.2, 1.2, 1.5, 1.8])
        for label, col in zip(["SEV", "SOURCE IP", "THREAT TYPE", "CONFIDENCE", "ACTION", "TOP FEATURE", "IMPORTANCE"], header_cols):
            col.markdown(f'<span style="font-size:0.63rem;color:#6b7280;letter-spacing:0.07em;">{label}</span>', unsafe_allow_html=True)

        for _, row in threat_rows.iterrows():
            sev   = row["severity"]
            sc    = SEV_COLOUR.get(sev, "#888")
            ac    = ATTACK_COLOUR.get(row["threat_type"], "#888")
            cols  = st.columns([0.3, 1.8, 1.5, 1.2, 1.2, 1.5, 1.8])
            cols[0].markdown(f'<span style="color:{sc};font-size:0.9rem;">■</span>', unsafe_allow_html=True)
            cols[1].markdown(f'<code style="font-size:0.75rem;background:#181b1f;color:#9ca3af;">{row["src_ip"]}</code>', unsafe_allow_html=True)
            cols[2].markdown(f'<span style="color:{ac};font-size:0.77rem;font-weight:500;">{row["threat_type"]}</span>', unsafe_allow_html=True)
            cols[3].markdown(f'<span style="font-size:0.77rem;color:#d0d5dd;">{row["confidence"]:.1%}</span>', unsafe_allow_html=True)
            act_col = {"block_ip": PALETTE["accent_red"], "deep_inspect": PALETTE["accent_purple"],
                       "rate_limit": PALETTE["accent_blue"], "alert_soc": PALETTE["accent_yellow"],
                       "no_action": PALETTE["text_muted"]}
            cols[4].markdown(f'<code style="font-size:0.72rem;color:{act_col.get(row["action"], "#888")};">{row["action"]}</code>', unsafe_allow_html=True)
            if show_shap:
                cols[5].markdown(f'<code style="font-size:0.7rem;color:#6b7280;">{row["shap_top_feat"]}</code>', unsafe_allow_html=True)
                cols[6].markdown(f'<span style="font-size:0.77rem;color:#9ca3af;">{row["shap_importance"]:.3f}</span>', unsafe_allow_html=True)
    else:
        st.success("All traffic nominal. No threats in current window.")

    # ── Footer status bar ─────────────────────────────────────────────────
    st.divider()
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.markdown(f'<span style="font-size:0.68rem;color:#6b7280;">Cycle: <code style="color:#9ca3af;">{st.session_state.tick}</code></span>', unsafe_allow_html=True)
    fc2.markdown(f'<span style="font-size:0.68rem;color:#6b7280;">Buffer: <code style="color:#9ca3af;">{len(hist):,} flows</code></span>', unsafe_allow_html=True)
    fc3.markdown(f'<span style="font-size:0.68rem;color:#6b7280;">Speed: <code style="color:#9ca3af;">{sim_speed} flows/cycle</code></span>', unsafe_allow_html=True)
    fc4.markdown(f'<span style="font-size:0.68rem;color:#73bf69;">● Pipeline operational</span>', unsafe_allow_html=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────

if auto_refresh:
    refresh_ms = max(400, int(1200 / max(1, sim_speed // 8)))
    time.sleep(refresh_ms / 1000)
    st.rerun()
