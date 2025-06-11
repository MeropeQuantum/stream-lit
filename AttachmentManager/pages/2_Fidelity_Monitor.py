import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from utils.data_generator import DataGenerator
from utils.styling import (
    apply_enterprise_style, create_enterprise_metric_card, 
    create_status_badge, get_enterprise_plotly_theme,
    create_enterprise_header, create_alert_box, EnterpriseTheme
)

st.set_page_config(page_title="QuantumOS - Fidelity Monitor", page_icon="🎯", layout="wide")
apply_enterprise_style()

# Initialize data generator
if 'data_gen' not in st.session_state:
    st.session_state.data_gen = DataGenerator()
if 'fidelity_history' not in st.session_state:
    st.session_state.fidelity_history = []

st.markdown(create_enterprise_header(
    "Quantum Fidelity Monitor",
    "Comprehensive tracking and analysis of quantum gate and readout fidelities",
    "operational"
), unsafe_allow_html=True)

# Alert thresholds
st.sidebar.subheader("Alert Thresholds")
single_gate_threshold = st.sidebar.slider("Single Gate Fidelity (%)", 95.0, 99.9, 99.0, 0.1)
two_gate_threshold = st.sidebar.slider("Two-Qubit Gate Fidelity (%)", 90.0, 99.0, 95.0, 0.1)
readout_threshold = st.sidebar.slider("Readout Fidelity (%)", 95.0, 99.9, 99.0, 0.1)

# Control panel
col1, col2, col3, col4 = st.columns(4)

with col1:
    time_range = st.selectbox(
        "Time Range",
        options=["1h", "6h", "24h", "7d"],
        index=1
    )

with col2:
    gate_type = st.selectbox(
        "Gate Type",
        options=["All", "Single Qubit", "Two Qubit", "Readout"],
        index=0
    )

with col3:
    analysis_mode = st.selectbox(
        "Analysis Mode",
        options=["Real-time", "Historical", "Comparison"],
        index=0
    )

with col4:
    auto_refresh = st.checkbox("Auto Refresh", value=True)

st.markdown("---")

# Current fidelity overview
st.subheader("📊 Current Fidelity Overview")

col1, col2, col3, col4 = st.columns(4)

# Generate current fidelity values
current_single_fidelity = 99.5 + np.random.normal(0, 0.2)
current_two_fidelity = 95.8 + np.random.normal(0, 0.5)
current_readout_fidelity = 99.2 + np.random.normal(0, 0.3)
avg_process_fidelity = (current_single_fidelity + current_two_fidelity) / 2

fidelity_metrics = [
    ("Single Qubit Gates", f"{current_single_fidelity:.2f}%", f"{np.random.normal(0, 0.1):+.2f}%", "inverse" if current_single_fidelity < single_gate_threshold else "normal"),
    ("Two-Qubit Gates", f"{current_two_fidelity:.2f}%", f"{np.random.normal(0, 0.2):+.2f}%", "inverse" if current_two_fidelity < two_gate_threshold else "normal"),
    ("Readout Fidelity", f"{current_readout_fidelity:.2f}%", f"{np.random.normal(0, 0.15):+.2f}%", "inverse" if current_readout_fidelity < readout_threshold else "normal"),
    ("Process Fidelity", f"{avg_process_fidelity:.2f}%", f"{np.random.normal(0, 0.1):+.2f}%", "normal")
]

for col, (title, value, delta, delta_color) in zip([col1, col2, col3, col4], fidelity_metrics):
    with col:
        st.markdown(create_enterprise_metric_card(title, value, delta, delta_color), unsafe_allow_html=True)

# Historical fidelity trends
st.subheader("📈 Fidelity Trends")

# Generate time series data
time_points_map = {
    "1h": 60,
    "6h": 360,
    "24h": 1440,
    "7d": 10080
}

num_points = min(time_points_map[time_range], 200)  # Limit for performance
current_time = datetime.now()

if time_range == "7d":
    time_points = [current_time - timedelta(days=i/24) for i in range(num_points, 0, -1)]
elif time_range == "24h":
    time_points = [current_time - timedelta(hours=i/60) for i in range(num_points, 0, -1)]
else:
    time_points = [current_time - timedelta(minutes=i) for i in range(num_points, 0, -1)]

# Create fidelity trend plot
fig_trends = go.Figure()

# Single qubit fidelity
single_fidelities = [99.5 + np.random.normal(0, 0.2) for _ in range(num_points)]
fig_trends.add_trace(go.Scatter(
    x=time_points,
    y=single_fidelities,
    mode='lines',
    name='Single Qubit Gates',
    line=dict(width=2, color='#1f77b4')
))

# Two-qubit fidelity
two_fidelities = [95.8 + np.random.normal(0, 0.5) for _ in range(num_points)]
fig_trends.add_trace(go.Scatter(
    x=time_points,
    y=two_fidelities,
    mode='lines',
    name='Two-Qubit Gates',
    line=dict(width=2, color='#ff7f0e')
))

# Readout fidelity
readout_fidelities = [99.2 + np.random.normal(0, 0.3) for _ in range(num_points)]
fig_trends.add_trace(go.Scatter(
    x=time_points,
    y=readout_fidelities,
    mode='lines',
    name='Readout',
    line=dict(width=2, color='#2ca02c')
))

# Add threshold lines
fig_trends.add_hline(y=single_gate_threshold, line_dash="dash", line_color="red", 
                    annotation_text=f"Single Gate Threshold ({single_gate_threshold}%)")
fig_trends.add_hline(y=two_gate_threshold, line_dash="dash", line_color="orange",
                    annotation_text=f"Two-Qubit Threshold ({two_gate_threshold}%)")
fig_trends.add_hline(y=readout_threshold, line_dash="dash", line_color="green",
                    annotation_text=f"Readout Threshold ({readout_threshold}%)")

# Apply enterprise theme
enterprise_theme = get_enterprise_plotly_theme()
fig_trends.update_layout(
    title={
        'text': f"Fidelity Trends - {time_range}",
        'font': {'size': 18, 'color': '#ffffff'}
    },
    xaxis_title="Time",
    yaxis_title="Fidelity (%)",
    height=500,
    hovermode='x unified',
    **enterprise_theme['layout']
)

st.plotly_chart(fig_trends, use_container_width=True)

# Detailed analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Fidelity Distribution")
    
    # Create histogram of fidelity values
    fig_dist = go.Figure()
    
    fig_dist.add_trace(go.Histogram(
        x=single_fidelities,
        name='Single Qubit',
        opacity=0.7,
        nbinsx=20
    ))
    
    fig_dist.add_trace(go.Histogram(
        x=two_fidelities,
        name='Two-Qubit',
        opacity=0.7,
        nbinsx=20
    ))
    
    fig_dist.add_trace(go.Histogram(
        x=readout_fidelities,
        name='Readout',
        opacity=0.7,
        nbinsx=20
    ))
    
    fig_dist.update_layout(
        title="Fidelity Distribution",
        xaxis_title="Fidelity (%)",
        yaxis_title="Count",
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    st.subheader("⚠️ Alert Analysis")
    
    # Count violations
    single_violations = sum(1 for f in single_fidelities if f < single_gate_threshold)
    two_violations = sum(1 for f in two_fidelities if f < two_gate_threshold)
    readout_violations = sum(1 for f in readout_fidelities if f < readout_threshold)
    
    alert_data = pd.DataFrame({
        'Gate Type': ['Single Qubit', 'Two-Qubit', 'Readout'],
        'Violations': [single_violations, two_violations, readout_violations],
        'Total Measurements': [len(single_fidelities)] * 3,
        'Violation Rate (%)': [
            (single_violations / len(single_fidelities)) * 100,
            (two_violations / len(two_fidelities)) * 100,
            (readout_violations / len(readout_fidelities)) * 100
        ]
    })
    
    fig_violations = px.bar(
        alert_data,
        x='Gate Type',
        y='Violation Rate (%)',
        title="Threshold Violation Rates",
        color='Violation Rate (%)',
        color_continuous_scale='Reds'
    )
    
    fig_violations.update_layout(height=400)
    st.plotly_chart(fig_violations, use_container_width=True)

# Per-qubit analysis
st.subheader("🎛️ Per-Qubit Fidelity Analysis")

# Create per-qubit heatmap
qubits = list(range(8))
gate_types = ['X', 'Y', 'Z', 'H', 'T']

# Generate fidelity matrix
fidelity_matrix = np.random.normal(99.5, 0.3, (len(gate_types), len(qubits)))
fidelity_matrix = np.clip(fidelity_matrix, 95, 100)

fig_heatmap = go.Figure(data=go.Heatmap(
    z=fidelity_matrix,
    x=[f'Q{i}' for i in qubits],
    y=gate_types,
    colorscale='RdYlGn',
    zmid=99,
    colorbar=dict(title="Fidelity (%)")
))

fig_heatmap.update_layout(
    title="Single-Qubit Gate Fidelity by Qubit and Gate Type",
    xaxis_title="Qubit",
    yaxis_title="Gate Type",
    height=300
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# Export and summary
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Generate Report"):
        report_data = {
            'timestamp': datetime.now(),
            'single_qubit_avg': np.mean(single_fidelities),
            'two_qubit_avg': np.mean(two_fidelities),
            'readout_avg': np.mean(readout_fidelities),
            'violations': {
                'single': single_violations,
                'two': two_violations,
                'readout': readout_violations
            }
        }
        st.success("Report generated successfully!")
        st.json(report_data)

with col2:
    if st.button("📥 Export Data"):
        export_df = pd.DataFrame({
            'timestamp': time_points,
            'single_qubit_fidelity': single_fidelities,
            'two_qubit_fidelity': two_fidelities,
            'readout_fidelity': readout_fidelities
        })
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"fidelity_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col3:
    if st.button("🔄 Reset Thresholds"):
        st.rerun()

# Auto-refresh
if auto_refresh:
    time.sleep(2)
    st.rerun()
