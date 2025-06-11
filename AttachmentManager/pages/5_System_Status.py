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

st.set_page_config(page_title="QuantumOS - System Status", page_icon="🖥️", layout="wide")
apply_enterprise_style()

# Initialize data generator
if 'data_gen' not in st.session_state:
    st.session_state.data_gen = DataGenerator()

st.markdown(create_enterprise_header(
    "Quantum System Status Dashboard",
    "Comprehensive monitoring of quantum computer hardware and software systems",
    "operational"
), unsafe_allow_html=True)

# Auto-refresh control
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_interval = st.sidebar.selectbox("Refresh Interval", [2, 5, 10, 30], index=1)

# System overview metrics
st.subheader("🎛️ System Overview")

col1, col2, col3, col4, col5 = st.columns(5)

# Generate system metrics
system_uptime = 99.7 + np.random.normal(0, 0.1)
calibration_status = np.random.choice([95, 98, 99], p=[0.1, 0.3, 0.6])
error_rate = 0.5 + np.random.normal(0, 0.1)
queue_length = np.random.randint(0, 15)
active_jobs = np.random.randint(0, 8)

with col1:
    st.metric(
        "System Uptime",
        f"{system_uptime:.1f}%",
        f"{np.random.normal(0, 0.1):+.1f}%"
    )

with col2:
    st.metric(
        "Calibration Status",
        f"{calibration_status}%",
        f"{np.random.randint(-2, 3):+d}%"
    )

with col3:
    st.metric(
        "Error Rate",
        f"{error_rate:.2f}%",
        f"{np.random.normal(0, 0.05):+.2f}%"
    )

with col4:
    st.metric(
        "Queue Length",
        f"{queue_length}",
        f"{np.random.randint(-3, 4):+d}"
    )

with col5:
    st.metric(
        "Active Jobs",
        f"{active_jobs}",
        f"{np.random.randint(-2, 3):+d}"
    )

st.markdown("---")

# Hardware status
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌡️ Cryogenic System")
    
    # Temperature monitoring
    temperatures = {
        'Mixing Chamber': 0.015 + np.random.normal(0, 0.002),
        'Still': 0.7 + np.random.normal(0, 0.1),
        '4K Stage': 4.2 + np.random.normal(0, 0.3),
        '50K Stage': 52 + np.random.normal(0, 2)
    }
    
    temp_df = pd.DataFrame(list(temperatures.items()), columns=['Stage', 'Temperature (K)'])
    
    fig_temp = px.bar(
        temp_df,
        x='Stage',
        y='Temperature (K)',
        title='Dilution Refrigerator Temperatures',
        color='Temperature (K)',
        color_continuous_scale='Blues_r'
    )
    fig_temp.update_layout(height=400)
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Cryogenic status indicators
    st.markdown("**Status Indicators:**")
    for stage, temp in temperatures.items():
        if stage == 'Mixing Chamber':
            status = "🟢 Normal" if temp < 0.02 else "🟡 Warning" if temp < 0.05 else "🔴 Critical"
        elif stage == 'Still':
            status = "🟢 Normal" if temp < 1.0 else "🟡 Warning" if temp < 1.5 else "🔴 Critical"
        elif stage == '4K Stage':
            status = "🟢 Normal" if temp < 5.0 else "🟡 Warning" if temp < 6.0 else "🔴 Critical"
        else:  # 50K Stage
            status = "🟢 Normal" if temp < 60 else "🟡 Warning" if temp < 70 else "🔴 Critical"
        
        st.markdown(f"- **{stage}**: {status}")

with col2:
    st.subheader("⚡ Electronics Status")
    
    # Electronics monitoring
    electronics_status = {
        'AWG Channels': np.random.randint(95, 101),
        'Readout Lines': np.random.randint(90, 101),
        'Control Lines': np.random.randint(92, 101),
        'Local Oscillators': np.random.randint(88, 101),
        'Amplifiers': np.random.randint(85, 101),
        'Digitizers': np.random.randint(90, 101)
    }
    
    # Create electronics status chart
    components = list(electronics_status.keys())
    status_values = list(electronics_status.values())
    colors = ['green' if v >= 95 else 'orange' if v >= 90 else 'red' for v in status_values]
    
    fig_electronics = go.Figure(go.Bar(
        x=components,
        y=status_values,
        marker_color=colors,
        text=[f"{v}%" for v in status_values],
        textposition='outside'
    ))
    
    fig_electronics.update_layout(
        title="Electronics Health Status",
        yaxis_title="Health (%)",
        height=400,
        yaxis=dict(range=[0, 105])
    )
    
    st.plotly_chart(fig_electronics, use_container_width=True)
    
    # Electronics details
    st.markdown("**Component Details:**")
    for component, health in electronics_status.items():
        if health >= 95:
            st.success(f"✅ {component}: {health}% - Operational")
        elif health >= 90:
            st.warning(f"⚠️ {component}: {health}% - Degraded")
        else:
            st.error(f"❌ {component}: {health}% - Critical")

# Qubit status matrix
st.markdown("---")
st.subheader("🔬 Individual Qubit Status")

# Generate qubit status data
num_qubits = 8
qubit_metrics = []

for i in range(num_qubits):
    metrics = {
        'Qubit': f'Q{i}',
        'Frequency (GHz)': 4.5 + i * 0.2 + np.random.normal(0, 0.005),
        'T1 (μs)': 120 + np.random.normal(0, 15),
        'T2* (μs)': 85 + np.random.normal(0, 10),
        'Gate Fidelity (%)': 99.5 + np.random.normal(0, 0.3),
        'Readout Fidelity (%)': 99.2 + np.random.normal(0, 0.4),
        'Crosstalk (%)': 2.1 + np.random.normal(0, 0.5),
        'Status': np.random.choice(['Operational', 'Calibrating', 'Maintenance'], p=[0.8, 0.15, 0.05])
    }
    qubit_metrics.append(metrics)

qubit_df = pd.DataFrame(qubit_metrics)

# Display as styled dataframe
def style_status(val):
    if val == 'Operational':
        return 'background-color: #90EE90'
    elif val == 'Calibrating':
        return 'background-color: #FFE4B5'
    else:
        return 'background-color: #FFB6C1'

styled_df = qubit_df.style.map(style_status, subset=['Status'])
st.dataframe(styled_df, use_container_width=True)

# Connectivity map
st.markdown("---")
st.subheader("🔗 Qubit Connectivity")

col1, col2 = st.columns([2, 1])

with col1:
    # Create connectivity graph
    # Simple linear topology for demonstration
    fig_connectivity = go.Figure()
    
    # Qubit positions (circular layout)
    angles = np.linspace(0, 2*np.pi, num_qubits, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
    
    # Draw connections (nearest neighbors)
    for i in range(num_qubits):
        next_qubit = (i + 1) % num_qubits
        fig_connectivity.add_trace(go.Scatter(
            x=[x_pos[i], x_pos[next_qubit]],
            y=[y_pos[i], y_pos[next_qubit]],
            mode='lines',
            line=dict(width=2, color='gray'),
            showlegend=False
        ))
    
    # Draw qubits
    colors = ['green' if status == 'Operational' else 'orange' if status == 'Calibrating' else 'red' 
              for status in qubit_df['Status']]
    
    fig_connectivity.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers+text',
        marker=dict(size=30, color=colors, line=dict(width=2, color='black')),
        text=[f'Q{i}' for i in range(num_qubits)],
        textposition='middle center',
        textfont=dict(color='white', size=12),
        name='Qubits'
    ))
    
    fig_connectivity.update_layout(
        title="Qubit Connectivity Graph",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_connectivity, use_container_width=True)

with col2:
    st.subheader("📊 Connectivity Metrics")
    
    # Calculate connectivity metrics
    total_connections = num_qubits  # Ring topology
    active_connections = sum(1 for status in qubit_df['Status'] if status == 'Operational') 
    
    st.metric("Total Connections", total_connections)
    st.metric("Active Connections", active_connections)
    st.metric("Connectivity Rate", f"{(active_connections/total_connections)*100:.1f}%")
    
    # Two-qubit gate fidelities
    st.markdown("**Two-Qubit Gate Fidelities:**")
    for i in range(min(4, num_qubits-1)):  # Show first 4 connections
        fidelity = 95.5 + np.random.normal(0, 1.0)
        st.metric(f"Q{i}-Q{i+1}", f"{fidelity:.1f}%")

# System logs and alerts
st.markdown("---")
st.subheader("📋 System Logs & Alerts")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🚨 Recent Alerts")
    
    # Generate recent alerts
    alerts = [
        {
            'time': datetime.now() - timedelta(minutes=5),
            'level': 'INFO',
            'message': 'Calibration completed for Q3',
            'component': 'Calibration'
        },
        {
            'time': datetime.now() - timedelta(minutes=12),
            'level': 'WARNING',
            'message': 'T2* below threshold for Q5',
            'component': 'Coherence'
        },
        {
            'time': datetime.now() - timedelta(minutes=25),
            'level': 'INFO',
            'message': 'Job queue processed successfully',
            'component': 'Scheduler'
        },
        {
            'time': datetime.now() - timedelta(minutes=38),
            'level': 'SUCCESS',
            'message': 'Auto-tune completed',
            'component': 'Optimization'
        },
        {
            'time': datetime.now() - timedelta(hours=1, minutes=15),
            'level': 'ERROR',
            'message': 'Readout fidelity drop detected',
            'component': 'Readout'
        }
    ]
    
    for alert in alerts:
        time_str = alert['time'].strftime('%H:%M:%S')
        if alert['level'] == 'ERROR':
            st.error(f"🔴 [{time_str}] {alert['message']} ({alert['component']})")
        elif alert['level'] == 'WARNING':
            st.warning(f"🟡 [{time_str}] {alert['message']} ({alert['component']})")
        elif alert['level'] == 'SUCCESS':
            st.success(f"🟢 [{time_str}] {alert['message']} ({alert['component']})")
        else:
            st.info(f"🔵 [{time_str}] {alert['message']} ({alert['component']})")

with col2:
    st.subheader("💾 Performance Metrics")
    
    # System performance over time
    time_points = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
    performance_metrics = {
        'CPU Usage (%)': [60 + np.random.normal(0, 10) for _ in time_points],
        'Memory Usage (%)': [45 + np.random.normal(0, 8) for _ in time_points],
        'Network I/O (MB/s)': [12 + np.random.normal(0, 3) for _ in time_points]
    }
    
    fig_performance = go.Figure()
    
    colors = ['blue', 'green', 'red']
    for i, (metric, values) in enumerate(performance_metrics.items()):
        fig_performance.add_trace(go.Scatter(
            x=time_points,
            y=values,
            mode='lines',
            name=metric,
            line=dict(width=2, color=colors[i])
        ))
    
    fig_performance.update_layout(
        title="System Performance (24h)",
        xaxis_title="Time",
        yaxis_title="Usage (%)",
        height=300
    )
    
    st.plotly_chart(fig_performance, use_container_width=True)

# Export and maintenance
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Generate Report"):
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'system_uptime': system_uptime,
            'qubit_status': qubit_df.to_dict('records'),
            'temperatures': temperatures,
            'electronics_status': electronics_status
        }
        st.success("System report generated!")

with col2:
    if st.button("🔧 Schedule Maintenance"):
        st.warning("Maintenance scheduled for next available window")

with col3:
    if st.button("📥 Export Logs"):
        logs_df = pd.DataFrame(alerts)
        csv = logs_df.to_csv(index=False)
        st.download_button(
            label="Download Logs CSV",
            data=csv,
            file_name=f"system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col4:
    if st.button("🔄 Refresh All"):
        st.rerun()

# Auto-refresh mechanism
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
