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

st.set_page_config(page_title="QuantumOS - Qubit Telemetry", page_icon="📡", layout="wide")
apply_enterprise_style()

# Initialize data generator
if 'data_gen' not in st.session_state:
    st.session_state.data_gen = DataGenerator()

st.markdown(create_enterprise_header(
    "Qubit Telemetry Dashboard",
    "Real-time monitoring of individual qubit parameters and performance metrics",
    "operational"
), unsafe_allow_html=True)

# Enterprise control panel
st.markdown("### ⚙️ Control Panel")
col1, col2, col3, col4 = st.columns(4)

with col1:
    selected_qubits = st.multiselect(
        "Select Qubits",
        options=list(range(8)),
        default=[0, 1, 2, 3],
        format_func=lambda x: f"Qubit {x}"
    )

with col2:
    time_window = st.selectbox(
        "Time Window",
        options=[30, 60, 120, 300],
        format_func=lambda x: f"{x}s",
        index=1
    )

with col3:
    update_rate = st.selectbox(
        "Update Rate",
        options=[0.5, 1, 2, 5],
        format_func=lambda x: f"{x}s",
        index=1
    )

with col4:
    auto_update = st.checkbox("Auto Update", value=True)

st.markdown("---")

# Generate telemetry data
current_time = datetime.now()
time_points = [current_time - timedelta(seconds=i) for i in range(time_window, 0, -1)]

if selected_qubits:
    # Frequency monitoring
    st.subheader("🔊 Qubit Frequencies")
    
    fig_freq = go.Figure()
    
    for qubit_id in selected_qubits:
        frequencies = st.session_state.data_gen.generate_frequency_data(time_window)
        base_freq = 4.5 + qubit_id * 0.2  # Base frequency for each qubit
        
        fig_freq.add_trace(go.Scatter(
            x=time_points,
            y=frequencies + base_freq,
            mode='lines+markers',
            name=f'Qubit {qubit_id}',
            line=dict(width=2),
            hovertemplate=f'Qubit {qubit_id}<br>Time: %{{x}}<br>Frequency: %{{y:.6f}} GHz<extra></extra>'
        ))
    
    # Apply enterprise theme
    enterprise_theme = get_enterprise_plotly_theme()
    fig_freq.update_layout(
        title={
            'text': "Qubit Transition Frequencies",
            'font': {'size': 18, 'color': '#ffffff'}
        },
        xaxis_title="Time",
        yaxis_title="Frequency (GHz)",
        height=450,
        hovermode='x unified',
        **enterprise_theme['layout']
    )
    
    st.plotly_chart(fig_freq, use_container_width=True)
    
    # Multi-parameter dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Coherence Times")
        
        # T1 and T2 coherence times
        fig_coherence = make_subplots(
            rows=2, cols=1,
            subplot_titles=('T1 Relaxation Time', 'T2* Dephasing Time'),
            vertical_spacing=0.1
        )
        
        for qubit_id in selected_qubits:
            t1_data = st.session_state.data_gen.generate_coherence_data(time_window, 'T1')
            t2_data = st.session_state.data_gen.generate_coherence_data(time_window, 'T2*')
            
            fig_coherence.add_trace(
                go.Scatter(
                    x=time_points,
                    y=t1_data,
                    mode='lines',
                    name=f'Q{qubit_id} T1',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
            
            fig_coherence.add_trace(
                go.Scatter(
                    x=time_points,
                    y=t2_data,
                    mode='lines',
                    name=f'Q{qubit_id} T2*',
                    line=dict(width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig_coherence.update_xaxes(title_text="Time", row=2, col=1)
        fig_coherence.update_yaxes(title_text="T1 (μs)", row=1, col=1)
        fig_coherence.update_yaxes(title_text="T2* (μs)", row=2, col=1)
        fig_coherence.update_layout(height=500, **enterprise_theme['layout'])
        
        st.plotly_chart(fig_coherence, use_container_width=True)
    
    with col2:
        st.subheader("📊 Gate Performance")
        
        # Gate fidelity and timing
        fig_gates = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Single Qubit Gate Fidelity', 'gate Duration'),
            vertical_spacing=0.1
        )
        
        for qubit_id in selected_qubits:
            fidelity_data = st.session_state.data_gen.generate_fidelity_data(time_window)
            duration_data = st.session_state.data_gen.generate_gate_duration_data(time_window)
            
            fig_gates.add_trace(
                go.Scatter(
                    x=time_points,
                    y=fidelity_data,
                    mode='lines+markers',
                    name=f'Q{qubit_id} Fidelity',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
            
            fig_gates.add_trace(
                go.Scatter(
                    x=time_points,
                    y=duration_data,
                    mode='lines',
                    name=f'Q{qubit_id} Duration',
                    line=dict(width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig_gates.update_xaxes(title_text="Time", row=2, col=1)
        fig_gates.update_yaxes(title_text="Fidelity (%)", row=1, col=1)
        fig_gates.update_yaxes(title_text="Duration (ns)", row=2, col=1)
        fig_gates.update_layout(height=500)
        
        st.plotly_chart(fig_gates, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("📋 Current Metrics")
    
    metrics_data = []
    for qubit_id in selected_qubits:
        metrics_data.append({
            'Qubit': f'Q{qubit_id}',
            'Frequency (GHz)': f"{4.5 + qubit_id * 0.2 + np.random.normal(0, 0.001):.6f}",
            'T1 (μs)': f"{120 + np.random.normal(0, 10):.1f}",
            'T2* (μs)': f"{85 + np.random.normal(0, 8):.1f}",
            'Gate Fidelity (%)': f"{99.5 + np.random.normal(0, 0.2):.2f}",
            'Readout Fidelity (%)': f"{99.2 + np.random.normal(0, 0.3):.2f}",
            'Status': np.random.choice(['✅ Good', '⚠️ Warning', '✅ Good'], p=[0.8, 0.1, 0.1])
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True)
    
    # Export functionality
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("📥 Export CSV"):
            csv = df_metrics.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"qubit_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Export Plots"):
            st.info("Plots can be downloaded using the camera icon in each chart")
    
    # Auto-refresh
    if auto_update:
        time.sleep(update_rate)
        st.rerun()

else:
    st.warning("Please select at least one qubit to monitor.")
