import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime
import time
from utils.data_generator import DataGenerator
from utils.styling import (
    apply_enterprise_style, create_enterprise_metric_card, 
    create_status_badge, get_enterprise_plotly_theme,
    create_enterprise_header, create_alert_box, EnterpriseTheme
)

st.set_page_config(page_title="QuantumOS - Pulse Control", page_icon="⚡", layout="wide")
apply_enterprise_style()

# Initialize data generator
if 'data_gen' not in st.session_state:
    st.session_state.data_gen = DataGenerator()

# Initialize pulse parameters in session state
if 'pulse_params' not in st.session_state:
    st.session_state.pulse_params = {
        'amplitude': 0.5,
        'frequency': 4.8,
        'duration': 20.0,
        'phase': 0.0,
        'rise_time': 2.0,
        'fall_time': 2.0,
        'pulse_type': 'Gaussian'
    }

st.markdown(create_enterprise_header(
    "Pulse Control & Optimization",
    "Interactive control and real-time optimization of quantum control pulses",
    "operational"
), unsafe_allow_html=True)

# Sidebar controls
st.sidebar.subheader("Pulse Parameters")

# Pulse type selection
pulse_type = st.sidebar.selectbox(
    "Pulse Type",
    options=["Gaussian", "Square", "DRAG", "Cosine", "Hermite"],
    index=0
)

st.session_state.pulse_params['pulse_type'] = pulse_type

# Parameter controls
col1, col2 = st.sidebar.columns(2)

with col1:
    amplitude = st.slider(
        "Amplitude",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.pulse_params['amplitude'],
        step=0.01,
        key="amplitude_slider"
    )
    
    frequency = st.slider(
        "Frequency (GHz)",
        min_value=4.0,
        max_value=6.0,
        value=st.session_state.pulse_params['frequency'],
        step=0.01,
        key="frequency_slider"
    )
    
    duration = st.slider(
        "Duration (ns)",
        min_value=1.0,
        max_value=100.0,
        value=st.session_state.pulse_params['duration'],
        step=0.5,
        key="duration_slider"
    )

with col2:
    phase = st.slider(
        "Phase (rad)",
        min_value=0.0,
        max_value=2*np.pi,
        value=st.session_state.pulse_params['phase'],
        step=0.1,
        key="phase_slider"
    )
    
    rise_time = st.slider(
        "Rise Time (ns)",
        min_value=0.5,
        max_value=10.0,
        value=st.session_state.pulse_params['rise_time'],
        step=0.1,
        key="rise_time_slider"
    )
    
    fall_time = st.slider(
        "Fall Time (ns)",
        min_value=0.5,
        max_value=10.0,
        value=st.session_state.pulse_params['fall_time'],
        step=0.1,
        key="fall_time_slider"
    )

# Update session state
st.session_state.pulse_params.update({
    'amplitude': amplitude,
    'frequency': frequency,
    'duration': duration,
    'phase': phase,
    'rise_time': rise_time,
    'fall_time': fall_time
})

# Control buttons
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🔄 Apply Changes", type="primary"):
        st.success("Pulse parameters updated!")
        
with col2:
    if st.button("↩️ Reset to Default"):
        st.session_state.pulse_params = {
            'amplitude': 0.5,
            'frequency': 4.8,
            'duration': 20.0,
            'phase': 0.0,
            'rise_time': 2.0,
            'fall_time': 2.0,
            'pulse_type': 'Gaussian'
        }
        st.rerun()

# Auto-tune controls
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Auto-Tune")

target_gate = st.sidebar.selectbox(
    "Target Gate",
    options=["X", "Y", "Z", "H", "CNOT", "CZ"],
    index=0
)

if st.sidebar.button("🎯 Auto-Optimize"):
    with st.spinner("Optimizing pulse parameters..."):
        time.sleep(2)  # Simulate optimization
        # Add some random optimization
        st.session_state.pulse_params['amplitude'] *= (1 + np.random.normal(0, 0.05))
        st.session_state.pulse_params['duration'] *= (1 + np.random.normal(0, 0.1))
        st.session_state.pulse_params['amplitude'] = np.clip(st.session_state.pulse_params['amplitude'], 0, 1)
        st.sidebar.success(f"Optimized for {target_gate} gate!")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Pulse Waveform")
    
    # Generate pulse waveform based on parameters
    def generate_pulse_waveform(params):
        t = np.linspace(0, params['duration'], int(params['duration'] * 10))
        
        if params['pulse_type'] == 'Gaussian':
            sigma = params['duration'] / 6
            envelope = params['amplitude'] * np.exp(-((t - params['duration']/2)**2) / (2*sigma**2))
        elif params['pulse_type'] == 'Square':
            envelope = np.full_like(t, params['amplitude'])
            # Apply rise and fall times
            rise_samples = int(params['rise_time'] * 10)
            fall_samples = int(params['fall_time'] * 10)
            if rise_samples > 0:
                envelope[:rise_samples] *= np.linspace(0, 1, rise_samples)
            if fall_samples > 0:
                envelope[-fall_samples:] *= np.linspace(1, 0, fall_samples)
        elif params['pulse_type'] == 'DRAG':
            sigma = params['duration'] / 6
            gauss = np.exp(-((t - params['duration']/2)**2) / (2*sigma**2))
            deriv = -(t - params['duration']/2) / sigma**2 * gauss
            envelope = params['amplitude'] * (gauss + 0.2 * deriv)
        elif params['pulse_type'] == 'Cosine':
            envelope = params['amplitude'] * np.cos(2*np.pi * t / params['duration'] - np.pi/2)**2
        else:  # Hermite
            x = (t - params['duration']/2) / (params['duration']/6)
            envelope = params['amplitude'] * np.exp(-x**2/2) * (1 - x**2)
        
        # Add carrier frequency and phase
        carrier = np.cos(2*np.pi * params['frequency'] * t + params['phase'])
        waveform = envelope * carrier
        
        return t, envelope, waveform
    
    t, envelope, waveform = generate_pulse_waveform(st.session_state.pulse_params)
    
    # Create subplot with envelope and full waveform
    fig_pulse = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Pulse Envelope', 'Full Waveform (I/Q)'),
        vertical_spacing=0.1
    )
    
    # Envelope plot
    fig_pulse.add_trace(
        go.Scatter(
            x=t,
            y=envelope,
            mode='lines',
            name='Envelope',
            line=dict(width=3, color='red')
        ),
        row=1, col=1
    )
    
    # Full waveform plot
    fig_pulse.add_trace(
        go.Scatter(
            x=t,
            y=waveform,
            mode='lines',
            name='I Component',
            line=dict(width=2, color='blue')
        ),
        row=2, col=1
    )
    
    # Q component (90 degree phase shift)
    q_waveform = envelope * np.sin(2*np.pi * st.session_state.pulse_params['frequency'] * t + st.session_state.pulse_params['phase'])
    fig_pulse.add_trace(
        go.Scatter(
            x=t,
            y=q_waveform,
            mode='lines',
            name='Q Component',
            line=dict(width=2, color='orange')
        ),
        row=2, col=1
    )
    
    fig_pulse.update_xaxes(title_text="Time (ns)", row=2, col=1)
    fig_pulse.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig_pulse.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig_pulse.update_layout(height=600, showlegend=True)
    
    st.plotly_chart(fig_pulse, use_container_width=True)

with col2:
    st.subheader("📈 Pulse Metrics")
    
    # Calculate pulse metrics
    pulse_energy = np.trapezoid(envelope**2, t)
    peak_power = np.max(envelope**2)
    rms_amplitude = np.sqrt(np.mean(envelope**2))
    bandwidth = 1 / st.session_state.pulse_params['duration']  # Approximation
    
    pulse_metrics = [
        ("Pulse Energy", f"{pulse_energy:.3f} a.u.", None, "normal"),
        ("Peak Power", f"{peak_power:.3f} a.u.", None, "normal"),
        ("RMS Amplitude", f"{rms_amplitude:.3f}", None, "normal"),
        ("Est. Bandwidth", f"{bandwidth:.2f} GHz", None, "normal")
    ]
    
    for title, value, delta, delta_color in pulse_metrics:
        st.markdown(create_enterprise_metric_card(title, value, delta, delta_color), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Fidelity prediction (simulated)
    base_fidelity = 99.5
    amplitude_penalty = abs(st.session_state.pulse_params['amplitude'] - 0.5) * 2
    duration_penalty = abs(st.session_state.pulse_params['duration'] - 20) * 0.1
    predicted_fidelity = base_fidelity - amplitude_penalty - duration_penalty
    predicted_fidelity = max(predicted_fidelity, 90.0)
    
    st.metric(
        "Predicted Fidelity",
        f"{predicted_fidelity:.2f}%",
        f"{predicted_fidelity - 99.0:+.2f}%"
    )
    
    # Status indicators
    st.markdown("---")
    st.subheader("🚦 Status")
    
    if predicted_fidelity > 99:
        st.success("✅ Optimal parameters")
    elif predicted_fidelity > 95:
        st.warning("⚠️ Suboptimal parameters")
    else:
        st.error("❌ Poor parameters")

# Pulse sequence designer
st.markdown("---")
st.subheader("🔧 Pulse Sequence Designer")

col1, col2, col3 = st.columns(3)

with col1:
    sequence_length = st.number_input("Sequence Length", min_value=1, max_value=10, value=3)

with col2:
    gate_sequence = st.text_input("Gate Sequence", value="X-Y-X", help="Enter gates separated by '-'")

with col3:
    if st.button("🏗️ Build Sequence"):
        st.info(f"Building sequence: {gate_sequence}")

# Sequence visualization
if gate_sequence:
    gates = gate_sequence.split('-')
    fig_sequence = go.Figure()
    
    total_time = 0
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, gate in enumerate(gates[:sequence_length]):
        gate_duration = st.session_state.pulse_params['duration']
        t_gate = np.linspace(total_time, total_time + gate_duration, int(gate_duration * 5))
        
        # Simple gate waveform
        if gate.upper() == 'X':
            waveform = st.session_state.pulse_params['amplitude'] * np.sin(2*np.pi * t_gate / gate_duration)
        elif gate.upper() == 'Y':
            waveform = st.session_state.pulse_params['amplitude'] * np.cos(2*np.pi * t_gate / gate_duration)
        else:
            waveform = st.session_state.pulse_params['amplitude'] * np.ones_like(t_gate) * 0.5
        
        fig_sequence.add_trace(go.Scatter(
            x=t_gate,
            y=waveform,
            mode='lines',
            name=f'{gate} Gate',
            line=dict(width=2, color=colors[i % len(colors)])
        ))
        
        total_time += gate_duration + 5  # 5ns gap between gates
    
    fig_sequence.update_layout(
        title=f"Pulse Sequence: {gate_sequence}",
        xaxis_title="Time (ns)",
        yaxis_title="Amplitude",
        height=300
    )
    
    st.plotly_chart(fig_sequence, use_container_width=True)

# Parameter optimization history
st.markdown("---")
st.subheader("📊 Optimization History")

# Simulate optimization history
if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []

# Add current parameters to history
current_params = st.session_state.pulse_params.copy()
current_params['timestamp'] = datetime.now()
current_params['fidelity'] = predicted_fidelity

if len(st.session_state.optimization_history) == 0 or \
   st.session_state.optimization_history[-1]['amplitude'] != current_params['amplitude']:
    st.session_state.optimization_history.append(current_params)

# Keep only last 20 entries
if len(st.session_state.optimization_history) > 20:
    st.session_state.optimization_history = st.session_state.optimization_history[-20:]

if st.session_state.optimization_history:
    history_df = pd.DataFrame(st.session_state.optimization_history)
    
    fig_history = go.Figure()
    fig_history.add_trace(go.Scatter(
        x=list(range(len(history_df))),
        y=history_df['fidelity'],
        mode='lines+markers',
        name='Predicted Fidelity',
        line=dict(width=2)
    ))
    
    fig_history.update_layout(
        title="Fidelity vs Optimization Steps",
        xaxis_title="Optimization Step",
        yaxis_title="Predicted Fidelity (%)",
        height=300
    )
    
    st.plotly_chart(fig_history, use_container_width=True)
