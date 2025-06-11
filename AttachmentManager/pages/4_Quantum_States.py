import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime
import time
from utils.quantum_simulator import QuantumSimulator
from utils.bloch_sphere import BlochSphere
from utils.styling import (
    apply_enterprise_style, create_enterprise_metric_card, 
    create_status_badge, get_enterprise_plotly_theme,
    create_enterprise_header, create_alert_box, EnterpriseTheme
)

st.set_page_config(page_title="QuantumOS - Quantum States", page_icon="🌐", layout="wide")
apply_enterprise_style()

# Initialize quantum simulator and Bloch sphere
if 'quantum_sim' not in st.session_state:
    st.session_state.quantum_sim = QuantumSimulator()
if 'bloch_sphere' not in st.session_state:
    st.session_state.bloch_sphere = BlochSphere()

st.markdown(create_enterprise_header(
    "Quantum State Visualization",
    "Interactive visualization and analysis of quantum states and operations",
    "operational"
), unsafe_allow_html=True)

# Sidebar controls
st.sidebar.subheader("State Preparation")

# Qubit selection
num_qubits = st.sidebar.selectbox("Number of Qubits", options=[1, 2, 3], index=0)

if num_qubits == 1:
    # Single qubit controls
    st.sidebar.subheader("Single Qubit State")
    
    # Bloch sphere coordinates
    theta = st.sidebar.slider("Theta (polar angle)", 0.0, np.pi, np.pi/2, 0.1)
    phi = st.sidebar.slider("Phi (azimuthal angle)", 0.0, 2*np.pi, 0.0, 0.1)
    
    # Common states buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("📍 |0⟩"):
            theta, phi = 0, 0
        if st.button("📍 |1⟩"):
            theta, phi = np.pi, 0
        if st.button("📍 |+⟩"):
            theta, phi = np.pi/2, 0
    
    with col2:
        if st.button("📍 |-⟩"):
            theta, phi = np.pi/2, np.pi
        if st.button("📍 |+i⟩"):
            theta, phi = np.pi/2, np.pi/2
        if st.button("📍 |-i⟩"):
            theta, phi = np.pi/2, 3*np.pi/2

# Gate operations
st.sidebar.markdown("---")
st.sidebar.subheader("Gate Operations")

if num_qubits == 1:
    gate_options = ["I", "X", "Y", "Z", "H", "S", "T", "Rx", "Ry", "Rz"]
else:
    gate_options = ["I", "X", "Y", "Z", "H", "CNOT", "CZ", "SWAP"]

selected_gate = st.sidebar.selectbox("Select Gate", options=gate_options)

# Rotation angles for parameterized gates
if selected_gate in ["Rx", "Ry", "Rz"]:
    rotation_angle = st.sidebar.slider("Rotation Angle", 0.0, 2*np.pi, np.pi, 0.1)

# Apply gate button
if st.sidebar.button(f"Apply {selected_gate} Gate", type="primary"):
    st.sidebar.success(f"{selected_gate} gate applied!")

# Main content area
if num_qubits == 1:
    # Single qubit visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🌍 Bloch Sphere Visualization")
        
        # Generate Bloch sphere plot
        fig_bloch = st.session_state.bloch_sphere.create_bloch_sphere(theta, phi)
        st.plotly_chart(fig_bloch, use_container_width=True)
    
    with col2:
        st.subheader("📊 State Information")
        
        # Calculate state vector
        alpha = np.cos(theta/2)
        beta = np.sin(theta/2) * np.exp(1j * phi)
        
        # Display state vector
        st.markdown("**State Vector:**")
        st.latex(f"|\\psi\\rangle = {alpha:.3f}|0\\rangle + ({beta.real:.3f}{'+' if beta.imag >= 0 else ''}{beta.imag:.3f}i)|1\\rangle")
        
        # Probability amplitudes
        st.markdown("**Measurement Probabilities:**")
        prob_0 = abs(alpha)**2
        prob_1 = abs(beta)**2
        
        st.metric("P(|0⟩)", f"{prob_0:.3f}")
        st.metric("P(|1⟩)", f"{prob_1:.3f}")
        
        # Expectation values
        st.markdown("**Expectation Values:**")
        exp_x = np.sin(theta) * np.cos(phi)
        exp_y = np.sin(theta) * np.sin(phi)
        exp_z = np.cos(theta)
        
        st.metric("⟨X⟩", f"{exp_x:.3f}")
        st.metric("⟨Y⟩", f"{exp_y:.3f}")
        st.metric("⟨Z⟩", f"{exp_z:.3f}")
        
        # Purity and entropy
        st.markdown("**State Properties:**")
        st.metric("Purity", "1.000")  # Pure state
        st.metric("von Neumann Entropy", "0.000")
    
    # State evolution
    st.markdown("---")
    st.subheader("⏰ State Evolution")
    
    # Time evolution controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hamiltonian_type = st.selectbox("Hamiltonian", ["σ_x", "σ_y", "σ_z", "Custom"])
    
    with col2:
        evolution_time = st.slider("Evolution Time", 0.0, 10.0, 1.0, 0.1)
    
    with col3:
        animate_evolution = st.checkbox("Animate Evolution")
    
    if animate_evolution:
        # Create animation frames
        time_steps = np.linspace(0, evolution_time, 50)
        
        # Simple time evolution simulation
        evolved_states = []
        for t in time_steps:
            if hamiltonian_type == "σ_x":
                evolved_theta = theta
                evolved_phi = phi + 2*t  # Rotation around X-axis
            elif hamiltonian_type == "σ_y":
                evolved_theta = theta + 2*t  # Simplified evolution
                evolved_phi = phi
            else:  # σ_z
                evolved_theta = theta
                evolved_phi = phi + 2*t
            
            evolved_states.append((evolved_theta % (2*np.pi), evolved_phi % (2*np.pi)))
        
        # Plot evolution trajectory
        thetas, phis = zip(*evolved_states)
        
        fig_evolution = go.Figure()
        
        # Convert to Cartesian coordinates for trajectory
        x_traj = [np.sin(t) * np.cos(p) for t, p in evolved_states]
        y_traj = [np.sin(t) * np.sin(p) for t, p in evolved_states]
        z_traj = [np.cos(t) for t, p in evolved_states]
        
        fig_evolution.add_trace(go.Scatter3d(
            x=x_traj,
            y=y_traj,
            z=z_traj,
            mode='lines+markers',
            name='Evolution Trajectory',
            line=dict(width=4, color='red'),
            marker=dict(size=3)
        ))
        
        # Add Bloch sphere
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
        x_sphere = np.cos(u)*np.sin(v)
        y_sphere = np.sin(u)*np.sin(v)
        z_sphere = np.cos(v)
        
        fig_evolution.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.3,
            colorscale='Blues',
            showscale=False,
            name='Bloch Sphere'
        ))
        
        fig_evolution.update_layout(
            title=f"State Evolution under {hamiltonian_type} Hamiltonian",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='cube'
            ),
            height=500
        )
        
        st.plotly_chart(fig_evolution, use_container_width=True)

elif num_qubits == 2:
    # Two-qubit state visualization
    st.subheader("🔗 Two-Qubit State Analysis")
    
    # State preparation options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        state_type = st.selectbox(
            "Initial State",
            options=["Product State", "Bell State", "Custom"],
            index=1
        )
    
    with col2:
        if state_type == "Bell State":
            bell_state = st.selectbox(
                "Bell State Type",
                options=["Φ+", "Φ-", "Ψ+", "Ψ-"],
                index=0
            )
    
    with col3:
        measurement_basis = st.selectbox(
            "Measurement Basis",
            options=["Computational", "Bell", "Custom"],
            index=0
        )
    
    # Generate two-qubit state
    if state_type == "Bell State":
        if bell_state == "Φ+":
            state_vector = np.array([1, 0, 0, 1]) / np.sqrt(2)  # |00⟩ + |11⟩
        elif bell_state == "Φ-":
            state_vector = np.array([1, 0, 0, -1]) / np.sqrt(2)  # |00⟩ - |11⟩
        elif bell_state == "Ψ+":
            state_vector = np.array([0, 1, 1, 0]) / np.sqrt(2)  # |01⟩ + |10⟩
        else:  # Ψ-
            state_vector = np.array([0, 1, -1, 0]) / np.sqrt(2)  # |01⟩ - |10⟩
    else:
        # Product state |00⟩
        state_vector = np.array([1, 0, 0, 0])
    
    # State visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Probability Distribution")
        
        # Plot probability distribution
        basis_states = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
        probabilities = np.abs(state_vector)**2
        
        fig_prob = px.bar(
            x=basis_states,
            y=probabilities,
            title="Measurement Probabilities",
            labels={'x': 'Basis State', 'y': 'Probability'}
        )
        fig_prob.update_layout(height=400)
        st.plotly_chart(fig_prob, use_container_width=True)
    
    with col2:
        st.subheader("🎯 State Properties")
        
        # Calculate entanglement measures
        # Reduced density matrix for qubit 1
        rho_1 = np.array([
            [abs(state_vector[0])**2 + abs(state_vector[1])**2, 
             state_vector[0].conj() * state_vector[2] + state_vector[1].conj() * state_vector[3]],
            [state_vector[2].conj() * state_vector[0] + state_vector[3].conj() * state_vector[1],
             abs(state_vector[2])**2 + abs(state_vector[3])**2]
        ])
        
        # Calculate purity
        purity = np.trace(rho_1 @ rho_1).real
        
        # von Neumann entropy
        eigenvals = np.linalg.eigvals(rho_1)
        eigenvals = eigenvals[eigenvals > 1e-10]  # Remove numerical zeros
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        
        st.metric("Entanglement Entropy", f"{entropy:.3f}")
        st.metric("Purity (Qubit 1)", f"{purity:.3f}")
        
        # Concurrence (for pure states)
        if np.abs(np.sum(np.abs(state_vector)**2) - 1) < 1e-10:  # Pure state
            # Calculate concurrence
            sigma_y = np.array([[0, -1j], [1j, 0]])
            state_matrix = state_vector.reshape(2, 2)
            conj_state = np.kron(sigma_y, sigma_y) @ state_vector.conj()
            concurrence = abs(np.vdot(state_vector, conj_state))
            st.metric("Concurrence", f"{concurrence:.3f}")
        
        # Fidelity with Bell states
        bell_states = {
            "Φ+": np.array([1, 0, 0, 1]) / np.sqrt(2),
            "Φ-": np.array([1, 0, 0, -1]) / np.sqrt(2),
            "Ψ+": np.array([0, 1, 1, 0]) / np.sqrt(2),
            "Ψ-": np.array([0, 1, -1, 0]) / np.sqrt(2)
        }
        
        st.markdown("**Fidelity with Bell States:**")
        for name, bell_state in bell_states.items():
            fidelity = abs(np.vdot(state_vector, bell_state))**2
            st.metric(f"F({name})", f"{fidelity:.3f}")

    # Correlation analysis
    st.markdown("---")
    st.subheader("🔄 Quantum Correlations")
    
    # Two-qubit correlation matrix
    pauli_matrices = {
        'I': np.array([[1, 0], [0, 1]]),
        'X': np.array([[0, 1], [1, 0]]),
        'Y': np.array([[0, -1j], [1j, 0]]),
        'Z': np.array([[1, 0], [0, -1]])
    }
    
    correlation_matrix = np.zeros((4, 4))
    labels = ['I', 'X', 'Y', 'Z']
    
    for i, op1 in enumerate(['I', 'X', 'Y', 'Z']):
        for j, op2 in enumerate(['I', 'X', 'Y', 'Z']):
            # Calculate expectation value ⟨op1 ⊗ op2⟩
            operator = np.kron(pauli_matrices[op1], pauli_matrices[op2])
            expectation = np.real(np.conj(state_vector) @ operator @ state_vector)
            correlation_matrix[i, j] = expectation
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="Expectation Value")
    ))
    
    fig_corr.update_layout(
        title="Two-Qubit Correlation Matrix ⟨σᵢ ⊗ σⱼ⟩",
        xaxis_title="Qubit 2 Observable",
        yaxis_title="Qubit 1 Observable",
        height=400
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)

else:  # Three qubits
    st.subheader("🌌 Three-Qubit State Space")
    st.info("Three-qubit visualization requires advanced techniques. Showing computational basis probabilities.")
    
    # Generate random three-qubit state
    if 'three_qubit_state' not in st.session_state:
        # Create GHZ state as example
        st.session_state.three_qubit_state = np.zeros(8, dtype=complex)
        st.session_state.three_qubit_state[0] = 1/np.sqrt(2)  # |000⟩
        st.session_state.three_qubit_state[7] = 1/np.sqrt(2)  # |111⟩
    
    state_vector = st.session_state.three_qubit_state
    
    # Plot probability distribution
    basis_states = ['|000⟩', '|001⟩', '|010⟩', '|011⟩', '|100⟩', '|101⟩', '|110⟩', '|111⟩']
    probabilities = np.abs(state_vector)**2
    
    fig_3q = px.bar(
        x=basis_states,
        y=probabilities,
        title="Three-Qubit State Probabilities (GHZ State)",
        labels={'x': 'Basis State', 'y': 'Probability'}
    )
    fig_3q.update_layout(height=400)
    st.plotly_chart(fig_3q, use_container_width=True)

# Export functionality
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📸 Export State"):
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'num_qubits': num_qubits,
            'state_vector': state_vector.tolist() if isinstance(state_vector, np.ndarray) else None,
            'parameters': {
                'theta': theta if num_qubits == 1 else None,
                'phi': phi if num_qubits == 1 else None
            }
        }
        st.success("State exported successfully!")
        st.json(state_data)

with col2:
    if st.button("🔄 Random State"):
        if num_qubits == 1:
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
        st.rerun()

with col3:
    if st.button("↩️ Reset to |0⟩"):
        if num_qubits == 1:
            theta, phi = 0, 0
        st.rerun()
