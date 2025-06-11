import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import numpy.typing as npt

class QuantumSimulator:
    """
    A comprehensive quantum simulator for generating realistic quantum control data
    and simulating quantum operations for the dashboard.
    """
    
    def __init__(self, num_qubits: int = 8):
        """
        Initialize the quantum simulator.
        
        Args:
            num_qubits: Number of qubits in the system
        """
        self.num_qubits = num_qubits
        self.reset_system()
        
        # Physical constants and parameters
        self.h_bar = 1.054571817e-34  # Reduced Planck constant
        self.k_b = 1.380649e-23      # Boltzmann constant
        
        # System parameters
        self.base_frequencies = np.linspace(4.2, 5.8, num_qubits)  # GHz
        self.coupling_strengths = np.random.uniform(0.01, 0.05, (num_qubits, num_qubits))  # GHz
        np.fill_diagonal(self.coupling_strengths, 0)  # No self-coupling
        
        # Noise parameters
        self.t1_times = np.random.uniform(80, 150, num_qubits)  # microseconds
        self.t2_times = np.random.uniform(60, 120, num_qubits)  # microseconds
        self.gate_fidelities = np.random.uniform(0.995, 0.999, num_qubits)
        
    def reset_system(self):
        """Reset the quantum system to ground state."""
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1.0  # |000...0⟩ state
        self.density_matrix = np.outer(self.state.conj(), self.state)
        
    def get_qubit_frequency(self, qubit_id: int, include_noise: bool = True) -> float:
        """
        Get the transition frequency of a specific qubit.
        
        Args:
            qubit_id: Index of the qubit
            include_noise: Whether to include frequency noise
            
        Returns:
            Qubit frequency in GHz
        """
        base_freq = self.base_frequencies[qubit_id]
        
        if include_noise:
            # Add frequency noise (typical 1/f noise and white noise)
            noise = np.random.normal(0, 0.001)  # 1 MHz RMS noise
            return base_freq + noise
        
        return base_freq
    
    def get_coherence_times(self, qubit_id: int, include_noise: bool = True) -> Tuple[float, float]:
        """
        Get T1 and T2* coherence times for a qubit.
        
        Args:
            qubit_id: Index of the qubit
            include_noise: Whether to include time variations
            
        Returns:
            Tuple of (T1, T2*) times in microseconds
        """
        t1_base = self.t1_times[qubit_id]
        t2_base = self.t2_times[qubit_id]
        
        if include_noise:
            # Add temporal fluctuations
            t1_noise = np.random.normal(0, t1_base * 0.1)
            t2_noise = np.random.normal(0, t2_base * 0.15)
            
            t1 = max(t1_base + t1_noise, 10)  # Minimum 10 μs
            t2 = max(t2_base + t2_noise, 5)   # Minimum 5 μs
            
            return t1, t2
        
        return t1_base, t2_base
    
    def calculate_gate_fidelity(self, gate_type: str, qubit_ids: List[int], 
                              pulse_params: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate the fidelity of a quantum gate operation.
        
        Args:
            gate_type: Type of gate ('X', 'Y', 'Z', 'H', 'CNOT', etc.)
            qubit_ids: List of qubit indices the gate acts on
            pulse_params: Optional pulse parameters affecting fidelity
            
        Returns:
            Gate fidelity (0 to 1)
        """
        if len(qubit_ids) == 1:
            # Single qubit gate
            base_fidelity = self.gate_fidelities[qubit_ids[0]]
            
            # Pulse parameter effects
            if pulse_params:
                amplitude_error = abs(pulse_params.get('amplitude', 0.5) - 0.5) * 0.01
                duration_error = abs(pulse_params.get('duration', 20) - 20) * 0.0001
                fidelity_loss = amplitude_error + duration_error
                base_fidelity -= fidelity_loss
                
        elif len(qubit_ids) == 2:
            # Two qubit gate - typically lower fidelity
            q1, q2 = qubit_ids
            base_fidelity = min(self.gate_fidelities[q1], self.gate_fidelities[q2]) * 0.96
            
            # Coupling strength affects two-qubit gates
            coupling = self.coupling_strengths[q1, q2]
            if coupling < 0.02:
                base_fidelity *= 0.98  # Weak coupling reduces fidelity
                
        else:
            # Multi-qubit gate (rarely used)
            base_fidelity = np.prod([self.gate_fidelities[q] for q in qubit_ids]) * 0.9
        
        # Add small random fluctuations
        noise = np.random.normal(0, 0.001)
        return np.clip(base_fidelity + noise, 0.9, 1.0)
    
    def simulate_readout(self, qubit_id: int, num_shots: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Simulate quantum measurement with realistic readout errors.
        
        Args:
            qubit_id: Index of the qubit to measure
            num_shots: Number of measurement shots
            
        Returns:
            Tuple of (measurement results, readout fidelity)
        """
        # Get the probability of measuring |1⟩
        prob_1 = self.get_measurement_probability(qubit_id)
        
        # Simulate readout with errors
        readout_fidelity = 0.99 + np.random.normal(0, 0.01)
        readout_fidelity = np.clip(readout_fidelity, 0.95, 1.0)
        
        # Generate ideal measurements
        ideal_results = np.random.binomial(1, prob_1, num_shots)
        
        # Add readout errors
        error_prob = 1 - readout_fidelity
        error_mask = np.random.random(num_shots) < error_prob
        results = ideal_results.copy()
        results[error_mask] = 1 - results[error_mask]  # Flip errored measurements
        
        return results, readout_fidelity
    
    def get_measurement_probability(self, qubit_id: int) -> float:
        """
        Get the probability of measuring a qubit in the |1⟩ state.
        
        Args:
            qubit_id: Index of the qubit
            
        Returns:
            Probability of measuring |1⟩
        """
        # Calculate reduced density matrix for the qubit
        prob_1 = 0.0
        
        for i in range(2**self.num_qubits):
            if (i >> qubit_id) & 1:  # Check if qubit is in |1⟩ state
                prob_1 += abs(self.state[i])**2
                
        return prob_1
    
    def apply_single_qubit_gate(self, gate_type: str, qubit_id: int, 
                               angle: Optional[float] = None) -> None:
        """
        Apply a single qubit gate to the quantum state.
        
        Args:
            gate_type: Type of gate ('X', 'Y', 'Z', 'H', 'Rx', 'Ry', 'Rz')
            qubit_id: Index of the target qubit
            angle: Rotation angle for parameterized gates
        """
        # Define Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        H = (X + Z) / np.sqrt(2)
        
        # Select gate matrix
        if gate_type == 'I':
            gate_matrix = I
        elif gate_type == 'X':
            gate_matrix = X
        elif gate_type == 'Y':
            gate_matrix = Y
        elif gate_type == 'Z':
            gate_matrix = Z
        elif gate_type == 'H':
            gate_matrix = H
        elif gate_type == 'Rx' and angle is not None:
            gate_matrix = np.cos(angle/2) * I - 1j * np.sin(angle/2) * X
        elif gate_type == 'Ry' and angle is not None:
            gate_matrix = np.cos(angle/2) * I - 1j * np.sin(angle/2) * Y
        elif gate_type == 'Rz' and angle is not None:
            gate_matrix = np.cos(angle/2) * I - 1j * np.sin(angle/2) * Z
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
        
        # Apply gate to the state vector
        self._apply_gate_to_state(gate_matrix, [qubit_id])
    
    def apply_two_qubit_gate(self, gate_type: str, control_qubit: int, target_qubit: int) -> None:
        """
        Apply a two-qubit gate to the quantum state.
        
        Args:
            gate_type: Type of gate ('CNOT', 'CZ', 'SWAP')
            control_qubit: Index of the control qubit
            target_qubit: Index of the target qubit
        """
        if gate_type == 'CNOT':
            # Controlled-X gate
            for i in range(2**self.num_qubits):
                if (i >> control_qubit) & 1:  # Control qubit is |1⟩
                    # Flip target qubit
                    j = i ^ (1 << target_qubit)
                    self.state[i], self.state[j] = self.state[j], self.state[i]
                    
        elif gate_type == 'CZ':
            # Controlled-Z gate
            for i in range(2**self.num_qubits):
                if ((i >> control_qubit) & 1) and ((i >> target_qubit) & 1):
                    self.state[i] *= -1
                    
        elif gate_type == 'SWAP':
            # Swap gate
            for i in range(2**self.num_qubits):
                control_bit = (i >> control_qubit) & 1
                target_bit = (i >> target_qubit) & 1
                
                if control_bit != target_bit:
                    # Swap the bits
                    j = i ^ (1 << control_qubit) ^ (1 << target_qubit)
                    if i < j:  # Avoid double swapping
                        self.state[i], self.state[j] = self.state[j], self.state[i]
    
    def _apply_gate_to_state(self, gate_matrix: np.ndarray, qubit_indices: List[int]) -> None:
        """
        Apply a gate matrix to specific qubits in the state vector.
        
        Args:
            gate_matrix: 2^n x 2^n gate matrix where n is the number of qubits
            qubit_indices: List of qubit indices the gate acts on
        """
        # This is a simplified implementation for single qubit gates
        # For a full implementation, we would need tensor product operations
        if len(qubit_indices) == 1:
            qubit_id = qubit_indices[0]
            new_state = np.zeros_like(self.state)
            
            for i in range(2**self.num_qubits):
                # Extract the bit values
                bit_val = (i >> qubit_id) & 1
                
                # Apply gate to this qubit
                if bit_val == 0:
                    # |0⟩ component
                    new_state[i] += gate_matrix[0, 0] * self.state[i]
                    new_state[i | (1 << qubit_id)] += gate_matrix[1, 0] * self.state[i]
                else:
                    # |1⟩ component
                    new_state[i] += gate_matrix[1, 1] * self.state[i]
                    new_state[i & ~(1 << qubit_id)] += gate_matrix[0, 1] * self.state[i]
            
            self.state = new_state
    
    def add_decoherence(self, time_step: float) -> None:
        """
        Add decoherence effects to the quantum state.
        
        Args:
            time_step: Time step in microseconds
        """
        for qubit_id in range(self.num_qubits):
            t1, t2 = self.get_coherence_times(qubit_id, include_noise=False)
            
            # T1 decay (amplitude damping)
            gamma1 = 1 / t1
            p1 = 1 - np.exp(-time_step / t1)
            
            # T2 dephasing
            gamma2 = 1 / t2
            p2 = 1 - np.exp(-time_step / t2)
            
            # Apply simplified decoherence (this is a basic model)
            # In practice, this would involve Kraus operators
            if np.random.random() < p1:
                # Apply amplitude damping with some probability
                prob_1 = self.get_measurement_probability(qubit_id)
                if prob_1 > 0.01:  # Only if qubit has significant |1⟩ probability
                    self.apply_single_qubit_gate('X', qubit_id)  # Simplified decay
            
            if np.random.random() < p2:
                # Apply dephasing with some probability
                phase = np.random.uniform(0, 2*np.pi)
                self.apply_single_qubit_gate('Rz', qubit_id, phase)
    
    def get_system_entropy(self) -> float:
        """
        Calculate the von Neumann entropy of the quantum state.
        
        Returns:
            von Neumann entropy in bits
        """
        # Calculate density matrix eigenvalues
        rho = np.outer(self.state.conj(), self.state)
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        
        # Calculate entropy
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        return entropy.real
    
    def get_entanglement_entropy(self, subsystem_qubits: List[int]) -> float:
        """
        Calculate the entanglement entropy of a subsystem.
        
        Args:
            subsystem_qubits: List of qubit indices forming the subsystem
            
        Returns:
            Entanglement entropy in bits
        """
        # This is a simplified calculation for demonstration
        # A full implementation would require partial trace calculations
        if len(subsystem_qubits) == 1:
            # Single qubit entropy
            prob_0 = 1 - self.get_measurement_probability(subsystem_qubits[0])
            prob_1 = self.get_measurement_probability(subsystem_qubits[0])
            
            if prob_0 > 1e-12 and prob_1 > 1e-12:
                return -(prob_0 * np.log2(prob_0) + prob_1 * np.log2(prob_1))
            else:
                return 0.0
        
        # For multi-qubit subsystems, return a placeholder
        return 0.5 * len(subsystem_qubits)  # Rough estimate
    
    def generate_process_tomography_data(self, gate_type: str, qubit_id: int) -> Dict[str, Any]:
        """
        Generate simulated process tomography data for a quantum gate.
        
        Args:
            gate_type: Type of gate to characterize
            qubit_id: Target qubit index
            
        Returns:
            Dictionary containing process tomography results
        """
        # Pauli basis states for tomography
        pauli_states = {
            'X+': (np.array([1, 1]) / np.sqrt(2), 'X+'),
            'X-': (np.array([1, -1]) / np.sqrt(2), 'X-'),
            'Y+': (np.array([1, 1j]) / np.sqrt(2), 'Y+'),
            'Y-': (np.array([1, -1j]) / np.sqrt(2), 'Y-'),
            'Z+': (np.array([1, 0]), 'Z+'),
            'Z-': (np.array([0, 1]), 'Z-')
        }
        
        results = {}
        
        for input_state, (state_vec, state_name) in pauli_states.items():
            # Set initial state
            self.reset_system()
            # Apply input state preparation (simplified)
            
            # Apply the gate
            self.apply_single_qubit_gate(gate_type, qubit_id)
            
            # Measure in all Pauli bases
            measurements = {}
            for basis in ['X', 'Y', 'Z']:
                # Simulate measurement in different bases
                prob = np.random.random()  # Simplified measurement simulation
                measurements[basis] = prob
                
            results[input_state] = measurements
        
        # Calculate process fidelity
        ideal_fidelity = self.calculate_gate_fidelity(gate_type, [qubit_id])
        results['process_fidelity'] = ideal_fidelity
        results['gate_type'] = gate_type
        results['qubit_id'] = qubit_id
        
        return results
