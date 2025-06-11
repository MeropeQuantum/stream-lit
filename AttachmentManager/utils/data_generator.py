import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd

class DataGenerator:
    """
    Data generator for realistic quantum control telemetry and metrics.
    Generates authentic-looking quantum system data for dashboard visualization.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Base parameters for realistic quantum system data
        self.base_frequency = 4.8  # GHz
        self.frequency_drift_rate = 0.001  # GHz per hour
        self.temperature_base = 0.015  # Kelvin (mixing chamber)
        
        # Coherence time parameters
        self.t1_base_range = (80, 150)  # microseconds
        self.t2_base_range = (60, 120)  # microseconds
        
        # Fidelity parameters
        self.single_gate_fidelity_base = 0.995
        self.two_gate_fidelity_base = 0.955
        self.readout_fidelity_base = 0.992
        
        # Noise characteristics
        self.frequency_noise_amplitude = 0.001  # GHz RMS
        self.coherence_time_noise = 0.15  # 15% RMS variation
        self.fidelity_noise = 0.002  # 0.2% RMS variation
        
        # System state tracking
        self.last_calibration_time = datetime.now() - timedelta(hours=2)
        self.drift_accumulator = 0.0
        
    def generate_frequency_data(self, num_points: int, 
                              time_span_seconds: Optional[float] = None) -> np.ndarray:
        """
        Generate realistic qubit frequency data over time.
        
        Args:
            num_points: Number of data points to generate
            time_span_seconds: Time span in seconds (default: num_points seconds)
            
        Returns:
            Array of frequency values in GHz
        """
        if time_span_seconds is None:
            time_span_seconds = num_points
            
        # Time points
        time_points = np.linspace(0, time_span_seconds, num_points)
        
        # Base frequency with slow drift
        drift_rate = self.frequency_drift_rate / 3600  # Convert to per second
        frequency_drift = drift_rate * time_points
        
        # Add 1/f noise (low frequency drift)
        f_noise = self._generate_1f_noise(num_points, time_span_seconds) * 0.0001
        
        # Add white noise
        white_noise = np.random.normal(0, self.frequency_noise_amplitude, num_points)
        
        # Periodic calibration corrections (every 4 hours typically)
        calibration_period = 4 * 3600  # 4 hours in seconds
        calibration_corrections = 0.0002 * np.sin(2 * np.pi * time_points / calibration_period)
        
        # Combine all effects
        frequencies = (self.base_frequency + 
                      frequency_drift + 
                      f_noise + 
                      white_noise + 
                      calibration_corrections)
        
        return frequencies
    
    def generate_coherence_data(self, num_points: int, coherence_type: str = 'T1') -> np.ndarray:
        """
        Generate realistic coherence time data (T1 or T2*).
        
        Args:
            num_points: Number of data points
            coherence_type: Either 'T1' or 'T2*'
            
        Returns:
            Array of coherence times in microseconds
        """
        if coherence_type.upper() == 'T1':
            base_range = self.t1_base_range
        else:  # T2*
            base_range = self.t2_base_range
            
        # Base coherence time with some variation
        base_time = np.random.uniform(base_range[0], base_range[1])
        
        # Temperature-dependent variations
        temperature_effect = self._generate_temperature_coherence_effect(num_points)
        
        # Charge noise effects (more prominent in T2*)
        if coherence_type.upper() == 'T2*':
            charge_noise = self._generate_charge_noise_effect(num_points) * 0.3
        else:
            charge_noise = self._generate_charge_noise_effect(num_points) * 0.1
            
        # Random telegraph noise (occasional jumps)
        rtn_effect = self._generate_rtn_effect(num_points, base_time * 0.2)
        
        # Combine effects
        coherence_times = base_time * (1 + temperature_effect + charge_noise + rtn_effect)
        
        # Add measurement noise
        measurement_noise = np.random.normal(0, base_time * 0.05, num_points)
        coherence_times += measurement_noise
        
        # Ensure positive values
        coherence_times = np.maximum(coherence_times, base_time * 0.3)
        
        return coherence_times
    
    def generate_fidelity_data(self, num_points: int, gate_type: str = 'single') -> np.ndarray:
        """
        Generate realistic gate fidelity data.
        
        Args:
            num_points: Number of data points
            gate_type: Type of gate ('single', 'two_qubit', 'readout')
            
        Returns:
            Array of fidelity values (0-1 scale, converted to percentage)
        """
        if gate_type == 'single':
            base_fidelity = self.single_gate_fidelity_base
        elif gate_type == 'two_qubit':
            base_fidelity = self.two_gate_fidelity_base
        else:  # readout
            base_fidelity = self.readout_fidelity_base
            
        # Slow drift due to calibration aging
        drift_factor = self._generate_calibration_drift(num_points)
        
        # Temperature fluctuation effects
        temp_effect = self._generate_temperature_fidelity_effect(num_points)
        
        # Pulse amplitude/timing errors
        pulse_errors = np.random.normal(0, self.fidelity_noise * 0.5, num_points)
        
        # Crosstalk effects (more prominent in two-qubit gates)
        if gate_type == 'two_qubit':
            crosstalk_effect = np.random.normal(0, 0.003, num_points)
        else:
            crosstalk_effect = np.random.normal(0, 0.001, num_points)
            
        # Combine effects
        fidelities = base_fidelity + drift_factor + temp_effect + pulse_errors + crosstalk_effect
        
        # Add occasional calibration improvements
        calibration_jumps = self._generate_calibration_jumps(num_points, 0.002)
        fidelities += calibration_jumps
        
        # Ensure realistic bounds
        fidelities = np.clip(fidelities, 0.85, 0.999)
        
        # Convert to percentage
        return fidelities * 100
    
    def generate_gate_duration_data(self, num_points: int) -> np.ndarray:
        """
        Generate realistic gate duration data.
        
        Args:
            num_points: Number of data points
            
        Returns:
            Array of gate durations in nanoseconds
        """
        base_duration = 20.0  # nanoseconds
        
        # Temperature-dependent timing variations
        temp_variation = np.random.normal(0, 0.1, num_points)
        
        # Control electronics jitter
        jitter = np.random.normal(0, 0.05, num_points)
        
        # Calibration-dependent adjustments
        cal_adjustments = np.random.normal(0, 0.2, num_points)
        
        durations = base_duration + temp_variation + jitter + cal_adjustments
        
        # Ensure positive durations
        durations = np.maximum(durations, 15.0)
        
        return durations
    
    def generate_temperature_data(self, num_points: int, stage: str = 'mixing_chamber') -> np.ndarray:
        """
        Generate realistic cryogenic temperature data.
        
        Args:
            num_points: Number of data points
            stage: Temperature stage ('mixing_chamber', 'still', '4k', '50k')
            
        Returns:
            Array of temperatures in Kelvin
        """
        stage_temps = {
            'mixing_chamber': 0.015,
            'still': 0.7,
            '4k': 4.2,
            '50k': 52.0
        }
        
        base_temp = stage_temps.get(stage, 0.015)
        
        # Temperature stability varies by stage
        if stage == 'mixing_chamber':
            stability = 0.001  # Very stable
            regulation_period = 300  # 5 minute regulation cycle
        elif stage == 'still':
            stability = 0.05
            regulation_period = 600
        elif stage == '4k':
            stability = 0.2
            regulation_period = 1800
        else:  # 50k
            stability = 2.0
            regulation_period = 3600
            
        # Generate time-dependent variations
        time_points = np.arange(num_points)
        
        # Regulation cycles
        regulation_variation = stability * 0.3 * np.sin(2 * np.pi * time_points / regulation_period)
        
        # Random fluctuations
        random_fluctuations = np.random.normal(0, stability * 0.5, num_points)
        
        # Occasional temperature excursions
        excursions = self._generate_temperature_excursions(num_points, base_temp * 0.1)
        
        temperatures = base_temp + regulation_variation + random_fluctuations + excursions
        
        # Ensure physically reasonable temperatures
        temperatures = np.maximum(temperatures, base_temp * 0.5)
        
        return temperatures
    
    def generate_pulse_sequence_data(self, sequence_length: int, 
                                   pulse_type: str = 'gaussian') -> Dict[str, np.ndarray]:
        """
        Generate realistic pulse sequence data.
        
        Args:
            sequence_length: Number of pulses in sequence
            pulse_type: Type of pulse ('gaussian', 'square', 'drag')
            
        Returns:
            Dictionary containing I/Q data and timing information
        """
        # Pulse parameters
        pulse_duration = 20.0  # nanoseconds
        pulse_spacing = 5.0    # nanoseconds gap between pulses
        sampling_rate = 2.0    # GS/s
        
        total_duration = sequence_length * (pulse_duration + pulse_spacing)
        num_samples = int(total_duration * sampling_rate)
        
        time_axis = np.linspace(0, total_duration, num_samples)
        i_data = np.zeros(num_samples)
        q_data = np.zeros(num_samples)
        
        for pulse_idx in range(sequence_length):
            pulse_start = pulse_idx * (pulse_duration + pulse_spacing)
            pulse_end = pulse_start + pulse_duration
            
            # Find sample indices for this pulse
            start_idx = int(pulse_start * sampling_rate)
            end_idx = int(pulse_end * sampling_rate)
            
            if end_idx > num_samples:
                end_idx = num_samples
                
            pulse_samples = end_idx - start_idx
            pulse_time = np.linspace(0, pulse_duration, pulse_samples)
            
            # Generate pulse envelope
            if pulse_type == 'gaussian':
                sigma = pulse_duration / 6
                envelope = np.exp(-((pulse_time - pulse_duration/2)**2) / (2*sigma**2))
            elif pulse_type == 'square':
                envelope = np.ones(pulse_samples)
            elif pulse_type == 'drag':
                sigma = pulse_duration / 6
                gauss = np.exp(-((pulse_time - pulse_duration/2)**2) / (2*sigma**2))
                deriv = -(pulse_time - pulse_duration/2) / sigma**2 * gauss
                envelope = gauss + 0.2 * deriv
            else:
                envelope = np.ones(pulse_samples)
            
            # Add amplitude and phase variations
            amplitude = 0.5 + np.random.normal(0, 0.02)  # 4% amplitude noise
            phase = np.random.uniform(0, 2*np.pi)
            
            # Carrier frequency (assuming 100 MHz IF)
            carrier_freq = 0.1  # GHz
            carrier = np.cos(2*np.pi * carrier_freq * pulse_time + phase)
            
            # I/Q components
            i_component = amplitude * envelope * np.cos(phase)
            q_component = amplitude * envelope * np.sin(phase)
            
            # Add to sequence
            i_data[start_idx:end_idx] = i_component
            q_data[start_idx:end_idx] = q_component
        
        # Add AWG noise and imperfections
        awg_noise_i = np.random.normal(0, 0.01, num_samples)
        awg_noise_q = np.random.normal(0, 0.01, num_samples)
        
        i_data += awg_noise_i
        q_data += awg_noise_q
        
        return {
            'time': time_axis,
            'i_data': i_data,
            'q_data': q_data,
            'sequence_length': sequence_length,
            'pulse_type': pulse_type
        }
    
    # Helper methods for generating realistic noise and effects
    
    def _generate_1f_noise(self, num_points: int, time_span: float) -> np.ndarray:
        """Generate 1/f noise using spectral shaping."""
        frequencies = np.fft.fftfreq(num_points, time_span / num_points)
        frequencies[0] = 1e-10  # Avoid division by zero
        
        # 1/f power spectral density
        psd = 1.0 / np.abs(frequencies)
        psd[0] = psd[1]  # Set DC component
        
        # Generate white noise and shape it
        white_noise = np.random.normal(0, 1, num_points)
        noise_fft = np.fft.fft(white_noise)
        shaped_fft = noise_fft * np.sqrt(psd)
        
        return np.real(np.fft.ifft(shaped_fft))
    
    def _generate_temperature_coherence_effect(self, num_points: int) -> np.ndarray:
        """Generate temperature-dependent coherence variations."""
        # Temperature fluctuations affect T1 and T2* differently
        temp_variation = np.random.normal(0, 0.05, num_points)  # 5% temperature effect
        return temp_variation
    
    def _generate_charge_noise_effect(self, num_points: int) -> np.ndarray:
        """Generate charge noise effects on coherence."""
        # Charge noise typically has 1/f characteristics
        return self._generate_1f_noise(num_points, num_points) * 0.1
    
    def _generate_rtn_effect(self, num_points: int, amplitude: float) -> np.ndarray:
        """Generate random telegraph noise effects."""
        # Occasional sudden jumps in coherence time
        rtn = np.zeros(num_points)
        jump_probability = 0.01  # 1% chance per point
        
        for i in range(num_points):
            if np.random.random() < jump_probability:
                # Duration of the jump
                duration = np.random.randint(10, 50)
                end_idx = min(i + duration, num_points)
                rtn[i:end_idx] = np.random.choice([-1, 1]) * amplitude / 100
                
        return rtn
    
    def _generate_calibration_drift(self, num_points: int) -> np.ndarray:
        """Generate slow drift due to calibration aging."""
        # Linear drift with small fluctuations
        drift_rate = -0.0001 / 3600  # Slow degradation per second
        time_points = np.arange(num_points)
        
        linear_drift = drift_rate * time_points
        fluctuations = self._generate_1f_noise(num_points, num_points) * 0.0001
        
        return linear_drift + fluctuations
    
    def _generate_temperature_fidelity_effect(self, num_points: int) -> np.ndarray:
        """Generate temperature effects on fidelity."""
        temp_noise = np.random.normal(0, 0.001, num_points)
        return temp_noise
    
    def _generate_calibration_jumps(self, num_points: int, amplitude: float) -> np.ndarray:
        """Generate occasional calibration improvement jumps."""
        jumps = np.zeros(num_points)
        jump_probability = 0.005  # 0.5% chance per point
        
        for i in range(num_points):
            if np.random.random() < jump_probability:
                jumps[i:] += amplitude  # Persistent improvement
                
        return jumps
    
    def _generate_temperature_excursions(self, num_points: int, amplitude: float) -> np.ndarray:
        """Generate occasional temperature excursions."""
        excursions = np.zeros(num_points)
        excursion_probability = 0.002  # 0.2% chance per point
        
        for i in range(num_points):
            if np.random.random() < excursion_probability:
                # Duration and shape of excursion
                duration = np.random.randint(20, 100)
                end_idx = min(i + duration, num_points)
                
                # Exponential decay back to baseline
                decay_profile = np.exp(-np.arange(duration) / (duration / 3))
                excursion_amplitude = amplitude * np.random.uniform(0.5, 2.0)
                
                if end_idx - i < len(decay_profile):
                    decay_profile = decay_profile[:end_idx - i]
                    
                excursions[i:end_idx] += excursion_amplitude * decay_profile
                
        return excursions
    
    def generate_system_health_metrics(self) -> Dict[str, Any]:
        """
        Generate comprehensive system health metrics.
        
        Returns:
            Dictionary containing various system health indicators
        """
        current_time = datetime.now()
        
        metrics = {
            'timestamp': current_time,
            'uptime_hours': (current_time - self.last_calibration_time).total_seconds() / 3600,
            'overall_health': np.random.uniform(85, 99),
            'temperature_stability': np.random.uniform(95, 100),
            'frequency_stability': np.random.uniform(90, 99),
            'coherence_stability': np.random.uniform(88, 97),
            'fidelity_trend': np.random.choice(['improving', 'stable', 'degrading'], p=[0.2, 0.6, 0.2]),
            'calibration_needed': np.random.choice([True, False], p=[0.1, 0.9]),
            'error_rate': np.random.uniform(0.1, 1.5),
            'queue_efficiency': np.random.uniform(80, 98),
            'hardware_status': {
                'dilution_refrigerator': np.random.choice(['optimal', 'good', 'warning'], p=[0.7, 0.25, 0.05]),
                'electronics': np.random.choice(['optimal', 'good', 'warning'], p=[0.6, 0.35, 0.05]),
                'lasers': np.random.choice(['optimal', 'good', 'warning'], p=[0.8, 0.18, 0.02]),
                'microwave_sources': np.random.choice(['optimal', 'good', 'warning'], p=[0.75, 0.22, 0.03])
            }
        }
        
        return metrics
    
    def export_telemetry_data(self, duration_hours: float = 24) -> pd.DataFrame:
        """
        Export comprehensive telemetry data for analysis.
        
        Args:
            duration_hours: Duration of data to export in hours
            
        Returns:
            Pandas DataFrame with telemetry data
        """
        num_points = int(duration_hours * 60)  # One point per minute
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=duration_hours)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=num_points)
        
        # Generate data for multiple qubits
        data_records = []
        
        for qubit_id in range(8):  # 8 qubits
            frequencies = self.generate_frequency_data(num_points, duration_hours * 3600)
            t1_times = self.generate_coherence_data(num_points, 'T1')
            t2_times = self.generate_coherence_data(num_points, 'T2*')
            fidelities = self.generate_fidelity_data(num_points, 'single')
            
            for i, timestamp in enumerate(timestamps):
                record = {
                    'timestamp': timestamp,
                    'qubit_id': qubit_id,
                    'frequency_ghz': frequencies[i] + qubit_id * 0.2,  # Offset for each qubit
                    't1_us': t1_times[i],
                    't2_us': t2_times[i],
                    'gate_fidelity_percent': fidelities[i],
                    'readout_fidelity_percent': self.generate_fidelity_data(1, 'readout')[0]
                }
                data_records.append(record)
        
        return pd.DataFrame(data_records)
