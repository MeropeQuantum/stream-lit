"""
Database integration utilities for the QuantumOS dashboard.
Handles data synchronization between live systems and the database.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import time
from database.operations import quantum_db
from .data_generator import DataGenerator

class DatabaseManager:
    """Manages database operations and data synchronization for the dashboard."""
    
    def __init__(self):
        self.data_gen = DataGenerator()
        self.is_collecting = False
        self.collection_thread = None
        
    def start_data_collection(self, interval_seconds: int = 30):
        """Start background data collection and storage."""
        if self.is_collecting:
            return False
            
        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._data_collection_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.collection_thread.start()
        return True
    
    def stop_data_collection(self):
        """Stop background data collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
    
    def _data_collection_loop(self, interval_seconds: int):
        """Background loop for collecting and storing telemetry data."""
        while self.is_collecting:
            try:
                # Collect and store qubit telemetry
                self._collect_qubit_telemetry()
                
                # Collect and store system metrics
                self._collect_system_metrics()
                
                # Collect and store temperature readings
                self._collect_temperature_readings()
                
                # Generate occasional alerts
                if np.random.random() < 0.1:  # 10% chance per cycle
                    self._generate_system_alert()
                    
                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"Error in data collection loop: {e}")
                time.sleep(interval_seconds)
    
    def _collect_qubit_telemetry(self):
        """Collect and store qubit telemetry data for all qubits."""
        num_qubits = 8
        telemetry_data = []
        
        for qubit_id in range(num_qubits):
            # Generate realistic telemetry data
            frequency = 4.5 + qubit_id * 0.2 + np.random.normal(0, 0.001)
            t1_coherence = 120 + np.random.normal(0, 15)
            t2_coherence = 85 + np.random.normal(0, 10)
            gate_fidelity = 99.5 + np.random.normal(0, 0.3)
            readout_fidelity = 99.2 + np.random.normal(0, 0.4)
            temperature = 0.015 + np.random.normal(0, 0.002)
            
            # Determine status based on thresholds
            status = 'operational'
            if gate_fidelity < 99.0 or readout_fidelity < 98.5:
                status = 'degraded'
            elif t1_coherence < 80 or t2_coherence < 60:
                status = 'maintenance_required'
            
            telemetry_data.append({
                'qubit_id': qubit_id,
                'frequency_ghz': frequency,
                't1_coherence_us': max(t1_coherence, 50),
                't2_coherence_us': max(t2_coherence, 30),
                'gate_fidelity_percent': np.clip(gate_fidelity, 95, 100),
                'readout_fidelity_percent': np.clip(readout_fidelity, 95, 100),
                'temperature_mk': max(temperature, 0.010),
                'status': status
            })
        
        quantum_db.store_qubit_telemetry(telemetry_data)
    
    def _collect_system_metrics(self):
        """Collect and store system-wide metrics."""
        metrics = {
            'uptime_percent': 99.7 + np.random.normal(0, 0.1),
            'error_rate_percent': 0.5 + np.random.normal(0, 0.1),
            'queue_length': np.random.randint(0, 15),
            'active_jobs': np.random.randint(0, 8),
            'cpu_usage_percent': 60 + np.random.normal(0, 10),
            'memory_usage_percent': 45 + np.random.normal(0, 8),
            'network_io_mbps': 12 + np.random.normal(0, 3)
        }
        
        # Ensure realistic bounds
        metrics['uptime_percent'] = np.clip(metrics['uptime_percent'], 95, 100)
        metrics['error_rate_percent'] = np.clip(metrics['error_rate_percent'], 0, 5)
        metrics['cpu_usage_percent'] = np.clip(metrics['cpu_usage_percent'], 0, 100)
        metrics['memory_usage_percent'] = np.clip(metrics['memory_usage_percent'], 0, 100)
        metrics['network_io_mbps'] = max(metrics['network_io_mbps'], 0)
        
        quantum_db.store_system_metrics(metrics)
    
    def _collect_temperature_readings(self):
        """Collect and store cryogenic temperature readings."""
        stages = {
            'mixing_chamber': 0.015,
            'still': 0.7,
            '4k': 4.2,
            '50k': 52.0
        }
        
        readings = []
        for stage, base_temp in stages.items():
            if stage == 'mixing_chamber':
                temp = base_temp + np.random.normal(0, 0.001)
                status = 'normal' if temp < 0.02 else 'warning'
            elif stage == 'still':
                temp = base_temp + np.random.normal(0, 0.05)
                status = 'normal' if temp < 1.0 else 'warning'
            elif stage == '4k':
                temp = base_temp + np.random.normal(0, 0.2)
                status = 'normal' if temp < 5.0 else 'warning'
            else:  # 50k
                temp = base_temp + np.random.normal(0, 2.0)
                status = 'normal' if temp < 60 else 'warning'
            
            readings.append({
                'stage': stage,
                'temperature_k': max(temp, base_temp * 0.5),
                'status': status
            })
        
        quantum_db.store_temperature_readings(readings)
    
    def _generate_system_alert(self):
        """Generate realistic system alerts."""
        alert_types = ['info', 'warning', 'success']
        components = ['Calibration', 'Coherence', 'Scheduler', 'Optimization', 'Readout']
        
        messages = {
            'info': [
                "Calibration scheduled in 2h",
                "Daily backup completed",
                "System maintenance window approaching",
                "New quantum algorithm available"
            ],
            'warning': [
                "Q3 coherence below threshold",
                "Temperature excursion detected",
                "High error rate on readout line 2",
                "Memory usage approaching limit"
            ],
            'success': [
                "Auto-tune completed",
                "Calibration improved fidelity by 0.2%",
                "System recovery successful",
                "Optimization cycle completed"
            ]
        }
        
        alert_type = np.random.choice(alert_types)
        component = np.random.choice(components)
        message = np.random.choice(messages[alert_type])
        
        priority = 'low'
        if alert_type == 'warning':
            priority = np.random.choice(['medium', 'high'], p=[0.7, 0.3])
        
        alert_data = {
            'alert_type': alert_type,
            'component': component,
            'message': message,
            'priority': priority,
            'alert_metadata': {
                'generated_by': 'automated_system',
                'auto_generated': True
            }
        }
        
        quantum_db.create_system_alert(alert_data)
    
    def get_dashboard_data(self, hours: int = 24) -> Dict[str, Any]:
        """Retrieve comprehensive dashboard data from the database."""
        try:
            # Get recent telemetry data
            telemetry_df = quantum_db.get_recent_qubit_telemetry(hours=hours)
            
            # Get system metrics
            metrics_df = quantum_db.get_system_metrics_history(hours=hours)
            
            # Get temperature data
            temp_df = quantum_db.get_temperature_history(hours=hours)
            
            # Get active alerts
            alerts_df = quantum_db.get_active_alerts()
            
            # Get fidelity trends
            fidelity_trends = quantum_db.get_fidelity_trends(hours=hours)
            
            # Get system health summary
            health_summary = quantum_db.get_system_health_summary()
            
            return {
                'telemetry': telemetry_df,
                'system_metrics': metrics_df,
                'temperatures': temp_df,
                'alerts': alerts_df,
                'fidelity_trends': fidelity_trends,
                'health_summary': health_summary,
                'data_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"Error retrieving dashboard data: {e}")
            return {}
    
    def store_pulse_sequence_execution(self, sequence_name: str, pulse_params: Dict, 
                                     target_qubits: List[int], predicted_fidelity: float) -> str:
        """Store a pulse sequence execution."""
        sequence_data = {
            'sequence_name': sequence_name,
            'target_qubits': target_qubits,
            'pulse_parameters': pulse_params,
            'predicted_fidelity': predicted_fidelity,
            'status': 'executed'
        }
        
        sequence_id = quantum_db.store_pulse_sequence(sequence_data)
        
        # Simulate execution results
        actual_fidelity = predicted_fidelity + np.random.normal(0, 0.5)
        execution_time = pulse_params.get('duration', 20) * len(target_qubits)
        
        results = {
            'actual_fidelity': np.clip(actual_fidelity, 90, 100),
            'execution_time_ns': execution_time,
            'status': 'completed'
        }
        
        quantum_db.update_pulse_sequence_results(sequence_id, results)
        return sequence_id
    
    def store_quantum_state_snapshot(self, state_name: str, num_qubits: int, 
                                   state_data: Dict) -> str:
        """Store a quantum state snapshot."""
        state_record = {
            'state_name': state_name,
            'num_qubits': num_qubits,
            'state_vector': state_data.get('state_vector'),
            'fidelity': state_data.get('fidelity'),
            'purity': state_data.get('purity', 1.0),
            'entanglement_entropy': state_data.get('entanglement_entropy', 0.0),
            'measurement_results': state_data.get('measurements')
        }
        
        return quantum_db.store_quantum_state(state_record)
    
    def get_calibration_recommendations(self) -> List[Dict]:
        """Analyze recent data and provide calibration recommendations."""
        try:
            # Get recent telemetry data
            recent_data = quantum_db.get_recent_qubit_telemetry(hours=1)
            
            if recent_data.empty:
                return []
            
            recommendations = []
            
            # Analyze each qubit for calibration needs
            for qubit_id in recent_data['qubit_id'].unique():
                qubit_data = recent_data[recent_data['qubit_id'] == qubit_id]
                
                avg_gate_fidelity = qubit_data['gate_fidelity_percent'].mean()
                avg_readout_fidelity = qubit_data['readout_fidelity_percent'].mean()
                avg_t1 = qubit_data['t1_coherence_us'].mean()
                avg_t2 = qubit_data['t2_coherence_us'].mean()
                
                # Check for calibration needs
                if avg_gate_fidelity < 99.0:
                    recommendations.append({
                        'type': 'gate_calibration',
                        'target_qubits': [int(qubit_id)],
                        'priority': 'high' if avg_gate_fidelity < 98.5 else 'medium',
                        'reason': f'Gate fidelity {avg_gate_fidelity:.2f}% below threshold',
                        'estimated_improvement': '0.3-0.5%'
                    })
                
                if avg_readout_fidelity < 99.0:
                    recommendations.append({
                        'type': 'readout_calibration',
                        'target_qubits': [int(qubit_id)],
                        'priority': 'medium',
                        'reason': f'Readout fidelity {avg_readout_fidelity:.2f}% below threshold',
                        'estimated_improvement': '0.2-0.4%'
                    })
                
                if avg_t1 < 100:
                    recommendations.append({
                        'type': 'coherence_optimization',
                        'target_qubits': [int(qubit_id)],
                        'priority': 'medium',
                        'reason': f'T1 coherence {avg_t1:.1f}μs below optimal range',
                        'estimated_improvement': '10-20μs'
                    })
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating calibration recommendations: {e}")
            return []

# Global database manager instance
db_manager = DatabaseManager()