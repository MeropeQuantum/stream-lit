"""
Database models for the QuantumOS Enterprise Control Platform.
Defines SQLAlchemy models for storing quantum telemetry, system metrics, and operational data.
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid
import os

Base = declarative_base()

class QubitTelemetry(Base):
    """Store real-time qubit telemetry data."""
    __tablename__ = 'qubit_telemetry'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    qubit_id = Column(Integer, nullable=False, index=True)
    frequency_ghz = Column(Float, nullable=False)
    t1_coherence_us = Column(Float, nullable=False)
    t2_coherence_us = Column(Float, nullable=False)
    gate_fidelity_percent = Column(Float, nullable=False)
    readout_fidelity_percent = Column(Float, nullable=False)
    temperature_mk = Column(Float, nullable=True)
    status = Column(String(50), default='operational')
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'timestamp': self.timestamp.isoformat(),
            'qubit_id': self.qubit_id,
            'frequency_ghz': self.frequency_ghz,
            't1_coherence_us': self.t1_coherence_us,
            't2_coherence_us': self.t2_coherence_us,
            'gate_fidelity_percent': self.gate_fidelity_percent,
            'readout_fidelity_percent': self.readout_fidelity_percent,
            'temperature_mk': self.temperature_mk,
            'status': self.status
        }

class SystemMetrics(Base):
    """Store system-wide performance metrics."""
    __tablename__ = 'system_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    uptime_percent = Column(Float, nullable=False)
    error_rate_percent = Column(Float, nullable=False)
    queue_length = Column(Integer, nullable=False)
    active_jobs = Column(Integer, nullable=False)
    cpu_usage_percent = Column(Float, nullable=True)
    memory_usage_percent = Column(Float, nullable=True)
    network_io_mbps = Column(Float, nullable=True)
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'timestamp': self.timestamp.isoformat(),
            'uptime_percent': self.uptime_percent,
            'error_rate_percent': self.error_rate_percent,
            'queue_length': self.queue_length,
            'active_jobs': self.active_jobs,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_percent': self.memory_usage_percent,
            'network_io_mbps': self.network_io_mbps
        }

class CryogenicTemperatures(Base):
    """Store cryogenic system temperature readings."""
    __tablename__ = 'cryogenic_temperatures'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    stage = Column(String(50), nullable=False)  # mixing_chamber, still, 4k, 50k
    temperature_k = Column(Float, nullable=False)
    status = Column(String(50), default='normal')
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'timestamp': self.timestamp.isoformat(),
            'stage': self.stage,
            'temperature_k': self.temperature_k,
            'status': self.status
        }

class PulseSequences(Base):
    """Store quantum pulse sequence configurations and results."""
    __tablename__ = 'pulse_sequences'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    sequence_name = Column(String(100), nullable=False)
    target_qubits = Column(JSONB, nullable=False)  # List of qubit IDs
    pulse_parameters = Column(JSONB, nullable=False)  # Pulse config as JSON
    predicted_fidelity = Column(Float, nullable=True)
    actual_fidelity = Column(Float, nullable=True)
    execution_time_ns = Column(Float, nullable=True)
    status = Column(String(50), default='pending')
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'timestamp': self.timestamp.isoformat(),
            'sequence_name': self.sequence_name,
            'target_qubits': self.target_qubits,
            'pulse_parameters': self.pulse_parameters,
            'predicted_fidelity': self.predicted_fidelity,
            'actual_fidelity': self.actual_fidelity,
            'execution_time_ns': self.execution_time_ns,
            'status': self.status
        }

class SystemAlerts(Base):
    """Store system alerts and notifications."""
    __tablename__ = 'system_alerts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)  # info, warning, error, success
    component = Column(String(100), nullable=False)  # Component that generated alert
    message = Column(Text, nullable=False)
    priority = Column(String(20), default='low')  # low, medium, high, critical
    acknowledged = Column(Boolean, default=False)
    resolved = Column(Boolean, default=False)
    alert_metadata = Column(JSONB, nullable=True)  # Additional alert data
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'component': self.component,
            'message': self.message,
            'priority': self.priority,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'metadata': self.alert_metadata
        }

class CalibrationRuns(Base):
    """Store calibration run data and results."""
    __tablename__ = 'calibration_runs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    calibration_type = Column(String(100), nullable=False)  # frequency, readout, gate, etc.
    target_qubits = Column(JSONB, nullable=False)  # List of affected qubits
    parameters_before = Column(JSONB, nullable=True)
    parameters_after = Column(JSONB, nullable=False)
    improvement_metrics = Column(JSONB, nullable=True)
    duration_minutes = Column(Float, nullable=True)
    status = Column(String(50), default='completed')
    notes = Column(Text, nullable=True)
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'timestamp': self.timestamp.isoformat(),
            'calibration_type': self.calibration_type,
            'target_qubits': self.target_qubits,
            'parameters_before': self.parameters_before,
            'parameters_after': self.parameters_after,
            'improvement_metrics': self.improvement_metrics,
            'duration_minutes': self.duration_minutes,
            'status': self.status,
            'notes': self.notes
        }

class QuantumStates(Base):
    """Store quantum state snapshots and measurements."""
    __tablename__ = 'quantum_states'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    state_name = Column(String(100), nullable=True)
    num_qubits = Column(Integer, nullable=False)
    state_vector = Column(JSONB, nullable=True)  # Complex amplitudes as JSON
    density_matrix = Column(JSONB, nullable=True)  # For mixed states
    fidelity = Column(Float, nullable=True)
    purity = Column(Float, nullable=True)
    entanglement_entropy = Column(Float, nullable=True)
    measurement_results = Column(JSONB, nullable=True)
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'timestamp': self.timestamp.isoformat(),
            'state_name': self.state_name,
            'num_qubits': self.num_qubits,
            'state_vector': self.state_vector,
            'density_matrix': self.density_matrix,
            'fidelity': self.fidelity,
            'purity': self.purity,
            'entanglement_entropy': self.entanglement_entropy,
            'measurement_results': self.measurement_results
        }

# Database connection and session management
def get_database_url():
    """Get database URL from environment variables."""
    return os.getenv('DATABASE_URL')

def create_database_engine():
    """Create SQLAlchemy engine for database connections."""
    database_url = get_database_url()
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=False  # Set to True for SQL query logging
    )
    return engine

def get_session():
    """Get database session."""
    engine = create_database_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def init_database():
    """Initialize database tables."""
    engine = create_database_engine()
    Base.metadata.create_all(engine)
    return engine