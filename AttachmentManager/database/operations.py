"""
Database operations for the QuantumOS Enterprise Control Platform.
Provides high-level functions for storing and retrieving quantum telemetry data.
"""

from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_, or_
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

from .models import (
    QubitTelemetry, SystemMetrics, CryogenicTemperatures, PulseSequences,
    SystemAlerts, CalibrationRuns, QuantumStates, get_session, init_database
)

class QuantumDataStore:
    """High-level interface for quantum data storage and retrieval."""
    
    def __init__(self):
        """Initialize the data store and ensure database tables exist."""
        init_database()
    
    def store_qubit_telemetry(self, telemetry_data: List[Dict]) -> bool:
        """Store multiple qubit telemetry measurements."""
        session = get_session()
        try:
            for data in telemetry_data:
                record = QubitTelemetry(**data)
                session.add(record)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error storing qubit telemetry: {e}")
            return False
        finally:
            session.close()
    
    def get_recent_qubit_telemetry(self, hours: int = 24, qubit_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Retrieve recent qubit telemetry data."""
        session = get_session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            query = session.query(QubitTelemetry).filter(QubitTelemetry.timestamp >= cutoff_time)
            
            if qubit_ids:
                query = query.filter(QubitTelemetry.qubit_id.in_(qubit_ids))
            
            results = query.order_by(desc(QubitTelemetry.timestamp)).all()
            
            if not results:
                return pd.DataFrame()
            
            data = [record.to_dict() for record in results]
            return pd.DataFrame(data)
        
        except Exception as e:
            print(f"Error retrieving qubit telemetry: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def store_system_metrics(self, metrics: Dict) -> bool:
        """Store system performance metrics."""
        session = get_session()
        try:
            record = SystemMetrics(**metrics)
            session.add(record)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error storing system metrics: {e}")
            return False
        finally:
            session.close()
    
    def get_system_metrics_history(self, hours: int = 24) -> pd.DataFrame:
        """Retrieve system metrics history."""
        session = get_session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            results = session.query(SystemMetrics).filter(
                SystemMetrics.timestamp >= cutoff_time
            ).order_by(desc(SystemMetrics.timestamp)).all()
            
            if not results:
                return pd.DataFrame()
            
            data = [record.to_dict() for record in results]
            return pd.DataFrame(data)
        
        except Exception as e:
            print(f"Error retrieving system metrics: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def store_temperature_readings(self, readings: List[Dict]) -> bool:
        """Store cryogenic temperature readings."""
        session = get_session()
        try:
            for reading in readings:
                record = CryogenicTemperatures(**reading)
                session.add(record)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error storing temperature readings: {e}")
            return False
        finally:
            session.close()
    
    def get_temperature_history(self, hours: int = 24, stage: Optional[str] = None) -> pd.DataFrame:
        """Retrieve temperature history."""
        session = get_session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            query = session.query(CryogenicTemperatures).filter(
                CryogenicTemperatures.timestamp >= cutoff_time
            )
            
            if stage:
                query = query.filter(CryogenicTemperatures.stage == stage)
            
            results = query.order_by(desc(CryogenicTemperatures.timestamp)).all()
            
            if not results:
                return pd.DataFrame()
            
            data = [record.to_dict() for record in results]
            return pd.DataFrame(data)
        
        except Exception as e:
            print(f"Error retrieving temperature history: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def store_pulse_sequence(self, sequence_data: Dict) -> str:
        """Store pulse sequence configuration and return sequence ID."""
        session = get_session()
        try:
            record = PulseSequences(**sequence_data)
            session.add(record)
            session.commit()
            session.refresh(record)
            return str(record.id)
        except Exception as e:
            session.rollback()
            print(f"Error storing pulse sequence: {e}")
            return ""
        finally:
            session.close()
    
    def update_pulse_sequence_results(self, sequence_id: str, results: Dict) -> bool:
        """Update pulse sequence with execution results."""
        session = get_session()
        try:
            record = session.query(PulseSequences).filter(
                PulseSequences.id == sequence_id
            ).first()
            
            if record:
                for key, value in results.items():
                    setattr(record, key, value)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error updating pulse sequence results: {e}")
            return False
        finally:
            session.close()
    
    def get_pulse_sequences(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve recent pulse sequences."""
        session = get_session()
        try:
            results = session.query(PulseSequences).order_by(
                desc(PulseSequences.timestamp)
            ).limit(limit).all()
            
            if not results:
                return pd.DataFrame()
            
            data = [record.to_dict() for record in results]
            return pd.DataFrame(data)
        
        except Exception as e:
            print(f"Error retrieving pulse sequences: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def create_system_alert(self, alert_data: Dict) -> str:
        """Create a new system alert."""
        session = get_session()
        try:
            record = SystemAlerts(**alert_data)
            session.add(record)
            session.commit()
            session.refresh(record)
            return str(record.id)
        except Exception as e:
            session.rollback()
            print(f"Error creating system alert: {e}")
            return ""
        finally:
            session.close()
    
    def get_active_alerts(self, limit: int = 50) -> pd.DataFrame:
        """Retrieve active system alerts."""
        session = get_session()
        try:
            results = session.query(SystemAlerts).filter(
                SystemAlerts.resolved == False
            ).order_by(desc(SystemAlerts.timestamp)).limit(limit).all()
            
            if not results:
                return pd.DataFrame()
            
            data = [record.to_dict() for record in results]
            return pd.DataFrame(data)
        
        except Exception as e:
            print(f"Error retrieving active alerts: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        session = get_session()
        try:
            record = session.query(SystemAlerts).filter(
                SystemAlerts.id == alert_id
            ).first()
            
            if record:
                record.acknowledged = True
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error acknowledging alert: {e}")
            return False
        finally:
            session.close()
    
    def store_calibration_run(self, calibration_data: Dict) -> str:
        """Store calibration run data."""
        session = get_session()
        try:
            record = CalibrationRuns(**calibration_data)
            session.add(record)
            session.commit()
            session.refresh(record)
            return str(record.id)
        except Exception as e:
            session.rollback()
            print(f"Error storing calibration run: {e}")
            return ""
        finally:
            session.close()
    
    def get_calibration_history(self, days: int = 30) -> pd.DataFrame:
        """Retrieve calibration run history."""
        session = get_session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            results = session.query(CalibrationRuns).filter(
                CalibrationRuns.timestamp >= cutoff_time
            ).order_by(desc(CalibrationRuns.timestamp)).all()
            
            if not results:
                return pd.DataFrame()
            
            data = [record.to_dict() for record in results]
            return pd.DataFrame(data)
        
        except Exception as e:
            print(f"Error retrieving calibration history: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def store_quantum_state(self, state_data: Dict) -> str:
        """Store quantum state snapshot."""
        session = get_session()
        try:
            record = QuantumStates(**state_data)
            session.add(record)
            session.commit()
            session.refresh(record)
            return str(record.id)
        except Exception as e:
            session.rollback()
            print(f"Error storing quantum state: {e}")
            return ""
        finally:
            session.close()
    
    def get_quantum_states(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve recent quantum states."""
        session = get_session()
        try:
            results = session.query(QuantumStates).order_by(
                desc(QuantumStates.timestamp)
            ).limit(limit).all()
            
            if not results:
                return pd.DataFrame()
            
            data = [record.to_dict() for record in results]
            return pd.DataFrame(data)
        
        except Exception as e:
            print(f"Error retrieving quantum states: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def get_fidelity_trends(self, hours: int = 24, qubit_ids: Optional[List[int]] = None) -> Dict[str, pd.DataFrame]:
        """Retrieve fidelity trend data for analysis."""
        session = get_session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Get gate fidelity trends from telemetry
            query = session.query(QubitTelemetry).filter(
                QubitTelemetry.timestamp >= cutoff_time
            )
            
            if qubit_ids:
                query = query.filter(QubitTelemetry.qubit_id.in_(qubit_ids))
            
            telemetry_results = query.order_by(QubitTelemetry.timestamp).all()
            
            # Get pulse sequence fidelities
            pulse_results = session.query(PulseSequences).filter(
                PulseSequences.timestamp >= cutoff_time,
                PulseSequences.actual_fidelity.isnot(None)
            ).order_by(PulseSequences.timestamp).all()
            
            trends = {}
            
            if telemetry_results:
                telemetry_data = [record.to_dict() for record in telemetry_results]
                trends['gate_fidelity'] = pd.DataFrame(telemetry_data)
            
            if pulse_results:
                pulse_data = [record.to_dict() for record in pulse_results]
                trends['sequence_fidelity'] = pd.DataFrame(pulse_data)
            
            return trends
        
        except Exception as e:
            print(f"Error retrieving fidelity trends: {e}")
            return {}
        finally:
            session.close()
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get a comprehensive system health summary."""
        session = get_session()
        try:
            summary = {}
            
            # Recent system metrics
            recent_metrics = session.query(SystemMetrics).order_by(
                desc(SystemMetrics.timestamp)
            ).first()
            
            if recent_metrics:
                summary['system'] = recent_metrics.to_dict()
            
            # Qubit status counts
            recent_time = datetime.utcnow() - timedelta(hours=1)
            qubit_status = session.query(
                QubitTelemetry.status,
                func.count(QubitTelemetry.qubit_id.distinct()).label('count')
            ).filter(
                QubitTelemetry.timestamp >= recent_time
            ).group_by(QubitTelemetry.status).all()
            
            summary['qubit_status'] = {status: count for status, count in qubit_status}
            
            # Active alerts by priority
            alert_counts = session.query(
                SystemAlerts.priority,
                func.count(SystemAlerts.id).label('count')
            ).filter(
                SystemAlerts.resolved == False
            ).group_by(SystemAlerts.priority).all()
            
            summary['alerts'] = {priority: count for priority, count in alert_counts}
            
            # Temperature status
            temp_status = session.query(CryogenicTemperatures).filter(
                CryogenicTemperatures.timestamp >= recent_time
            ).order_by(desc(CryogenicTemperatures.timestamp)).all()
            
            if temp_status:
                summary['temperatures'] = {record.stage: record.to_dict() for record in temp_status}
            
            return summary
        
        except Exception as e:
            print(f"Error retrieving system health summary: {e}")
            return {}
        finally:
            session.close()

# Global instance for easy access
quantum_db = QuantumDataStore()