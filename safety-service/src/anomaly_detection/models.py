"""
Anomaly Detection Models
Data models for anomaly detection system
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class BaseAnomalyModel(BaseModel):
    """Base model for anomaly detection"""

    model_config = {"arbitrary_types_allowed": True}

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AnomalyDetectionRequest(BaseAnomalyModel):
    """Request for anomaly detection"""

    system_metrics: Dict[str, float]
    user_behavior: Optional[Dict[str, Any]] = None
    time_window: int = 3600  # seconds


class AnomalyResult(BaseAnomalyModel):
    """Anomaly detection result"""

    anomaly_score: float
    is_anomaly: bool
    anomaly_type: str
    confidence: float
    features: Dict[str, float]
    explanation: Optional[str] = None


class AnomalyDetectionResult(BaseAnomalyModel):
    """Comprehensive anomaly detection result"""

    system_anomalies: List[AnomalyResult]
    user_anomalies: List[AnomalyResult]
    overall_anomaly_score: float
    requires_attention: bool
    detection_timestamp: datetime
