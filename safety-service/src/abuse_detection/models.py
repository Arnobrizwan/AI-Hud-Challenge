"""
Abuse Detection Models
Data models for abuse detection system
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class BaseAbuseModel(BaseModel):
    """Base model for abuse detection"""

    model_config = {"arbitrary_types_allowed": True}

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ActivityData(BaseAbuseModel):
    """User activity data"""

    recent_activities: List[Dict[str, Any]]
    connection_data: Optional[Dict[str, Any]] = None
    user_features: Dict[str, Any]
    activity_features: Dict[str, Any]


class AbuseDetectionRequest(BaseAbuseModel):
    """Request for abuse detection"""

    user_id: str
    activity_data: ActivityData
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None


class BehavioralSignals(BaseAbuseModel):
    """Behavioral analysis signals"""

    anomaly_score: float
    pattern_violations: List[str]
    time_anomalies: List[str]
    frequency_anomalies: List[str]


class GraphSignals(BaseAbuseModel):
    """Graph-based analysis signals"""

    abuse_probability: float
    suspicious_connections: List[str]
    network_anomalies: List[str]
    collusion_indicators: List[str]


class MLPrediction(BaseAbuseModel):
    """ML-based abuse prediction"""

    model_config = {"protected_namespaces": ()}

    abuse_probability: float
    confidence: float
    feature_importance: Dict[str, float]
    model_version: str


class RuleViolation(BaseAbuseModel):
    """Rule violation details"""

    rule_id: str
    rule_name: str
    violation_type: str
    severity: str
    details: Dict[str, Any]


class MitigationAction(BaseAbuseModel):
    """Abuse mitigation action"""

    action_type: str
    parameters: Dict[str, Any]
    priority: int
    duration: Optional[int] = None


class AbuseDetectionResult(BaseAbuseModel):
    """Result of abuse detection"""

    user_id: str
    abuse_score: float
    threat_level: str
    behavioral_signals: BehavioralSignals
    graph_signals: GraphSignals
    ml_prediction: MLPrediction
    rule_violations: List[RuleViolation]
    reputation_score: float
    response_actions: List[MitigationAction]
    detection_timestamp: datetime


class MitigationResult(BaseAbuseModel):
    """Result of mitigation actions"""

    user_id: str
    applied_actions: List[MitigationAction]
    mitigation_timestamp: datetime
