"""
Safety Engine Data Models
Comprehensive data models for safety monitoring and response
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field


# Enums
class ThreatLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModerationAction(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    FLAG = "flag"
    BLOCK = "block"
    REMOVE = "remove"


class IncidentStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ComplianceType(str, Enum):
    GDPR = "gdpr"
    CONTENT_POLICY = "content_policy"
    PRIVACY = "privacy"
    SECURITY = "security"


class ActionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    ALREADY_ACTIVE = "already_active"


class EscalationLevel(str, Enum):
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"
    LEVEL_4 = "level_4"
    LEVEL_5 = "level_5"


class CommunicationChannel(str, Enum):
    EMAIL = "email"
    SLACK = "slack"
    PHONE = "phone"
    SMS = "sms"
    WEBHOOK = "webhook"


# Base Models
class BaseSafetyModel(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


# Request Models
class SafetyMonitoringRequest(BaseSafetyModel):
    user_id: str
    content: str
    features: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class DriftDetectionRequest(BaseSafetyModel):
    reference_data: Optional[pd.DataFrame] = None
    current_data: Optional[pd.DataFrame] = None
    features_to_monitor: List[str]
    reference_labels: Optional[pd.Series] = None
    current_labels: Optional[pd.Series] = None
    reference_predictions: Optional[pd.Series] = None
    current_predictions: Optional[pd.Series] = None
    reference_model: Optional[Any] = None
    current_model: Optional[Any] = None


class AbuseDetectionRequest(BaseSafetyModel):
    user_id: str
    activity_data: Optional["ActivityData"] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None


class ContentModerationRequest(BaseSafetyModel):
    content: "ContentItem"
    user_id: str
    content_type: str = "text"
    priority: str = "normal"


class AnomalyDetectionRequest(BaseSafetyModel):
    system_metrics: Dict[str, float]
    user_behavior: Optional[Dict[str, Any]] = None
    time_window: int = 3600  # seconds


class RateLimitRequest(BaseSafetyModel):
    user_id: str
    endpoint: str
    ip_address: str
    request_size: int = 1
    current_load: float = 0.0


class ComplianceRequest(BaseSafetyModel):
    check_gdpr: bool = True
    check_content_policy: bool = True
    check_privacy: bool = True
    data_activities: Optional[List[Dict[str, Any]]] = None
    content_items: Optional[List["ContentItem"]] = None
    user_data: Optional[Dict[str, Any]] = None


# Data Models
class ActivityData(BaseSafetyModel):
    recent_activities: List[Dict[str, Any]]
    connection_data: Optional[Dict[str, Any]] = None
    user_features: Dict[str, Any]
    activity_features: Dict[str, Any]


class ContentItem(BaseModel):
    id: str
    text_content: Optional[str] = None
    image_urls: Optional[List[str]] = None
    video_urls: Optional[List[str]] = None
    external_urls: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


# Response Models
class SafetyStatus(BaseSafetyModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    drift_status: Optional[Any] = None
    abuse_status: Optional[Any] = None
    content_status: Optional[Any] = None
    anomaly_status: Optional[Any] = None
    rate_limit_status: Optional[Any] = None
    requires_intervention: bool = False


class MitigationResult(BaseSafetyModel):
    """Result of abuse mitigation actions"""

    user_id: str
    applied_actions: List[Any]
    mitigation_timestamp: datetime


class TextModerationResult(BaseSafetyModel):
    """Result of text content moderation"""

    text: str
    toxicity_score: float
    hate_speech_score: float
    spam_score: float
    misinformation_score: float
    external_results: Optional[Dict[str, Any]] = None
    detected_issues: List[str]


class SafetyIncident(BaseSafetyModel):
    """Safety incident"""

    id: str
    incident_type: str
    severity: str
    status: str
    description: str
    affected_systems: List[str]
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ResponseResult(BaseSafetyModel):
    """Response action result"""

    action_id: str
    status: str
    message: str
    execution_time: Optional[float] = None
    error_details: Optional[str] = None
    completed_at: Optional[datetime] = None


class DriftAnalysisResult(BaseSafetyModel):
    data_drift: "DataDriftResult"
    concept_drift: "ConceptDriftResult"
    prediction_drift: "PredictionDriftResult"
    importance_drift: "ImportanceDriftResult"
    overall_severity: float = Field(ge=0.0, le=1.0)
    requires_action: bool = False
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


class DataDriftResult(BaseSafetyModel):
    feature_results: Dict[str, "FeatureDriftResult"]
    overall_drift_score: float = Field(ge=0.0, le=1.0)
    drifted_features: List[str]


class FeatureDriftResult(BaseSafetyModel):
    feature_name: str
    test_results: Dict[str, "StatisticalTestResult"]
    drift_magnitude: float
    is_drifted: bool
    drift_score: float = Field(ge=0.0, le=1.0)


class StatisticalTestResult(BaseSafetyModel):
    test_name: str
    p_value: float
    statistic: float
    is_significant: bool
    drift_score: float = Field(ge=0.0, le=1.0)


class ConceptDriftResult(BaseSafetyModel):
    drift_detected: bool
    drift_score: float = Field(ge=0.0, le=1.0)
    affected_features: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionDriftResult(BaseSafetyModel):
    drift_detected: bool
    drift_score: float = Field(ge=0.0, le=1.0)
    prediction_accuracy_change: float
    confidence: float = Field(ge=0.0, le=1.0)


class ImportanceDriftResult(BaseSafetyModel):
    drift_detected: bool
    drift_score: float = Field(ge=0.0, le=1.0)
    feature_importance_changes: Dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


class AbuseDetectionResult(BaseSafetyModel):
    user_id: str
    abuse_score: float = Field(ge=0.0, le=1.0)
    threat_level: ThreatLevel
    behavioral_signals: "BehavioralSignals"
    graph_signals: "GraphSignals"
    ml_prediction: "MLPrediction"
    rule_violations: List["RuleViolation"]
    reputation_score: float = Field(ge=0.0, le=1.0)
    response_actions: List["MitigationAction"]
    detection_timestamp: datetime = Field(default_factory=datetime.utcnow)


class BehavioralSignals(BaseSafetyModel):
    anomaly_score: float = Field(ge=0.0, le=1.0)
    velocity_anomaly: bool = False
    pattern_anomaly: bool = False
    frequency_anomaly: bool = False
    time_anomaly: bool = False


class GraphSignals(BaseSafetyModel):
    abuse_probability: float = Field(ge=0.0, le=1.0)
    suspicious_connections: int = 0
    cluster_anomaly: bool = False
    centrality_anomaly: bool = False


class MLPrediction(BaseSafetyModel):
    abuse_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    feature_importance: Dict[str, float]
    model_version: str


class RuleViolation(BaseSafetyModel):
    rule_id: str
    rule_name: str
    violation_type: str
    severity: ThreatLevel
    description: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MitigationAction(BaseSafetyModel):
    action_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    expires_at: Optional[datetime] = None


class ModerationResult(BaseSafetyModel):
    content_id: str
    overall_safety_score: float = Field(ge=0.0, le=1.0)
    moderation_results: Dict[str, Any]
    recommended_action: ModerationAction
    violations: List["ContentViolation"]
    moderation_timestamp: datetime = Field(default_factory=datetime.utcnow)


class ContentViolation(BaseSafetyModel):
    violation_type: str
    severity: ThreatLevel
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    affected_content: str


class AnomalyDetectionResult(BaseSafetyModel):
    anomaly_score: float = Field(ge=0.0, le=1.0)
    anomalies_detected: List["Anomaly"]
    system_health: str
    recommendations: List[str]


class Anomaly(BaseSafetyModel):
    metric_name: str
    anomaly_type: str
    severity: ThreatLevel
    value: float
    expected_value: float
    deviation: float


class RateLimitResult(BaseSafetyModel):
    user_id: str
    endpoint: str
    is_rate_limited: bool
    triggered_limits: List[str]
    remaining_capacity: int
    retry_after: int
    check_timestamp: datetime = Field(default_factory=datetime.utcnow)


class ComplianceReport(BaseSafetyModel):
    overall_compliance_score: float = Field(ge=0.0, le=1.0)
    compliance_results: Dict[str, Any]
    violations: List["ComplianceViolation"]
    recommendations: List[str]
    report_timestamp: datetime = Field(default_factory=datetime.utcnow)


class ComplianceViolation(BaseSafetyModel):
    violation_type: ComplianceType
    severity: ThreatLevel
    description: str
    affected_data: Optional[Dict[str, Any]] = None
    remediation_required: bool = True


class IncidentResponse(BaseSafetyModel):
    incident_id: str
    classification: "IncidentClassification"
    response_plan: "ResponsePlan"
    immediate_actions: List["ResponseAction"]
    monitoring_config: "MonitoringConfig"
    response_timestamp: datetime = Field(default_factory=datetime.utcnow)


class IncidentClassification(BaseSafetyModel):
    incident_type: str
    severity: ThreatLevel
    category: str
    requires_escalation: bool = False
    estimated_resolution_time: int  # minutes


class ResponsePlan(BaseSafetyModel):
    plan_id: str
    actions: List["ResponseAction"]
    timeline: List["TimelineItem"]
    success_criteria: List[str]


class ResponseAction(BaseSafetyModel):
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    priority: int
    dependencies: List[str] = []


class TimelineItem(BaseSafetyModel):
    phase: str
    duration_minutes: int
    actions: List[str]


class MonitoringConfig(BaseSafetyModel):
    metrics_to_monitor: List[str]
    alert_thresholds: Dict[str, float]
    check_interval: int  # seconds


# Update forward references
SafetyMonitoringRequest.model_rebuild()
DriftDetectionRequest.model_rebuild()
AbuseDetectionRequest.model_rebuild()
ContentModerationRequest.model_rebuild()
AnomalyDetectionRequest.model_rebuild()
RateLimitRequest.model_rebuild()
ComplianceRequest.model_rebuild()
