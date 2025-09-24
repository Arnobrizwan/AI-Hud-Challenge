"""
Monitoring Models - Data models for model monitoring and alerting
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertType(str, Enum):
    PERFORMANCE = "performance"
    DRIFT = "drift"
    DATA_QUALITY = "data_quality"
    SYSTEM = "system"
    CUSTOM = "custom"


class MetricThreshold(BaseModel):
    """Metric threshold configuration"""

    metric_name: str
    operator: str  # greater_than, less_than, equals, not_equals
    threshold: float
    duration_minutes: int = 5  # Duration threshold must be exceeded


class AlertRule(BaseModel):
    """Alert rule configuration"""

    name: str
    description: Optional[str] = None
    alert_type: AlertType
    severity: AlertSeverity
    metric_name: str
    operator: str
    threshold: float
    duration_minutes: int = 5
    enabled: bool = True
    notification_channels: List[str] = Field(default_factory=list)


class MonitoringConfig(BaseModel):
    """Configuration for model monitoring"""

    model_name: str
    monitoring_interval_seconds: int = 60
    performance_monitoring_interval_seconds: int = 300  # 5 minutes
    drift_detection_enabled: bool = True
    drift_detection_interval_hours: int = 24
    alert_rules: List[AlertRule] = Field(default_factory=list)
    metrics_to_track: List[str] = Field(default_factory=list)
    retention_days: int = 30


class ModelMetrics(BaseModel):
    """Model performance metrics"""

    model_name: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    error_rate: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PerformanceAlert(BaseModel):
    """Performance alert"""

    id: str
    model_name: str
    alert_type: AlertType = AlertType.PERFORMANCE
    severity: AlertSeverity
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    status: AlertStatus = AlertStatus.ACTIVE
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


class DriftAlert(BaseModel):
    """Data drift alert"""

    id: str
    model_name: str
    alert_type: AlertType = AlertType.DRIFT
    severity: AlertSeverity
    message: str
    drift_score: float
    affected_features: int
    status: AlertStatus = AlertStatus.ACTIVE
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


class DataQualityAlert(BaseModel):
    """Data quality alert"""

    id: str
    model_name: str
    alert_type: AlertType = AlertType.DATA_QUALITY
    severity: AlertSeverity
    message: str
    quality_score: float
    threshold: float
    affected_features: List[str] = Field(default_factory=list)
    status: AlertStatus = AlertStatus.ACTIVE
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


class ModelHealth(BaseModel):
    """Model health status"""

    model_name: str
    status: str  # healthy, degraded, unhealthy
    health_score: float  # 0.0 to 1.0
    active_alerts: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = Field(default_factory=dict)


class MonitoringDashboard(BaseModel):
    """Monitoring dashboard configuration"""

    id: str
    name: str
    model_name: str
    dashboard_url: str
    panels: List[Dict[str, Any]] = Field(default_factory=list)
    refresh_interval: int = 30  # seconds
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DriftDetectionResult(BaseModel):
    """Data drift detection result"""

    model_name: str
    overall_drift_score: float
    feature_drift_scores: Dict[str, float] = Field(default_factory=dict)
    drift_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    is_drift_detected: bool = False
    reference_data_period: Optional[Dict[str, datetime]] = None
    current_data_period: Optional[Dict[str, datetime]] = None


class PerformanceReport(BaseModel):
    """Model performance report"""

    model_name: str
    start_time: datetime
    end_time: datetime
    metrics_summary: Dict[str, Any] = Field(default_factory=dict)
    alerts_summary: List[Dict[str, Any]] = Field(default_factory=list)
    health_score: Optional[float] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    report_url: Optional[str] = None


class MonitoringJob(BaseModel):
    """Monitoring job instance"""

    id: str
    model_name: str
    job_type: str  # metrics_collection, drift_detection, alert_evaluation
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    results: Dict[str, Any] = Field(default_factory=dict)


class MetricDataPoint(BaseModel):
    """Individual metric data point"""

    metric_name: str
    value: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = Field(default_factory=dict)


class AlertNotification(BaseModel):
    """Alert notification"""

    alert_id: str
    notification_type: str  # email, slack, webhook, etc.
    recipient: str
    message: str
    sent_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "sent"  # sent, failed, pending
    error_message: Optional[str] = None
