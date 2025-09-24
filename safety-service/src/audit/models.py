"""
Audit Models
Data models for audit logging and reporting
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field

class AuditEventType(str, Enum):
    """Types of audit events"""
    SAFETY_CHECK = "safety_check"
    DRIFT_DETECTION = "drift_detection"
    ABUSE_DETECTION = "abuse_detection"
    CONTENT_MODERATION = "content_moderation"
    RATE_LIMITING = "rate_limiting"
    COMPLIANCE_CHECK = "compliance_check"
    INCIDENT_RESPONSE = "incident_response"
    SYSTEM_ACCESS = "system_access"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"

class AuditSeverity(str, Enum):
    """Audit event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditStatus(str, Enum):
    """Audit event status"""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    INFO = "info"

class AuditEvent(BaseModel):
    """Individual audit event"""
    event_id: str = Field(..., description="Unique event identifier")
    event_type: AuditEventType = Field(..., description="Type of audit event")
    severity: AuditSeverity = Field(..., description="Event severity level")
    status: AuditStatus = Field(..., description="Event status")
    timestamp: datetime = Field(..., description="Event timestamp")
    user_id: Optional[str] = Field(None, description="User ID associated with event")
    session_id: Optional[str] = Field(None, description="Session ID associated with event")
    request_id: Optional[str] = Field(None, description="Request ID associated with event")
    source_ip: Optional[str] = Field(None, description="Source IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    resource: Optional[str] = Field(None, description="Resource accessed or modified")
    action: str = Field(..., description="Action performed")
    description: str = Field(..., description="Event description")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional event details")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Event metadata")
    tags: List[str] = Field(default_factory=list, description="Event tags for categorization")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for related events")
    parent_event_id: Optional[str] = Field(None, description="Parent event ID for event hierarchy")
    duration_ms: Optional[float] = Field(None, description="Event duration in milliseconds")
    error_code: Optional[str] = Field(None, description="Error code if event failed")
    error_message: Optional[str] = Field(None, description="Error message if event failed")

class AuditLog(BaseModel):
    """Audit log containing multiple events"""
    log_id: str = Field(..., description="Unique log identifier")
    log_type: str = Field(..., description="Type of audit log")
    start_time: datetime = Field(..., description="Log start time")
    end_time: datetime = Field(..., description="Log end time")
    events: List[AuditEvent] = Field(..., description="List of audit events")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Log summary")
    total_events: int = Field(..., description="Total number of events")
    success_count: int = Field(..., description="Number of successful events")
    failure_count: int = Field(..., description="Number of failed events")
    warning_count: int = Field(..., description="Number of warning events")
    info_count: int = Field(..., description="Number of info events")

class AuditReport(BaseModel):
    """Audit report with analysis and insights"""
    report_id: str = Field(..., description="Unique report identifier")
    report_type: str = Field(..., description="Type of audit report")
    generated_at: datetime = Field(..., description="Report generation timestamp")
    period_start: datetime = Field(..., description="Report period start")
    period_end: datetime = Field(..., description="Report period end")
    total_events: int = Field(..., description="Total events in period")
    events_by_type: Dict[str, int] = Field(..., description="Event count by type")
    events_by_severity: Dict[str, int] = Field(..., description="Event count by severity")
    events_by_status: Dict[str, int] = Field(..., description="Event count by status")
    top_users: List[Dict[str, Any]] = Field(..., description="Top users by activity")
    top_resources: List[Dict[str, Any]] = Field(..., description="Top resources by access")
    top_actions: List[Dict[str, Any]] = Field(..., description="Top actions performed")
    security_events: List[AuditEvent] = Field(..., description="Security-related events")
    compliance_events: List[AuditEvent] = Field(..., description="Compliance-related events")
    anomalies: List[Dict[str, Any]] = Field(..., description="Detected anomalies")
    recommendations: List[str] = Field(..., description="Security recommendations")
    summary: str = Field(..., description="Report summary")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional report details")

class AuditQuery(BaseModel):
    """Query parameters for audit log retrieval"""
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    event_types: Optional[List[AuditEventType]] = Field(None, description="Event type filter")
    severities: Optional[List[AuditSeverity]] = Field(None, description="Severity filter")
    statuses: Optional[List[AuditStatus]] = Field(None, description="Status filter")
    user_ids: Optional[List[str]] = Field(None, description="User ID filter")
    resources: Optional[List[str]] = Field(None, description="Resource filter")
    actions: Optional[List[str]] = Field(None, description="Action filter")
    tags: Optional[List[str]] = Field(None, description="Tag filter")
    correlation_id: Optional[str] = Field(None, description="Correlation ID filter")
    limit: Optional[int] = Field(1000, description="Maximum number of events to return")
    offset: Optional[int] = Field(0, description="Number of events to skip")
    sort_by: Optional[str] = Field("timestamp", description="Field to sort by")
    sort_order: Optional[str] = Field("desc", description="Sort order (asc/desc)")

class AuditMetrics(BaseModel):
    """Audit metrics and statistics"""
    total_events: int = Field(..., description="Total number of events")
    events_today: int = Field(..., description="Events today")
    events_this_week: int = Field(..., description="Events this week")
    events_this_month: int = Field(..., description="Events this month")
    success_rate: float = Field(..., description="Overall success rate")
    failure_rate: float = Field(..., description="Overall failure rate")
    average_duration: float = Field(..., description="Average event duration in ms")
    top_event_types: List[Dict[str, Any]] = Field(..., description="Top event types")
    top_users: List[Dict[str, Any]] = Field(..., description="Top users by activity")
    security_events_count: int = Field(..., description="Number of security events")
    compliance_events_count: int = Field(..., description="Number of compliance events")
    anomaly_count: int = Field(..., description="Number of anomalies detected")
    last_updated: datetime = Field(..., description="Last metrics update")

class AuditAlert(BaseModel):
    """Audit alert for suspicious or critical events"""
    alert_id: str = Field(..., description="Unique alert identifier")
    alert_type: str = Field(..., description="Type of alert")
    severity: AuditSeverity = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    triggered_at: datetime = Field(..., description="Alert trigger timestamp")
    event_ids: List[str] = Field(..., description="Related event IDs")
    user_id: Optional[str] = Field(None, description="User ID associated with alert")
    resource: Optional[str] = Field(None, description="Resource associated with alert")
    action: Optional[str] = Field(None, description="Action associated with alert")
    details: Dict[str, Any] = Field(default_factory=dict, description="Alert details")
    status: str = Field("active", description="Alert status")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged alert")
    acknowledged_at: Optional[datetime] = Field(None, description="Alert acknowledgment timestamp")
    resolved_by: Optional[str] = Field(None, description="User who resolved alert")
    resolved_at: Optional[datetime] = Field(None, description="Alert resolution timestamp")
    resolution_notes: Optional[str] = Field(None, description="Alert resolution notes")

class AuditConfiguration(BaseModel):
    """Audit system configuration"""
    enabled: bool = Field(True, description="Whether audit logging is enabled")
    log_level: str = Field("INFO", description="Minimum log level to capture")
    retention_days: int = Field(90, description="Number of days to retain audit logs")
    max_events_per_log: int = Field(10000, description="Maximum events per audit log")
    compression_enabled: bool = Field(True, description="Whether to compress audit logs")
    encryption_enabled: bool = Field(True, description="Whether to encrypt audit logs")
    real_time_alerts: bool = Field(True, description="Whether to enable real-time alerts")
    alert_thresholds: Dict[str, Any] = Field(default_factory=dict, description="Alert thresholds")
    excluded_events: List[str] = Field(default_factory=list, description="Event types to exclude")
    included_events: List[str] = Field(default_factory=list, description="Event types to include")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom audit fields")
