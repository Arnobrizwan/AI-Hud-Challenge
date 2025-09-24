"""
Incident Response Models
Data models for incident response system
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BaseIncidentModel(BaseModel):
    """Base model for incident response"""

    model_config = {"arbitrary_types_allowed": True}

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SafetyIncident(BaseIncidentModel):
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


class IncidentClassification(BaseIncidentModel):
    """Incident classification"""

    incident_type: str
    severity: str
    impact_assessment: Dict[str, str]
    requires_escalation: bool
    confidence: float
    classification_timestamp: datetime


class ResponseAction(BaseIncidentModel):
    """Response action"""

    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    priority: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ResponseResult(BaseIncidentModel):
    """Response action result"""

    action_id: str
    status: str
    message: str
    execution_time: Optional[float] = None
    error_details: Optional[str] = None
    completed_at: Optional[datetime] = None


class TimelineItem(BaseIncidentModel):
    """Response timeline item"""

    phase: str
    duration_minutes: int
    actions: List[str]


class ResponsePlan(BaseIncidentModel):
    """Incident response plan"""

    plan_id: str
    actions: List[ResponseAction]
    timeline: List[TimelineItem]
    success_criteria: List[str]


class MonitoringConfig(BaseIncidentModel):
    """Incident monitoring configuration"""

    metrics_to_monitor: List[str]
    alert_thresholds: Dict[str, float]
    check_interval: int


class IncidentResponse(BaseIncidentModel):
    """Incident response"""

    incident_id: str
    classification: IncidentClassification
    response_plan: ResponsePlan
    immediate_actions: List[ResponseAction]
    monitoring_config: MonitoringConfig
    response_timestamp: datetime
