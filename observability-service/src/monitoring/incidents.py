"""
Incident management and post-mortem automation
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IncidentStatus(Enum):
    """Incident status"""

    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"


class IncidentSeverity(Enum):
    """Incident severity"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class IncidentType(Enum):
    """Incident type"""

    SERVICE_DOWN = "service_down"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_LOSS = "data_loss"
    SECURITY_BREACH = "security_breach"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN = "unknown"


@dataclass
class Incident:
    """Incident definition"""

    id: str
    title: str
    description: str
    status: IncidentStatus
    severity: IncidentSeverity
    incident_type: IncidentType
    created_at: datetime
    created_by: str
    assigned_to: Optional[str] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    affected_services: List[str] = field(default_factory=list)
    affected_users: Optional[int] = None
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PostMortem:
    """Post-mortem report"""

    id: str
    incident_id: str
    title: str
    summary: str
    timeline: List[Dict[str, Any]]
    root_cause: str
    impact: str
    resolution: str
    lessons_learned: List[str]
    action_items: List[Dict[str, Any]]
    created_at: datetime
    created_by: str
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    published: bool = False


@dataclass
class IncidentTemplate:
    """Incident response template"""

    id: str
    name: str
    incident_type: IncidentType
    severity: IncidentSeverity
    steps: List[Dict[str, Any]]
    escalation_policy: str
    notification_channels: List[str]
    runbooks: List[str] = field(default_factory=list)


class IncidentManager:
    """Main incident management system"""

    def __init__(self):
        self.incidents = {}
        self.post_mortems = {}
        self.templates = {}
        self.is_initialized = False

    async def initialize(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Initialize incident management"""
        # Load incident templates
        await self.load_incident_templates(config.get("templates", []) if config else [])

        self.is_initialized = True
        logger.info("Incident manager initialized")

    async def load_incident_templates(self, templates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Load incident response templates"""
        for template_data in templates:
            template = IncidentTemplate(
                id=template_data["id"],
                name=template_data["name"],
                incident_type=IncidentType(template_data["incident_type"]),
                severity=IncidentSeverity(template_data["severity"]),
                steps=template_data["steps"],
                escalation_policy=template_data["escalation_policy"],
                notification_channels=template_data["notification_channels"],
                runbooks=template_data.get("runbooks", []),
            )
            self.templates[template.id] = template

    async def create_incident(self, incident_data: Dict[str, Any]) -> Incident:
        """Create new incident"""

        incident = Incident(
            id=str(uuid.uuid4()),
            title=incident_data["title"],
            description=incident_data["description"],
            status=IncidentStatus.OPEN,
            severity=IncidentSeverity(incident_data["severity"]),
            incident_type=IncidentType(incident_data["incident_type"]),
            created_at=datetime.utcnow(),
            created_by=incident_data.get("created_by", "system"),
            affected_services=incident_data.get("affected_services", []),
            affected_users=incident_data.get("affected_users"),
            tags=incident_data.get("tags", []),
            metadata=incident_data.get("metadata", {}),
        )

        # Add initial timeline entry
        incident.timeline.append(
            {
                "timestamp": incident.created_at.isoformat(),
                "action": "incident_created",
                "user": incident.created_by,
                "description": f"Incident created: {incident.title}",
            }
        )

        # Store incident
        self.incidents[incident.id] = incident

        # Trigger incident response
        await self.trigger_incident_response(incident)

        logger.info(f"Created incident {incident.id}: {incident.title}")
        return incident

    async def create_emergency_incident(
        self, incident_type: str, severity: str, description: str
    ) -> Incident:
        """Create emergency incident with immediate response"""

        incident_data = {
            "title": f"EMERGENCY: {incident_type}",
            "description": description,
            "severity": severity,
            "incident_type": incident_type,
            "created_by": "system",
            "tags": ["emergency", "auto-created"],
        }

        incident = await self.create_incident(incident_data)

        # Set to emergency severity if not already
        if severity == "emergency":
            incident.severity = IncidentSeverity.EMERGENCY

        # Trigger immediate response
        await self.trigger_emergency_response(incident.id)

        return incident

    async def trigger_incident_response(self, incident: Incident) -> Dict[str, Any]:
    """Trigger incident response procedures"""
        # Find applicable template
        template = self._find_incident_template(incident)

        if template:
            # Execute template steps
            await self._execute_incident_template(incident, template)

        # Update incident status
        incident.status = IncidentStatus.INVESTIGATING
        incident.timeline.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "investigation_started",
                "user": "system",
                "description": "Incident investigation started",
            }
        )

        # Send notifications
        await self._send_incident_notifications(incident)

    async def trigger_emergency_response(self, incident_id: str) -> Dict[str, Any]:
    """Trigger emergency response procedures"""
        incident = self.incidents.get(incident_id)
        if not incident:
            logger.error(
                f"Incident {incident_id} not found for emergency response")
            return

        # Escalate to emergency procedures
        incident.severity = IncidentSeverity.EMERGENCY
        incident.status = IncidentStatus.INVESTIGATING

        # Add emergency timeline entry
        incident.timeline.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "emergency_response_triggered",
                "user": "system",
                "description": "Emergency response procedures activated",
            }
        )

        # Send emergency notifications
        await self._send_emergency_notifications(incident)

        # Execute emergency runbooks
        await self._execute_emergency_runbooks(incident)

    def _find_incident_template(
            self, incident: Incident) -> Optional[IncidentTemplate]:
        """Find applicable incident template"""

        for template in self.templates.values():
            if (
                template.incident_type == incident.incident_type
                and template.severity == incident.severity
            ):
                return template

        # Fallback to severity-based template
        for template in self.templates.values():
            if template.severity == incident.severity:
                return template

        return None

    async def _execute_incident_template(
            self,
            incident: Incident,
            template: IncidentTemplate):
         -> Dict[str, Any]:"""Execute incident response template"""

        for step in template.steps:
            try:
    await self._execute_template_step(incident, step)
            except Exception as e:
                logger.error(
                    f"Failed to execute template step {step['name']}: {str(e)}")

    async def _execute_template_step(
            self, incident: Incident, step: Dict[str, Any]):
         -> Dict[str, Any]:"""Execute individual template step"""

        step_type = step.get("type")

        if step_type == "notification":
    await self._send_notification(incident, step)
        elif step_type == "escalation":
    await self._escalate_incident(incident, step)
        elif step_type == "runbook":
    await self._execute_runbook(incident, step)
        elif step_type == "assignment":
    await self._assign_incident(incident, step)
        else:
            logger.warning(f"Unknown template step type: {step_type}")

    async def _send_notification(
            self, incident: Incident, step: Dict[str, Any]):
         -> Dict[str, Any]:"""Send incident notification"""

        channels = step.get("channels", [])
        message = step.get(
            "message",
            f"Incident {incident.id}: {incident.title}")

        # This would integrate with notification system
        logger.info(f"Sending notification to {channels}: {message}")

    async def _escalate_incident(
            self, incident: Incident, step: Dict[str, Any]):
         -> Dict[str, Any]:"""Escalate incident"""

        escalation_level = step.get("level", 1)

        incident.timeline.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "escalated",
                "user": "system",
                "description": f"Incident escalated to level {escalation_level}",
            })

        logger.info(
            f"Escalated incident {incident.id} to level {escalation_level}")

    async def _execute_runbook(self, incident: Incident, step: Dict[str, Any]) -> Dict[str, Any]:
    """Execute incident runbook"""
        runbook_id = step.get("runbook_id")
        if runbook_id:
            # This would integrate with runbook engine
            logger.info(
                f"Executing runbook {runbook_id} for incident {incident.id}")

    async def _assign_incident(self, incident: Incident, step: Dict[str, Any]) -> Dict[str, Any]:
    """Assign incident to team member"""
        assignee = step.get("assignee")
        if assignee:
            incident.assigned_to = assignee
            incident.timeline.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "assigned",
                    "user": "system",
                    "description": f"Incident assigned to {assignee}",
                }
            )

    async def _send_incident_notifications(self, incident: Incident) -> Dict[str, Any]:
    """Send incident notifications"""
        # This would integrate with notification system
        logger.info(f"Sending incident notifications for {incident.id}")

    async def _send_emergency_notifications(self, incident: Incident) -> Dict[str, Any]:
    """Send emergency notifications"""
        # This would send to emergency channels
        logger.info(f"Sending emergency notifications for {incident.id}")

    async def _execute_emergency_runbooks(self, incident: Incident) -> Dict[str, Any]:
    """Execute emergency runbooks"""
        # This would execute emergency response runbooks
        logger.info(f"Executing emergency runbooks for {incident.id}")

    async def update_incident_status(
            self,
            incident_id: str,
            status: IncidentStatus,
            user: str,
            notes: Optional[str] = None):
         -> Dict[str, Any]:"""Update incident status"""

        incident = self.incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        old_status = incident.status
        incident.status = status

        # Add timeline entry
        incident.timeline.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "status_updated",
                "user": user,
                "description": f"Status changed from {old_status.value} to {status.value}",
                "notes": notes,
            })

        # Handle status-specific actions
        if status == IncidentStatus.RESOLVED:
            incident.resolved_at = datetime.utcnow()
            await self._handle_incident_resolution(incident)
        elif status == IncidentStatus.CLOSED:
            incident.closed_at = datetime.utcnow()
            await self._handle_incident_closure(incident)

    async def _handle_incident_resolution(self, incident: Incident) -> Dict[str, Any]:
    """Handle incident resolution"""
        # Trigger post-mortem creation for high severity incidents
        if incident.severity in [
            IncidentSeverity.HIGH,
            IncidentSeverity.CRITICAL,
            IncidentSeverity.EMERGENCY,
        ]:
    await self.create_post_mortem(incident.id)

        # Send resolution notifications
        logger.info(f"Incident {incident.id} resolved")

    async def _handle_incident_closure(self, incident: Incident) -> Dict[str, Any]:
    """Handle incident closure"""
        # Archive incident data
        logger.info(f"Incident {incident.id} closed")

    async def create_post_mortem(self, incident_id: str) -> PostMortem:
        """Create post-mortem report for incident"""

        incident = self.incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        post_mortem = PostMortem(
            id=str(uuid.uuid4()),
            incident_id=incident_id,
            title=f"Post-mortem: {incident.title}",
            summary=f"Post-mortem report for incident {incident_id}",
            timeline=incident.timeline.copy(),
            root_cause=incident.root_cause or "To be determined",
            impact=f"Affected {incident.affected_users or 'unknown'} users",
            resolution=incident.resolution or "To be documented",
            lessons_learned=[],
            action_items=[],
            created_at=datetime.utcnow(),
            created_by="system",
        )

        self.post_mortems[post_mortem.id] = post_mortem

        logger.info(
            f"Created post-mortem {post_mortem.id} for incident {incident_id}")
        return post_mortem

    async def update_post_mortem(
        self, post_mortem_id: str, updates: Dict[str, Any], user: str
    ) -> PostMortem:
        """Update post-mortem report"""

        post_mortem = self.post_mortems.get(post_mortem_id)
        if not post_mortem:
            raise ValueError(f"Post-mortem {post_mortem_id} not found")

        # Update fields
        for field, value in updates.items():
            if hasattr(post_mortem, field):
                setattr(post_mortem, field, value)

        # Update timeline
        post_mortem.timeline.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "post_mortem_updated",
                "user": user,
                "description": f"Post-mortem updated by {user}",
            }
        )

        return post_mortem

    async def get_active_incidents(self) -> List[Incident]:
        """Get all active incidents"""

        active_statuses = [
            IncidentStatus.OPEN,
            IncidentStatus.INVESTIGATING,
            IncidentStatus.IDENTIFIED,
            IncidentStatus.MONITORING,
        ]

        return [incident for incident in self.incidents.values(
        ) if incident.status in active_statuses]

    async def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID"""
        return self.incidents.get(incident_id)

    async def get_incidents_by_severity(
            self, severity: IncidentSeverity) -> List[Incident]:
        """Get incidents by severity"""
        return [incident for incident in self.incidents.values()
                if incident.severity == severity]

    async def get_incidents_by_service(
            self, service_name: str) -> List[Incident]:
        """Get incidents affecting specific service"""
        return [
            incident
            for incident in self.incidents.values()
            if service_name in incident.affected_services
        ]

    async def search_incidents(self, query: str) -> List[Incident]:
        """Search incidents by title or description"""

        query_lower = query.lower()
        return [incident for incident in self.incidents.values() if (
            query_lower in incident.title.lower() or query_lower in incident.description.lower())]

    async def get_incident_metrics(
        self, time_window: timedelta = timedelta(days=30)
    ) -> Dict[str, Any]:
    """Get incident metrics for time window"""
        cutoff_time = datetime.utcnow() - time_window
        recent_incidents = [incident for incident in self.incidents.values(
        ) if incident.created_at >= cutoff_time]

        # Calculate metrics
        total_incidents = len(recent_incidents)
        resolved_incidents = len(
            [i for i in recent_incidents if i.status == IncidentStatus.RESOLVED]
        )
        avg_resolution_time = self._calculate_avg_resolution_time(
            recent_incidents)

        # Group by severity
        severity_counts = {}
        for severity in IncidentSeverity:
            severity_counts[severity.value] = len(
                [i for i in recent_incidents if i.severity == severity]
            )

        # Group by type
        type_counts = {}
        for incident_type in IncidentType:
            type_counts[incident_type.value] = len(
                [i for i in recent_incidents if i.incident_type == incident_type]
            )

        return {
            "total_incidents": total_incidents,
            "resolved_incidents": resolved_incidents,
            "resolution_rate": (
                (resolved_incidents / total_incidents * 100) if total_incidents > 0 else 0),
            "avg_resolution_time_hours": avg_resolution_time,
            "severity_breakdown": severity_counts,
            "type_breakdown": type_counts,
            "time_window_days": time_window.days,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _calculate_avg_resolution_time(
            self, incidents: List[Incident]) -> float:
        """Calculate average resolution time in hours"""

        resolved_incidents = [
            i for i in incidents if i.status == IncidentStatus.RESOLVED and i.resolved_at]

        if not resolved_incidents:
            return 0.0

        total_time = sum((i.resolved_at - i.created_at).total_seconds()
                         for i in resolved_incidents)

        return total_time / len(resolved_incidents) / 3600  # Convert to hours

    async def cleanup(self) -> Dict[str, Any]:
    """Cleanup incident manager"""
        self.is_initialized = False
        logger.info("Incident manager cleaned up")
