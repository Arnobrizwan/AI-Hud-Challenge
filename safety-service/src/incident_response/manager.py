"""
Incident Response Manager
Automated incident response and escalation
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from safety_engine.config import get_settings
from safety_engine.models import (
    IncidentClassification,
    IncidentResponse,
    MonitoringConfig,
    ResponseAction,
    ResponsePlan,
    SafetyIncident,
    TimelineItem,
)

from .classifier import IncidentClassifier
from .communication import CommunicationManager
from .escalation import EscalationManager
from .orchestrator import ResponseOrchestrator

logger = logging.getLogger(__name__)


class IncidentResponseManager:
    """Automated incident response and escalation"""

    def __init__(self):
        self.config = get_settings()
        self.is_initialized = False

        # Incident storage
        self.active_incidents: Dict[str, SafetyIncident] = {}
        self.resolved_incidents: Dict[str, SafetyIncident] = {}

        # Response components
        self.incident_classifier = IncidentClassifier()
        self.response_orchestrator = ResponseOrchestrator()
        self.escalation_manager = EscalationManager()
        self.communication_manager = CommunicationManager()

        # Incident counter
        self.incident_counter = 0

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the incident response manager"""
        try:
            # Initialize all components
            await self.incident_classifier.initialize()
            await self.response_orchestrator.initialize()
            await self.escalation_manager.initialize()
            await self.communication_manager.initialize()

            # Start background tasks
            asyncio.create_task(self.incident_monitoring_task())
            asyncio.create_task(self.incident_cleanup_task())

            self.is_initialized = True
            logger.info("Incident response manager initialized")

        except Exception as e:
            logger.error(
                f"Failed to initialize incident response manager: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup resources"""
        try:
    await self.incident_classifier.cleanup()
            await self.response_orchestrator.cleanup()
            await self.escalation_manager.cleanup()
            await self.communication_manager.cleanup()

            self.active_incidents.clear()
            self.resolved_incidents.clear()

            self.is_initialized = False
            logger.info("Incident response manager cleanup completed")

        except Exception as e:
            logger.error(
                f"Error during incident response manager cleanup: {str(e)}")

    async def handle_safety_incident(
            self, incident: SafetyIncident) -> IncidentResponse:
        """Handle safety incident with automated response"""

        if not self.is_initialized:
            raise RuntimeError("Incident response manager not initialized")

        try:
            # Generate incident ID if not provided
            if not incident.id:
                incident.id = self.generate_incident_id()

            # Classify incident severity and type
            classification = await self.incident_classifier.classify_incident(incident)

            # Generate response plan
            response_plan = await self.generate_response_plan(incident, classification)

            # Execute immediate response actions
            immediate_actions = await self.execute_immediate_response(response_plan)

            # Set up monitoring for incident resolution
            monitoring_config = await self.setup_incident_monitoring(incident)

            # Handle escalation if needed
            if classification.requires_escalation:
    await self.escalation_manager.escalate_incident(incident, classification)

            # Communicate with stakeholders
            await self.communication_manager.notify_stakeholders(incident, response_plan)

            # Store incident
            self.active_incidents[incident.id] = incident

            # Create incident response
            response = IncidentResponse(
                incident_id=incident.id,
                classification=classification,
                response_plan=response_plan,
                immediate_actions=immediate_actions,
                monitoring_config=monitoring_config,
                response_timestamp=datetime.utcnow(),
            )

            logger.info(
                f"Incident {incident.id} handled with response plan {response_plan.plan_id}"
            )

            return response

        except Exception as e:
            logger.error(f"Incident handling failed: {str(e)}")
            raise

    async def create_drift_incident(self, drift_status: Any) -> SafetyIncident:
        """Create incident for drift detection"""
        try:
            incident = SafetyIncident(
                id=self.generate_incident_id(),
                incident_type="data_drift",
                severity="high" if drift_status.overall_severity > 0.8 else "medium",
                description=f"Data drift detected with severity {drift_status.overall_severity:.2f}",
                affected_systems=[
                    "drift_detection",
                    "ml_models"],
                detected_at=datetime.utcnow(),
                status="open",
                metadata={
                    "drift_severity": drift_status.overall_severity,
                    "drifted_features": drift_status.data_drift.drifted_features,
                    "requires_action": drift_status.requires_action,
                },
            )

            return incident

        except Exception as e:
            logger.error(f"Drift incident creation failed: {str(e)}")
            raise

    async def create_abuse_incident(self, abuse_status: Any) -> SafetyIncident:
        """Create incident for abuse detection"""
        try:
            incident = SafetyIncident(
                id=self.generate_incident_id(),
                incident_type="abuse_detection",
                severity=abuse_status.threat_level,
                description=f"Abuse detected for user {abuse_status.user_id} with score {abuse_status.abuse_score:.2f}",
                affected_systems=[
                    "abuse_detection",
                    "user_management"],
                detected_at=datetime.utcnow(),
                status="open",
                metadata={
                    "user_id": abuse_status.user_id,
                    "abuse_score": abuse_status.abuse_score,
                    "threat_level": abuse_status.threat_level,
                    "rule_violations": len(
                        abuse_status.rule_violations),
                },
            )

            return incident

        except Exception as e:
            logger.error(f"Abuse incident creation failed: {str(e)}")
            raise

    async def create_content_incident(
            self, content_status: Any) -> SafetyIncident:
        """Create incident for content safety"""
        try:
            incident = SafetyIncident(
                id=self.generate_incident_id(),
                incident_type="content_safety",
                severity="high" if content_status.overall_safety_score < 0.5 else "medium",
                description=f"Content safety violation detected with score {content_status.overall_safety_score:.2f}",
                affected_systems=[
                    "content_moderation",
                    "content_management"],
                detected_at=datetime.utcnow(),
                status="open",
                metadata={
                    "content_id": content_status.content_id,
                    "safety_score": content_status.overall_safety_score,
                    "violations": len(
                        content_status.violations),
                    "recommended_action": content_status.recommended_action,
                },
            )

            return incident

        except Exception as e:
            logger.error(f"Content incident creation failed: {str(e)}")
            raise

    async def generate_response_plan(
        self, incident: SafetyIncident, classification: IncidentClassification
    ) -> ResponsePlan:
        """Generate response plan for incident"""
        try:
            plan_id = f"plan_{incident.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            # Generate actions based on incident type and severity
            actions = await self.generate_response_actions(incident, classification)

            # Generate timeline
            timeline = await self.generate_response_timeline(incident, classification)

            # Generate success criteria
            success_criteria = await self.generate_success_criteria(incident, classification)

            response_plan = ResponsePlan(
                plan_id=plan_id,
                actions=actions,
                timeline=timeline,
                success_criteria=success_criteria,
            )

            return response_plan

        except Exception as e:
            logger.error(f"Response plan generation failed: {str(e)}")
            raise

    async def generate_response_actions(
        self, incident: SafetyIncident, classification: IncidentClassification
    ) -> List[ResponseAction]:
        """Generate response actions for incident"""
        try:
            actions = []

            # Immediate actions based on incident type
            if incident.incident_type == "data_drift":
                actions.extend(
                    [
                        ResponseAction(
                            action_id=f"action_{incident.id}_1",
                            action_type="retrain_models",
                            parameters={"priority": "high"},
                            priority=1,
                        ),
                        ResponseAction(
                            action_id=f"action_{incident.id}_2",
                            action_type="notify_data_team",
                            parameters={"severity": classification.severity},
                            priority=2,
                        ),
                    ]
                )

            elif incident.incident_type == "abuse_detection":
                actions.extend(
                    [
                        ResponseAction(
                            action_id=f"action_{incident.id}_1",
                            action_type="apply_mitigation",
                            parameters={
                                "user_id": incident.metadata.get("user_id")},
                            priority=1,
                        ),
                        ResponseAction(
                            action_id=f"action_{incident.id}_2",
                            action_type="review_user_activity",
                            parameters={
                                "user_id": incident.metadata.get("user_id")},
                            priority=2,
                        ),
                    ])

            elif incident.incident_type == "content_safety":
                actions.extend(
                    [
                        ResponseAction(
                            action_id=f"action_{incident.id}_1",
                            action_type="moderate_content",
                            parameters={
                                "content_id": incident.metadata.get("content_id")},
                            priority=1,
                        ),
                        ResponseAction(
                            action_id=f"action_{incident.id}_2",
                            action_type="review_content_policy",
                            parameters={
                                "violations": incident.metadata.get("violations")},
                            priority=2,
                        ),
                    ])

            # Add escalation action if needed
            if classification.requires_escalation:
                actions.append(
                    ResponseAction(
                        action_id=f"action_{incident.id}_escalate",
                        action_type="escalate_incident",
                        parameters={"severity": classification.severity},
                        priority=3,
                    )
                )

            return actions

        except Exception as e:
            logger.error(f"Response actions generation failed: {str(e)}")
            return []

    async def generate_response_timeline(
        self, incident: SafetyIncident, classification: IncidentClassification
    ) -> List[TimelineItem]:
        """Generate response timeline for incident"""
        try:
            timeline = []

            # Immediate response phase
            timeline.append(
                TimelineItem(
                    phase="immediate_response",
                    duration_minutes=15,
                    actions=[
                        "detect_incident",
                        "classify_severity",
                        "execute_immediate_actions"],
                ))

            # Investigation phase
            timeline.append(
                TimelineItem(
                    phase="investigation",
                    duration_minutes=60 if classification.severity == "high" else 30,
                    actions=[
                        "investigate_root_cause",
                        "assess_impact",
                        "gather_evidence"],
                ))

            # Resolution phase
            timeline.append(
                TimelineItem(
                    phase="resolution",
                    duration_minutes=120 if classification.severity == "high" else 60,
                    actions=[
                        "implement_fix",
                        "verify_resolution",
                        "update_monitoring"],
                ))

            # Follow-up phase
            timeline.append(
                TimelineItem(
                    phase="follow_up",
                    duration_minutes=30,
                    actions=[
                        "document_incident",
                        "update_procedures",
                        "conduct_post_mortem"],
                ))

            return timeline

        except Exception as e:
            logger.error(f"Response timeline generation failed: {str(e)}")
            return []

    async def generate_success_criteria(
        self, incident: SafetyIncident, classification: IncidentClassification
    ) -> List[str]:
        """Generate success criteria for incident resolution"""
        try:
            criteria = []

            # General success criteria
            criteria.extend(
                [
                    "Incident is fully resolved and system is stable",
                    "Root cause has been identified and addressed",
                    "All affected systems are operational",
                    "No data loss or corruption occurred",
                    "Incident documentation is complete",
                ]
            )

            # Type-specific criteria
            if incident.incident_type == "data_drift":
                criteria.extend(
                    [
                        "ML models have been retrained with new data",
                        "Drift detection thresholds have been updated",
                        "Model performance has been validated",
                    ]
                )

            elif incident.incident_type == "abuse_detection":
                criteria.extend(
                    [
                        "Abuse has been mitigated and user is contained",
                        "Abuse detection rules have been updated",
                        "User activity has been reviewed and documented",
                    ]
                )

            elif incident.incident_type == "content_safety":
                criteria.extend(
                    [
                        "Content has been properly moderated",
                        "Content policy violations have been addressed",
                        "Content moderation system has been updated",
                    ]
                )

            return criteria

        except Exception as e:
            logger.error(f"Success criteria generation failed: {str(e)}")
            return []

    async def execute_immediate_response(
            self, response_plan: ResponsePlan) -> List[ResponseAction]:
        """Execute immediate response actions"""
        try:
            executed_actions = []

            # Execute high-priority actions immediately
            immediate_actions = [
                action for action in response_plan.actions if action.priority <= 2]

            for action in immediate_actions:
                try:
    await self.response_orchestrator.execute_action(action)
                    executed_actions.append(action)
                    logger.info(
                        f"Executed immediate action: {action.action_type}")
                except Exception as e:
                    logger.error(
                        f"Failed to execute action {action.action_type}: {str(e)}")

            return executed_actions

        except Exception as e:
            logger.error(f"Immediate response execution failed: {str(e)}")
            return []

    async def setup_incident_monitoring(
            self, incident: SafetyIncident) -> MonitoringConfig:
        """Set up monitoring for incident resolution"""
        try:
            monitoring_config = MonitoringConfig(
                metrics_to_monitor=[
                    "system_health",
                    "error_rate",
                    "response_time",
                    "user_activity",
                ],
                alert_thresholds={
                    "error_rate": 0.05,
                    "response_time": 1000,
                    "system_health": 0.8},
                check_interval=60,  # 1 minute
            )

            return monitoring_config

        except Exception as e:
            logger.error(f"Incident monitoring setup failed: {str(e)}")
            raise

    async def resolve_incident(
            self,
            incident_id: str,
            resolution_notes: str) -> bool:
        """Resolve an incident"""
        try:
            if incident_id not in self.active_incidents:
                logger.warning(
                    f"Incident {incident_id} not found in active incidents")
                return False

            incident = self.active_incidents[incident_id]
            incident.status = "resolved"
            incident.resolved_at = datetime.utcnow()
            incident.resolution_notes = resolution_notes

            # Move to resolved incidents
            self.resolved_incidents[incident_id] = incident
            del self.active_incidents[incident_id]

            logger.info(f"Incident {incident_id} resolved")
            return True

        except Exception as e:
            logger.error(f"Incident resolution failed: {str(e)}")
            return False

    async def get_active_incidents(self) -> List[Dict[str, Any]]:
        """Get active incidents"""
        try:
            incidents = []
            for incident in self.active_incidents.values():
                incidents.append(
                    {
                        "id": incident.id,
                        "type": incident.incident_type,
                        "severity": incident.severity,
                        "status": incident.status,
                        "detected_at": incident.detected_at.isoformat(),
                        "description": incident.description,
                        "affected_systems": incident.affected_systems,
                    }
                )

            return incidents

        except Exception as e:
            logger.error(f"Active incidents retrieval failed: {str(e)}")
            return []

    async def cleanup_resolved_incidents(self) -> Dict[str, Any]:
        """Cleanup old resolved incidents"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=30)  # Keep for 30 days

            old_incidents = [
                incident_id
                for incident_id, incident in self.resolved_incidents.items()
                if incident.resolved_at and incident.resolved_at < cutoff_date
            ]

            for incident_id in old_incidents:
                del self.resolved_incidents[incident_id]

            if old_incidents:
                logger.info(
                    f"Cleaned up {len(old_incidents)} old resolved incidents")

        except Exception as e:
            logger.error(f"Incident cleanup failed: {str(e)}")

    async def incident_monitoring_task(self) -> Dict[str, Any]:
        """Background task for incident monitoring"""
        while True:
            try:
    await asyncio.sleep(300)  # Check every 5 minutes

                if not self.is_initialized:
                    break

                # Check for incidents that need attention
                for incident in self.active_incidents.values():
                    if incident.status == "open":
                        # Check if incident is stale
                        if incident.detected_at < datetime.utcnow() - timedelta(hours=24):
                            logger.warning(
                                f"Incident {incident.id} has been open for over 24 hours"
                            )

            except Exception as e:
                logger.error(f"Incident monitoring task failed: {str(e)}")
                await asyncio.sleep(300)

    async def incident_cleanup_task(self) -> Dict[str, Any]:
        """Background task for incident cleanup"""
        while True:
            try:
    await asyncio.sleep(3600)  # Run every hour

                if not self.is_initialized:
                    break

                await self.cleanup_resolved_incidents()

            except Exception as e:
                logger.error(f"Incident cleanup task failed: {str(e)}")
                await asyncio.sleep(3600)

    def generate_incident_id(self) -> str:
        """Generate unique incident ID"""
        self.incident_counter += 1
        return f"incident_{self.incident_counter}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    async def get_incident_statistics(self) -> Dict[str, Any]:
        """Get incident statistics"""
        try:
            return {
                "active_incidents": len(self.active_incidents),
                "resolved_incidents": len(self.resolved_incidents),
                "total_incidents": len(self.active_incidents) + len(self.resolved_incidents),
                "incidents_by_type": self.get_incidents_by_type(),
                "incidents_by_severity": self.get_incidents_by_severity(),
                "average_resolution_time": self.calculate_average_resolution_time(),
            }

        except Exception as e:
            logger.error(f"Incident statistics calculation failed: {str(e)}")
            return {"error": str(e)}

    def get_incidents_by_type(self) -> Dict[str, int]:
        """Get incident count by type"""
        try:
            type_counts = {}

            # Count active incidents
            for incident in self.active_incidents.values():
                type_counts[incident.incident_type] = type_counts.get(
                    incident.incident_type, 0) + 1

            # Count resolved incidents
            for incident in self.resolved_incidents.values():
                type_counts[incident.incident_type] = type_counts.get(
                    incident.incident_type, 0) + 1

            return type_counts

        except Exception as e:
            logger.error(f"Incident type counting failed: {str(e)}")
            return {}

    def get_incidents_by_severity(self) -> Dict[str, int]:
        """Get incident count by severity"""
        try:
            severity_counts = {}

            # Count active incidents
            for incident in self.active_incidents.values():
                severity_counts[incident.severity] = severity_counts.get(
                    incident.severity, 0) + 1

            # Count resolved incidents
            for incident in self.resolved_incidents.values():
                severity_counts[incident.severity] = severity_counts.get(
                    incident.severity, 0) + 1

            return severity_counts

        except Exception as e:
            logger.error(f"Incident severity counting failed: {str(e)}")
            return {}

    def calculate_average_resolution_time(self) -> float:
        """Calculate average incident resolution time in hours"""
        try:
            resolved_incidents = [
                incident
                for incident in self.resolved_incidents.values()
                if incident.resolved_at and incident.detected_at
            ]

            if not resolved_incidents:
                return 0.0

            total_time = sum(
                (incident.resolved_at - incident.detected_at).total_seconds()
                for incident in resolved_incidents
            )

            return total_time / len(resolved_incidents) / \
                3600  # Convert to hours

        except Exception as e:
            logger.error(
                f"Average resolution time calculation failed: {str(e)}")
            return 0.0
