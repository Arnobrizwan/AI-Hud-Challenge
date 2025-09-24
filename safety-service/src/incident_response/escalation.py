"""
Escalation Manager
Handle incident escalation and stakeholder notification
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from safety_engine.config import get_settings
from safety_engine.models import EscalationLevel, IncidentClassification, SafetyIncident

logger = logging.getLogger(__name__)


class EscalationManager:
    """Handle incident escalation and stakeholder notification"""

    def __init__(self):
        self.config = get_settings()
        self.is_initialized = False

        # Escalation levels
        self.escalation_levels = {
            "level_1": {
                "name": "On-Call Engineer",
                "response_time": 15,  # minutes
                "contacts": ["oncall@company.com"],
                "channels": ["email", "slack"],
            },
            "level_2": {
                "name": "Team Lead",
                "response_time": 30,  # minutes
                "contacts": ["team-lead@company.com"],
                "channels": ["email", "slack", "phone"],
            },
            "level_3": {
                "name": "Engineering Manager",
                "response_time": 60,  # minutes
                "contacts": ["eng-manager@company.com"],
                "channels": ["email", "slack", "phone"],
            },
            "level_4": {
                "name": "Director",
                "response_time": 120,  # minutes
                "contacts": ["director@company.com"],
                "channels": ["email", "slack", "phone"],
            },
            "level_5": {
                "name": "C-Level",
                "response_time": 240,  # minutes
                "contacts": ["ceo@company.com", "cto@company.com"],
                "channels": ["email", "slack", "phone", "sms"],
            },
        }

        # Escalation rules
        self.escalation_rules = {
            "critical": {
                "immediate_level": "level_3",
                "escalation_chain": ["level_1", "level_2", "level_3", "level_4", "level_5"],
                "escalation_intervals": [5, 15, 30, 60],  # minutes
            },
            "high": {
                "immediate_level": "level_2",
                "escalation_chain": ["level_1", "level_2", "level_3", "level_4"],
                "escalation_intervals": [10, 30, 60],  # minutes
            },
            "medium": {
                "immediate_level": "level_1",
                "escalation_chain": ["level_1", "level_2", "level_3"],
                "escalation_intervals": [30, 60],  # minutes
            },
            "low": {
                "immediate_level": "level_1",
                "escalation_chain": ["level_1", "level_2"],
                "escalation_intervals": [60],  # minutes
            },
        }

        # Escalation tracking
        self.active_escalations: Dict[str, Dict[str, Any]] = {}
        self.escalation_history: List[Dict[str, Any]] = []

        # Notification channels
        self.notification_channels = {
            "email": self.send_email_notification,
            "slack": self.send_slack_notification,
            "phone": self.send_phone_notification,
            "sms": self.send_sms_notification,
        }

    async def initialize(self):
        """Initialize the escalation manager"""
        try:
            # Initialize notification services
            await self.initialize_notification_services()

            # Start background tasks
            asyncio.create_task(self.escalation_monitoring_task())
            asyncio.create_task(self.escalation_cleanup_task())

            self.is_initialized = True
            logger.info("Escalation manager initialized")

        except Exception as e:
            logger.error(f"Failed to initialize escalation manager: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear escalation tracking
            self.active_escalations.clear()
            self.escalation_history.clear()

            self.is_initialized = False
            logger.info("Escalation manager cleanup completed")

        except Exception as e:
            logger.error(f"Error during escalation manager cleanup: {str(e)}")

    async def escalate_incident(
        self, incident: SafetyIncident, classification: IncidentClassification
    ) -> bool:
        """Escalate an incident based on classification"""

        if not self.is_initialized:
            raise RuntimeError("Escalation manager not initialized")

        try:
            # Check if incident is already escalated
            if incident.id in self.active_escalations:
                logger.warning(f"Incident {incident.id} is already escalated")
                return False

            # Get escalation configuration
            escalation_config = self.escalation_rules.get(classification.severity)
            if not escalation_config:
                logger.error(f"No escalation configuration for severity: {classification.severity}")
                return False

            # Create escalation record
            escalation_id = (
                f"escalation_{incident.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )
            escalation_record = {
                "escalation_id": escalation_id,
                "incident_id": incident.id,
                "severity": classification.severity,
                "current_level": escalation_config["immediate_level"],
                "escalation_chain": escalation_config["escalation_chain"],
                "escalation_intervals": escalation_config["escalation_intervals"],
                "escalation_start": datetime.utcnow(),
                "last_escalation": datetime.utcnow(),
                "escalation_count": 0,
                "status": "active",
                "notifications_sent": [],
            }

            # Store escalation record
            self.active_escalations[incident.id] = escalation_record

            # Send immediate notification
            await self.send_immediate_notification(
                incident, classification, escalation_config["immediate_level"]
            )

            # Schedule next escalation
            await self.schedule_next_escalation(incident.id, escalation_config)

            logger.info(
                f"Incident {incident.id} escalated to {escalation_config['immediate_level']}"
            )

            return True

        except Exception as e:
            logger.error(f"Incident escalation failed: {str(e)}")
            return False

    async def send_immediate_notification(
        self, incident: SafetyIncident, classification: IncidentClassification, level: str
    ) -> bool:
        """Send immediate notification for escalation"""
        try:
            # Get escalation level configuration
            level_config = self.escalation_levels.get(level)
            if not level_config:
                logger.error(f"Unknown escalation level: {level}")
                return False

            # Create notification message
            message = self.create_escalation_message(incident, classification, level)

            # Send notifications through all channels
            notification_results = []
            for channel in level_config["channels"]:
                if channel in self.notification_channels:
                    try:
                        result = await self.notification_channels[channel](
                            level_config["contacts"], message, incident.id
                        )
                        notification_results.append(result)
                    except Exception as e:
                        logger.error(f"Notification channel {channel} failed: {str(e)}")
                        notification_results.append(False)

            # Update escalation record
            if incident.id in self.active_escalations:
                self.active_escalations[incident.id]["notifications_sent"].append(
                    {
                        "level": level,
                        "timestamp": datetime.utcnow(),
                        "channels": level_config["channels"],
                        "success": any(notification_results),
                    }
                )

            return any(notification_results)

        except Exception as e:
            logger.error(f"Immediate notification failed: {str(e)}")
            return False

    async def schedule_next_escalation(
        self, incident_id: str, escalation_config: Dict[str, Any]
    ) -> None:
        """Schedule next escalation in the chain"""
        try:
            if incident_id not in self.active_escalations:
                return

            escalation_record = self.active_escalations[incident_id]
            current_level = escalation_record["current_level"]
            escalation_chain = escalation_record["escalation_chain"]
            escalation_intervals = escalation_record["escalation_intervals"]

            # Find current level index
            current_index = (
                escalation_chain.index(current_level) if current_level in escalation_chain else 0
            )

            # Check if there's a next level
            if current_index + 1 < len(escalation_chain):
                next_level = escalation_chain[current_index + 1]
                interval_minutes = (
                    escalation_intervals[current_index]
                    if current_index < len(escalation_intervals)
                    else 60
                )

                # Schedule next escalation
                asyncio.create_task(
                    self.delayed_escalation(incident_id, next_level, interval_minutes)
                )

        except Exception as e:
            logger.error(f"Next escalation scheduling failed: {str(e)}")

    async def delayed_escalation(
        self, incident_id: str, next_level: str, delay_minutes: int
    ) -> None:
        """Execute delayed escalation"""
        try:
            # Wait for the delay period
            await asyncio.sleep(delay_minutes * 60)

            # Check if escalation is still active
            if incident_id not in self.active_escalations:
                return

            escalation_record = self.active_escalations[incident_id]

            # Check if incident is still unresolved
            if escalation_record["status"] != "active":
                return

            # Update escalation level
            escalation_record["current_level"] = next_level
            escalation_record["last_escalation"] = datetime.utcnow()
            escalation_record["escalation_count"] += 1

            # Send notification for next level
            level_config = self.escalation_levels.get(next_level)
            if level_config:
                message = self.create_escalation_message(None, None, next_level)
                await self.send_escalation_notification(
                    level_config["contacts"], message, incident_id, next_level
                )

            # Schedule next escalation if needed
            escalation_config = self.escalation_rules.get(escalation_record["severity"])
            if escalation_config:
                await self.schedule_next_escalation(incident_id, escalation_config)

            logger.info(f"Incident {incident_id} escalated to {next_level}")

        except Exception as e:
            logger.error(f"Delayed escalation failed: {str(e)}")

    async def send_escalation_notification(
        self, contacts: List[str], message: str, incident_id: str, level: str
    ) -> bool:
        """Send escalation notification"""
        try:
            level_config = self.escalation_levels.get(level)
            if not level_config:
                return False

            # Send notifications through all channels
            notification_results = []
            for channel in level_config["channels"]:
                if channel in self.notification_channels:
                    try:
                        result = await self.notification_channels[channel](
                            contacts, message, incident_id
                        )
                        notification_results.append(result)
                    except Exception as e:
                        logger.error(f"Escalation notification channel {channel} failed: {str(e)}")
                        notification_results.append(False)

            return any(notification_results)

        except Exception as e:
            logger.error(f"Escalation notification failed: {str(e)}")
            return False

    def create_escalation_message(
        self,
        incident: Optional[SafetyIncident],
        classification: Optional[IncidentClassification],
        level: str,
    ) -> str:
        """Create escalation message"""
        try:
            level_config = self.escalation_levels.get(level, {})
            level_name = level_config.get("name", level)

            if incident and classification:
                message = f"""
🚨 INCIDENT ESCALATION - {level_name.upper()} 🚨

Incident ID: {incident.id}
Type: {incident.incident_type}
Severity: {classification.severity.upper()}
Status: {incident.status}
Detected: {incident.detected_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

Description: {incident.description}

Affected Systems: {', '.join(incident.affected_systems)}

Impact Assessment:
{self.format_impact_assessment(classification.impact_assessment)}

Response Required: {level_config.get('response_time', 'Unknown')} minutes

Please take immediate action to resolve this incident.
                """.strip()
            else:
                message = f"""
🚨 INCIDENT ESCALATION - {level_name.upper()} 🚨

Incident ID: {incident_id if 'incident_id' in locals() else 'Unknown'}
Level: {level_name}
Escalated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

This incident has been escalated to your level and requires immediate attention.
                """.strip()

            return message

        except Exception as e:
            logger.error(f"Escalation message creation failed: {str(e)}")
            return f"Incident escalation to {level} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"

    def format_impact_assessment(self, impact_assessment: Dict[str, str]) -> str:
        """Format impact assessment for message"""
        try:
            if not impact_assessment:
                return "Impact assessment not available"

            formatted = []
            for category, level in impact_assessment.items():
                formatted.append(f"  {category.replace('_', ' ').title()}: {level.upper()}")

            return "\n".join(formatted)

        except Exception as e:
            logger.error(f"Impact assessment formatting failed: {str(e)}")
            return "Impact assessment formatting error"

    async def resolve_escalation(self, incident_id: str, resolution_notes: str) -> bool:
        """Resolve an escalation"""
        try:
            if incident_id not in self.active_escalations:
                logger.warning(f"Escalation for incident {incident_id} not found")
                return False

            escalation_record = self.active_escalations[incident_id]
            escalation_record["status"] = "resolved"
            escalation_record["resolved_at"] = datetime.utcnow()
            escalation_record["resolution_notes"] = resolution_notes

            # Move to history
            self.escalation_history.append(escalation_record)
            del self.active_escalations[incident_id]

            logger.info(f"Escalation for incident {incident_id} resolved")
            return True

        except Exception as e:
            logger.error(f"Escalation resolution failed: {str(e)}")
            return False

    async def get_active_escalations(self) -> List[Dict[str, Any]]:
        """Get active escalations"""
        try:
            return list(self.active_escalations.values())

        except Exception as e:
            logger.error(f"Active escalations retrieval failed: {str(e)}")
            return []

    async def get_escalation_statistics(self) -> Dict[str, Any]:
        """Get escalation statistics"""
        try:
            return {
                "active_escalations": len(self.active_escalations),
                "total_escalations": len(self.active_escalations) + len(self.escalation_history),
                "escalations_by_level": self.get_escalations_by_level(),
                "escalations_by_severity": self.get_escalations_by_severity(),
                "average_escalation_time": self.calculate_average_escalation_time(),
            }

        except Exception as e:
            logger.error(f"Escalation statistics calculation failed: {str(e)}")
            return {"error": str(e)}

    def get_escalations_by_level(self) -> Dict[str, int]:
        """Get escalation count by level"""
        try:
            level_counts = {}

            # Count active escalations
            for escalation in self.active_escalations.values():
                level = escalation["current_level"]
                level_counts[level] = level_counts.get(level, 0) + 1

            # Count historical escalations
            for escalation in self.escalation_history:
                level = escalation["current_level"]
                level_counts[level] = level_counts.get(level, 0) + 1

            return level_counts

        except Exception as e:
            logger.error(f"Escalation level counting failed: {str(e)}")
            return {}

    def get_escalations_by_severity(self) -> Dict[str, int]:
        """Get escalation count by severity"""
        try:
            severity_counts = {}

            # Count active escalations
            for escalation in self.active_escalations.values():
                severity = escalation["severity"]
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            # Count historical escalations
            for escalation in self.escalation_history:
                severity = escalation["severity"]
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            return severity_counts

        except Exception as e:
            logger.error(f"Escalation severity counting failed: {str(e)}")
            return {}

    def calculate_average_escalation_time(self) -> float:
        """Calculate average escalation resolution time in hours"""
        try:
            resolved_escalations = [
                escalation
                for escalation in self.escalation_history
                if escalation.get("resolved_at") and escalation.get("escalation_start")
            ]

            if not resolved_escalations:
                return 0.0

            total_time = sum(
                (escalation["resolved_at"] - escalation["escalation_start"]).total_seconds()
                for escalation in resolved_escalations
            )

            return total_time / len(resolved_escalations) / 3600  # Convert to hours

        except Exception as e:
            logger.error(f"Average escalation time calculation failed: {str(e)}")
            return 0.0

    async def escalation_monitoring_task(self):
        """Background task for monitoring escalations"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                if not self.is_initialized:
                    break

                # Check for stale escalations
                current_time = datetime.utcnow()
                stale_escalations = []

                for incident_id, escalation in self.active_escalations.items():
                    # Check if escalation has been active too long
                    if (
                        current_time - escalation["escalation_start"]
                    ).total_seconds() > 24 * 3600:  # 24 hours
                        stale_escalations.append(incident_id)

                # Handle stale escalations
                for incident_id in stale_escalations:
                    logger.warning(
                        f"Escalation for incident {incident_id} has been active for over 24 hours"
                    )
                    # Could implement additional handling here

            except Exception as e:
                logger.error(f"Escalation monitoring task failed: {str(e)}")
                await asyncio.sleep(300)

    async def escalation_cleanup_task(self):
        """Background task for cleaning up old escalations"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                if not self.is_initialized:
                    break

                # Clean up old escalation history (keep for 30 days)
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                old_escalations = [
                    escalation
                    for escalation in self.escalation_history
                    if escalation.get("resolved_at") and escalation["resolved_at"] < cutoff_date
                ]

                for escalation in old_escalations:
                    self.escalation_history.remove(escalation)

                if old_escalations:
                    logger.info(f"Cleaned up {len(old_escalations)} old escalations")

            except Exception as e:
                logger.error(f"Escalation cleanup task failed: {str(e)}")
                await asyncio.sleep(3600)

    async def initialize_notification_services(self):
        """Initialize notification services"""
        try:
            # Placeholder for notification service initialization
            logger.info("Notification services initialized")

        except Exception as e:
            logger.error(f"Notification service initialization failed: {str(e)}")
            raise

    async def send_email_notification(
        self, contacts: List[str], message: str, incident_id: str
    ) -> bool:
        """Send email notification"""
        try:
            # Simulate email sending
            await asyncio.sleep(0.1)
            logger.info(f"Email notification sent to {contacts} for incident {incident_id}")
            return True

        except Exception as e:
            logger.error(f"Email notification failed: {str(e)}")
            return False

    async def send_slack_notification(
        self, contacts: List[str], message: str, incident_id: str
    ) -> bool:
        """Send Slack notification"""
        try:
            # Simulate Slack notification
            await asyncio.sleep(0.1)
            logger.info(f"Slack notification sent to {contacts} for incident {incident_id}")
            return True

        except Exception as e:
            logger.error(f"Slack notification failed: {str(e)}")
            return False

    async def send_phone_notification(
        self, contacts: List[str], message: str, incident_id: str
    ) -> bool:
        """Send phone notification"""
        try:
            # Simulate phone notification
            await asyncio.sleep(0.1)
            logger.info(f"Phone notification sent to {contacts} for incident {incident_id}")
            return True

        except Exception as e:
            logger.error(f"Phone notification failed: {str(e)}")
            return False

    async def send_sms_notification(
        self, contacts: List[str], message: str, incident_id: str
    ) -> bool:
        """Send SMS notification"""
        try:
            # Simulate SMS notification
            await asyncio.sleep(0.1)
            logger.info(f"SMS notification sent to {contacts} for incident {incident_id}")
            return True

        except Exception as e:
            logger.error(f"SMS notification failed: {str(e)}")
            return False
