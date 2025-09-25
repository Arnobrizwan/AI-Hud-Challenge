"""
Communication Manager
Handle stakeholder communication and notifications
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from safety_engine.config import get_settings
from safety_engine.models import CommunicationChannel, ResponsePlan, SafetyIncident

logger = logging.getLogger(__name__)


class CommunicationManager:
    """Handle stakeholder communication and notifications"""

    def __init__(self):
        self.config = get_settings()
        self.is_initialized = False

        # Communication channels
        self.communication_channels = {
            "email": {
                "enabled": True,
                "priority": "high",
                "handler": self.send_email_communication,
            },
            "slack": {
                "enabled": True,
                "priority": "high",
                "handler": self.send_slack_communication,
            },
            "phone": {
                "enabled": True,
                "priority": "critical",
                "handler": self.send_phone_communication,
            },
            "sms": {
                "enabled": True,
                "priority": "critical",
                "handler": self.send_sms_communication,
            },
            "webhook": {
                "enabled": True,
                "priority": "medium",
                "handler": self.send_webhook_communication,
            },
        }

        # Stakeholder groups
        self.stakeholder_groups = {
            "incident_response_team": {
                "name": "Incident Response Team",
                "channels": ["email", "slack", "phone"],
                "contacts": ["incident-response@company.com"],
                "escalation_levels": ["level_1", "level_2", "level_3"],
            },
            "engineering_team": {
                "name": "Engineering Team",
                "channels": ["email", "slack"],
                "contacts": ["engineering@company.com"],
                "escalation_levels": ["level_2", "level_3", "level_4"],
            },
            "management_team": {
                "name": "Management Team",
                "channels": ["email", "slack", "phone"],
                "contacts": ["management@company.com"],
                "escalation_levels": ["level_3", "level_4", "level_5"],
            },
            "executive_team": {
                "name": "Executive Team",
                "channels": ["email", "phone", "sms"],
                "contacts": ["executives@company.com"],
                "escalation_levels": ["level_4", "level_5"],
            },
            "external_stakeholders": {
                "name": "External Stakeholders",
                "channels": ["email", "webhook"],
                "contacts": ["external@company.com"],
                "escalation_levels": ["level_3", "level_4", "level_5"],
            },
        }

        # Communication templates
        self.communication_templates = {
            "incident_detected": {
                "subject": "🚨 Incident Detected: {incident_type}",
                "body": """
An incident has been detected in the system.

Incident Details:
- ID: {incident_id}
- Type: {incident_type}
- Severity: {severity}
- Status: {status}
- Detected: {detected_at}
- Description: {description}

Affected Systems: {affected_systems}

The incident response team has been notified and is investigating.
                """.strip(),
            },
            "incident_escalated": {
                "subject": "⚠️ Incident Escalated: {incident_id}",
                "body": """
An incident has been escalated to your level.

Incident Details:
- ID: {incident_id}
- Type: {incident_type}
- Severity: {severity}
- Current Level: {escalation_level}
- Escalated: {escalated_at}

Description: {description}

Please take immediate action to resolve this incident.
                """.strip(),
            },
            "incident_resolved": {
                "subject": "✅ Incident Resolved: {incident_id}",
                "body": """
An incident has been resolved.

Incident Details:
- ID: {incident_id}
- Type: {incident_type}
- Severity: {severity}
- Resolved: {resolved_at}
- Resolution Time: {resolution_time}

Resolution Notes: {resolution_notes}

The incident has been closed and normal operations have resumed.
                """.strip(),
            },
            "incident_update": {
                "subject": "📋 Incident Update: {incident_id}",
                "body": """
An update on the ongoing incident.

Incident Details:
- ID: {incident_id}
- Type: {incident_type}
- Severity: {severity}
- Status: {status}
- Last Updated: {updated_at}

Update: {update_message}

The incident response team continues to work on resolution.
                """.strip(),
            },
        }

        # Communication tracking
        self.communication_log: List[Dict[str, Any]] = []
        self.active_communications: Dict[str, Dict[str, Any]] = {}

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the communication manager"""
        try:
            # Initialize communication services
            except Exception as e:
                pass

            await self.initialize_communication_services()

            # Start background tasks
            asyncio.create_task(self.communication_monitoring_task())
            asyncio.create_task(self.communication_cleanup_task())

            self.is_initialized = True
            logger.info("Communication manager initialized")

        except Exception as e:
            logger.error(
                f"Failed to initialize communication manager: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup resources"""
        try:
            # Clear communication tracking
            except Exception as e:
                pass

            self.communication_log.clear()
            self.active_communications.clear()

            self.is_initialized = False
            logger.info("Communication manager cleanup completed")

        except Exception as e:
            logger.error(
                f"Error during communication manager cleanup: {str(e)}")

    async def notify_stakeholders(
        self, incident: SafetyIncident, response_plan: ResponsePlan
    ) -> bool:
        """Notify stakeholders about an incident"""

        if not self.is_initialized:
            raise RuntimeError("Communication manager not initialized")

        try:
            # Determine which stakeholder groups to notify
            except Exception as e:
                pass

            stakeholder_groups = await self.determine_stakeholder_groups(incident)

            # Create communication messages
            messages = await self.create_communication_messages(incident, response_plan)

            # Send notifications to each stakeholder group
            notification_results = []
            for group_name in stakeholder_groups:
                group_config = self.stakeholder_groups.get(group_name)
                if not group_config:
                    continue

                # Send notification to group
                result = await self.send_group_notification(
                    group_name, group_config, messages, incident
                )
                notification_results.append(result)

            # Log communication
            await self.log_communication(incident, stakeholder_groups, notification_results)

            return any(notification_results)

        except Exception as e:
            logger.error(f"Stakeholder notification failed: {str(e)}")
            return False

    async def determine_stakeholder_groups(
            self, incident: SafetyIncident) -> List[str]:
        """Determine which stakeholder groups to notify"""
        try:
            groups = []
            except Exception as e:
                pass


            # Always notify incident response team
            groups.append("incident_response_team")

            # Determine additional groups based on incident severity
            if incident.severity in ["high", "critical"]:
                groups.append("engineering_team")

            if incident.severity == "critical":
                groups.append("management_team")
                groups.append("executive_team")

            # Check if external notification is needed
            if incident.metadata and incident.metadata.get(
                    "requires_external_notification", False):
                groups.append("external_stakeholders")

            return groups

        except Exception as e:
            logger.error(f"Stakeholder group determination failed: {str(e)}")
            return ["incident_response_team"]

    async def create_communication_messages(
        self, incident: SafetyIncident, response_plan: ResponsePlan
    ) -> Dict[str, str]:
        """Create communication messages for different channels"""
        try:
            messages = {}
            except Exception as e:
                pass


            # Create incident detected message
            incident_message = self.communication_templates["incident_detected"]
            messages["incident_detected"] = {
                "subject": incident_message["subject"].format(
                    incident_type=incident.incident_type,
                    incident_id=incident.id),
                "body": incident_message["body"].format(
                    incident_id=incident.id,
                    incident_type=incident.incident_type,
                    severity=incident.severity,
                    status=incident.status,
                    detected_at=incident.detected_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    description=incident.description,
                    affected_systems=", ".join(
                        incident.affected_systems),
                ),
            }

            # Create escalation message if needed
            if incident.metadata and incident.metadata.get("escalation_level"):
                escalation_message = self.communication_templates["incident_escalated"]
                messages["incident_escalated"] = {
                    "subject": escalation_message["subject"].format(
                        incident_id=incident.id),
                    "body": escalation_message["body"].format(
                        incident_id=incident.id,
                        incident_type=incident.incident_type,
                        severity=incident.severity,
                        escalation_level=incident.metadata["escalation_level"],
                        escalated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                        description=incident.description,
                    ),
                }

            return messages

        except Exception as e:
            logger.error(f"Communication message creation failed: {str(e)}")
            return {}

    async def send_group_notification(
        self,
        group_name: str,
        group_config: Dict[str, Any],
        messages: Dict[str, str],
        incident: SafetyIncident,
    ) -> bool:
        """Send notification to a stakeholder group"""
        try:
            # Get group channels
            except Exception as e:
                pass

            channels = group_config.get("channels", [])
            contacts = group_config.get("contacts", [])

            # Send notification through each channel
            notification_results = []
            for channel in channels:
                if channel in self.communication_channels:
                    channel_config = self.communication_channels[channel]
                    if channel_config["enabled"]:
                        # Get appropriate message for channel
                        message_key = self.get_message_key_for_channel(
                            channel, incident)
                        if message_key in messages:
                            message = messages[message_key]

                            # Send notification
                            result = await channel_config["handler"](
                                contacts, message, incident.id, group_name
                            )
                            notification_results.append(result)

            return any(notification_results)

        except Exception as e:
            logger.error(
                f"Group notification failed for {group_name}: {str(e)}")
            return False

    def get_message_key_for_channel(
            self,
            channel: str,
            incident: SafetyIncident) -> str:
        """Get appropriate message key for communication channel"""
        try:
            # Critical channels get escalation messages
            except Exception as e:
                pass

            if channel in ["phone", "sms"]:
                return "incident_escalated"

            # Default to incident detected
            return "incident_detected"

        except Exception as e:
            logger.error(f"Message key determination failed: {str(e)}")
            return "incident_detected"

    async def send_incident_update(
        self, incident_id: str, update_message: str, severity: str = "medium"
    ) -> bool:
        """Send incident update to stakeholders"""
        try:
            # Get incident details
            except Exception as e:
                pass

            incident = await self.get_incident_details(incident_id)
            if not incident:
                logger.warning(f"Incident {incident_id} not found for update")
                return False

            # Create update message
            update_template = self.communication_templates["incident_update"]
            message = {
                "subject": update_template["subject"].format(
                    incident_id=incident_id),
                "body": update_template["body"].format(
                    incident_id=incident_id,
                    incident_type=incident.get(
                        "type",
                        "Unknown"),
                    severity=severity,
                    status=incident.get(
                        "status",
                        "Unknown"),
                    updated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    update_message=update_message,
                ),
            }

            # Determine stakeholder groups for update
            stakeholder_groups = await self.determine_stakeholder_groups_for_update(severity)

            # Send update to each group
            notification_results = []
            for group_name in stakeholder_groups:
                group_config = self.stakeholder_groups.get(group_name)
                if group_config:
                    result = await self.send_group_notification(
                        group_name, group_config, {"incident_update": message}, incident
                    )
                    notification_results.append(result)

            # Log communication
            await self.log_communication(
                incident, stakeholder_groups, notification_results, "update"
            )

            return any(notification_results)

        except Exception as e:
            logger.error(f"Incident update failed: {str(e)}")
            return False

    async def determine_stakeholder_groups_for_update(
            self, severity: str) -> List[str]:
        """Determine stakeholder groups for incident update"""
        try:
            groups = ["incident_response_team"]
            except Exception as e:
                pass


            if severity in ["high", "critical"]:
                groups.append("engineering_team")

            if severity == "critical":
                groups.append("management_team")

            return groups

        except Exception as e:
            logger.error(
                f"Stakeholder group determination for update failed: {str(e)}")
            return ["incident_response_team"]

    async def send_incident_resolution(
        self, incident_id: str, resolution_notes: str, resolution_time: str
    ) -> bool:
        """Send incident resolution notification"""
        try:
            # Get incident details
            except Exception as e:
                pass

            incident = await self.get_incident_details(incident_id)
            if not incident:
                logger.warning(
                    f"Incident {incident_id} not found for resolution")
                return False

            # Create resolution message
            resolution_template = self.communication_templates["incident_resolved"]
            message = {
                "subject": resolution_template["subject"].format(
                    incident_id=incident_id),
                "body": resolution_template["body"].format(
                    incident_id=incident_id,
                    incident_type=incident.get(
                        "type",
                        "Unknown"),
                    severity=incident.get(
                        "severity",
                        "Unknown"),
                    resolved_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    resolution_time=resolution_time,
                    resolution_notes=resolution_notes,
                ),
            }

            # Get all stakeholder groups that were notified
            stakeholder_groups = incident.get(
                "notified_groups", ["incident_response_team"])

            # Send resolution to each group
            notification_results = []
            for group_name in stakeholder_groups:
                group_config = self.stakeholder_groups.get(group_name)
                if group_config:
                    result = await self.send_group_notification(
                        group_name, group_config, {"incident_resolved": message}, incident
                    )
                    notification_results.append(result)

            # Log communication
            await self.log_communication(
                incident, stakeholder_groups, notification_results, "resolution"
            )

            return any(notification_results)

        except Exception as e:
            logger.error(f"Incident resolution notification failed: {str(e)}")
            return False

    async def log_communication(
        self,
        incident: SafetyIncident,
        stakeholder_groups: List[str],
        notification_results: List[bool],
        communication_type: str = "notification",
    ) -> None:
        """Log communication activity"""
        try:
            communication_record = {
            except Exception as e:
                pass

                "communication_id": f"comm_{uuid.uuid4().hex[:8]}",
                "incident_id": incident.id,
                "communication_type": communication_type,
                "stakeholder_groups": stakeholder_groups,
                "notification_results": notification_results,
                "success": any(notification_results),
                "timestamp": datetime.utcnow(),
            }

            self.communication_log.append(communication_record)

            # Keep only last 1000 communications
            if len(self.communication_log) > 1000:
                self.communication_log = self.communication_log[-1000:]

        except Exception as e:
            logger.error(f"Communication logging failed: {str(e)}")

    async def get_incident_details(
            self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Get incident details for communication"""
        try:
            # Placeholder for getting incident details
            except Exception as e:
                pass

            # In a real implementation, this would query the incident database
            return {
                "id": incident_id,
                "type": "unknown",
                "severity": "medium",
                "status": "open",
                "notified_groups": ["incident_response_team"],
            }

        except Exception as e:
            logger.error(f"Incident details retrieval failed: {str(e)}")
            return None

    async def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication statistics"""
        try:
            return {
            except Exception as e:
                pass

                "total_communications": len(
                    self.communication_log),
                "successful_communications": sum(
                    1 for comm in self.communication_log if comm["success"]),
                "communications_by_type": self.get_communications_by_type(),
                "communications_by_group": self.get_communications_by_group(),
                "average_response_time": self.calculate_average_response_time(),
            }

        except Exception as e:
            logger.error(
                f"Communication statistics calculation failed: {str(e)}")
            return {"error": str(e)}

    def get_communications_by_type(self) -> Dict[str, int]:
        """Get communication count by type"""
        try:
            type_counts = {}
            except Exception as e:
                pass

            for comm in self.communication_log:
                comm_type = comm["communication_type"]
                type_counts[comm_type] = type_counts.get(comm_type, 0) + 1
            return type_counts

        except Exception as e:
            logger.error(f"Communication type counting failed: {str(e)}")
            return {}

    def get_communications_by_group(self) -> Dict[str, int]:
        """Get communication count by stakeholder group"""
        try:
            group_counts = {}
            except Exception as e:
                pass

            for comm in self.communication_log:
                for group in comm["stakeholder_groups"]:
                    group_counts[group] = group_counts.get(group, 0) + 1
            return group_counts

        except Exception as e:
            logger.error(f"Communication group counting failed: {str(e)}")
            return {}

    def calculate_average_response_time(self) -> float:
        """Calculate average response time in minutes"""
        try:
            # Placeholder for response time calculation
            except Exception as e:
                pass

            # In a real implementation, this would calculate actual response
            # times
            return 15.0  # 15 minutes average

        except Exception as e:
            logger.error(f"Average response time calculation failed: {str(e)}")
            return 0.0

    async def communication_monitoring_task(self) -> Dict[str, Any]:
        """Background task for monitoring communications"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    pass


                if not self.is_initialized:
                    break

                # Check for failed communications
                failed_communications = [
                    comm
                    # Last 100 communications
                    for comm in self.communication_log[-100:]
                    if not comm["success"]
                ]

                if failed_communications:
                    logger.warning(
                        f"Found {len(failed_communications)} failed communications in the last 100"
                    )

            except Exception as e:
                logger.error(f"Communication monitoring task failed: {str(e)}")
                await asyncio.sleep(300)

    async def communication_cleanup_task(self) -> Dict[str, Any]:
        """Background task for cleaning up old communications"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                except Exception as e:
                    pass


                if not self.is_initialized:
                    break

                # Clean up old communications (keep for 30 days)
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                old_communications = [
                    comm for comm in self.communication_log if comm["timestamp"] < cutoff_date]

                for comm in old_communications:
                    self.communication_log.remove(comm)

                if old_communications:
                    logger.info(
                        f"Cleaned up {len(old_communications)} old communications")

            except Exception as e:
                logger.error(f"Communication cleanup task failed: {str(e)}")
                await asyncio.sleep(3600)

    async def initialize_communication_services(self) -> Dict[str, Any]:
        """Initialize communication services"""
        try:
            # Placeholder for communication service initialization
            except Exception as e:
                pass

            logger.info("Communication services initialized")

        except Exception as e:
            logger.error(
                f"Communication service initialization failed: {str(e)}")
            raise

    async def send_email_communication(self,
                                       contacts: List[str],
                                       message: Dict[str,
                                                     str],
                                       incident_id: str,
                                       group_name: str) -> bool:
        """Send email communication"""
        try:
            # Simulate email sending
            except Exception as e:
                pass

            await asyncio.sleep(0.1)
            logger.info(
                f"Email sent to {contacts} for incident {incident_id} (group: {group_name})"
            )
            return True

        except Exception as e:
            logger.error(f"Email communication failed: {str(e)}")
            return False

    async def send_slack_communication(self,
                                       contacts: List[str],
                                       message: Dict[str,
                                                     str],
                                       incident_id: str,
                                       group_name: str) -> bool:
        """Send Slack communication"""
        try:
            # Simulate Slack notification
            except Exception as e:
                pass

            await asyncio.sleep(0.1)
            logger.info(
                f"Slack message sent to {contacts} for incident {incident_id} (group: {group_name})"
            )
            return True

        except Exception as e:
            logger.error(f"Slack communication failed: {str(e)}")
            return False

    async def send_phone_communication(self,
                                       contacts: List[str],
                                       message: Dict[str,
                                                     str],
                                       incident_id: str,
                                       group_name: str) -> bool:
        """Send phone communication"""
        try:
            # Simulate phone call
            except Exception as e:
                pass

            await asyncio.sleep(0.1)
            logger.info(
                f"Phone call made to {contacts} for incident {incident_id} (group: {group_name})"
            )
            return True

        except Exception as e:
            logger.error(f"Phone communication failed: {str(e)}")
            return False

    async def send_sms_communication(self,
                                     contacts: List[str],
                                     message: Dict[str,
                                                   str],
                                     incident_id: str,
                                     group_name: str) -> bool:
        """Send SMS communication"""
        try:
            # Simulate SMS sending
            except Exception as e:
                pass

            await asyncio.sleep(0.1)
            logger.info(
                f"SMS sent to {contacts} for incident {incident_id} (group: {group_name})")
            return True

        except Exception as e:
            logger.error(f"SMS communication failed: {str(e)}")
            return False

    async def send_webhook_communication(self,
                                         contacts: List[str],
                                         message: Dict[str,
                                                       str],
                                         incident_id: str,
                                         group_name: str) -> bool:
        """Send webhook communication"""
        try:
            # Simulate webhook call
            except Exception as e:
                pass

            await asyncio.sleep(0.1)
            logger.info(
                f"Webhook sent to {contacts} for incident {incident_id} (group: {group_name})"
            )
            return True

        except Exception as e:
            logger.error(f"Webhook communication failed: {str(e)}")
            return False
