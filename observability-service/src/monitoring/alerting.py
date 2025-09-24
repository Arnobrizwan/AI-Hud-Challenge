"""
Intelligent alerting system with escalation and routing
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status"""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert definition"""

    id: str
    type: str
    severity: AlertSeverity
    message: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0
    correlation_id: Optional[str] = None


@dataclass
class AlertRule:
    """Alerting rule definition"""

    id: str
    name: str
    description: str
    condition: str
    severity: AlertSeverity
    threshold: float
    evaluation_window: int  # seconds
    notification_channels: List[str]
    escalation_policy: Optional[str] = None
    suppression_rules: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    enabled: bool = True


@dataclass
class NotificationChannel:
    """Notification channel configuration"""

    id: str
    name: str
    type: str  # slack, email, pagerduty, webhook
    config: Dict[str, Any]
    enabled: bool = True


@dataclass
class EscalationPolicy:
    """Escalation policy definition"""

    id: str
    name: str
    levels: List[Dict[str, Any]]  # Each level has delay, channels, etc.
    max_escalations: int = 5


@dataclass
class AlertProcessingResult:
    """Result of alert processing"""

    alert_id: str
    action: str
    severity: Optional[AlertSeverity] = None
    notifications_sent: int = 0
    escalation_scheduled: bool = False
    reason: Optional[str] = None


class AlertCorrelationEngine:
    """Alert correlation and deduplication"""

    def __init__(self):
        self.active_alerts = {}
        self.correlation_rules = []

    async def correlate_alert(self, alert: Alert) -> List[Alert]:
        """Correlate alert with existing alerts"""

        correlated_alerts = []

        # Check for similar alerts within time window
        time_window = timedelta(minutes=5)
        cutoff_time = alert.timestamp - time_window

        for existing_alert in self.active_alerts.values():
            if (
                existing_alert.type == alert.type
                and existing_alert.source == alert.source
                and existing_alert.timestamp > cutoff_time
            ):
                correlated_alerts.append(existing_alert)

        return correlated_alerts

    def add_correlation_rule(self, rule: Dict[str, Any]):
        """Add correlation rule"""
        self.correlation_rules.append(rule)


class SuppressionRuleEngine:
    """Alert suppression rules"""

    def __init__(self):
        self.suppression_rules = []
        self.active_suppressions = {}

    async def should_suppress(self, alert: Alert, correlated_alerts: List[Alert]) -> bool:
        """Check if alert should be suppressed"""

        # Check active suppressions
        for suppression_id, suppression in self.active_suppressions.items():
            if self._matches_suppression(alert, suppression):
                return True

        # Check suppression rules
        for rule in self.suppression_rules:
            if self._matches_rule(alert, rule):
                # Create active suppression
                suppression_id = str(uuid.uuid4())
                self.active_suppressions[suppression_id] = {
                    "rule": rule,
                    "created_at": datetime.utcnow(),
                    "expires_at": datetime.utcnow()
                    + timedelta(minutes=rule.get("duration_minutes", 30)),
                }
                return True

        return False

    def _matches_suppression(self, alert: Alert, suppression: Dict[str, Any]) -> bool:
        """Check if alert matches active suppression"""
        rule = suppression["rule"]

        # Check if suppression has expired
        if datetime.utcnow() > suppression["expires_at"]:
            return False

        return self._matches_rule(alert, rule)

    def _matches_rule(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Check if alert matches suppression rule"""

        # Check alert type
        if rule.get("alert_type") and alert.type != rule["alert_type"]:
            return False

        # Check severity
        if rule.get("min_severity") and alert.severity.value < rule["min_severity"]:
            return False

        # Check source
        if rule.get("source_pattern") and not self._matches_pattern(
            alert.source, rule["source_pattern"]
        ):
            return False

        return True

    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Simple pattern matching"""
        return pattern in text


class NotificationRouter:
    """Route alerts to appropriate notification channels"""

    def __init__(self):
        self.channels = {}
        self.routing_rules = []

    async def route_alert(self, alert: Alert, severity: AlertSeverity) -> Dict[str, Any]:
        """Route alert to appropriate channels"""

        channels = []

        # Determine channels based on severity and routing rules
        if severity == AlertSeverity.EMERGENCY:
            channels = self._get_emergency_channels()
        elif severity == AlertSeverity.CRITICAL:
            channels = self._get_critical_channels()
        elif severity == AlertSeverity.HIGH:
            channels = self._get_high_channels()
        else:
            channels = self._get_standard_channels()

        # Apply routing rules
        for rule in self.routing_rules:
            if self._matches_routing_rule(alert, rule):
                channels.extend(rule.get("channels", []))

        return {"channels": channels, "routing_reason": f"Severity: {severity.value}"}

    def _get_emergency_channels(self) -> List[str]:
        """Get emergency notification channels"""
        return [ch_id for ch_id, ch in self.channels.items() if ch.get("emergency_enabled", False)]

    def _get_critical_channels(self) -> List[str]:
        """Get critical notification channels"""
        return [ch_id for ch_id, ch in self.channels.items() if ch.get("critical_enabled", True)]

    def _get_high_channels(self) -> List[str]:
        """Get high severity notification channels"""
        return [ch_id for ch_id, ch in self.channels.items() if ch.get("high_enabled", True)]

    def _get_standard_channels(self) -> List[str]:
        """Get standard notification channels"""
        return [ch_id for ch_id, ch in self.channels.items() if ch.get("standard_enabled", True)]

    def _matches_routing_rule(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Check if alert matches routing rule"""

        # Check alert type
        if rule.get("alert_type") and alert.type != rule["alert_type"]:
            return False

        # Check source pattern
        if rule.get("source_pattern") and not self._matches_pattern(
            alert.source, rule["source_pattern"]
        ):
            return False

        return True

    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Simple pattern matching"""
        return pattern in text


class EscalationEngine:
    """Alert escalation management"""

    def __init__(self):
        self.escalation_policies = {}
        self.active_escalations = {}

    async def setup_escalation(self, alert: Alert, routing_decision: Dict[str, Any]):
        """Set up escalation for alert"""

        if not alert.escalation_level:
            alert.escalation_level = 0

        # Find escalation policy
        policy = self._find_escalation_policy(alert)
        if not policy:
            return

        # Schedule escalation
        escalation_id = str(uuid.uuid4())
        self.active_escalations[escalation_id] = {
            "alert_id": alert.id,
            "policy_id": policy["id"],
            "current_level": alert.escalation_level,
            "next_escalation": datetime.utcnow()
            + timedelta(minutes=policy["levels"][alert.escalation_level]["delay_minutes"]),
            "created_at": datetime.utcnow(),
        }

        # Schedule escalation task
        asyncio.create_task(self._handle_escalation(escalation_id))

    def _find_escalation_policy(self, alert: Alert) -> Optional[Dict[str, Any]]:
        """Find escalation policy for alert"""

        # Simple policy selection - in practice, this would be more sophisticated
        for policy in self.escalation_policies.values():
            if policy.get("alert_types", []):
                if alert.type in policy["alert_types"]:
                    return policy
            else:
                return policy  # Default policy

        return None

    async def _handle_escalation(self, escalation_id: str):
        """Handle escalation process"""

        escalation = self.active_escalations.get(escalation_id)
        if not escalation:
            return

        # Wait for escalation time
        await asyncio.sleep((escalation["next_escalation"] - datetime.utcnow()).total_seconds())

        # Check if alert is still active
        # In practice, you'd check the actual alert status

        # Escalate to next level
        policy = self.escalation_policies[escalation["policy_id"]]
        current_level = escalation["current_level"]

        if current_level < len(policy["levels"]) - 1:
            next_level = current_level + 1
            level_config = policy["levels"][next_level]

            # Send notifications for next level
            # This would integrate with notification system

            # Update escalation
            escalation["current_level"] = next_level
            escalation["next_escalation"] = datetime.utcnow() + timedelta(
                minutes=level_config["delay_minutes"]
            )

            # Schedule next escalation
            asyncio.create_task(self._handle_escalation(escalation_id))
        else:
            # Max escalation reached
            del self.active_escalations[escalation_id]


class AlertingSystem:
    """Main alerting system"""

    def __init__(self):
        self.alert_manager = AlertManager()
        self.notification_router = NotificationRouter()
        self.escalation_engine = EscalationEngine()
        self.alert_correlation = AlertCorrelationEngine()
        self.suppression_rules = SuppressionRuleEngine()
        self.is_initialized = False

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize alerting system"""

        # Load alerting rules
        await self.load_alerting_rules(config.get("rules_path", "config/alert_rules.json"))

        # Configure notification channels
        await self.configure_notification_channels(config.get("channels", []))

        # Set up escalation policies
        await self.setup_escalation_policies(config.get("escalation_policies", []))

        # Start alert processing
        await self.start_alert_processing()

        self.is_initialized = True
        logger.info("Alerting system initialized")

    async def load_alerting_rules(self, rules_path: str):
        """Load alerting rules from configuration"""

        try:
            with open(rules_path, "r") as f:
                rules_data = json.load(f)

            for rule_data in rules_data:
                rule = AlertRule(
                    id=rule_data["id"],
                    name=rule_data["name"],
                    description=rule_data["description"],
                    condition=rule_data["condition"],
                    severity=AlertSeverity(rule_data["severity"]),
                    threshold=rule_data["threshold"],
                    evaluation_window=rule_data["evaluation_window"],
                    notification_channels=rule_data["notification_channels"],
                    escalation_policy=rule_data.get("escalation_policy"),
                    suppression_rules=rule_data.get("suppression_rules", []),
                )

                await self.alert_manager.store_rule(rule)

        except FileNotFoundError:
            logger.warning(f"Alert rules file not found: {rules_path}")
        except Exception as e:
            logger.error(f"Failed to load alerting rules: {str(e)}")

    async def configure_notification_channels(self, channels_config: List[Dict[str, Any]]):
        """Configure notification channels"""

        for channel_config in channels_config:
            channel = NotificationChannel(
                id=channel_config["id"],
                name=channel_config["name"],
                type=channel_config["type"],
                config=channel_config["config"],
                enabled=channel_config.get("enabled", True),
            )

            self.notification_router.channels[channel.id] = channel_config

    async def setup_escalation_policies(self, policies_config: List[Dict[str, Any]]):
        """Set up escalation policies"""

        for policy_config in policies_config:
            policy = EscalationPolicy(
                id=policy_config["id"],
                name=policy_config["name"],
                levels=policy_config["levels"],
                max_escalations=policy_config.get("max_escalations", 5),
            )

            self.escalation_engine.escalation_policies[policy.id] = policy_config

    async def start_alert_processing(self):
        """Start background alert processing"""
        asyncio.create_task(self._alert_processing_loop())

    async def _alert_processing_loop(self):
        """Background alert processing loop"""
        while True:
            try:
                await self.process_pending_alerts()
                await asyncio.sleep(10)  # Process every 10 seconds
            except Exception as e:
                logger.error(f"Alert processing loop error: {str(e)}")
                await asyncio.sleep(10)

    async def process_alert(self, alert: Alert) -> AlertProcessingResult:
        """Process incoming alert with intelligent routing"""

        # Correlate with existing alerts
        correlated_alerts = await self.alert_correlation.correlate_alert(alert)

        # Check suppression rules
        if await self.suppression_rules.should_suppress(alert, correlated_alerts):
            return AlertProcessingResult(
                alert_id=alert.id, action="suppressed", reason="matched_suppression_rule"
            )

        # Determine severity and priority
        alert_severity = await self.calculate_alert_severity(alert, correlated_alerts)

        # Route to appropriate channels
        routing_decision = await self.notification_router.route_alert(alert, alert_severity)

        # Send notifications
        notification_results = []
        for channel_id in routing_decision["channels"]:
            result = await self.send_notification(alert, channel_id)
            notification_results.append(result)

        # Set up escalation if needed
        if alert_severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            await self.escalation_engine.setup_escalation(alert, routing_decision)

        return AlertProcessingResult(
            alert_id=alert.id,
            action="processed",
            severity=alert_severity,
            notifications_sent=len(notification_results),
            escalation_scheduled=alert_severity
            in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY],
        )

    async def calculate_alert_severity(
        self, alert: Alert, correlated_alerts: List[Alert]
    ) -> AlertSeverity:
        """Calculate alert severity based on context"""

        # Start with base severity
        severity = alert.severity

        # Increase severity based on correlation
        if len(correlated_alerts) > 3:
            if severity == AlertSeverity.LOW:
                severity = AlertSeverity.MEDIUM
            elif severity == AlertSeverity.MEDIUM:
                severity = AlertSeverity.HIGH

        # Increase severity based on frequency
        recent_alerts = [
            a for a in correlated_alerts if a.timestamp > datetime.utcnow() - timedelta(minutes=10)
        ]

        if len(recent_alerts) > 5:
            if severity == AlertSeverity.HIGH:
                severity = AlertSeverity.CRITICAL

        return severity

    async def send_notification(self, alert: Alert, channel_id: str) -> Dict[str, Any]:
        """Send notification to specific channel"""

        channel = self.notification_router.channels.get(channel_id)
        if not channel or not channel.get("enabled"):
            return {"success": False, "reason": "channel_disabled"}

        try:
            # This would integrate with actual notification services
            # For now, just log the notification

            logger.info(f"Sending notification for alert {alert.id} to channel {channel_id}")

            return {
                "success": True,
                "channel_id": channel_id,
                "sent_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to send notification to {channel_id}: {str(e)}")
            return {"success": False, "reason": str(e)}

    async def create_alert(self, alert_data: Dict[str, Any]) -> Alert:
        """Create new alert"""

        alert = Alert(
            id=str(uuid.uuid4()),
            type=alert_data["type"],
            severity=AlertSeverity(alert_data["severity"]),
            message=alert_data["message"],
            source=alert_data["source"],
            timestamp=datetime.utcnow(),
            metadata=alert_data.get("metadata", {}),
            correlation_id=alert_data.get("correlation_id"),
        )

        # Process alert
        await self.process_alert(alert)

        return alert

    async def process_pending_alerts(self):
        """Process pending alerts"""
        # This would process alerts from a queue or database
        pass

    async def cleanup(self):
        """Cleanup alerting system"""
        self.is_initialized = False
        logger.info("Alerting system cleaned up")


class AlertManager:
    """Alert storage and management"""

    def __init__(self):
        self.alerts = {}
        self.rules = {}

    async def store_alert(self, alert: Alert):
        """Store alert"""
        self.alerts[alert.id] = alert

    async def store_rule(self, rule: AlertRule):
        """Store alerting rule"""
        self.rules[rule.id] = rule

    async def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID"""
        return self.alerts.get(alert_id)

    async def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        return [alert for alert in self.alerts.values() if alert.status == AlertStatus.ACTIVE]
