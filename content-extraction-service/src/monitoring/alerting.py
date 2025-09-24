"""
Alerting system for content extraction service.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from .metrics import metrics


class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert data structure."""

    id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    component: str
    metric_name: str
    threshold_value: float
    current_value: float
    created_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class AlertRule:
    """Alert rule definition."""

    def __init__(
        self,
        name: str,
        metric_name: str,
        threshold: float,
        comparison: str,
        severity: AlertSeverity,
        component: str,
        description: str,
        evaluation_interval: int = 60,
        cooldown_period: int = 300,
    ):
        """Initialize alert rule."""
        self.name = name
        self.metric_name = metric_name
        self.threshold = threshold
        self.comparison = comparison  # '>', '<', '>=', '<=', '==', '!='
        self.severity = severity
        self.component = component
        self.description = description
        self.evaluation_interval = evaluation_interval
        self.cooldown_period = cooldown_period
        self.last_evaluation = None
        self.last_alert_time = None

    def evaluate(self, current_value: float) -> bool:
        """Evaluate if alert should be triggered."""
        now = datetime.utcnow()

        # Check if enough time has passed since last evaluation
        if (self.last_evaluation and (
                now - self.last_evaluation).total_seconds() < self.evaluation_interval):
            return False

        # Check if in cooldown period
        if (self.last_alert_time and (
                now - self.last_alert_time).total_seconds() < self.cooldown_period):
            return False

        # Evaluate condition
        if self.comparison == ">":
            return current_value > self.threshold
        elif self.comparison == "<":
            return current_value < self.threshold
        elif self.comparison == ">=":
            return current_value >= self.threshold
        elif self.comparison == "<=":
            return current_value <= self.threshold
        elif self.comparison == "==":
            return current_value == self.threshold
        elif self.comparison == "!=":
            return current_value != self.threshold

        return False

    def update_evaluation_time(self):
        """Update last evaluation time."""
        self.last_evaluation = datetime.utcnow()

    def update_alert_time(self):
        """Update last alert time."""
        self.last_alert_time = datetime.utcnow()


class AlertManager:
    """Alert management system."""

    def __init__(self):
        """Initialize alert manager."""
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[AlertRule] = []
        self.notification_handlers: List[Callable] = []
        self.is_running = False
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default alert rules."""
        # High error rate
        self.add_alert_rule(
            AlertRule(
                name="high_error_rate",
                metric_name="error_rate",
                threshold=0.05,  # 5%
                comparison=">",
                severity=AlertSeverity.HIGH,
                component="service",
                description="Error rate is above 5%",
                evaluation_interval=60,
                cooldown_period=300,
            )
        )

        # Low cache hit rate
        self.add_alert_rule(
            AlertRule(
                name="low_cache_hit_rate",
                metric_name="cache_hit_rate",
                threshold=0.7,  # 70%
                comparison="<",
                severity=AlertSeverity.MEDIUM,
                component="cache",
                description="Cache hit rate is below 70%",
                evaluation_interval=120,
                cooldown_period=600,
            )
        )

        # High memory usage
        self.add_alert_rule(
            AlertRule(
                name="high_memory_usage",
                metric_name="memory_usage_bytes",
                threshold=1.5 * 1024 * 1024 * 1024,  # 1.5GB
                comparison=">",
                severity=AlertSeverity.HIGH,
                component="system",
                description="Memory usage is above 1.5GB",
                evaluation_interval=60,
                cooldown_period=300,
            )
        )

        # High processing time
        self.add_alert_rule(
            AlertRule(
                name="high_processing_time",
                metric_name="average_processing_time",
                threshold=30.0,  # 30 seconds
                comparison=">",
                severity=AlertSeverity.MEDIUM,
                component="extraction",
                description="Average processing time is above 30 seconds",
                evaluation_interval=120,
                cooldown_period=600,
            )
        )

        # Queue size too large
        self.add_alert_rule(
            AlertRule(
                name="large_queue_size",
                metric_name="queue_size",
                threshold=100,
                comparison=">",
                severity=AlertSeverity.MEDIUM,
                component="queue",
                description="Queue size is above 100 tasks",
                evaluation_interval=60,
                cooldown_period=300,
            )
        )

    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def add_notification_handler(self, handler: Callable):
        """Add notification handler."""
        self.notification_handlers.append(handler)
        logger.info("Added notification handler")

    async def start_monitoring(self) -> Dict[str, Any]:
        """Start alert monitoring."""
        if self.is_running:
            return

        self.is_running = True
        logger.info("Started alert monitoring")

        while self.is_running:
            try:
    await self._evaluate_alerts()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Alert monitoring error: {str(e)}")
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> Dict[str, Any]:
        """Stop alert monitoring."""
        self.is_running = False
        logger.info("Stopped alert monitoring")

    async def _evaluate_alerts(self) -> Dict[str, Any]:
        """Evaluate all alert rules."""
        for rule in self.alert_rules:
            try:
                current_value = await self._get_metric_value(rule.metric_name)

                if rule.evaluate(current_value):
    await self._trigger_alert(rule, current_value)
                    rule.update_alert_time()

                rule.update_evaluation_time()

            except Exception as e:
                logger.error(
                    f"Alert evaluation failed for {rule.name}: {str(e)}")

    async def _get_metric_value(self, metric_name: str) -> float:
        """Get current metric value."""
        if metric_name == "error_rate":
            return metrics.get_error_rate()
        elif metric_name == "cache_hit_rate":
            return metrics.get_cache_hit_rate()
        elif metric_name == "memory_usage_bytes":
            return metrics.memory_usage_bytes._value.get()
        elif metric_name == "average_processing_time":
            return metrics.get_average_processing_time()
        elif metric_name == "queue_size":
            return metrics.queue_size._value.get()
        else:
            return 0.0

    async def _trigger_alert(self, rule: AlertRule, current_value: float) -> Dict[str, Any]:
        """Trigger alert."""
        alert_id = f"{rule.name}_{int(datetime.utcnow().timestamp())}"

        alert = Alert(
            id=alert_id,
            title=f"{rule.severity.value.upper()}: {rule.name}",
            description=rule.description,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            component=rule.component,
            metric_name=rule.metric_name,
            threshold_value=rule.threshold,
            current_value=current_value,
            created_at=datetime.utcnow(),
            metadata={"rule_name": rule.name, "comparison": rule.comparison},
        )

        self.alerts[alert_id] = alert

        # Send notifications
        await self._send_notifications(alert)

        logger.warning(
            f"Alert triggered: {alert.title} (value: {current_value})")

    async def _send_notifications(self, alert: Alert) -> Dict[str, Any]:
        """Send alert notifications."""
        for handler in self.notification_handlers:
            try:
    await handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {str(e)}")

    async def resolve_alert(self, alert_id: str) -> Dict[str, Any]:
        """Resolve alert."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            logger.info(f"Alert resolved: {alert.title}")

    async def suppress_alert(self, alert_id: str) -> Dict[str, Any]:
        """Suppress alert."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            logger.info(f"Alert suppressed: {alert.title}")

    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return [alert for alert in self.alerts.values() if alert.status ==
                AlertStatus.ACTIVE]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity."""
        return [alert for alert in self.alerts.values()
                if alert.severity == severity]

    def get_alerts_by_component(self, component: str) -> List[Alert]:
        """Get alerts by component."""
        return [alert for alert in self.alerts.values()
                if alert.component == component]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())

        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len(
                self.get_alerts_by_severity(severity))

        component_counts = {}
        components = set(alert.component for alert in self.alerts.values())
        for component in components:
            component_counts[component] = len(
                self.get_alerts_by_component(component))

        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "resolved_alerts": total_alerts - active_alerts,
            "severity_counts": severity_counts,
            "component_counts": component_counts,
            "alert_rules_count": len(self.alert_rules),
        }


# Global alert manager instance
alert_manager = AlertManager()


# Notification handlers
async def log_notification_handler(alert: Alert) -> Dict[str, Any]:
    """Log notification handler."""
    logger.warning(f"ALERT: {alert.title} - {alert.description}")


async def email_notification_handler(alert: Alert) -> Dict[str, Any]:
    """Email notification handler (placeholder)."""
    # In production, this would send actual emails
    logger.info(f"EMAIL ALERT: {alert.title}")


async def slack_notification_handler(alert: Alert) -> Dict[str, Any]:
    """Slack notification handler (placeholder)."""
    # In production, this would send Slack messages
    logger.info(f"SLACK ALERT: {alert.title}")


# Register default notification handlers
alert_manager.add_notification_handler(log_notification_handler)
alert_manager.add_notification_handler(email_notification_handler)
alert_manager.add_notification_handler(slack_notification_handler)
