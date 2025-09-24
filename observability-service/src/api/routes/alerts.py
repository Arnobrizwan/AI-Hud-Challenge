"""
Alerts API routes
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel

from monitoring.alerting import Alert, AlertingSystem, AlertSeverity, AlertStatus
from monitoring.observability_engine import observability_engine

logger = logging.getLogger(__name__)

router = APIRouter()


class AlertRequest(BaseModel):
    """Alert creation request"""

    type: str
    severity: str
    message: str
    source: str
    metadata: Dict[str, Any] = {}
    correlation_id: Optional[str] = None


class AlertResponse(BaseModel):
    """Alert response"""

    id: str
    type: str
    severity: str
    message: str
    source: str
    status: str
    timestamp: str
    metadata: Dict[str, Any]
    correlation_id: Optional[str] = None


class AlertRuleRequest(BaseModel):
    """Alert rule creation request"""

    name: str
    description: str
    condition: str
    severity: str
    threshold: float
    evaluation_window: int
    notification_channels: List[str]
    escalation_policy: Optional[str] = None


class AlertRuleResponse(BaseModel):
    """Alert rule response"""

    id: str
    name: str
    description: str
    condition: str
    severity: str
    threshold: float
    evaluation_window: int
    notification_channels: List[str]
    escalation_policy: Optional[str] = None
    created_at: str
    enabled: bool


class AlertProcessingResult(BaseModel):
    """Alert processing result"""

    alert_id: str
    action: str
    severity: Optional[str] = None
    notifications_sent: int = 0
    escalation_scheduled: bool = False
    reason: Optional[str] = None


@router.post("/", response_model=AlertResponse)
async def create_alert(alert_request: AlertRequest):
    """Create new alert"""

    try:
        if not observability_engine or not observability_engine.alerting_system:
            raise HTTPException(status_code=503, detail="Alerting system not available")

        # Create alert
        alert_data = {
            "type": alert_request.type,
            "severity": alert_request.severity,
            "message": alert_request.message,
            "source": alert_request.source,
            "metadata": alert_request.metadata,
            "correlation_id": alert_request.correlation_id,
        }

        alert = await observability_engine.alerting_system.create_alert(alert_data)

        return AlertResponse(
            id=alert.id,
            type=alert.type,
            severity=alert.severity.value,
            message=alert.message,
            source=alert.source,
            status=alert.status.value,
            timestamp=alert.timestamp.isoformat(),
            metadata=alert.metadata,
            correlation_id=alert.correlation_id,
        )

    except Exception as e:
        logger.error(f"Failed to create alert: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create alert: {str(e)}")


@router.get("/", response_model=List[AlertResponse])
async def get_alerts(
    status: Optional[str] = Query(None, description="Filter by alert status"),
    severity: Optional[str] = Query(None, description="Filter by alert severity"),
    source: Optional[str] = Query(None, description="Filter by alert source"),
    limit: int = Query(100, description="Maximum number of alerts to return"),
):
    """Get alerts with optional filtering"""

    try:
        if not observability_engine or not observability_engine.alerting_system:
            raise HTTPException(status_code=503, detail="Alerting system not available")

        # Get alerts from alert manager
        alert_manager = observability_engine.alerting_system.alert_manager
        all_alerts = list(alert_manager.alerts.values())

        # Apply filters
        filtered_alerts = all_alerts

        if status:
            filtered_alerts = [a for a in filtered_alerts if a.status.value == status]

        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity.value == severity]

        if source:
            filtered_alerts = [a for a in filtered_alerts if a.source == source]

        # Limit results
        filtered_alerts = filtered_alerts[:limit]

        # Convert to response format
        alerts = []
        for alert in filtered_alerts:
            alerts.append(
                AlertResponse(
                    id=alert.id,
                    type=alert.type,
                    severity=alert.severity.value,
                    message=alert.message,
                    source=alert.source,
                    status=alert.status.value,
                    timestamp=alert.timestamp.isoformat(),
                    metadata=alert.metadata,
                    correlation_id=alert.correlation_id,
                )
            )

        return alerts

    except Exception as e:
        logger.error(f"Failed to get alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: str):
    """Get specific alert by ID"""

    try:
        if not observability_engine or not observability_engine.alerting_system:
            raise HTTPException(status_code=503, detail="Alerting system not available")

        # Get alert from alert manager
        alert_manager = observability_engine.alerting_system.alert_manager
        alert = await alert_manager.get_alert(alert_id)

        if not alert:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        return AlertResponse(
            id=alert.id,
            type=alert.type,
            severity=alert.severity.value,
            message=alert.message,
            source=alert.source,
            status=alert.status.value,
            timestamp=alert.timestamp.isoformat(),
            metadata=alert.metadata,
            correlation_id=alert.correlation_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert {alert_id}: {str(e)}")


@router.put("/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user: str = Body(..., embed=True)):
    """Acknowledge alert"""

    try:
        if not observability_engine or not observability_engine.alerting_system:
            raise HTTPException(status_code=503, detail="Alerting system not available")

        # Get alert
        alert_manager = observability_engine.alerting_system.alert_manager
        alert = await alert_manager.get_alert(alert_id)

        if not alert:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        # Acknowledge alert
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = user
        alert.acknowledged_at = datetime.utcnow()

        return {
            "message": f"Alert {alert_id} acknowledged by {user}",
            "alert_id": alert_id,
            "acknowledged_by": user,
            "acknowledged_at": alert.acknowledged_at.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to acknowledge alert {alert_id}: {str(e)}"
        )


@router.put("/{alert_id}/resolve")
async def resolve_alert(alert_id: str, user: str = Body(..., embed=True)):
    """Resolve alert"""

    try:
        if not observability_engine or not observability_engine.alerting_system:
            raise HTTPException(status_code=503, detail="Alerting system not available")

        # Get alert
        alert_manager = observability_engine.alerting_system.alert_manager
        alert = await alert_manager.get_alert(alert_id)

        if not alert:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        # Resolve alert
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()

        return {
            "message": f"Alert {alert_id} resolved by {user}",
            "alert_id": alert_id,
            "resolved_by": user,
            "resolved_at": alert.resolved_at.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert {alert_id}: {str(e)}")


@router.post("/rules", response_model=AlertRuleResponse)
async def create_alert_rule(rule_request: AlertRuleRequest):
    """Create new alert rule"""

    try:
        if not observability_engine or not observability_engine.alerting_system:
            raise HTTPException(status_code=503, detail="Alerting system not available")

        # Create alert rule
        rule_config = {
            "name": rule_request.name,
            "description": rule_request.description,
            "condition": rule_request.condition,
            "severity": rule_request.severity,
            "threshold": rule_request.threshold,
            "evaluation_window": rule_request.evaluation_window,
            "notification_channels": rule_request.notification_channels,
            "escalation_policy": rule_request.escalation_policy,
        }

        rule = await observability_engine.alerting_system.create_alerting_rule(rule_config)

        return AlertRuleResponse(
            id=rule.id,
            name=rule.name,
            description=rule.description,
            condition=rule.condition,
            severity=rule.severity.value,
            threshold=rule.threshold,
            evaluation_window=rule.evaluation_window,
            notification_channels=rule.notification_channels,
            escalation_policy=rule.escalation_policy,
            created_at=rule.created_at.isoformat(),
            enabled=rule.enabled,
        )

    except Exception as e:
        logger.error(f"Failed to create alert rule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create alert rule: {str(e)}")


@router.get("/rules", response_model=List[AlertRuleResponse])
async def get_alert_rules():
    """Get all alert rules"""

    try:
        if not observability_engine or not observability_engine.alerting_system:
            raise HTTPException(status_code=503, detail="Alerting system not available")

        # Get alert rules from alert manager
        alert_manager = observability_engine.alerting_system.alert_manager
        rules = list(alert_manager.rules.values())

        # Convert to response format
        rule_responses = []
        for rule in rules:
            rule_responses.append(
                AlertRuleResponse(
                    id=rule.id,
                    name=rule.name,
                    description=rule.description,
                    condition=rule.condition,
                    severity=rule.severity.value,
                    threshold=rule.threshold,
                    evaluation_window=rule.evaluation_window,
                    notification_channels=rule.notification_channels,
                    escalation_policy=rule.escalation_policy,
                    created_at=rule.created_at.isoformat(),
                    enabled=rule.enabled,
                )
            )

        return rule_responses

    except Exception as e:
        logger.error(f"Failed to get alert rules: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert rules: {str(e)}")


@router.get("/rules/{rule_id}", response_model=AlertRuleResponse)
async def get_alert_rule(rule_id: str):
    """Get specific alert rule by ID"""

    try:
        if not observability_engine or not observability_engine.alerting_system:
            raise HTTPException(status_code=503, detail="Alerting system not available")

        # Get alert rule from alert manager
        alert_manager = observability_engine.alerting_system.alert_manager
        rule = alert_manager.rules.get(rule_id)

        if not rule:
            raise HTTPException(status_code=404, detail=f"Alert rule {rule_id} not found")

        return AlertRuleResponse(
            id=rule.id,
            name=rule.name,
            description=rule.description,
            condition=rule.condition,
            severity=rule.severity.value,
            threshold=rule.threshold,
            evaluation_window=rule.evaluation_window,
            notification_channels=rule.notification_channels,
            escalation_policy=rule.escalation_policy,
            created_at=rule.created_at.isoformat(),
            enabled=rule.enabled,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alert rule {rule_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert rule {rule_id}: {str(e)}")


@router.put("/rules/{rule_id}/enable")
async def enable_alert_rule(rule_id: str):
    """Enable alert rule"""

    try:
        if not observability_engine or not observability_engine.alerting_system:
            raise HTTPException(status_code=503, detail="Alerting system not available")

        # Get alert rule
        alert_manager = observability_engine.alerting_system.alert_manager
        rule = alert_manager.rules.get(rule_id)

        if not rule:
            raise HTTPException(status_code=404, detail=f"Alert rule {rule_id} not found")

        # Enable rule
        rule.enabled = True

        return {"message": f"Alert rule {rule_id} enabled", "rule_id": rule_id, "enabled": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable alert rule {rule_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to enable alert rule {rule_id}: {str(e)}"
        )


@router.put("/rules/{rule_id}/disable")
async def disable_alert_rule(rule_id: str):
    """Disable alert rule"""

    try:
        if not observability_engine or not observability_engine.alerting_system:
            raise HTTPException(status_code=503, detail="Alerting system not available")

        # Get alert rule
        alert_manager = observability_engine.alerting_system.alert_manager
        rule = alert_manager.rules.get(rule_id)

        if not rule:
            raise HTTPException(status_code=404, detail=f"Alert rule {rule_id} not found")

        # Disable rule
        rule.enabled = False

        return {"message": f"Alert rule {rule_id} disabled", "rule_id": rule_id, "enabled": False}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable alert rule {rule_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to disable alert rule {rule_id}: {str(e)}"
        )


@router.get("/active", response_model=List[AlertResponse])
async def get_active_alerts():
    """Get active alerts"""

    try:
        if not observability_engine or not observability_engine.alerting_system:
            raise HTTPException(status_code=503, detail="Alerting system not available")

        # Get active alerts
        active_alerts = await observability_engine.alerting_system.alert_manager.get_active_alerts()

        # Convert to response format
        alerts = []
        for alert in active_alerts:
            alerts.append(
                AlertResponse(
                    id=alert.id,
                    type=alert.type,
                    severity=alert.severity.value,
                    message=alert.message,
                    source=alert.source,
                    status=alert.status.value,
                    timestamp=alert.timestamp.isoformat(),
                    metadata=alert.metadata,
                    correlation_id=alert.correlation_id,
                )
            )

        return alerts

    except Exception as e:
        logger.error(f"Failed to get active alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active alerts: {str(e)}")


@router.get("/stats")
async def get_alert_statistics(
    time_window: int = Query(3600, description="Time window in seconds")
):
    """Get alert statistics"""

    try:
        if not observability_engine or not observability_engine.alerting_system:
            raise HTTPException(status_code=503, detail="Alerting system not available")

        # Get alert statistics
        alert_manager = observability_engine.alerting_system.alert_manager
        all_alerts = list(alert_manager.alerts.values())

        # Filter by time window
        cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
        recent_alerts = [a for a in all_alerts if a.timestamp >= cutoff_time]

        # Calculate statistics
        total_alerts = len(recent_alerts)
        active_alerts = len([a for a in recent_alerts if a.status == AlertStatus.ACTIVE])
        resolved_alerts = len([a for a in recent_alerts if a.status == AlertStatus.RESOLVED])

        # Group by severity
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len(
                [a for a in recent_alerts if a.severity == severity]
            )

        # Group by source
        source_counts = {}
        for alert in recent_alerts:
            source = alert.source
            source_counts[source] = source_counts.get(source, 0) + 1

        return {
            "time_window_seconds": time_window,
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "resolved_alerts": resolved_alerts,
            "severity_breakdown": severity_counts,
            "source_breakdown": source_counts,
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get alert statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert statistics: {str(e)}")


@router.post("/test")
async def test_alert_system():
    """Test alert system by creating a test alert"""

    try:
        if not observability_engine or not observability_engine.alerting_system:
            raise HTTPException(status_code=503, detail="Alerting system not available")

        # Create test alert
        test_alert_data = {
            "type": "test_alert",
            "severity": "low",
            "message": "Test alert to verify alerting system functionality",
            "source": "test_system",
            "metadata": {"test": True},
        }

        alert = await observability_engine.alerting_system.create_alert(test_alert_data)

        return {
            "message": "Test alert created successfully",
            "alert_id": alert.id,
            "alert_type": alert.type,
            "severity": alert.severity.value,
            "timestamp": alert.timestamp.isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to create test alert: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create test alert: {str(e)}")
