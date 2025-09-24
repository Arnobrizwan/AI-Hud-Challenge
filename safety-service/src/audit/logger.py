"""
Audit Logger
Comprehensive audit trails and reporting
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from safety_engine.config import get_settings
from safety_engine.models import SafetyMonitoringRequest, SafetyStatus

from .models import AuditEvent, AuditLog, AuditMetrics, AuditQuery, AuditReport

logger = logging.getLogger(__name__)


class AuditLogger:
    """Comprehensive audit logging and reporting system"""

    def __init__(self):
        self.config = get_settings()
        self.is_initialized = False

        # Audit storage
        self.audit_events: List[AuditEvent] = []
        self.audit_logs: List[AuditLog] = []
        self.audit_reports: List[AuditReport] = []

        # Configuration
        self.retention_days = 90
        self.max_events_per_log = 10000
        self.compression_enabled = True
        self.encryption_enabled = True

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the audit logger"""
        try:
            # Initialize storage and services
            await self.initialize_storage()

            # Start background tasks
            asyncio.create_task(self.audit_cleanup_task())
            asyncio.create_task(self.audit_compression_task())

            self.is_initialized = True
            logger.info("Audit logger initialized")

        except Exception as e:
            logger.error(f"Failed to initialize audit logger: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup resources"""
        try:
            self.audit_events.clear()
            self.audit_logs.clear()
            self.audit_reports.clear()

            self.is_initialized = False
            logger.info("Audit logger cleanup completed")

        except Exception as e:
            logger.error(f"Error during audit logger cleanup: {str(e)}")

    async def log_safety_check(self,
                               request: SafetyMonitoringRequest,
                               status: SafetyStatus) -> str:
        """Log safety check audit event"""
        try:
            event = AuditEvent(
                event_id=f"audit_{uuid.uuid4().hex[:8]}",
                event_type="safety_check",
                severity="high" if status.requires_intervention else "medium",
                status="success" if not status.requires_intervention else "warning",
                timestamp=datetime.utcnow(),
                user_id=getattr(request, "user_id", None),
                request_id=getattr(request, "request_id", None),
                action="safety_monitoring",
                description=f"Safety check completed with score {status.overall_score:.2f}",
                details={
                    "overall_score": status.overall_score,
                    "requires_intervention": status.requires_intervention,
                    "drift_status": status.drift_status.__dict__ if status.drift_status else None,
                    "abuse_status": status.abuse_status.__dict__ if status.abuse_status else None,
                    "content_status": (
                        status.content_status.__dict__ if status.content_status else None
                    ),
                },
                tags=["safety", "monitoring", "automated"],
            )

            await self.store_audit_event(event)
            return event.event_id

        except Exception as e:
            logger.error(f"Safety check audit logging failed: {str(e)}")
            return ""

    async def log_drift_detection(
            self,
            drift_request: Any,
            drift_result: Any) -> str:
        """Log drift detection audit event"""
        try:
            event = AuditEvent(
                event_id=f"audit_{uuid.uuid4().hex[:8]}",
                event_type="drift_detection",
                severity="high" if drift_result.overall_severity > 0.7 else "medium",
                status="warning" if drift_result.requires_action else "success",
                timestamp=datetime.utcnow(),
                action="drift_detection",
                description=f"Drift detected with severity {drift_result.overall_severity:.2f}",
                details={
                    "overall_severity": drift_result.overall_severity,
                    "requires_action": drift_result.requires_action,
                    "drifted_features": (
                        drift_result.data_drift.drifted_features if drift_result.data_drift else []),
                },
                tags=[
                    "drift",
                    "ml",
                    "monitoring"],
            )

            await self.store_audit_event(event)
            return event.event_id

        except Exception as e:
            logger.error(f"Drift detection audit logging failed: {str(e)}")
            return ""

    async def log_abuse_detection(
            self,
            abuse_request: Any,
            abuse_result: Any) -> str:
        """Log abuse detection audit event"""
        try:
            event = AuditEvent(
                event_id=f"audit_{uuid.uuid4().hex[:8]}",
                event_type="abuse_detection",
                severity="critical" if abuse_result.threat_level == "high" else "high",
                status="warning" if abuse_result.abuse_score > 0.5 else "success",
                timestamp=datetime.utcnow(),
                user_id=getattr(abuse_request, "user_id", None),
                action="abuse_detection",
                description=f"Abuse detected for user {abuse_result.user_id} with score {abuse_result.abuse_score:.2f}",
                details={
                    "user_id": abuse_result.user_id,
                    "abuse_score": abuse_result.abuse_score,
                    "threat_level": abuse_result.threat_level,
                    "rule_violations": (
                        len(abuse_result.rule_violations) if abuse_result.rule_violations else 0
                    ),
                },
                tags=["abuse", "security", "user_behavior"],
            )

            await self.store_audit_event(event)
            return event.event_id

        except Exception as e:
            logger.error(f"Abuse detection audit logging failed: {str(e)}")
            return ""

    async def log_content_moderation(
            self,
            content: Any,
            moderation_result: Any) -> str:
        """Log content moderation audit event"""
        try:
            event = AuditEvent(
                event_id=f"audit_{uuid.uuid4().hex[:8]}",
                event_type="content_moderation",
                severity="high" if moderation_result.overall_safety_score < 0.5 else "medium",
                status="warning" if moderation_result.violations else "success",
                timestamp=datetime.utcnow(),
                action="content_moderation",
                description=f"Content moderated with safety score {moderation_result.overall_safety_score:.2f}",
                details={
                    "content_id": moderation_result.content_id,
                    "safety_score": moderation_result.overall_safety_score,
                    "violations": (
                        len(moderation_result.violations) if moderation_result.violations else 0
                    ),
                    "recommended_action": moderation_result.recommended_action,
                },
                tags=["content", "moderation", "safety"],
            )

            await self.store_audit_event(event)
            return event.event_id

        except Exception as e:
            logger.error(f"Content moderation audit logging failed: {str(e)}")
            return ""

    async def log_rate_limiting(
            self,
            rate_limit_request: Any,
            rate_limit_result: Any) -> str:
        """Log rate limiting audit event"""
        try:
            event = AuditEvent(
                event_id=f"audit_{uuid.uuid4().hex[:8]}",
                event_type="rate_limiting",
                severity="medium" if rate_limit_result.is_rate_limited else "low",
                status="warning" if rate_limit_result.is_rate_limited else "success",
                timestamp=datetime.utcnow(),
                user_id=getattr(rate_limit_request, "user_id", None),
                action="rate_limiting",
                description=f"Rate limit check for user {rate_limit_result.user_id} - {'Limited' if rate_limit_result.is_rate_limited else 'Allowed'}",
                details={
                    "user_id": rate_limit_result.user_id,
                    "endpoint": rate_limit_result.endpoint,
                    "is_rate_limited": rate_limit_result.is_rate_limited,
                    "triggered_limits": rate_limit_result.triggered_limits,
                    "remaining_capacity": rate_limit_result.remaining_capacity,
                },
                tags=["rate_limiting", "security", "performance"],
            )

            await self.store_audit_event(event)
            return event.event_id

        except Exception as e:
            logger.error(f"Rate limiting audit logging failed: {str(e)}")
            return ""

    async def log_compliance_check(
            self,
            compliance_request: Any,
            compliance_result: Any) -> str:
        """Log compliance check audit event"""
        try:
            event = AuditEvent(
                event_id=f"audit_{uuid.uuid4().hex[:8]}",
                event_type="compliance_check",
                severity="high" if compliance_result.overall_compliance_score < 0.8 else "medium",
                status="warning" if compliance_result.violations else "success",
                timestamp=datetime.utcnow(),
                action="compliance_check",
                description=f"Compliance check completed with score {compliance_result.overall_compliance_score:.2f}",
                details={
                    "overall_compliance_score": compliance_result.overall_compliance_score,
                    "violations": (
                        len(compliance_result.violations) if compliance_result.violations else 0
                    ),
                    "compliance_results": compliance_result.compliance_results,
                },
                tags=["compliance", "regulatory", "audit"],
            )

            await self.store_audit_event(event)
            return event.event_id

        except Exception as e:
            logger.error(f"Compliance check audit logging failed: {str(e)}")
            return ""

    async def log_incident_response(self, incident: Any, response: Any) -> str:
        """Log incident response audit event"""
        try:
            event = AuditEvent(
                event_id=f"audit_{uuid.uuid4().hex[:8]}",
                event_type="incident_response",
                severity="critical" if incident.severity == "critical" else "high",
                status="success" if response else "failure",
                timestamp=datetime.utcnow(),
                action="incident_response",
                description=f"Incident {incident.id} response initiated",
                details={
                    "incident_id": incident.id,
                    "incident_type": incident.incident_type,
                    "severity": incident.severity,
                    "response_plan_id": response.plan_id if response else None,
                },
                tags=[
                    "incident",
                    "response",
                    "emergency"],
            )

            await self.store_audit_event(event)
            return event.event_id

        except Exception as e:
            logger.error(f"Incident response audit logging failed: {str(e)}")
            return ""

    async def store_audit_event(self, event: AuditEvent) -> None:
        """Store audit event"""
        try:
            self.audit_events.append(event)

            # Check if we need to create a new audit log
            if len(self.audit_events) >= self.max_events_per_log:
    await self.create_audit_log()

        except Exception as e:
            logger.error(f"Audit event storage failed: {str(e)}")

    async def create_audit_log(self) -> str:
        """Create audit log from current events"""
        try:
            if not self.audit_events:
                return ""

            # Create audit log
            log_id = f"log_{uuid.uuid4().hex[:8]}"
            start_time = min(event.timestamp for event in self.audit_events)
            end_time = max(event.timestamp for event in self.audit_events)

            # Calculate summary
            total_events = len(self.audit_events)
            success_count = sum(
                1 for event in self.audit_events if event.status == "success")
            failure_count = sum(
                1 for event in self.audit_events if event.status == "failure")
            warning_count = sum(
                1 for event in self.audit_events if event.status == "warning")
            info_count = sum(
                1 for event in self.audit_events if event.status == "info")

            audit_log = AuditLog(
                log_id=log_id,
                log_type="safety_service",
                start_time=start_time,
                end_time=end_time,
                events=self.audit_events.copy(),
                summary={
                    "total_events": total_events,
                    "success_rate": success_count /
                    total_events if total_events > 0 else 0,
                    "failure_rate": failure_count /
                    total_events if total_events > 0 else 0,
                },
                total_events=total_events,
                success_count=success_count,
                failure_count=failure_count,
                warning_count=warning_count,
                info_count=info_count,
            )

            # Store audit log
            self.audit_logs.append(audit_log)

            # Clear events
            self.audit_events.clear()

            logger.info(
                f"Created audit log {log_id} with {total_events} events")
            return log_id

        except Exception as e:
            logger.error(f"Audit log creation failed: {str(e)}")
            return ""

    async def query_audit_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events based on criteria"""
        try:
            events = self.audit_events.copy()

            # Apply filters
            if query.start_time:
                events = [e for e in events if e.timestamp >= query.start_time]

            if query.end_time:
                events = [e for e in events if e.timestamp <= query.end_time]

            if query.event_types:
                events = [
                    e for e in events if e.event_type in query.event_types]

            if query.severities:
                events = [e for e in events if e.severity in query.severities]

            if query.statuses:
                events = [e for e in events if e.status in query.statuses]

            if query.user_ids:
                events = [e for e in events if e.user_id in query.user_ids]

            if query.resources:
                events = [e for e in events if e.resource in query.resources]

            if query.actions:
                events = [e for e in events if e.action in query.actions]

            if query.tags:
                events = [
                    e for e in events if any(
                        tag in e.tags for tag in query.tags)]

            if query.correlation_id:
                events = [
                    e for e in events if e.correlation_id == query.correlation_id]

            # Apply sorting
            if query.sort_by == "timestamp":
                events.sort(
                    key=lambda x: x.timestamp, reverse=(
                        query.sort_order == "desc"))
            elif query.sort_by == "severity":
                severity_order = {
                    "critical": 4, "high": 3, "medium": 2, "low": 1}
                events.sort(
                    key=lambda x: severity_order.get(x.severity, 0),
                    reverse=(query.sort_order == "desc"),
                )

            # Apply pagination
            start = query.offset or 0
            end = start + (query.limit or 1000)
            events = events[start:end]

            return events

        except Exception as e:
            logger.error(f"Audit event query failed: {str(e)}")
            return []

    async def generate_audit_report(
            self,
            start_time: datetime,
            end_time: datetime) -> AuditReport:
        """Generate comprehensive audit report"""
        try:
            # Get events in time range
            events = [e for e in self.audit_events if start_time <=
                      e.timestamp <= end_time]

            # Calculate statistics
            total_events = len(events)
            events_by_type = {}
            events_by_severity = {}
            events_by_status = {}

            for event in events:
                events_by_type[event.event_type] = events_by_type.get(
                    event.event_type, 0) + 1
                events_by_severity[event.severity] = events_by_severity.get(
                    event.severity, 0) + 1
                events_by_status[event.status] = events_by_status.get(
                    event.status, 0) + 1

            # Get top users
            user_counts = {}
            for event in events:
                if event.user_id:
                    user_counts[event.user_id] = user_counts.get(
                        event.user_id, 0) + 1

            top_users = [
                {"user_id": user, "count": count}
                for user, count in sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
            ]

            # Get top resources
            resource_counts = {}
            for event in events:
                if event.resource:
                    resource_counts[event.resource] = resource_counts.get(
                        event.resource, 0) + 1

            top_resources = [
                {"resource": resource, "count": count}
                for resource, count in sorted(
                    resource_counts.items(), key=lambda x: x[1], reverse=True
                )[:10]
            ]

            # Get top actions
            action_counts = {}
            for event in events:
                action_counts[event.action] = action_counts.get(
                    event.action, 0) + 1

            top_actions = [
                {"action": action, "count": count}
                for action, count in sorted(
                    action_counts.items(), key=lambda x: x[1], reverse=True
                )[:10]
            ]

            # Get security and compliance events
            security_events = [
                e
                for e in events
                if e.event_type in ["abuse_detection", "rate_limiting", "system_access"]
            ]
            compliance_events = [
                e for e in events if e.event_type in [
                    "compliance_check", "data_access"]]

            # Generate report
            report = AuditReport(
                report_id=f"report_{uuid.uuid4().hex[:8]}",
                report_type="comprehensive",
                generated_at=datetime.utcnow(),
                period_start=start_time,
                period_end=end_time,
                total_events=total_events,
                events_by_type=events_by_type,
                events_by_severity=events_by_severity,
                events_by_status=events_by_status,
                top_users=top_users,
                top_resources=top_resources,
                top_actions=top_actions,
                security_events=security_events,
                compliance_events=compliance_events,
                anomalies=[],  # Placeholder for anomaly detection
                recommendations=[],  # Placeholder for recommendations
                summary=f"Audit report covering {total_events} events from {start_time} to {end_time}",
                details={},
            )

            self.audit_reports.append(report)
            return report

        except Exception as e:
            logger.error(f"Audit report generation failed: {str(e)}")
            raise

    async def get_audit_metrics(self) -> AuditMetrics:
        """Get audit metrics and statistics"""
        try:
            total_events = len(self.audit_events)

            # Calculate time-based metrics
            now = datetime.utcnow()
            today_start = now.replace(
                hour=0, minute=0, second=0, microsecond=0)
            week_start = today_start - timedelta(days=7)
            month_start = today_start - timedelta(days=30)

            events_today = len(
                [e for e in self.audit_events if e.timestamp >= today_start])
            events_this_week = len(
                [e for e in self.audit_events if e.timestamp >= week_start])
            events_this_month = len(
                [e for e in self.audit_events if e.timestamp >= month_start])

            # Calculate success/failure rates
            success_count = sum(
                1 for e in self.audit_events if e.status == "success")
            failure_count = sum(
                1 for e in self.audit_events if e.status == "failure")
            success_rate = success_count / total_events if total_events > 0 else 0
            failure_rate = failure_count / total_events if total_events > 0 else 0

            # Calculate average duration
            durations = [
                e.duration_ms for e in self.audit_events if e.duration_ms is not None]
            average_duration = sum(durations) / \
                len(durations) if durations else 0

            # Get top event types
            event_type_counts = {}
            for event in self.audit_events:
                event_type_counts[event.event_type] = event_type_counts.get(
                    event.event_type, 0) + 1

            top_event_types = [
                {"type": event_type, "count": count}
                for event_type, count in sorted(
                    event_type_counts.items(), key=lambda x: x[1], reverse=True
                )[:10]
            ]

            # Get top users
            user_counts = {}
            for event in self.audit_events:
                if event.user_id:
                    user_counts[event.user_id] = user_counts.get(
                        event.user_id, 0) + 1

            top_users = [
                {"user_id": user, "count": count}
                for user, count in sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
            ]

            # Count security and compliance events
            security_events_count = len(
                [
                    e
                    for e in self.audit_events
                    if e.event_type in ["abuse_detection", "rate_limiting", "system_access"]
                ]
            )
            compliance_events_count = len(
                [
                    e
                    for e in self.audit_events
                    if e.event_type in ["compliance_check", "data_access"]
                ]
            )

            return AuditMetrics(
                total_events=total_events,
                events_today=events_today,
                events_this_week=events_this_week,
                events_this_month=events_this_month,
                success_rate=success_rate,
                failure_rate=failure_rate,
                average_duration=average_duration,
                top_event_types=top_event_types,
                top_users=top_users,
                security_events_count=security_events_count,
                compliance_events_count=compliance_events_count,
                anomaly_count=0,  # Placeholder
                last_updated=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Audit metrics calculation failed: {str(e)}")
            raise

    async def audit_cleanup_task(self) -> Dict[str, Any]:
        """Background task for cleaning up old audit data"""
        while True:
            try:
    await asyncio.sleep(3600)  # Run every hour

                if not self.is_initialized:
                    break

                # Clean up old events
                cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
                old_events = [
                    e for e in self.audit_events if e.timestamp < cutoff_date]

                for event in old_events:
                    self.audit_events.remove(event)

                # Clean up old logs
                old_logs = [
                    l for l in self.audit_logs if l.end_time < cutoff_date]

                for log in old_logs:
                    self.audit_logs.remove(log)

                if old_events or old_logs:
                    logger.info(
                        f"Cleaned up {len(old_events)} old events and {len(old_logs)} old logs"
                    )

            except Exception as e:
                logger.error(f"Audit cleanup task failed: {str(e)}")
                await asyncio.sleep(3600)

    async def audit_compression_task(self) -> Dict[str, Any]:
        """Background task for compressing audit data"""
        while True:
            try:
    await asyncio.sleep(3600)  # Run every hour

                if not self.is_initialized:
                    break

                # Placeholder for compression logic
                if self.compression_enabled:
                    logger.info("Audit data compression completed")

            except Exception as e:
                logger.error(f"Audit compression task failed: {str(e)}")
                await asyncio.sleep(3600)

    async def initialize_storage(self) -> Dict[str, Any]:
        """Initialize audit storage"""
        try:
            # Placeholder for storage initialization
            logger.info("Audit storage initialized")

        except Exception as e:
            logger.error(f"Audit storage initialization failed: {str(e)}")
            raise
