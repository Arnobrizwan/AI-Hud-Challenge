"""
Monitoring and metrics for Safety Service
"""

import logging
from typing import Dict, Any

from prometheus_client import Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

# Metrics
safety_requests_total = Counter(
    'safety_requests_total',
    'Total number of safety requests',
    ['service', 'endpoint', 'status']
)

safety_response_time = Histogram(
    'safety_response_time_seconds',
    'Safety service response time',
    ['service', 'endpoint']
)

safety_active_incidents = Gauge(
    'safety_active_incidents',
    'Number of active safety incidents',
    ['severity']
)

safety_drift_detected = Counter(
    'safety_drift_detected_total',
    'Total number of drift detections',
    ['type', 'severity']
)

safety_abuse_detected = Counter(
    'safety_abuse_detected_total',
    'Total number of abuse detections',
    ['type', 'severity']
)

safety_content_moderated = Counter(
    'safety_content_moderated_total',
    'Total number of content moderation actions',
    ['action', 'reason']
)

safety_rate_limits_hit = Counter(
    'safety_rate_limits_hit_total',
    'Total number of rate limit hits',
    ['endpoint', 'limit_type']
)

safety_compliance_violations = Counter(
    'safety_compliance_violations_total',
    'Total number of compliance violations',
    ['regulation', 'severity']
)


def setup_monitoring():
    """Setup monitoring configuration"""
    logger.info("Setting up safety monitoring")
    return True


def get_metrics() -> str:
    """Get Prometheus metrics"""
    return generate_latest().decode('utf-8')


def record_request(service: str, endpoint: str, status: str, duration: float):
    """Record request metrics"""
    safety_requests_total.labels(
        service=service,
        endpoint=endpoint,
        status=status
    ).inc()
    
    safety_response_time.labels(
        service=service,
        endpoint=endpoint
    ).observe(duration)


def record_incident(severity: str, count: int = 1):
    """Record incident metrics"""
    safety_active_incidents.labels(severity=severity).set(count)


def record_drift(drift_type: str, severity: str):
    """Record drift detection"""
    safety_drift_detected.labels(
        type=drift_type,
        severity=severity
    ).inc()


def record_abuse(abuse_type: str, severity: str):
    """Record abuse detection"""
    safety_abuse_detected.labels(
        type=abuse_type,
        severity=severity
    ).inc()


def record_content_moderation(action: str, reason: str):
    """Record content moderation action"""
    safety_content_moderated.labels(
        action=action,
        reason=reason
    ).inc()


def record_rate_limit(endpoint: str, limit_type: str):
    """Record rate limit hit"""
    safety_rate_limits_hit.labels(
        endpoint=endpoint,
        limit_type=limit_type
    ).inc()


def record_compliance_violation(regulation: str, severity: str):
    """Record compliance violation"""
    safety_compliance_violations.labels(
        regulation=regulation,
        severity=severity
    ).inc()
