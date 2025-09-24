"""
Configuration settings for observability service
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Service configuration
    service_name: str = "observability-service"
    version: str = "1.0.0"
    debug: bool = False
    port: int = 8000
    host: str = "0.0.0.0"

    # Database configuration
    database_url: str = "postgresql://user:password@localhost:5432/observability"
    redis_url: str = "redis://localhost:6379/0"

    # Elasticsearch configuration
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index_prefix: str = "observability"

    # Prometheus configuration
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"

    # Jaeger configuration
    jaeger_host: str = "localhost"
    jaeger_port: int = 14268

    # Zipkin configuration
    zipkin_endpoint: str = "http://localhost:9411/api/v2/spans"

    # OTLP configuration
    otlp_endpoint: str = "http://localhost:4317"

    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "json"

    # Security configuration
    secret_key: str = "your-secret-key-here"
    jwt_secret: str = "your-jwt-secret-here"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # Monitoring configuration
    health_check_interval: int = 30  # seconds
    metrics_collection_interval: int = 30  # seconds
    alert_processing_interval: int = 10  # seconds
    slo_monitoring_interval: int = 300  # seconds
    cost_monitoring_interval: int = 3600  # seconds
    chaos_experiment_interval: int = 1800  # seconds

    # Service endpoints
    service_endpoints: Dict[str, str] = field(
        default_factory=lambda: {
            "ingestion-service": "http://ingestion-service:8000",
            "content-extraction-service": "http://content-extraction-service:8000",
            "content-enrichment-service": "http://content-enrichment-service:8000",
            "deduplication-service": "http://deduplication-service:8000",
            "personalization-service": "http://personalization-service:8000",
            "summarization-service": "http://summarization-service:8000",
            "notification-service": "http://notification-service:8000",
            "feedback-service": "http://feedback-service:8000",
            "evaluation-service": "http://evaluation-service:8000",
            "safety-service": "http://safety-service:8000",
        }
    )

    # Notification channels
    notification_channels: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "slack": {
                "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                "channel": "#alerts",
                "enabled": True,
            },
            "email": {
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "your-email@gmail.com",
                "password": "your-app-password",
                "enabled": False,
            },
            "pagerduty": {"integration_key": "your-pagerduty-key", "enabled": False},
        }
    )

    # Alerting configuration
    alerting_rules_path: str = "config/alert_rules.json"
    escalation_policies_path: str = "config/escalation_policies.json"

    # Runbook configuration
    runbooks_path: str = "config/runbooks.json"
    approval_policies_path: str = "config/approval_policies.json"

    # SLO configuration
    slo_definitions_path: str = "config/slo_definitions.json"

    # Cost monitoring configuration
    cost_data_sources: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "aws": {
                "access_key_id": "your-aws-access-key",
                "secret_access_key": "your-aws-secret-key",
                "region": "us-east-1",
                "enabled": False,
            },
            "gcp": {
                "project_id": "your-gcp-project",
                "service_account_key": "path/to/service-account.json",
                "enabled": False,
            },
            "azure": {
                "subscription_id": "your-azure-subscription",
                "client_id": "your-azure-client-id",
                "client_secret": "your-azure-client-secret",
                "tenant_id": "your-azure-tenant-id",
                "enabled": False,
            },
        }
    )

    # Chaos engineering configuration
    chaos_experiments_path: str = "config/chaos_experiments.json"
    chaos_enabled: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False


@dataclass
class MetricsConfig:
    """Metrics collection configuration"""

    prometheus_port: int = 9090
    collection_interval: int = 30
    custom_metrics_enabled: bool = True
    business_metrics_enabled: bool = True
    system_metrics_enabled: bool = True


@dataclass
class TracingConfig:
    """Distributed tracing configuration"""

    jaeger_enabled: bool = True
    jaeger_host: str = "localhost"
    jaeger_port: int = 14268
    zipkin_enabled: bool = False
    zipkin_endpoint: str = "http://localhost:9411/api/v2/spans"
    otlp_enabled: bool = False
    otlp_endpoint: str = "http://localhost:4317"
    sampling_rate: float = 0.1
    service_name: str = "observability-service"


@dataclass
class LoggingConfig:
    """Logging configuration"""

    elasticsearch_enabled: bool = True
    elasticsearch_host: str = "localhost"
    elasticsearch_port: int = 9200
    elasticsearch_index_prefix: str = "observability-logs"
    redis_enabled: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    log_level: str = "INFO"
    structured_logging: bool = True
    log_rotation: bool = True
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class AlertingConfig:
    """Alerting configuration"""

    rules_path: str = "config/alert_rules.json"
    channels: List[Dict[str, Any]] = field(default_factory=list)
    escalation_policies: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RunbookConfig:
    """Runbook configuration"""

    runbooks_path: str = "config/runbooks.json"
    execution_config: Dict[str, Any] = field(default_factory=dict)
    approval_policies: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SLOConfig:
    """SLO monitoring configuration"""

    slo_definitions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class IncidentConfig:
    """Incident management configuration"""

    templates: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CostConfig:
    """Cost monitoring configuration"""

    data_sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ChaosConfig:
    """Chaos engineering configuration"""

    experiments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ObservabilityConfig:
    """Complete observability configuration"""

    metrics_config: MetricsConfig
    tracing_config: TracingConfig
    logging_config: LoggingConfig
    alerting_config: AlertingConfig
    runbook_config: RunbookConfig
    slo_config: SLOConfig
    incident_config: IncidentConfig
    cost_config: CostConfig
    chaos_config: ChaosConfig


@dataclass
class ObservabilityStatus:
    """Observability system status"""

    initialized_components: int
    failed_components: int
    status: str
    initialization_timestamp: str


def get_settings() -> Settings:
    """Get application settings"""
    return Settings()


def create_observability_config(settings: Settings) -> ObservabilityConfig:
    """Create observability configuration from settings"""

    return ObservabilityConfig(
        metrics_config=MetricsConfig(
            prometheus_port=settings.prometheus_port,
            collection_interval=settings.metrics_collection_interval,
            custom_metrics_enabled=True,
            business_metrics_enabled=True,
            system_metrics_enabled=True,
        ),
        tracing_config=TracingConfig(
            jaeger_enabled=True,
            jaeger_host=settings.jaeger_host,
            jaeger_port=settings.jaeger_port,
            zipkin_enabled=False,
            zipkin_endpoint=settings.zipkin_endpoint,
            otlp_enabled=False,
            otlp_endpoint=settings.otlp_endpoint,
            sampling_rate=0.1,
            service_name=settings.service_name,
        ),
        logging_config=LoggingConfig(
            elasticsearch_enabled=True,
            elasticsearch_host="localhost",
            elasticsearch_port=9200,
            elasticsearch_index_prefix=settings.elasticsearch_index_prefix,
            redis_enabled=True,
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            log_level=settings.log_level,
            structured_logging=True,
            log_rotation=True,
            max_log_size=10 * 1024 * 1024,
            backup_count=5,
        ),
        alerting_config=AlertingConfig(
            rules_path=settings.alerting_rules_path,
            channels=list(settings.notification_channels.values()),
            escalation_policies=[],
        ),
        runbook_config=RunbookConfig(
            runbooks_path=settings.runbooks_path, execution_config={}, approval_policies=[]
        ),
        slo_config=SLOConfig(slo_definitions=[]),
        incident_config=IncidentConfig(templates=[]),
        cost_config=CostConfig(data_sources=settings.cost_data_sources),
        chaos_config=ChaosConfig(experiments=[]),
    )
