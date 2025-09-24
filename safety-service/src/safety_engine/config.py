"""
Safety Engine Configuration
Configuration management for the safety monitoring system
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class SafetySettings(BaseSettings):
    """Safety service configuration settings"""

    # Service Configuration
    service_name: str = "safety-service"
    version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Database Configuration
    database_url: str = Field(
        env="DATABASE_URL", default="postgresql://user:password@localhost:5432/safety_db"
    )
    redis_url: str = Field(env="REDIS_URL", default="redis://localhost:6379/0")

    # Safety Thresholds
    drift_severity_threshold: float = Field(default=0.6, env="DRIFT_SEVERITY_THRESHOLD")
    abuse_score_threshold: float = Field(default=0.7, env="ABUSE_SCORE_THRESHOLD")
    content_safety_threshold: float = Field(default=0.8, env="CONTENT_SAFETY_THRESHOLD")
    anomaly_score_threshold: float = Field(default=0.7, env="ANOMALY_SCORE_THRESHOLD")
    overall_safety_threshold: float = Field(default=0.8, env="OVERALL_SAFETY_THRESHOLD")

    # Rate Limiting Configuration
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    default_rate_limit: int = Field(default=100, env="DEFAULT_RATE_LIMIT")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # seconds
    burst_limit: int = Field(default=10, env="BURST_LIMIT")

    # Drift Detection Configuration
    drift_detection_enabled: bool = Field(default=True, env="DRIFT_DETECTION_ENABLED")
    drift_check_interval: int = Field(default=3600, env="DRIFT_CHECK_INTERVAL")  # seconds
    min_samples_for_drift: int = Field(default=100, env="MIN_SAMPLES_FOR_DRIFT")
    drift_confidence_threshold: float = Field(default=0.95, env="DRIFT_CONFIDENCE_THRESHOLD")

    # Abuse Detection Configuration
    abuse_detection_enabled: bool = Field(default=True, env="ABUSE_DETECTION_ENABLED")
    behavioral_analysis_window: int = Field(
        default=86400, env="BEHAVIORAL_ANALYSIS_WINDOW"
    )  # seconds
    graph_analysis_depth: int = Field(default=3, env="GRAPH_ANALYSIS_DEPTH")
    reputation_decay_rate: float = Field(default=0.1, env="REPUTATION_DECAY_RATE")

    # Content Moderation Configuration
    content_moderation_enabled: bool = Field(default=True, env="CONTENT_MODERATION_ENABLED")
    external_moderation_apis: bool = Field(default=False, env="EXTERNAL_MODERATION_APIS")
    moderation_confidence_threshold: float = Field(
        default=0.8, env="MODERATION_CONFIDENCE_THRESHOLD"
    )
    auto_moderation_enabled: bool = Field(default=True, env="AUTO_MODERATION_ENABLED")

    # Anomaly Detection Configuration
    anomaly_detection_enabled: bool = Field(default=True, env="ANOMALY_DETECTION_ENABLED")
    anomaly_detection_window: int = Field(default=3600, env="ANOMALY_DETECTION_WINDOW")  # seconds
    anomaly_sensitivity: float = Field(default=0.5, env="ANOMALY_SENSITIVITY")

    # Compliance Configuration
    gdpr_compliance_enabled: bool = Field(default=True, env="GDPR_COMPLIANCE_ENABLED")
    content_policy_enabled: bool = Field(default=True, env="CONTENT_POLICY_ENABLED")
    privacy_monitoring_enabled: bool = Field(default=True, env="PRIVACY_MONITORING_ENABLED")
    audit_log_retention_days: int = Field(default=365, env="AUDIT_LOG_RETENTION_DAYS")

    # Incident Response Configuration
    incident_response_enabled: bool = Field(default=True, env="INCIDENT_RESPONSE_ENABLED")
    auto_escalation_enabled: bool = Field(default=True, env="AUTO_ESCALATION_ENABLED")
    incident_cleanup_interval: int = Field(default=3600, env="INCIDENT_CLEANUP_INTERVAL")  # seconds

    # Monitoring Configuration
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    health_check_interval: int = Field(default=60, env="HEALTH_CHECK_INTERVAL")  # seconds

    # ML Model Configuration
    model_update_interval: int = Field(default=86400, env="MODEL_UPDATE_INTERVAL")  # seconds
    model_confidence_threshold: float = Field(default=0.8, env="MODEL_CONFIDENCE_THRESHOLD")
    enable_model_retraining: bool = Field(default=True, env="ENABLE_MODEL_RETRAINING")

    # Performance Configuration
    max_concurrent_checks: int = Field(default=100, env="MAX_CONCURRENT_CHECKS")
    check_timeout: int = Field(default=30, env="CHECK_TIMEOUT")  # seconds
    cache_ttl: int = Field(default=300, env="CACHE_TTL")  # seconds

    # External API Configuration
    external_api_timeout: int = Field(default=10, env="EXTERNAL_API_TIMEOUT")  # seconds
    external_api_retries: int = Field(default=3, env="EXTERNAL_API_RETRIES")
    external_api_rate_limit: int = Field(default=1000, env="EXTERNAL_API_RATE_LIMIT")

    # Security Configuration
    api_key_required: bool = Field(default=False, env="API_KEY_REQUIRED")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")

    # Feature Flags
    features: Dict[str, bool] = Field(
        default={
            "drift_detection": True,
            "abuse_detection": True,
            "content_moderation": True,
            "anomaly_detection": True,
            "rate_limiting": True,
            "compliance_monitoring": True,
            "incident_response": True,
            "audit_logging": True,
        }
    )

    # Configuration for pydantic-settings
    model_config = {"env_file": ".env", "case_sensitive": False, "protected_namespaces": ()}


class DriftDetectionConfig:
    """Drift detection specific configuration"""

    def __init__(self, settings: SafetySettings):
        self.enabled = settings.drift_detection_enabled
        self.check_interval = settings.drift_check_interval
        self.min_samples = settings.min_samples_for_drift
        self.confidence_threshold = settings.drift_confidence_threshold
        self.severity_threshold = settings.drift_severity_threshold

        # Statistical test configurations
        self.ks_test_alpha = 0.05
        self.chi_square_alpha = 0.05
        self.psi_threshold = 0.2
        self.wasserstein_threshold = 0.1

        # Feature importance monitoring
        self.importance_drift_threshold = 0.1
        self.importance_change_threshold = 0.2


class AbuseDetectionConfig:
    """Abuse detection specific configuration"""

    def __init__(self, settings: SafetySettings):
        self.enabled = settings.abuse_detection_enabled
        self.analysis_window = settings.behavioral_analysis_window
        self.graph_depth = settings.graph_analysis_depth
        self.reputation_decay = settings.reputation_decay_rate
        self.score_threshold = settings.abuse_score_threshold

        # Behavioral analysis thresholds
        self.velocity_threshold = 0.8
        self.pattern_threshold = 0.7
        self.frequency_threshold = 0.6
        self.time_threshold = 0.5

        # Graph analysis thresholds
        self.suspicious_connection_threshold = 5
        self.cluster_anomaly_threshold = 0.8
        self.centrality_anomaly_threshold = 0.7


class ContentModerationConfig:
    """Content moderation specific configuration"""

    def __init__(self, settings: SafetySettings):
        self.enabled = settings.content_moderation_enabled
        self.external_apis = settings.external_moderation_apis
        self.confidence_threshold = settings.moderation_confidence_threshold
        self.auto_moderation = settings.auto_moderation_enabled
        self.safety_threshold = settings.content_safety_threshold

        # Moderation thresholds
        self.toxicity_threshold = 0.7
        self.hate_speech_threshold = 0.8
        self.spam_threshold = 0.6
        self.misinformation_threshold = 0.7
        self.adult_content_threshold = 0.8
        self.violence_threshold = 0.7


class AnomalyDetectionConfig:
    """Anomaly detection specific configuration"""

    def __init__(self, settings: SafetySettings):
        self.enabled = settings.anomaly_detection_enabled
        self.detection_window = settings.anomaly_detection_window
        self.sensitivity = settings.anomaly_sensitivity
        self.score_threshold = settings.anomaly_score_threshold

        # Anomaly detection parameters
        self.isolation_forest_contamination = 0.1
        self.one_class_svm_nu = 0.1
        self.lstm_sequence_length = 50
        self.autoencoder_threshold = 0.1


class RateLimitingConfig:
    """Rate limiting specific configuration"""

    def __init__(self, settings: SafetySettings):
        self.enabled = settings.rate_limit_enabled
        self.default_limit = settings.default_rate_limit
        self.window_size = settings.rate_limit_window
        self.burst_limit = settings.burst_limit

        # Rate limiting strategies
        self.sliding_window_enabled = True
        self.token_bucket_enabled = True
        self.adaptive_limiting_enabled = True
        self.geolocation_limiting_enabled = True

        # Dynamic rate limiting
        self.reputation_multiplier_min = 0.1
        self.reputation_multiplier_max = 2.0
        self.abuse_penalty_min = 0.1
        self.abuse_penalty_max = 1.0


class ComplianceConfig:
    """Compliance monitoring specific configuration"""

    def __init__(self, settings: SafetySettings):
        self.gdpr_enabled = settings.gdpr_compliance_enabled
        self.content_policy_enabled = settings.content_policy_enabled
        self.privacy_enabled = settings.privacy_monitoring_enabled
        self.audit_retention_days = settings.audit_log_retention_days

        # GDPR specific settings
        self.data_retention_period = 2555  # days (7 years)
        self.consent_required = True
        self.right_to_be_forgotten = True
        self.data_portability = True

        # Content policy settings
        self.prohibited_content_types = [
            "hate_speech",
            "harassment",
            "violence",
            "adult_content",
            "spam",
            "misinformation",
            "copyright_violation",
        ]

        # Privacy settings
        self.pii_detection_enabled = True
        self.encryption_required = True
        self.access_logging_enabled = True


@lru_cache()
def get_settings() -> SafetySettings:
    """Get safety settings (cached)"""
    return SafetySettings()


@lru_cache()
def get_drift_config() -> DriftDetectionConfig:
    """Get drift detection configuration (cached)"""
    return DriftDetectionConfig(get_settings())


@lru_cache()
def get_abuse_config() -> AbuseDetectionConfig:
    """Get abuse detection configuration (cached)"""
    return AbuseDetectionConfig(get_settings())


@lru_cache()
def get_content_config() -> ContentModerationConfig:
    """Get content moderation configuration (cached)"""
    return ContentModerationConfig(get_settings())


@lru_cache()
def get_anomaly_config() -> AnomalyDetectionConfig:
    """Get anomaly detection configuration (cached)"""
    return AnomalyDetectionConfig(get_settings())


@lru_cache()
def get_rate_limit_config() -> RateLimitingConfig:
    """Get rate limiting configuration (cached)"""
    return RateLimitingConfig(get_settings())


@lru_cache()
def get_compliance_config() -> ComplianceConfig:
    """Get compliance configuration (cached)"""
    return ComplianceConfig(get_settings())
