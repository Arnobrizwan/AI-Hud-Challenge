"""
Application settings and configuration
"""

from typing import List, Optional

from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings"""

    # Application
    app_name: str = "Feedback Service"
    debug: bool = False
    version: str = "1.0.0"

    # Database
    database_url: str = "postgresql://feedback_user:feedback_pass@localhost:5432/feedback_db"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_max_connections: int = 10

    # Elasticsearch
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index_prefix: str = "feedback"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_feedback_topic: str = "feedback-events"
    kafka_analytics_topic: str = "analytics-events"

    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    # WebSocket
    websocket_max_connections: int = 1000
    websocket_heartbeat_interval: int = 30

    # ML Models
    model_cache_size: int = 100
    model_update_interval: int = 3600  # seconds

    # Quality thresholds
    quality_threshold_low: float = 0.3
    quality_threshold_medium: float = 0.6
    quality_threshold_high: float = 0.8

    # Processing limits
    max_feedback_per_second: int = 10000
    max_annotation_tasks_per_user: int = 50
    max_campaign_tasks: int = 10000

    # Monitoring
    prometheus_port: int = 9090
    metrics_interval: int = 60  # seconds

    # File uploads
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = ["image/jpeg", "image/png", "text/plain", "application/pdf"]

    # External services
    external_feedback_webhook_url: Optional[str] = None
    external_ml_service_url: Optional[str] = None

    @validator("cors_origins", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings"""
    return Settings()
