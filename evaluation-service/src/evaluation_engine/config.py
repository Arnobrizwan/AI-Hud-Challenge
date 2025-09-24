"""
Configuration management for Evaluation Suite Microservice
"""

from typing import List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings"""

    # Application settings
    APP_NAME: str = "Evaluation Suite Microservice"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    # Security settings
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"], env="ALLOWED_ORIGINS"
    )
    ALLOWED_HOSTS: List[str] = Field(default=["localhost", "127.0.0.1"], env="ALLOWED_HOSTS")

    # Database settings
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(default=10, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")

    # Redis cache settings
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    REDIS_POOL_SIZE: int = Field(default=10, env="REDIS_POOL_SIZE")

    # BigQuery settings
    BIGQUERY_PROJECT_ID: str = Field(..., env="BIGQUERY_PROJECT_ID")
    BIGQUERY_DATASET_ID: str = Field(..., env="BIGQUERY_DATASET_ID")
    BIGQUERY_CREDENTIALS_PATH: Optional[str] = Field(default=None, env="BIGQUERY_CREDENTIALS_PATH")

    # MLflow settings
    MLFLOW_TRACKING_URI: str = Field(..., env="MLFLOW_TRACKING_URI")
    MLFLOW_REGISTRY_URI: str = Field(..., env="MLFLOW_REGISTRY_URI")
    MLFLOW_EXPERIMENT_NAME: str = Field(default="evaluation-suite", env="MLFLOW_EXPERIMENT_NAME")

    # Vertex AI settings
    VERTEX_AI_PROJECT_ID: str = Field(..., env="VERTEX_AI_PROJECT_ID")
    VERTEX_AI_LOCATION: str = Field(default="us-central1", env="VERTEX_AI_LOCATION")
    VERTEX_AI_CREDENTIALS_PATH: Optional[str] = Field(default=None, env="VERTEX_AI_CREDENTIALS_PATH")

    # Monitoring settings
    PROMETHEUS_ENDPOINT: str = Field(default="http://localhost:9090", env="PROMETHEUS_ENDPOINT")
    GRAFANA_ENDPOINT: str = Field(default="http://localhost:3000", env="GRAFANA_ENDPOINT")

    # Alerting settings
    ALERT_WEBHOOK_URL: Optional[str] = Field(default=None, env="ALERT_WEBHOOK_URL")
    SLACK_WEBHOOK_URL: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    EMAIL_SMTP_HOST: Optional[str] = Field(default=None, env="EMAIL_SMTP_HOST")
    EMAIL_SMTP_PORT: int = Field(default=587, env="EMAIL_SMTP_PORT")
    EMAIL_SMTP_USER: Optional[str] = Field(default=None, env="EMAIL_SMTP_USER")
    EMAIL_SMTP_PASSWORD: Optional[str] = Field(default=None, env="EMAIL_SMTP_PASSWORD")

    # Evaluation settings
    MAX_CONCURRENT_EVALUATIONS: int = Field(default=10, env="MAX_CONCURRENT_EVALUATIONS")
    EVALUATION_TIMEOUT_SECONDS: int = Field(default=3600, env="EVALUATION_TIMEOUT_SECONDS")
    CACHE_TTL_SECONDS: int = Field(default=3600, env="CACHE_TTL_SECONDS")

    # Statistical testing settings
    DEFAULT_ALPHA: float = Field(default=0.05, env="DEFAULT_ALPHA")
    DEFAULT_POWER: float = Field(default=0.8, env="DEFAULT_POWER")
    BOOTSTRAP_SAMPLES: int = Field(default=1000, env="BOOTSTRAP_SAMPLES")
    CONFIDENCE_LEVEL: float = Field(default=0.95, env="CONFIDENCE_LEVEL")

    # Drift detection settings
    DRIFT_SIGNIFICANCE_LEVEL: float = Field(default=0.05, env="DRIFT_SIGNIFICANCE_LEVEL")
    DRIFT_ALERT_THRESHOLD: float = Field(default=0.7, env="DRIFT_ALERT_THRESHOLD")
    DRIFT_WINDOW_SIZE: int = Field(default=1000, env="DRIFT_WINDOW_SIZE")

    # A/B testing settings
    MIN_EXPERIMENT_DURATION_DAYS: int = Field(default=7, env="MIN_EXPERIMENT_DURATION_DAYS")
    MAX_EXPERIMENT_DURATION_DAYS: int = Field(default=30, env="MAX_EXPERIMENT_DURATION_DAYS")
    MIN_SAMPLE_SIZE_PER_VARIANT: int = Field(default=1000, env="MIN_SAMPLE_SIZE_PER_VARIANT")

    # Performance settings
    MAX_BATCH_SIZE: int = Field(default=10000, env="MAX_BATCH_SIZE")
    ENABLE_PARALLEL_PROCESSING: bool = Field(default=True, env="ENABLE_PARALLEL_PROCESSING")
    MAX_WORKERS: int = Field(default=4, env="MAX_WORKERS")

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_database_url() -> str:
    """Get database URL from settings"""
    return get_settings().DATABASE_URL


def get_redis_url() -> str:
    """Get Redis URL from settings"""
    return get_settings().REDIS_URL


def get_bigquery_config() -> dict:
    """Get BigQuery configuration"""
    settings = get_settings()
    return {
        "project_id": settings.BIGQUERY_PROJECT_ID,
        "dataset_id": settings.BIGQUERY_DATASET_ID,
        "credentials_path": settings.BIGQUERY_CREDENTIALS_PATH,
    }


def get_mlflow_config() -> dict:
    """Get MLflow configuration"""
    settings = get_settings()
    return {
        "tracking_uri": settings.MLFLOW_TRACKING_URI,
        "registry_uri": settings.MLFLOW_REGISTRY_URI,
        "experiment_name": settings.MLFLOW_EXPERIMENT_NAME,
    }


def get_vertex_ai_config() -> dict:
    """Get Vertex AI configuration"""
    settings = get_settings()
    return {
        "project_id": settings.VERTEX_AI_PROJECT_ID,
        "location": settings.VERTEX_AI_LOCATION,
        "credentials_path": settings.VERTEX_AI_CREDENTIALS_PATH,
    }
