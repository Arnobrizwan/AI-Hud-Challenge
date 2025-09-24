"""Configuration settings for Content Enrichment Service."""

from typing import List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""

    # Application
    app_name: str = "Content Enrichment Service"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")

    # API Configuration
    api_prefix: str = "/api/v1"
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")

    # Database Configuration
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")

    # Redis Configuration
    redis_url: str = Field(..., env="REDIS_URL")
    redis_max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    redis_ttl: int = Field(default=3600, env="REDIS_TTL")

    # Google Cloud Configuration
    google_cloud_project: str = Field(..., env="GOOGLE_CLOUD_PROJECT")
    vertex_ai_location: str = Field(default="us-central1", env="VERTEX_AI_LOCATION")
    service_account_path: Optional[str] = Field(default=None, env="SERVICE_ACCOUNT_PATH")

    # MLflow Configuration
    mlflow_tracking_uri: str = Field(..., env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="content-enrichment", env="MLFLOW_EXPERIMENT_NAME")

    # Model Configuration
    model_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    max_text_length: int = Field(default=10000, env="MAX_TEXT_LENGTH")
    batch_size: int = Field(default=32, env="BATCH_SIZE")

    # Performance Configuration
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")
    enrichment_timeout: int = Field(default=30, env="ENRICHMENT_TIMEOUT")

    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")

    # A/B Testing Configuration
    enable_ab_testing: bool = Field(default=True, env="ENABLE_AB_TESTING")
    ab_test_traffic_split: float = Field(default=0.5, env="AB_TEST_TRAFFIC_SPLIT")

    # Content Processing Configuration
    supported_languages: List[str] = Field(
        default=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
        env="SUPPORTED_LANGUAGES",
    )

    # Entity Recognition Configuration
    entity_confidence_threshold: float = Field(default=0.3, env="ENTITY_CONFIDENCE_THRESHOLD")
    max_entities_per_document: int = Field(default=100, env="MAX_ENTITIES_PER_DOCUMENT")

    # Topic Classification Configuration
    topic_confidence_threshold: float = Field(default=0.3, env="TOPIC_CONFIDENCE_THRESHOLD")
    max_topics_per_document: int = Field(default=10, env="MAX_TOPICS_PER_DOCUMENT")

    # Sentiment Analysis Configuration
    sentiment_confidence_threshold: float = Field(default=0.5, env="SENTIMENT_CONFIDENCE_THRESHOLD")

    # Content Quality Configuration
    readability_threshold: float = Field(default=0.6, env="READABILITY_THRESHOLD")
    trust_score_threshold: float = Field(default=0.5, env="TRUST_SCORE_THRESHOLD")

    # Rate Limiting Configuration
    rate_limit_requests: int = Field(default=1000, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")

    # Security Configuration
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
