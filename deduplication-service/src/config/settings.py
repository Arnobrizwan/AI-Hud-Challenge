"""Configuration settings for the deduplication service."""

from typing import List

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""

    # Application
    app_name: str = "Deduplication Service"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")

    # Redis
    redis_url: str = Field(..., env="REDIS_URL")
    redis_max_connections: int = Field(default=100, env="REDIS_MAX_CONNECTIONS")
    redis_socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")

    # Similarity thresholds
    similarity_threshold: float = Field(default=0.85, env="SIMILARITY_THRESHOLD")
    lsh_threshold: float = Field(default=0.7, env="LSH_THRESHOLD")
    content_similarity_threshold: float = Field(default=0.8, env="CONTENT_SIMILARITY_THRESHOLD")
    title_similarity_threshold: float = Field(default=0.9, env="TITLE_SIMILARITY_THRESHOLD")

    # Clustering parameters
    clustering_eps: float = Field(default=0.3, env="CLUSTERING_EPS")
    clustering_min_samples: int = Field(default=2, env="CLUSTERING_MIN_SAMPLES")
    max_cluster_size: int = Field(default=100, env="MAX_CLUSTER_SIZE")

    # LSH parameters
    lsh_num_perm: int = Field(default=128, env="LSH_NUM_PERM")
    lsh_num_bands: int = Field(default=16, env="LSH_NUM_BANDS")
    lsh_band_size: int = Field(default=8, env="LSH_BAND_SIZE")

    # Embedding model
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")

    # Processing
    batch_size: int = Field(default=100, env="BATCH_SIZE")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    processing_timeout: int = Field(default=300, env="PROCESSING_TIMEOUT")

    # Temporal decay
    temporal_decay_half_life_hours: int = Field(default=24, env="TEMPORAL_DECAY_HALF_LIFE_HOURS")

    # Quality scoring weights
    title_weight: float = Field(default=0.4, env="TITLE_WEIGHT")
    content_weight: float = Field(default=0.4, env="CONTENT_WEIGHT")
    entity_weight: float = Field(default=0.2, env="ENTITY_WEIGHT")

    # Representative selection weights
    centrality_weight: float = Field(default=0.3, env="CENTRALITY_WEIGHT")
    quality_weight: float = Field(default=0.25, env="QUALITY_WEIGHT")
    freshness_weight: float = Field(default=0.2, env="FRESHNESS_WEIGHT")
    source_authority_weight: float = Field(default=0.15, env="SOURCE_AUTHORITY_WEIGHT")
    completeness_weight: float = Field(default=0.1, env="COMPLETENESS_WEIGHT")

    # Monitoring
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")

    # Rate limiting
    rate_limit_requests: int = Field(default=1000, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")

    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # CORS
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    allowed_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"], env="ALLOWED_METHODS")
    allowed_headers: List[str] = Field(default=["*"], env="ALLOWED_HEADERS")

    # Cache settings
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    cache_max_size: int = Field(default=10000, env="CACHE_MAX_SIZE")

    # Bloom filter settings
    bloom_filter_capacity: int = Field(default=1000000, env="BLOOM_FILTER_CAPACITY")
    bloom_filter_error_rate: float = Field(default=0.01, env="BLOOM_FILTER_ERROR_RATE")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
