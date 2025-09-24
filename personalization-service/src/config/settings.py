"""Configuration settings for the personalization service."""


from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""

    # Database settings
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/personalization", env="DATABASE_URL"
    )

    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # Application settings
    app_name: str = "Personalization Service"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # Model settings
    collaborative_factors: int = Field(default=100, env="COLLABORATIVE_FACTORS")
    collaborative_regularization: float = Field(default=0.01, env="COLLABORATIVE_REGULARIZATION")
    collaborative_iterations: int = Field(default=50, env="COLLABORATIVE_ITERATIONS")

    # Content-based filtering settings
    content_embedding_dim: int = Field(default=384, env="CONTENT_EMBEDDING_DIM")
    tfidf_max_features: int = Field(default=10000, env="TFIDF_MAX_FEATURES")

    # Bandit settings
    bandit_alpha: float = Field(default=1.0, env="BANDIT_ALPHA")
    bandit_beta: float = Field(default=1.0, env="BANDIT_BETA")
    epsilon: float = Field(default=0.1, env="EPSILON")

    # Diversity settings
    diversity_threshold: float = Field(default=0.3, env="DIVERSITY_THRESHOLD")
    serendipity_weight: float = Field(default=0.2, env="SERENDIPITY_WEIGHT")

    # Privacy settings
    privacy_epsilon: float = Field(default=1.0, env="PRIVACY_EPSILON")
    privacy_delta: float = Field(default=1e-5, env="PRIVACY_DELTA")

    # Cache settings
    profile_cache_ttl: int = Field(default=3600, env="PROFILE_CACHE_TTL")
    model_cache_ttl: int = Field(default=86400, env="MODEL_CACHE_TTL")

    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")

    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
