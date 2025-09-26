"""
Configuration management for the Foundations & Guards microservice.
Handles environment-based settings with validation and type safety.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    # Application Settings
    APP_NAME: str = "Foundations & Guards Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    ENVIRONMENT: str = Field(default="development", description="Application environment")

    # Server Configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    WORKERS: int = Field(default=1, description="Number of worker processes")

    # Security Settings
    SECRET_KEY: str = Field(
        default="dev-secret-key-change-in-production", description="Secret key for JWT signing"
    )
    JWT_ALGORITHM: str = Field(default="RS256", description="JWT algorithm")
    JWT_EXPIRE_MINUTES: int = Field(default=30, description="JWT expiration in minutes")

    # Firebase Configuration
    FIREBASE_PROJECT_ID: str = Field(default="dev-project-id", description="Firebase project ID")
    FIREBASE_CREDENTIALS_PATH: Optional[str] = Field(
        default=None, description="Path to Firebase service account JSON"
    )
    FIREBASE_CREDENTIALS_JSON: Optional[str] = Field(
        default=None, description="Firebase service account JSON as string"
    )

    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    REDIS_MAX_CONNECTIONS: int = Field(default=20, description="Max Redis connections")

    # Rate Limiting Configuration
    RATE_LIMIT_PER_USER: int = Field(default=1000, description="Requests per minute per user")
    RATE_LIMIT_PER_IP: int = Field(default=10000, description="Requests per minute per IP")
    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60, description="Rate limit window in seconds")

    # Request Configuration
    MAX_REQUEST_SIZE: int = Field(default=10485760, description="Max request size in bytes (10MB)")
    REQUEST_TIMEOUT: int = Field(default=30, description="Request timeout in seconds")

    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins",
    )
    CORS_CREDENTIALS: bool = Field(default=True, description="Allow CORS credentials")
    CORS_METHODS: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"], description="Allowed CORS methods"
    )
    CORS_HEADERS: List[str] = Field(default=["*"], description="Allowed CORS headers")

    # Security Headers
    ENABLE_SECURITY_HEADERS: bool = Field(default=True, description="Enable security headers")
    CSP_POLICY: str = Field(
        default=(
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'"
        ),
        description="Content Security Policy",
    )

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format (json/text)")
    LOG_FILE: Optional[str] = Field(default=None, description="Log file path")

    # Metrics Configuration
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    METRICS_PATH: str = Field(default="/metrics", description="Metrics endpoint path")

    # Health Check Configuration
    HEALTH_CHECK_PATH: str = Field(default="/health", description="Health check endpoint")
    HEALTH_CHECK_DEPENDENCIES: bool = Field(
        default=True, description="Include dependency checks in health endpoint"
    )

    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(
        default=5, description="Circuit breaker failure threshold"
    )
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(
        default=60, description="Circuit breaker recovery timeout in seconds"
    )
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION: tuple = Field(
        default=(Exception,), description="Expected exceptions for circuit breaker"
    )

    # GCP Configuration
    GCP_PROJECT_ID: Optional[str] = Field(default=None, description="GCP Project ID")
    GCP_REGION: str = Field(default="us-central1", description="GCP Region")
    CLOUD_RUN_SERVICE_NAME: str = Field(
        default="foundations-guards-service", description="Cloud Run service name"
    )

    @validator("ENVIRONMENT")
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        allowed_environments = ["development", "staging", "production"]
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level setting."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()

    @validator("LOG_FORMAT")
    def validate_log_format(cls, v: str) -> str:
        """Validate log format setting."""
        allowed_formats = ["json", "text"]
        if v not in allowed_formats:
            raise ValueError(f"Log format must be one of: {allowed_formats}")
        return v

    @validator("JWT_ALGORITHM")
    def validate_jwt_algorithm(cls, v: str) -> str:
        """Validate JWT algorithm."""
        allowed_algorithms = ["HS256", "RS256", "ES256"]
        if v not in allowed_algorithms:
            raise ValueError(f"JWT algorithm must be one of: {allowed_algorithms}")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"

    @property
    def database_url(self) -> str:
        """Get database URL (Redis in this case)."""
        return self.REDIS_URL


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Convenience function for accessing settings
settings = get_settings()
