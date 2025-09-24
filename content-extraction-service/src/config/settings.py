"""
Configuration management for the Content Extraction & Cleanup microservice.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration with environment variable support."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore")

    # Application Settings
    APP_NAME: str = "Content Extraction & Cleanup Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    ENVIRONMENT: str = Field(default="development", description="Application environment")

    # Server Configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    WORKERS: int = Field(default=1, description="Number of worker processes")

    # Google Cloud Configuration
    GCP_PROJECT_ID: str = Field(description="GCP Project ID")
    GCP_REGION: str = Field(default="us-central1", description="GCP Region")
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = Field(default=None, description="Path to GCP credentials")

    # Cloud Storage Configuration
    CLOUD_STORAGE_BUCKET: str = Field(
        default="content-extraction-cache", description="Cloud Storage bucket for caching"
    )
    CACHE_RETENTION_DAYS: int = Field(default=30, description="Cache retention period in days")

    # Cloud Tasks Configuration
    CLOUD_TASKS_QUEUE: str = Field(default="content-extraction-queue", description="Cloud Tasks queue name")
    TASK_RETRY_ATTEMPTS: int = Field(default=3, description="Number of task retry attempts")
    TASK_RETRY_DELAY: int = Field(default=60, description="Task retry delay in seconds")

    # Document AI Configuration
    DOCUMENT_AI_PROCESSOR_ID: str = Field(description="Document AI processor ID")
    DOCUMENT_AI_LOCATION: str = Field(default="us", description="Document AI location")

    # Firestore Configuration
    FIRESTORE_COLLECTION_EXTRACTIONS: str = Field(default="extractions", description="Firestore extractions collection")
    FIRESTORE_COLLECTION_TASKS: str = Field(default="tasks", description="Firestore tasks collection")

    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    REDIS_MAX_CONNECTIONS: int = Field(default=20, description="Max Redis connections")

    # Content Processing Configuration
    MAX_CONTENT_LENGTH: int = Field(default=10485760, description="Max content length in bytes (10MB)")
    MAX_IMAGE_SIZE: int = Field(default=5242880, description="Max image size in bytes (5MB)")
    MAX_IMAGES_PER_ARTICLE: int = Field(default=20, description="Max images per article")
    MIN_CONTENT_LENGTH: int = Field(default=100, description="Min content length in characters")
    MAX_CONTENT_LENGTH_CHARS: int = Field(default=100000, description="Max content length in characters")

    # Quality Scoring Configuration
    MIN_QUALITY_SCORE: float = Field(default=0.3, description="Minimum quality score threshold")
    SPAM_THRESHOLD: float = Field(default=0.7, description="Spam detection threshold")
    READABILITY_THRESHOLD: float = Field(default=30.0, description="Minimum readability score")

    # Image Processing Configuration
    IMAGE_QUALITY: int = Field(default=85, description="JPEG quality for image optimization")
    IMAGE_MAX_WIDTH: int = Field(default=1920, description="Max image width")
    IMAGE_MAX_HEIGHT: int = Field(default=1080, description="Max image height")
    IMAGE_FORMATS: List[str] = Field(default=["JPEG", "PNG", "WebP"], description="Supported image formats")

    # Playwright Configuration
    PLAYWRIGHT_HEADLESS: bool = Field(default=True, description="Run Playwright in headless mode")
    PLAYWRIGHT_TIMEOUT: int = Field(default=30000, description="Playwright timeout in milliseconds")
    PLAYWRIGHT_VIEWPORT_WIDTH: int = Field(default=1920, description="Viewport width")
    PLAYWRIGHT_VIEWPORT_HEIGHT: int = Field(default=1080, description="Viewport height")
    PLAYWRIGHT_USER_AGENT: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        description="User agent for Playwright",
    )

    # Content Security Configuration
    ALLOWED_DOMAINS: List[str] = Field(default=[], description="Allowed domains for extraction")
    BLOCKED_DOMAINS: List[str] = Field(default=[], description="Blocked domains")
    ALLOWED_CONTENT_TYPES: List[str] = Field(
        default=[
            "text/html",
            "application/pdf",
            "text/plain",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ],
        description="Allowed content types",
    )

    # Rate Limiting Configuration
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=60, description="Rate limit per minute")
    RATE_LIMIT_BURST: int = Field(default=10, description="Rate limit burst size")
    RATE_LIMIT_BACKOFF_FACTOR: float = Field(default=2.0, description="Exponential backoff factor")

    # Processing Configuration
    MAX_CONCURRENT_EXTRACTIONS: int = Field(default=50, description="Max concurrent extractions")
    EXTRACTION_TIMEOUT: int = Field(default=300, description="Extraction timeout in seconds")
    BATCH_SIZE: int = Field(default=100, description="Batch size for processing")

    # Language Detection Configuration
    SUPPORTED_LANGUAGES: List[str] = Field(
        default=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
        description="Supported languages",
    )
    DEFAULT_LANGUAGE: str = Field(default="en", description="Default language")

    # Monitoring Configuration
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    METRICS_PATH: str = Field(default="/metrics", description="Metrics endpoint path")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format (json/text)")

    # Health Check Configuration
    HEALTH_CHECK_PATH: str = Field(default="/health", description="Health check endpoint")
    HEALTH_CHECK_DEPENDENCIES: bool = Field(default=True, description="Include dependency checks")

    # Content Sanitization Configuration
    ALLOWED_HTML_TAGS: List[str] = Field(
        default=[
            "p",
            "br",
            "strong",
            "em",
            "u",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "ul",
            "ol",
            "li",
            "blockquote",
            "a",
            "img",
        ],
        description="Allowed HTML tags",
    )
    ALLOWED_HTML_ATTRIBUTES: List[str] = Field(
        default=["href", "src", "alt", "title", "class", "id"],
        description="Allowed HTML attributes",
    )

    # Anti-bot Detection Configuration
    ENABLE_ANTI_BOT_DETECTION: bool = Field(default=True, description="Enable anti-bot detection")
    ANTI_BOT_DELAY: float = Field(default=1.0, description="Delay between requests to avoid detection")
    ROTATE_USER_AGENTS: bool = Field(default=True, description="Rotate user agents")

    # Cookie Consent Configuration
    HANDLE_COOKIE_CONSENT: bool = Field(default=True, description="Handle cookie consent dialogs")
    COOKIE_CONSENT_SELECTORS: List[str] = Field(
        default=[
            "button[class*='accept']",
            "button[class*='consent']",
            "button[class*='cookie']",
            "#cookie-accept",
            "#consent-accept",
        ],
        description="Cookie consent button selectors",
    )

    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed_environments = ["development", "staging", "production"]
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level setting."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Convenience function for accessing settings
settings = get_settings()
