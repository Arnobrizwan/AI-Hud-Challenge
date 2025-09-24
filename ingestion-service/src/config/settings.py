"""
Configuration management for the Ingestion & Normalization microservice.
"""

from functools import lru_cache
from typing import List, Optional, Dict, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # Application Settings
    APP_NAME: str = "News Ingestion & Normalization Service"
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
    
    # Cloud Pub/Sub Configuration
    PUBSUB_TOPIC_INGESTION: str = Field(default="news-ingestion", description="Pub/Sub topic for ingestion")
    PUBSUB_TOPIC_NORMALIZATION: str = Field(default="news-normalization", description="Pub/Sub topic for normalization")
    PUBSUB_SUBSCRIPTION_INGESTION: str = Field(default="news-ingestion-sub", description="Pub/Sub subscription")
    
    # Firestore Configuration
    FIRESTORE_COLLECTION_ARTICLES: str = Field(default="articles", description="Firestore articles collection")
    FIRESTORE_COLLECTION_SOURCES: str = Field(default="sources", description="Firestore sources collection")
    FIRESTORE_COLLECTION_METADATA: str = Field(default="metadata", description="Firestore metadata collection")
    
    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    REDIS_MAX_CONNECTIONS: int = Field(default=20, description="Max Redis connections")
    
    # Content Processing Configuration
    MAX_CONTENT_LENGTH: int = Field(default=10485760, description="Max content length in bytes (10MB)")
    MAX_ARTICLE_AGE_DAYS: int = Field(default=30, description="Max article age in days")
    MIN_WORD_COUNT: int = Field(default=50, description="Minimum word count for articles")
    MAX_WORD_COUNT: int = Field(default=50000, description="Maximum word count for articles")
    
    # Rate Limiting Configuration
    DEFAULT_RATE_LIMIT: int = Field(default=60, description="Default requests per minute")
    RATE_LIMIT_BACKOFF_FACTOR: float = Field(default=2.0, description="Exponential backoff factor")
    MAX_RETRY_ATTEMPTS: int = Field(default=3, description="Maximum retry attempts")
    RETRY_DELAY_SECONDS: int = Field(default=1, description="Initial retry delay in seconds")
    
    # HTTP Client Configuration
    HTTP_TIMEOUT: int = Field(default=30, description="HTTP request timeout in seconds")
    HTTP_MAX_CONNECTIONS: int = Field(default=100, description="Max HTTP connections")
    HTTP_KEEPALIVE_TIMEOUT: int = Field(default=5, description="HTTP keepalive timeout")
    USER_AGENT: str = Field(default="NewsBot/1.0 (+https://example.com/bot)", description="Default user agent")
    
    # Content Normalization
    DEFAULT_LANGUAGE: str = Field(default="en", description="Default language code")
    SUPPORTED_LANGUAGES: List[str] = Field(default=["en", "es", "fr", "de", "it", "pt"], description="Supported languages")
    TIMEZONE: str = Field(default="UTC", description="Default timezone")
    
    # Duplicate Detection
    DUPLICATE_THRESHOLD: float = Field(default=0.8, description="Similarity threshold for duplicates")
    CONTENT_HASH_ALGORITHM: str = Field(default="sha256", description="Hash algorithm for content")
    
    # Web Scraping Configuration
    ENABLE_WEB_SCRAPING: bool = Field(default=True, description="Enable web scraping")
    SCRAPING_TIMEOUT: int = Field(default=30, description="Scraping timeout in seconds")
    SCRAPING_DELAY: float = Field(default=1.0, description="Delay between scraping requests")
    RESPECT_ROBOTS_TXT: bool = Field(default=True, description="Respect robots.txt")
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    METRICS_PATH: str = Field(default="/metrics", description="Metrics endpoint path")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format (json/text)")
    
    # Health Check Configuration
    HEALTH_CHECK_PATH: str = Field(default="/health", description="Health check endpoint")
    HEALTH_CHECK_DEPENDENCIES: bool = Field(default=True, description="Include dependency checks")
    
    # Content Sources Configuration
    SOURCES_CONFIG_PATH: str = Field(default="config/sources.yaml", description="Path to sources configuration")
    ENABLE_RSS_FEEDS: bool = Field(default=True, description="Enable RSS/Atom feed processing")
    ENABLE_JSON_FEEDS: bool = Field(default=True, description="Enable JSON Feed processing")
    ENABLE_API_SOURCES: bool = Field(default=True, description="Enable API source processing")
    ENABLE_WEB_SCRAPING: bool = Field(default=True, description="Enable web scraping sources")
    
    # Performance Configuration
    BATCH_SIZE: int = Field(default=100, description="Batch size for processing")
    MAX_CONCURRENT_TASKS: int = Field(default=50, description="Max concurrent processing tasks")
    PROCESSING_INTERVAL: int = Field(default=60, description="Processing interval in seconds")
    
    # Security Configuration
    ALLOWED_DOMAINS: List[str] = Field(default=[], description="Allowed domains for scraping")
    BLOCKED_DOMAINS: List[str] = Field(default=[], description="Blocked domains")
    ALLOWED_CONTENT_TYPES: List[str] = Field(
        default=["text/html", "text/xml", "application/xml", "application/rss+xml", "application/atom+xml"],
        description="Allowed content types"
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
