"""
Configuration settings for the Content Extraction & Cleanup microservice.
"""

import os
from typing import List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = Field("Content Extraction & Cleanup Service", env="APP_NAME")
    APP_VERSION: str = Field("1.0.0", env="APP_VERSION")
    DEBUG: bool = Field(False, env="DEBUG")
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    
    # Server
    HOST: str = Field("0.0.0.0", env="HOST")
    PORT: int = Field(8000, env="PORT")
    WORKERS: int = Field(1, env="WORKERS")
    
    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    REDIS_URL: str = Field(..., env="REDIS_URL")
    
    # Google Cloud
    GCP_PROJECT_ID: Optional[str] = Field(None, env="GCP_PROJECT_ID")
    GCP_REGION: str = Field("us-central1", env="GCP_REGION")
    
    # Document AI
    DOCUMENT_AI_PROCESSOR_ID: Optional[str] = Field(None, env="DOCUMENT_AI_PROCESSOR_ID")
    DOCUMENT_AI_LOCATION: str = Field("us", env="DOCUMENT_AI_LOCATION")
    
    # Cloud Tasks
    CLOUD_TASKS_QUEUE: str = Field("content-extraction-queue", env="CLOUD_TASKS_QUEUE")
    CLOUD_TASKS_LOCATION: str = Field("us-central1", env="CLOUD_TASKS_LOCATION")
    
    # Content Processing
    MAX_CONTENT_LENGTH: int = Field(10485760, env="MAX_CONTENT_LENGTH")  # 10MB
    MIN_WORD_COUNT: int = Field(50, env="MIN_WORD_COUNT")
    MAX_WORD_COUNT: int = Field(50000, env="MAX_WORD_COUNT")
    DEFAULT_QUALITY_THRESHOLD: float = Field(0.5, env="DEFAULT_QUALITY_THRESHOLD")
    
    # Image Processing
    MAX_IMAGE_SIZE: int = Field(5242880, env="MAX_IMAGE_SIZE")  # 5MB
    SUPPORTED_IMAGE_FORMATS: List[str] = Field(
        ["jpg", "jpeg", "png", "gif", "webp"], env="SUPPORTED_IMAGE_FORMATS"
    )
    IMAGE_QUALITY_THRESHOLD: float = Field(0.7, env="IMAGE_QUALITY_THRESHOLD")
    
    # HTML Processing
    HTML_CLEANER_CONFIG: dict = Field(
        {
            "remove_scripts": True,
            "remove_styles": True,
            "remove_comments": True,
            "remove_forms": True,
            "remove_ads": True,
            "preserve_links": True,
            "preserve_images": True,
        },
        env="HTML_CLEANER_CONFIG"
    )
    
    # Readability
    READABILITY_MODEL: str = Field("en", env="READABILITY_MODEL")
    MIN_READABILITY_SCORE: float = Field(0.3, env="MIN_READABILITY_SCORE")
    
    # Language Detection
    SUPPORTED_LANGUAGES: List[str] = Field(
        ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
        env="SUPPORTED_LANGUAGES"
    )
    LANGUAGE_CONFIDENCE_THRESHOLD: float = Field(0.8, env="LANGUAGE_CONFIDENCE_THRESHOLD")
    
    # Caching
    CACHE_TTL_SECONDS: int = Field(3600, env="CACHE_TTL_SECONDS")  # 1 hour
    CACHE_MAX_SIZE: int = Field(1000, env="CACHE_MAX_SIZE")
    CACHE_CLEANUP_INTERVAL: int = Field(300, env="CACHE_CLEANUP_INTERVAL")  # 5 minutes
    
    # Batch Processing
    BATCH_SIZE: int = Field(100, env="BATCH_SIZE")
    MAX_PARALLEL_WORKERS: int = Field(4, env="MAX_PARALLEL_WORKERS")
    BATCH_TIMEOUT_SECONDS: int = Field(300, env="BATCH_TIMEOUT_SECONDS")  # 5 minutes
    
    # Quality Analysis
    QUALITY_ANALYSIS_MODEL: str = Field("default", env="QUALITY_ANALYSIS_MODEL")
    SENTIMENT_ANALYSIS_MODEL: str = Field("default", env="SENTIMENT_ANALYSIS_MODEL")
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = Field(True, env="PROMETHEUS_ENABLED")
    METRICS_PORT: int = Field(9090, env="METRICS_PORT")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(1000, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    
    # Security
    SECRET_KEY: str = Field("your-secret-key", env="SECRET_KEY")
    ALLOWED_ORIGINS: List[str] = Field(["*"], env="ALLOWED_ORIGINS")
    
    # External APIs
    PERSPECTIVE_API_KEY: Optional[str] = Field(None, env="PERSPECTIVE_API_KEY")
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()