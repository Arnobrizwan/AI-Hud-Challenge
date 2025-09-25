"""
Configuration settings for the Realtime Interfaces microservice.
"""

import os
from typing import List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = Field("Realtime Interfaces Service", env="APP_NAME")
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
    
    # External Services
    RANKING_SERVICE_URL: str = Field("http://ranking:8000", env="RANKING_SERVICE_URL")
    PERSONALIZATION_SERVICE_URL: str = Field("http://personalization:8000", env="PERSONALIZATION_SERVICE_URL")
    DEDUPLICATION_SERVICE_URL: str = Field("http://deduplication:8000", env="DEDUPLICATION_SERVICE_URL")
    FEEDBACK_SERVICE_URL: str = Field("http://feedback:8000", env="FEEDBACK_SERVICE_URL")
    
    # Feed Configuration
    DEFAULT_FEED_LIMIT: int = Field(20, env="DEFAULT_FEED_LIMIT")
    MAX_FEED_LIMIT: int = Field(100, env="MAX_FEED_LIMIT")
    DEFAULT_TIME_RANGE: str = Field("24h", env="DEFAULT_TIME_RANGE")
    SUPPORTED_TIME_RANGES: List[str] = Field(
        ["1h", "24h", "7d", "30d"], env="SUPPORTED_TIME_RANGES"
    )
    
    # Personalization
    PERSONALIZATION_ENABLED: bool = Field(True, env="PERSONALIZATION_ENABLED")
    COLD_START_FALLBACK: bool = Field(True, env="COLD_START_FALLBACK")
    DIVERSITY_THRESHOLD: float = Field(0.3, env="DIVERSITY_THRESHOLD")
    
    # Caching
    CACHE_TTL_SECONDS: int = Field(300, env="CACHE_TTL_SECONDS")  # 5 minutes
    CACHE_MAX_SIZE: int = Field(10000, env="CACHE_MAX_SIZE")
    CACHE_CLEANUP_INTERVAL: int = Field(60, env="CACHE_CLEANUP_INTERVAL")  # 1 minute
    
    # WebSocket Configuration
    WEBSOCKET_HEARTBEAT_INTERVAL: int = Field(30, env="WEBSOCKET_HEARTBEAT_INTERVAL")  # 30 seconds
    WEBSOCKET_MAX_CONNECTIONS: int = Field(1000, env="WEBSOCKET_MAX_CONNECTIONS")
    WEBSOCKET_MESSAGE_SIZE_LIMIT: int = Field(1024 * 1024, env="WEBSOCKET_MESSAGE_SIZE_LIMIT")  # 1MB
    
    # Server-Sent Events
    SSE_HEARTBEAT_INTERVAL: int = Field(30, env="SSE_HEARTBEAT_INTERVAL")  # 30 seconds
    SSE_MAX_CONNECTIONS: int = Field(1000, env="SSE_MAX_CONNECTIONS")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(1000, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    RATE_LIMIT_PER_USER: int = Field(100, env="RATE_LIMIT_PER_USER")
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = Field(100, env="MAX_CONCURRENT_REQUESTS")
    REQUEST_TIMEOUT_SECONDS: int = Field(30, env="REQUEST_TIMEOUT_SECONDS")
    CONNECTION_POOL_SIZE: int = Field(20, env="CONNECTION_POOL_SIZE")
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = Field(True, env="PROMETHEUS_ENABLED")
    METRICS_PORT: int = Field(9090, env="METRICS_PORT")
    
    # Security
    SECRET_KEY: str = Field("your-secret-key", env="SECRET_KEY")
    ALLOWED_ORIGINS: List[str] = Field(["*"], env="ALLOWED_ORIGINS")
    JWT_SECRET: Optional[str] = Field(None, env="JWT_SECRET")
    JWT_ALGORITHM: str = Field("HS256", env="JWT_ALGORITHM")
    JWT_EXPIRATION_MINUTES: int = Field(60, env="JWT_EXPIRATION_MINUTES")
    
    # Content Filtering
    CONTENT_FILTER_ENABLED: bool = Field(True, env="CONTENT_FILTER_ENABLED")
    MIN_QUALITY_SCORE: float = Field(0.3, env="MIN_QUALITY_SCORE")
    BLOCKED_TOPICS: List[str] = Field([], env="BLOCKED_TOPICS")
    BLOCKED_SOURCES: List[str] = Field([], env="BLOCKED_SOURCES")
    
    # Real-time Updates
    UPDATE_BATCH_SIZE: int = Field(100, env="UPDATE_BATCH_SIZE")
    UPDATE_PROCESSING_INTERVAL: int = Field(5, env="UPDATE_PROCESSING_INTERVAL")  # 5 seconds
    MAX_UPDATE_AGE_SECONDS: int = Field(300, env="MAX_UPDATE_AGE_SECONDS")  # 5 minutes
    
    # Error Handling
    MAX_RETRIES: int = Field(3, env="MAX_RETRIES")
    RETRY_DELAY_SECONDS: int = Field(1, env="RETRY_DELAY_SECONDS")
    CIRCUIT_BREAKER_THRESHOLD: int = Field(5, env="CIRCUIT_BREAKER_THRESHOLD")
    CIRCUIT_BREAKER_TIMEOUT: int = Field(60, env="CIRCUIT_BREAKER_TIMEOUT")  # 60 seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
