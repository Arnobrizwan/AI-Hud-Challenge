"""
Configuration management for notification decisioning service.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""

    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    DEBUG: bool = Field(default=False, env="DEBUG")

    # Database settings
    DATABASE_URL: str = Field(
        default="postgresql://user:password@localhost/notification_db", env="DATABASE_URL"
    )

    # Redis settings
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # Firebase settings
    FIREBASE_CREDENTIALS_PATH: Optional[str] = Field(default=None, env="FIREBASE_CREDENTIALS_PATH")
    FIREBASE_PROJECT_ID: Optional[str] = Field(default=None, env="FIREBASE_PROJECT_ID")

    # Email settings
    SMTP_HOST: Optional[str] = Field(default=None, env="SMTP_HOST")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_USERNAME: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    SMTP_PASSWORD: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    SMTP_USE_TLS: bool = Field(default=True, env="SMTP_USE_TLS")

    # SMS settings
    TWILIO_ACCOUNT_SID: Optional[str] = Field(default=None, env="TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN: Optional[str] = Field(default=None, env="TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER: Optional[str] = Field(default=None, env="TWILIO_PHONE_NUMBER")

    # ML Model settings
    TIMING_MODEL_PATH: str = Field(default="models/timing_model.pkl", env="TIMING_MODEL_PATH")
    RELEVANCE_MODEL_PATH: str = Field(
        default="models/relevance_model.pkl", env="RELEVANCE_MODEL_PATH"
    )
    ENGAGEMENT_MODEL_PATH: str = Field(
        default="models/engagement_model.pkl", env="ENGAGEMENT_MODEL_PATH"
    )

    # Notification settings
    MAX_NOTIFICATIONS_PER_HOUR: int = Field(default=10, env="MAX_NOTIFICATIONS_PER_HOUR")
    MAX_NOTIFICATIONS_PER_DAY: int = Field(default=50, env="MAX_NOTIFICATIONS_PER_DAY")
    DEFAULT_RELEVANCE_THRESHOLD: float = Field(default=0.3, env="DEFAULT_RELEVANCE_THRESHOLD")

    # A/B Testing settings
    AB_TESTING_ENABLED: bool = Field(default=True, env="AB_TESTING_ENABLED")
    AB_TEST_EXPERIMENTS: List[str] = Field(
        default=["notification_strategy_v3", "timing_optimization_v2"], env="AB_TEST_EXPERIMENTS"
    )

    # Performance settings
    MAX_CONCURRENT_DECISIONS: int = Field(default=1000, env="MAX_CONCURRENT_DECISIONS")
    DECISION_TIMEOUT_SECONDS: int = Field(default=5, env="DECISION_TIMEOUT_SECONDS")

    # CORS settings
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"], env="ALLOWED_ORIGINS"
    )

    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()
