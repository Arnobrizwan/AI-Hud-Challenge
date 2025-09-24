"""
Configuration settings for the Summarization Service
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    SERVICE_NAME: str = "summarization-service"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # API configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["*"], 
        env="ALLOWED_ORIGINS"
    )
    
    # Vertex AI configuration
    GOOGLE_CLOUD_PROJECT: str = Field(env="GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_REGION: str = Field(default="us-central1", env="GOOGLE_CLOUD_REGION")
    VERTEX_AI_ENDPOINT: Optional[str] = Field(env="VERTEX_AI_ENDPOINT")
    
    # Model configuration
    MAX_CONTENT_LENGTH: int = Field(default=10000, env="MAX_CONTENT_LENGTH")
    MAX_SUMMARY_LENGTH: int = Field(default=500, env="MAX_SUMMARY_LENGTH")
    DEFAULT_TARGET_LENGTHS: List[int] = Field(default=[50, 120, 300])
    
    # Quality thresholds
    MIN_QUALITY_SCORE: float = Field(default=0.7, env="MIN_QUALITY_SCORE")
    MIN_CONSISTENCY_SCORE: float = Field(default=0.8, env="MIN_CONSISTENCY_SCORE")
    MAX_BIAS_SCORE: float = Field(default=0.3, env="MAX_BIAS_SCORE")
    
    # Performance configuration
    BATCH_SIZE: int = Field(default=8, env="BATCH_SIZE")
    MAX_CONCURRENT_REQUESTS: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # Redis configuration
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(env="REDIS_PASSWORD")
    
    # Monitoring configuration
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    # Logging configuration
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # A/B Testing configuration
    ENABLE_AB_TESTING: bool = Field(default=True, env="ENABLE_AB_TESTING")
    AB_TEST_TRAFFIC_SPLIT: float = Field(default=0.5, env="AB_TEST_TRAFFIC_SPLIT")
    
    # Multi-language configuration
    SUPPORTED_LANGUAGES: List[str] = Field(
        default=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
        env="SUPPORTED_LANGUAGES"
    )
    DEFAULT_LANGUAGE: str = Field(default="en", env="DEFAULT_LANGUAGE")
    
    # Model paths and configurations
    BERT_MODEL_PATH: str = Field(
        default="bert-base-uncased",
        env="BERT_MODEL_PATH"
    )
    T5_MODEL_PATH: str = Field(
        default="t5-base",
        env="T5_MODEL_PATH"
    )
    SPACY_MODEL: str = Field(
        default="en_core_web_lg",
        env="SPACY_MODEL"
    )
    
    # GPU configuration
    USE_GPU: bool = Field(default=True, env="USE_GPU")
    GPU_MEMORY_FRACTION: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
