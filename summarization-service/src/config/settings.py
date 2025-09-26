"""Configuration settings for summarization service."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Service configuration
    service_name: str = "summarization-service"
    version: str = "1.0.0"
    debug: bool = False
    
    # API configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model configuration
    max_summary_length: int = 200
    min_summary_length: int = 50
    default_compression_ratio: float = 0.3
    
    # CORS configuration
    allowed_origins: list = ["*"]
    
    # Logging configuration
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()