"""Configuration for evaluation engine."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    service_name: str = "evaluation-service"
    version: str = "1.0.0"
    debug: bool = False
    
    # CORS settings
    allowed_origins: list = ["*"]
    allowed_hosts: list = ["*"]
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra environment variables


def get_settings():
    """Get application settings."""
    return Settings()