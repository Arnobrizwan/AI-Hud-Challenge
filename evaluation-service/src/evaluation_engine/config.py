"""Configuration for evaluation engine."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    service_name: str = "evaluation-service"
    version: str = "1.0.0"
    debug: bool = False
    
    class Config:
        env_file = ".env"


def get_settings():
    """Get application settings."""
    return Settings()