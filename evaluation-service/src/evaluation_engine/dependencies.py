"""
Dependency injection for Evaluation Suite Microservice
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from .database import get_db_session
from .cache import get_cache_client
from .core import EvaluationEngine
from .config import get_settings


def get_database_dependency() -> Generator[Session, None, None]:
    """Get database session dependency"""
    db = get_db_session()
    try:
        yield db
    finally:
        db.close()


def get_cache_dependency():
    """Get cache client dependency"""
    return get_cache_client()


def get_evaluation_engine() -> EvaluationEngine:
    """Get evaluation engine dependency"""
    # This will be overridden in main.py with the actual instance
    raise HTTPException(
        status_code=503,
        detail="Evaluation engine not available"
    )


def get_settings_dependency():
    """Get settings dependency"""
    return get_settings()
