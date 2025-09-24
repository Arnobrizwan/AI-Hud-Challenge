"""
Database configuration and session management
"""

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from .config import get_database_url

logger = logging.getLogger(__name__)

# Database engine
engine = None
SessionLocal = None
Base = declarative_base()


def init_database():
    """Initialize database connection"""
    global engine, SessionLocal

    try:
        database_url = get_database_url()

        # Create engine with connection pooling
        engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )

        # Create session factory
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        # Create tables
        Base.metadata.create_all(bind=engine)

        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise


def get_db_session() -> Session:
    """Get database session"""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized")
    return SessionLocal()


@contextmanager
def get_db_context():
    """Get database session with context manager"""
    db = get_db_session()
    try:
        yield db
    finally:
        db.close()


def close_database():
    """Close database connections"""
    global engine
    if engine:
        engine.dispose()
        logger.info("Database connections closed")
