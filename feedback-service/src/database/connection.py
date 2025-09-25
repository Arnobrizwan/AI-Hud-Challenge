"""
Database connection and session management
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from ..config.settings import get_settings

logger = logging.getLogger(__name__)

# Database engine
engine = None
async_session_maker = None

# Base class for models
Base = declarative_base()


async def init_db() -> Dict[str, Any]:
    """Initialize database connection"""
    global engine, async_session_maker

    settings = get_settings()

    # Create async engine
    engine = create_async_engine(
        settings.database_url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        poolclass=NullPool if settings.debug else None,
        echo=settings.debug,
    )

    # Create session maker
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    logger.info("Database connection initialized")


async def close_db() -> Dict[str, Any]:
    """Close database connection"""
    global engine

    if engine:
        await engine.dispose()
        logger.info("Database connection closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    if not async_session_maker:
        raise RuntimeError("Database not initialized")

    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
