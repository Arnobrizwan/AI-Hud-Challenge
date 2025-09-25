"""Database management utilities."""

import asyncio
from typing import Optional

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from ..config.settings import settings


class DatabaseManager:
    """Database connection manager."""

    def __init__(self, database_url: str):
        """Initialize database manager.

        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
        self.pool = None

    async def initialize(self) -> Dict[str, Any]:
        """Initialize database connections."""
        # Create async engine
        self.engine = create_async_engine(
            self.database_url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            echo=settings.debug,
        )

        # Create session factory
        self.session_factory = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

        # Create connection pool
        self.pool = await asyncpg.create_pool(self.database_url, min_size=5, max_size=settings.database_pool_size)

    async def get_session(self) -> AsyncSession:
        """Get database session.

        Returns:
            Database session
        """
        if not self.session_factory:
            raise RuntimeError("Database not initialized")

        return self.session_factory()

    async def get_connection(self) -> Dict[str, Any]:
        """Get database connection.

        Returns:
            Database connection
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized")

        return await self.pool.acquire()

    async def close(self) -> Dict[str, Any]:
        """Close database connections."""
        if self.pool:
            await self.pool.close()

        if self.engine:
            await self.engine.dispose()
