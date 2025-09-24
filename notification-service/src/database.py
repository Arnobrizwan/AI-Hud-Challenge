"""
Database configuration and models for notification decisioning service.
"""

import asyncio
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    create_engine, MetaData
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .config import get_settings

Base = declarative_base()
metadata = MetaData()

# Database engines
_async_engine = None
_async_session_factory = None


class NotificationDecision(Base):
    """Database model for notification decisions."""
    __tablename__ = "notification_decisions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    notification_type = Column(String, nullable=False)
    should_send = Column(Boolean, nullable=False)
    reason = Column(String)
    delivery_channel = Column(String)
    priority = Column(String)
    score = Column(Float)
    threshold = Column(Float)
    processing_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)


class NotificationDelivery(Base):
    """Database model for notification deliveries."""
    __tablename__ = "notification_deliveries"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    notification_id = Column(String, nullable=False)
    channel = Column(String, nullable=False)
    success = Column(Boolean, nullable=False)
    delivered_at = Column(DateTime)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    engagement_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserPreferences(Base):
    """Database model for user notification preferences."""
    __tablename__ = "user_preferences"
    
    user_id = Column(String, primary_key=True)
    enabled_types = Column(JSON)
    delivery_channels = Column(JSON)
    quiet_hours_start = Column(Integer)
    quiet_hours_end = Column(Integer)
    timezone = Column(String, default="UTC")
    allow_emojis = Column(Boolean, default=True)
    max_daily_notifications = Column(Integer, default=50)
    max_hourly_notifications = Column(Integer, default=10)
    relevance_thresholds = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON)


class UserProfile(Base):
    """Database model for user profiles."""
    __tablename__ = "user_profiles"
    
    user_id = Column(String, primary_key=True)
    topic_preferences = Column(JSON)
    source_preferences = Column(JSON)
    location_preferences = Column(JSON)
    engagement_history = Column(JSON)
    device_info = Column(JSON)
    timezone = Column(String, default="UTC")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class NotificationAnalytics(Base):
    """Database model for notification analytics."""
    __tablename__ = "notification_analytics"
    
    id = Column(String, primary_key=True)
    notification_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    notification_type = Column(String, nullable=False)
    channel = Column(String, nullable=False)
    sent_at = Column(DateTime, nullable=False)
    delivered_at = Column(DateTime)
    opened_at = Column(DateTime)
    clicked_at = Column(DateTime)
    engagement_score = Column(Float)
    delivery_duration_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)


class ABTestAssignment(Base):
    """Database model for A/B test assignments."""
    __tablename__ = "ab_test_assignments"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    experiment_name = Column(String, nullable=False)
    variant_name = Column(String, nullable=False)
    parameters = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


async def get_async_engine():
    """Get async database engine."""
    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        _async_engine = create_async_engine(
            settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
            echo=settings.DEBUG,
            pool_size=20,
            max_overflow=30
        )
    return _async_engine


async def get_async_session_factory():
    """Get async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        engine = await get_async_engine()
        _async_session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
    return _async_session_factory


async def get_async_session() -> AsyncSession:
    """Get async database session."""
    session_factory = await get_async_session_factory()
    return session_factory()


async def init_db() -> None:
    """Initialize database tables."""
    engine = await get_async_engine()
    
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    print("Database initialized successfully")


async def close_db() -> None:
    """Close database connections."""
    global _async_engine
    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
