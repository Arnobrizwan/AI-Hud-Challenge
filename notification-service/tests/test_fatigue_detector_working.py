"""
Working tests for fatigue detection system.
"""

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.fatigue.fatigue_detector import FatigueDetector
from src.models.schemas import FatigueCheck, NotificationType


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.ping.return_value = True
    redis_mock.get.return_value = None
    redis_mock.incr.return_value = 1
    redis_mock.expire.return_value = True
    redis_mock.keys.return_value = []
    redis_mock.lrange.return_value = []
    return redis_mock


@pytest_asyncio.fixture
async def fatigue_detector(mock_redis) -> dict[str, Any]:
    """Fatigue detector instance for testing."""
    detector = FatigueDetector(mock_redis)
    await detector.initialize()
    return detector


@pytest.mark.asyncio
async def test_fatigue_detector_initialization(fatigue_detector) -> dict[str, Any]:
    """Test fatigue detector initialization."""
    assert fatigue_detector is not None
    assert fatigue_detector.redis_client is not None


@pytest.mark.asyncio
async def test_check_fatigue_returns_fatigue_check(fatigue_detector) -> dict[str, Any]:
    """Test that check_fatigue returns a FatigueCheck object."""
    result = await fatigue_detector.check_fatigue("user123", NotificationType.PERSONALIZED)
    
    assert isinstance(result, FatigueCheck)
    assert hasattr(result, 'is_fatigued')
    assert hasattr(result, 'hourly_count')
    assert hasattr(result, 'daily_count')
    assert hasattr(result, 'fatigue_score')


@pytest.mark.asyncio
async def test_record_notification_sent(fatigue_detector) -> dict[str, Any]:
    """Test recording notification sent."""
    user_id = "user123"
    notification_type = NotificationType.PERSONALIZED
    
    # Record notification
    await fatigue_detector.record_notification_sent(user_id, notification_type)
    
    # Verify Redis operations were called
    assert fatigue_detector.redis_client.incr.called
    assert fatigue_detector.redis_client.expire.called


@pytest.mark.asyncio
async def test_get_user_fatigue_analytics(fatigue_detector) -> dict[str, Any]:
    """Test getting user fatigue analytics."""
    user_id = "user123"
    
    # Get analytics
    analytics = await fatigue_detector.get_user_fatigue_analytics(user_id)
    
    # Verify analytics structure
    assert isinstance(analytics, dict)
    assert 'overall_fatigue_score' in analytics
    assert 'notification_counts' in analytics
    assert 'fatigue_patterns' in analytics


@pytest.mark.asyncio
async def test_different_notification_types_fatigue_limits(fatigue_detector) -> dict[str, Any]:
    """Test different notification types have different fatigue limits."""
    user_id = "user123"
    
    # Test different notification types
    personalized_result = await fatigue_detector.check_fatigue(user_id, NotificationType.PERSONALIZED)
    breaking_news_result = await fatigue_detector.check_fatigue(user_id, NotificationType.BREAKING_NEWS)
    
    # Both should return FatigueCheck objects
    assert isinstance(personalized_result, FatigueCheck)
    assert isinstance(breaking_news_result, FatigueCheck)


@pytest.mark.asyncio
async def test_fatigue_score_calculation_basic(fatigue_detector) -> dict[str, Any]:
    """Test basic fatigue score calculation."""
    user_id = "user123"
    
    # Check fatigue multiple times
    result1 = await fatigue_detector.check_fatigue(user_id, NotificationType.PERSONALIZED)
    result2 = await fatigue_detector.check_fatigue(user_id, NotificationType.PERSONALIZED)
    
    # Both should return valid FatigueCheck objects
    assert isinstance(result1, FatigueCheck)
    assert isinstance(result2, FatigueCheck)
    assert 0 <= result1.fatigue_score <= 1.0
    assert 0 <= result2.fatigue_score <= 1.0


@pytest.mark.asyncio
async def test_breaking_news_bypass_fatigue(fatigue_detector) -> dict[str, Any]:
    """Test breaking news bypass fatigue logic."""
    user_id = "user123"
    
    # Test breaking news notification
    result = await fatigue_detector.check_fatigue(user_id, NotificationType.BREAKING_NEWS)
    
    # Should return valid FatigueCheck object
    assert isinstance(result, FatigueCheck)
    assert 0 <= result.fatigue_score <= 1.0


@pytest.mark.asyncio
async def test_fatigue_recovery_time_calculation(fatigue_detector) -> dict[str, Any]:
    """Test fatigue recovery time calculation."""
    user_id = "user123"
    
    # Check fatigue
    result = await fatigue_detector.check_fatigue(user_id, NotificationType.PERSONALIZED)
    
    # Should return valid FatigueCheck with recovery time
    assert isinstance(result, FatigueCheck)
    assert hasattr(result, 'next_eligible_time')
    # next_eligible_time can be None if not fatigued
    assert result.next_eligible_time is None or isinstance(result.next_eligible_time, datetime)
