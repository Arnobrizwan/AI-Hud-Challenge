"""
Tests for fatigue detection system.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

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


@pytest.fixture
async def fatigue_detector(mock_redis):
    """Fatigue detector instance for testing."""
    detector = FatigueDetector(mock_redis)
    await detector.initialize()
    return detector


@pytest.mark.asyncio
async def test_check_fatigue_no_fatigue(fatigue_detector):
    """Test fatigue check when user is not fatigued."""

    # Mock Redis to return low counts
    fatigue_detector.redis_client.get = AsyncMock(return_value="2")  # Low count

    # Check fatigue
    result = await fatigue_detector.check_fatigue("user123", NotificationType.PERSONALIZED)

    # Assertions
    assert isinstance(result, FatigueCheck)
    assert result.is_fatigued is False
    assert result.hourly_count == 2
    assert result.daily_count == 2
    assert result.next_eligible_time is None


@pytest.mark.asyncio
async def test_check_fatigue_hourly_limit_exceeded(fatigue_detector):
    """Test fatigue check when hourly limit is exceeded."""

    # Mock Redis to return high hourly count
    fatigue_detector.redis_client.get = AsyncMock(side_effect=["5", "2"])  # High hourly, low daily

    # Check fatigue
    result = await fatigue_detector.check_fatigue("user123", NotificationType.PERSONALIZED)

    # Assertions
    assert result.is_fatigued is True
    assert result.hourly_count == 5
    assert result.daily_count == 2
    assert result.next_eligible_time is not None


@pytest.mark.asyncio
async def test_check_fatigue_daily_limit_exceeded(fatigue_detector):
    """Test fatigue check when daily limit is exceeded."""

    # Mock Redis to return high daily count
    fatigue_detector.redis_client.get = AsyncMock(side_effect=["2", "15"])  # Low hourly, high daily

    # Check fatigue
    result = await fatigue_detector.check_fatigue("user123", NotificationType.PERSONALIZED)

    # Assertions
    assert result.is_fatigued is True
    assert result.hourly_count == 2
    assert result.daily_count == 15
    assert result.next_eligible_time is not None


@pytest.mark.asyncio
async def test_record_notification_sent(fatigue_detector):
    """Test recording notification sent."""

    # Mock Redis operations
    fatigue_detector.redis_client.incr = AsyncMock(return_value=1)
    fatigue_detector.redis_client.expire = AsyncMock(return_value=True)

    # Record notification
    await fatigue_detector.record_notification_sent("user123", NotificationType.PERSONALIZED)

    # Assertions
    assert fatigue_detector.redis_client.incr.call_count == 2  # Hourly and daily
    assert fatigue_detector.redis_client.expire.call_count == 2


@pytest.mark.asyncio
async def test_get_user_fatigue_analytics(fatigue_detector):
    """Test getting user fatigue analytics."""

    # Mock Redis to return various counts
    fatigue_detector.redis_client.get = AsyncMock(return_value="0.5")  # Fatigue score
    fatigue_detector.redis_client.keys = AsyncMock(return_value=[])  # No keys

    # Get analytics
    analytics = await fatigue_detector.get_user_fatigue_analytics("user123")

    # Assertions
    assert analytics["user_id"] == "user123"
    assert "overall_fatigue_score" in analytics
    assert "notification_counts" in analytics
    assert "fatigue_patterns" in analytics
    assert "recommendations" in analytics


@pytest.mark.asyncio
async def test_fatigue_score_calculation(fatigue_detector):
    """Test fatigue score calculation."""

    # Test different scenarios
    test_cases = [
        (1, 5, 0.0, 0.2),  # Low counts, no user fatigue
        (5, 15, 0.0, 0.8),  # High counts, no user fatigue
        (1, 5, 0.9, 0.7),  # Low counts, high user fatigue
        (5, 15, 0.9, 1.0),  # High counts, high user fatigue
    ]

    for hourly, daily, user_fatigue, expected_min in test_cases:
        # Mock Redis
        fatigue_detector.redis_client.get = AsyncMock(
            side_effect=[str(hourly), str(daily), str(user_fatigue)]
        )

        # Check fatigue
        result = await fatigue_detector.check_fatigue("user123", NotificationType.PERSONALIZED)

        # Assertions
        assert result.fatigue_score >= expected_min
        assert 0.0 <= result.fatigue_score <= 1.0


@pytest.mark.asyncio
async def test_breaking_news_bypass_fatigue(fatigue_detector):
    """Test that breaking news can bypass fatigue."""

    # Mock Redis to return high counts (fatigued state)
    fatigue_detector.redis_client.get = AsyncMock(side_effect=["10", "50"])

    # Check fatigue for breaking news
    result = await fatigue_detector.check_fatigue("user123", NotificationType.BREAKING_NEWS)

    # Breaking news has higher thresholds, so should not be fatigued
    assert result.is_fatigued is False


@pytest.mark.asyncio
async def test_different_notification_types_fatigue_limits(fatigue_detector):
    """Test different fatigue limits for different notification types."""

    # Test breaking news (lower limits)
    result_breaking = await fatigue_detector.check_fatigue(
        "user123", NotificationType.BREAKING_NEWS
    )
    breaking_thresholds = fatigue_detector.fatigue_thresholds[NotificationType.BREAKING_NEWS]

    # Test personalized (higher limits)
    result_personalized = await fatigue_detector.check_fatigue(
        "user123", NotificationType.PERSONALIZED
    )
    personalized_thresholds = fatigue_detector.fatigue_thresholds[NotificationType.PERSONALIZED]

    # Assertions
    assert breaking_thresholds["daily"] < personalized_thresholds["daily"]
    assert breaking_thresholds["hourly"] < personalized_thresholds["hourly"]


@pytest.mark.asyncio
async def test_fatigue_recovery_time_calculation(fatigue_detector):
    """Test fatigue recovery time calculation."""

    # Mock Redis to return high counts
    fatigue_detector.redis_client.get = AsyncMock(side_effect=["10", "5"])  # High hourly, low daily

    # Check fatigue
    result = await fatigue_detector.check_fatigue("user123", NotificationType.PERSONALIZED)

    # Assertions
    assert result.is_fatigued is True
    assert result.next_eligible_time is not None

    # Next eligible time should be approximately 1 hour from now
    expected_time = datetime.utcnow() + timedelta(hours=1)
    time_diff = abs((result.next_eligible_time - expected_time).total_seconds())
    assert time_diff < 60  # Within 1 minute


@pytest.mark.asyncio
async def test_fatigue_detector_initialization(fatigue_detector):
    """Test fatigue detector initialization."""

    # Assertions
    assert fatigue_detector.redis_client is not None
    assert fatigue_detector.fatigue_thresholds is not None
    assert len(fatigue_detector.fatigue_thresholds) > 0

    # Check that all notification types have thresholds
    for notification_type in NotificationType:
        assert notification_type in fatigue_detector.fatigue_thresholds
        assert "daily" in fatigue_detector.fatigue_thresholds[notification_type]
        assert "hourly" in fatigue_detector.fatigue_thresholds[notification_type]
