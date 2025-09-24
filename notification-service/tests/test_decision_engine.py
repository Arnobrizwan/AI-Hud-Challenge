"""
Tests for notification decision engine.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.models.schemas import (
    NotificationCandidate, NewsItem, NotificationType, Priority
)
from src.decision_engine.engine import NotificationDecisionEngine


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.ping.return_value = True
    return redis_mock


@pytest.fixture
def sample_news_item():
    """Sample news item for testing."""
    return NewsItem(
        id="news1",
        title="Breaking: Important News",
        content="This is important news content",
        url="https://example.com/news1",
        published_at=datetime.utcnow(),
        category="breaking_news",
        topics=["politics", "urgent"],
        locations=["US"],
        source="reuters",
        is_breaking=True,
        urgency_score=0.9
    )


@pytest.fixture
def sample_candidate(sample_news_item):
    """Sample notification candidate for testing."""
    return NotificationCandidate(
        user_id="user123",
        content=sample_news_item,
        notification_type=NotificationType.BREAKING_NEWS,
        urgency_score=0.9,
        priority=Priority.URGENT
    )


@pytest.fixture
async def decision_engine(mock_redis):
    """Decision engine instance for testing."""
    engine = NotificationDecisionEngine(mock_redis)
    await engine.initialize()
    return engine


@pytest.mark.asyncio
async def test_process_notification_candidate_should_send(decision_engine, sample_candidate):
    """Test processing notification candidate that should be sent."""
    
    # Mock preference manager to return enabled preferences
    decision_engine.preference_manager.get_preferences = AsyncMock(return_value=MagicMock(
        is_notification_type_enabled=lambda x: True,
        get_threshold=lambda x: 0.3
    ))
    
    # Mock fatigue detector to return no fatigue
    decision_engine.fatigue_detector.check_fatigue = AsyncMock(return_value=MagicMock(
        is_fatigued=False
    ))
    
    # Mock relevance scorer to return high relevance
    decision_engine.relevance_scorer.score_relevance = AsyncMock(return_value=0.8)
    
    # Mock timing predictor
    decision_engine.timing_predictor.predict_optimal_time = AsyncMock(return_value=MagicMock(
        scheduled_time=datetime.utcnow() + timedelta(minutes=5),
        predicted_engagement=0.8
    ))
    
    # Mock delivery manager
    decision_engine.delivery_manager.select_optimal_channel = AsyncMock(return_value="push")
    
    # Mock content optimizer
    decision_engine.content_optimizer.optimize_notification_content = AsyncMock(return_value=MagicMock(
        title="🚨 Breaking: Important News",
        body="This is important news content",
        category="breaking_news",
        priority=Priority.URGENT
    ))
    
    # Mock A/B tester
    decision_engine.ab_tester.get_variant = AsyncMock(return_value="engaging")
    
    # Process candidate
    decision = await decision_engine.process_notification_candidate(sample_candidate)
    
    # Assertions
    assert decision.should_send is True
    assert decision.user_id == "user123"
    assert decision.delivery_channel == "push"
    assert decision.priority == Priority.URGENT


@pytest.mark.asyncio
async def test_process_notification_candidate_fatigue_detected(decision_engine, sample_candidate):
    """Test processing notification candidate when fatigue is detected."""
    
    # Mock preference manager
    decision_engine.preference_manager.get_preferences = AsyncMock(return_value=MagicMock(
        is_notification_type_enabled=lambda x: True,
        get_threshold=lambda x: 0.3
    ))
    
    # Mock fatigue detector to return fatigue
    decision_engine.fatigue_detector.check_fatigue = AsyncMock(return_value=MagicMock(
        is_fatigued=True,
        next_eligible_time=datetime.utcnow() + timedelta(hours=1)
    ))
    
    # Process candidate
    decision = await decision_engine.process_notification_candidate(sample_candidate)
    
    # Assertions
    assert decision.should_send is False
    assert decision.reason == "user_fatigue_detected"
    assert decision.next_eligible_time is not None


@pytest.mark.asyncio
async def test_process_notification_candidate_below_threshold(decision_engine, sample_candidate):
    """Test processing notification candidate below relevance threshold."""
    
    # Mock preference manager
    decision_engine.preference_manager.get_preferences = AsyncMock(return_value=MagicMock(
        is_notification_type_enabled=lambda x: True,
        get_threshold=lambda x: 0.9  # High threshold
    ))
    
    # Mock fatigue detector
    decision_engine.fatigue_detector.check_fatigue = AsyncMock(return_value=MagicMock(
        is_fatigued=False
    ))
    
    # Mock relevance scorer to return low relevance
    decision_engine.relevance_scorer.score_relevance = AsyncMock(return_value=0.2)
    
    # Process candidate
    decision = await decision_engine.process_notification_candidate(sample_candidate)
    
    # Assertions
    assert decision.should_send is False
    assert decision.reason == "below_relevance_threshold"
    assert decision.score < decision.threshold


@pytest.mark.asyncio
async def test_process_batch_notifications(decision_engine, sample_candidate):
    """Test processing batch notifications."""
    
    # Create multiple candidates
    candidates = [sample_candidate] * 3
    
    # Mock all dependencies
    decision_engine.preference_manager.get_preferences = AsyncMock(return_value=MagicMock(
        is_notification_type_enabled=lambda x: True,
        get_threshold=lambda x: 0.3
    ))
    
    decision_engine.fatigue_detector.check_fatigue = AsyncMock(return_value=MagicMock(
        is_fatigued=False
    ))
    
    decision_engine.relevance_scorer.score_relevance = AsyncMock(return_value=0.8)
    decision_engine.timing_predictor.predict_optimal_time = AsyncMock(return_value=MagicMock(
        scheduled_time=datetime.utcnow() + timedelta(minutes=5),
        predicted_engagement=0.8
    ))
    decision_engine.delivery_manager.select_optimal_channel = AsyncMock(return_value="push")
    decision_engine.content_optimizer.optimize_notification_content = AsyncMock(return_value=MagicMock(
        title="Test Title",
        body="Test Body",
        category="test",
        priority=Priority.MEDIUM
    ))
    decision_engine.ab_tester.get_variant = AsyncMock(return_value="engaging")
    
    # Process batch
    decisions = await decision_engine.process_batch_notifications(candidates)
    
    # Assertions
    assert len(decisions) == 3
    assert all(decision.should_send for decision in decisions)


@pytest.mark.asyncio
async def test_execute_notification_delivery_success(decision_engine):
    """Test successful notification delivery."""
    
    # Create decision
    decision = MagicMock()
    decision.should_send = True
    decision.user_id = "user123"
    decision.delivery_channel = "push"
    decision.content.category = "breaking_news"
    
    # Mock delivery manager
    decision_engine.delivery_manager.deliver_notification = AsyncMock(return_value=MagicMock(
        success=True,
        delivery_id="delivery123",
        channel="push",
        delivered_at=datetime.utcnow(),
        was_engaged=False
    ))
    
    # Mock fatigue detector
    decision_engine.fatigue_detector.record_notification_sent = AsyncMock()
    
    # Execute delivery
    result = await decision_engine.execute_notification_delivery(decision)
    
    # Assertions
    assert result.success is True
    assert result.delivery_id == "delivery123"
    assert result.channel == "push"


@pytest.mark.asyncio
async def test_execute_notification_delivery_retry_logic(decision_engine):
    """Test notification delivery with retry logic."""
    
    # Create decision
    decision = MagicMock()
    decision.should_send = True
    decision.user_id = "user123"
    decision.delivery_channel = "push"
    decision.content.category = "breaking_news"
    
    # Mock delivery manager to fail first two times, succeed on third
    decision_engine.delivery_manager.deliver_notification = AsyncMock(
        side_effect=[
            Exception("Network error"),
            Exception("Timeout error"),
            MagicMock(
                success=True,
                delivery_id="delivery123",
                channel="push",
                delivered_at=datetime.utcnow(),
                was_engaged=False
            )
        ]
    )
    
    # Mock fatigue detector
    decision_engine.fatigue_detector.record_notification_sent = AsyncMock()
    
    # Execute delivery
    result = await decision_engine.execute_notification_delivery(decision)
    
    # Assertions
    assert result.success is True
    assert result.delivery_id == "delivery123"
    assert decision_engine.delivery_manager.deliver_notification.call_count == 3


@pytest.mark.asyncio
async def test_execute_notification_delivery_max_retries_exceeded(decision_engine):
    """Test notification delivery when max retries are exceeded."""
    
    # Create decision
    decision = MagicMock()
    decision.should_send = True
    decision.user_id = "user123"
    decision.delivery_channel = "push"
    decision.content.category = "breaking_news"
    
    # Mock delivery manager to always fail
    decision_engine.delivery_manager.deliver_notification = AsyncMock(
        side_effect=Exception("Persistent error")
    )
    
    # Mock fatigue detector
    decision_engine.fatigue_detector.record_notification_sent = AsyncMock()
    
    # Execute delivery and expect exception
    with pytest.raises(Exception):
        await decision_engine.execute_notification_delivery(decision)
    
    # Assertions
    assert decision_engine.delivery_manager.deliver_notification.call_count == 3
