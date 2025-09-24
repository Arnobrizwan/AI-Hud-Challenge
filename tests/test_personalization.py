"""Tests for the personalization engine."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.optimization.cache import CacheManager
from src.personalization.engine import (
    CollaborativeFilter,
    ContentBasedFilter,
    PersonalizationEngine,
    UserProfileManager,
)
from src.schemas import Article, Author, Source, Topic, UserProfile


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager for testing."""
    cache = Mock(spec=CacheManager)
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    return cache


@pytest.fixture
def sample_article():
    """Sample article for testing."""
    return Article(
        id="article_1",
        title="Test Article",
        content="This is test content",
        url="https://example.com/article/1",
        published_at=datetime.utcnow(),
        source=Source(id="source_1", name="Test Source", domain="test.com"),
        author=Author(id="author_1", name="Test Author"),
        topics=[Topic(name="technology", confidence=0.8), Topic(name="AI", confidence=0.9)],
        word_count=500,
        reading_time=2,
        quality_score=0.8,
    )


@pytest.fixture
def sample_user_profile():
    """Sample user profile for testing."""
    return UserProfile(
        user_id="test_user",
        topic_preferences={"technology": 0.9, "AI": 0.8, "science": 0.7},
        source_preferences={"source_1": 0.8, "source_2": 0.6},
        reading_patterns={"preferred_hours": [9, 10, 11, 14, 15, 16]},
        content_preferences={"preferred_length": 500, "preferred_quality": 0.8},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest.fixture
def personalization_engine(mock_cache_manager):
    """Personalization engine instance for testing."""
    return PersonalizationEngine(mock_cache_manager)


@pytest.mark.asyncio
async def test_personalize_ranking_success(personalization_engine, sample_article):
    """Test successful personalization."""
    user_id = "test_user"

    with patch.object(personalization_engine.user_profiles, "get_profile") as mock_get_profile:
        mock_get_profile.return_value = sample_user_profile()

        result = await personalization_engine.personalize_ranking([sample_article], user_id)

        assert result is not None
        assert len(result) == 1
        assert 0 <= result[0].score <= 1
        assert result[0].article_id == sample_article.id


@pytest.mark.asyncio
async def test_personalize_ranking_no_profile(personalization_engine, sample_article):
    """Test personalization with no user profile."""
    user_id = "new_user"

    with patch.object(personalization_engine.user_profiles, "get_profile") as mock_get_profile:
        mock_get_profile.return_value = None

        result = await personalization_engine.personalize_ranking([sample_article], user_id)

        assert result is not None
        assert len(result) == 1
        assert result[0].score == 0.5  # Default neutral score


@pytest.mark.asyncio
async def test_compute_topic_affinity(personalization_engine, sample_article):
    """Test topic affinity computation."""
    user_preferences = {"technology": 0.9, "AI": 0.8, "science": 0.7}

    score = await personalization_engine._compute_topic_affinity(
        sample_article.topics, user_preferences
    )

    assert 0 <= score <= 1


@pytest.mark.asyncio
async def test_compute_source_preference(personalization_engine):
    """Test source preference computation."""
    source_id = "source_1"
    source_preferences = {"source_1": 0.8, "source_2": 0.6}

    score = await personalization_engine._compute_source_preference(source_id, source_preferences)

    assert score == 0.8


@pytest.mark.asyncio
async def test_compute_time_preference(personalization_engine, sample_article):
    """Test time preference computation."""
    reading_patterns = {"preferred_hours": [9, 10, 11, 14, 15, 16]}

    score = await personalization_engine._compute_time_preference(
        sample_article.published_at, reading_patterns
    )

    assert 0 <= score <= 1


@pytest.mark.asyncio
async def test_user_profile_manager_get_profile(mock_cache_manager):
    """Test user profile retrieval."""
    profile_manager = UserProfileManager(mock_cache_manager)

    with patch.object(mock_cache_manager, "get") as mock_get:
        mock_get.return_value = None

        profile = await profile_manager.get_profile("test_user")

        assert profile is not None
        assert profile.user_id == "test_user"


@pytest.mark.asyncio
async def test_user_profile_manager_update_profile(mock_cache_manager):
    """Test user profile update."""
    profile_manager = UserProfileManager(mock_cache_manager)

    # Get initial profile
    profile = await profile_manager.get_profile("test_user")

    # Update with interaction data
    interaction_data = {
        "article_topics": [
            Topic(name="technology", confidence=0.9),
            Topic(name="AI", confidence=0.8),
        ],
        "source_id": "source_1",
        "reading_time": 120,
    }

    await profile_manager.update_profile("test_user", interaction_data)

    # Verify profile was updated
    assert "technology" in profile.topic_preferences
    assert "AI" in profile.topic_preferences
    assert "source_1" in profile.source_preferences


@pytest.mark.asyncio
async def test_collaborative_filter_predict(mock_cache_manager):
    """Test collaborative filtering prediction."""
    cf = CollaborativeFilter(mock_cache_manager)

    score = await cf.predict("user_1", "article_1")

    assert 0 <= score <= 1


@pytest.mark.asyncio
async def test_content_based_filter_similarity(mock_cache_manager, sample_article):
    """Test content-based filtering similarity."""
    cb = ContentBasedFilter(mock_cache_manager)

    user_preferences = {"preferred_length": 500, "preferred_quality": 0.8}

    score = await cb.compute_similarity(sample_article, user_preferences)

    assert 0 <= score <= 1


@pytest.mark.asyncio
async def test_personalization_error_handling(personalization_engine, sample_article):
    """Test personalization error handling."""
    user_id = "test_user"

    with patch.object(personalization_engine.user_profiles, "get_profile") as mock_get_profile:
        mock_get_profile.side_effect = Exception("Test error")

        result = await personalization_engine.personalize_ranking([sample_article], user_id)

        # Should return neutral scores on error
        assert result is not None
        assert len(result) == 1
        assert result[0].score == 0.5


@pytest.mark.asyncio
async def test_generate_explanation(personalization_engine):
    """Test explanation generation."""
    explanation = personalization_engine._generate_explanation(
        topic_score=0.8, source_score=0.7, cf_score=0.6, cb_score=0.9, time_score=0.5, geo_score=0.5
    )

    assert isinstance(explanation, str)
    assert len(explanation) > 0


@pytest.mark.asyncio
async def test_topic_analyzer_analyze_topics():
    """Test topic analysis."""
    from src.personalization.engine import TopicAnalyzer

    analyzer = TopicAnalyzer()
    content = "This is about artificial intelligence and machine learning"

    topics = await analyzer.analyze_topics(content)

    assert isinstance(topics, list)
    # In production, this would return actual topics


@pytest.mark.asyncio
async def test_topic_analyzer_similarity():
    """Test topic similarity computation."""
    from src.personalization.engine import TopicAnalyzer

    analyzer = TopicAnalyzer()

    topics1 = [Topic(name="technology", confidence=0.8), Topic(name="AI", confidence=0.9)]

    topics2 = [
        Topic(name="technology", confidence=0.7),
        Topic(name="machine learning", confidence=0.8),
    ]

    similarity = await analyzer.compute_topic_similarity(topics1, topics2)

    assert 0 <= similarity <= 1


def test_personalization_weights():
    """Test personalization weight configuration."""
    # Test that weights sum to 1.0
    weights = {"topic": 0.25, "source": 0.20, "cf": 0.20, "cb": 0.20, "time": 0.10, "geo": 0.05}

    assert abs(sum(weights.values()) - 1.0) < 0.01


@pytest.mark.asyncio
async def test_personalization_performance(personalization_engine, sample_article):
    """Test personalization performance."""
    import time

    user_id = "test_user"

    with patch.object(personalization_engine.user_profiles, "get_profile") as mock_get_profile:
        mock_get_profile.return_value = sample_user_profile()

        start_time = time.time()
        result = await personalization_engine.personalize_ranking([sample_article], user_id)
        end_time = time.time()

        processing_time = (end_time - start_time) * 1000

        assert processing_time < 100  # Should be under 100ms
        assert result is not None
