"""Tests for the ranking engine."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.optimization.cache import CacheManager
from src.ranking.engine import ContentRankingEngine
from src.schemas import Article, Author, ContentType, RankingRequest, Source


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager for testing."""
    cache = Mock(spec=CacheManager)
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.get_hit_rate = Mock(return_value=0.8)
    return cache


@pytest.fixture
def sample_articles():
    """Sample articles for testing."""
    articles = []
    for i in range(5):
        article = Article(
            id=f"article_{i}",
            title=f"Test Article {i}",
            content=f"This is test content for article {i}",
            url=f"https://example.com/article/{i}",
            published_at=datetime.utcnow() - timedelta(hours=i),
            source=Source(
                id=f"source_{i % 2}", name=f"Source {i % 2}", domain=f"source{i % 2}.com"
            ),
            author=Author(id=f"author_{i}", name=f"Author {i}"),
            word_count=500 + i * 100,
            reading_time=2 + i,
            quality_score=0.5 + i * 0.1,
        )
        articles.append(article)
    return articles


@pytest.fixture
def sample_ranking_request():
    """Sample ranking request for testing."""
    return RankingRequest(
        user_id="test_user", query="test query", limit=10, enable_personalization=True
    )


@pytest.fixture
def ranking_engine(mock_cache_manager):
    """Ranking engine instance for testing."""
    return ContentRankingEngine(mock_cache_manager)


@pytest.mark.asyncio
async def test_rank_content_success(ranking_engine, sample_ranking_request):
    """Test successful content ranking."""
    with patch.object(ranking_engine, "_get_candidates") as mock_get_candidates:
        mock_get_candidates.return_value = []

        result = await ranking_engine.rank_content(sample_ranking_request)

        assert result is not None
        assert result.total_count == 0
        assert result.articles == []
        assert result.algorithm_variant is not None
        assert result.processing_time_ms >= 0


@pytest.mark.asyncio
async def test_rank_content_with_articles(ranking_engine, sample_ranking_request, sample_articles):
    """Test content ranking with articles."""
    with patch.object(ranking_engine, "_get_candidates") as mock_get_candidates:
        mock_get_candidates.return_value = sample_articles

        result = await ranking_engine.rank_content(sample_ranking_request)

        assert result is not None
        assert result.total_count == len(sample_articles)
        assert len(result.articles) <= sample_ranking_request.limit
        assert all(article.rank > 0 for article in result.articles)


@pytest.mark.asyncio
async def test_compute_ranking_features(ranking_engine, sample_articles, sample_ranking_request):
    """Test feature computation."""
    features = await ranking_engine.compute_ranking_features(
        sample_articles, sample_ranking_request
    )

    assert features is not None
    assert len(features) == len(sample_articles)
    assert all(len(feature_vector) > 0 for feature_vector in features)


@pytest.mark.asyncio
async def test_heuristic_ranking(ranking_engine, sample_articles, sample_ranking_request):
    """Test heuristic ranking algorithm."""
    features = [[0.5] * 20] * len(sample_articles)  # Dummy features

    result = await ranking_engine.heuristic_ranking(
        sample_articles, features, sample_ranking_request
    )

    assert result is not None
    assert len(result) <= sample_ranking_request.limit
    assert all(article.rank > 0 for article in result)
    assert all(0 <= article.score <= 1 for article in result)


@pytest.mark.asyncio
async def test_hybrid_ranking(ranking_engine, sample_articles, sample_ranking_request):
    """Test hybrid ranking algorithm."""
    features = [[0.5] * 20] * len(sample_articles)  # Dummy features

    result = await ranking_engine.hybrid_ranking(sample_articles, features, sample_ranking_request)

    assert result is not None
    assert len(result) <= sample_ranking_request.limit
    assert all(article.rank > 0 for article in result)
    assert all(0 <= article.score <= 1 for article in result)


@pytest.mark.asyncio
async def test_apply_ranking_constraints(ranking_engine, sample_articles, sample_ranking_request):
    """Test ranking constraints application."""
    # Create ranked articles
    ranked_articles = []
    for i, article in enumerate(sample_articles):
        from src.schemas import RankedArticle

        ranked_article = RankedArticle(
            article=article, rank=i + 1, score=0.5 + i * 0.1, explanation="Test ranking"
        )
        ranked_articles.append(ranked_article)

    result = await ranking_engine._apply_ranking_constraints(
        ranked_articles, sample_ranking_request
    )

    assert result is not None
    assert len(result) <= sample_ranking_request.limit
    assert all(article.rank > 0 for article in result)


@pytest.mark.asyncio
async def test_compute_heuristic_score(ranking_engine, sample_articles, sample_ranking_request):
    """Test heuristic score computation."""
    article = sample_articles[0]
    features = [0.5] * 20  # Dummy features

    score = await ranking_engine._compute_heuristic_score(article, features, sample_ranking_request)

    assert 0 <= score <= 1


@pytest.mark.asyncio
async def test_get_personalization_score(ranking_engine, sample_articles, sample_ranking_request):
    """Test personalization score computation."""
    article = sample_articles[0]

    with patch.object(
        ranking_engine.personalization_engine, "personalize_ranking"
    ) as mock_personalize:
        mock_personalize.return_value = [Mock(score=0.7)]

        score = await ranking_engine._get_personalization_score(article, "test_user")

        assert 0 <= score <= 1


@pytest.mark.asyncio
async def test_ranking_error_handling(ranking_engine, sample_ranking_request):
    """Test error handling in ranking."""
    with patch.object(ranking_engine, "_get_candidates") as mock_get_candidates:
        mock_get_candidates.side_effect = Exception("Test error")

        with pytest.raises(Exception):
            await ranking_engine.rank_content(sample_ranking_request)


@pytest.mark.asyncio
async def test_ml_ranking_fallback(ranking_engine, sample_articles, sample_ranking_request):
    """Test ML ranking fallback to heuristic."""
    features = [[0.5] * 20] * len(sample_articles)

    # Mock model not loaded
    ranking_engine.model_loaded = False

    result = await ranking_engine.ml_ranking(sample_articles, features, sample_ranking_request)

    assert result is not None
    assert len(result) <= sample_ranking_request.limit


@pytest.mark.asyncio
async def test_ranking_with_different_algorithms(
    ranking_engine, sample_articles, sample_ranking_request
):
    """Test ranking with different algorithm variants."""
    features = [[0.5] * 20] * len(sample_articles)

    # Test ML ranking
    ranking_engine.model_loaded = True
    ranking_engine.ranker_model = Mock()
    ranking_engine.ranker_model.predict = Mock(return_value=[0.8, 0.7, 0.6, 0.5, 0.4])

    result = await ranking_engine.ml_ranking(sample_articles, features, sample_ranking_request)

    assert result is not None
    assert len(result) <= sample_ranking_request.limit


def test_default_weights(ranking_engine):
    """Test default ranking weights."""
    weights = ranking_engine.default_weights

    assert "relevance" in weights
    assert "freshness" in weights
    assert "authority" in weights
    assert "personalization" in weights
    assert "diversity" in weights

    # Check weights sum to 1.0
    assert abs(sum(weights.values()) - 1.0) < 0.01


@pytest.mark.asyncio
async def test_ranking_performance(ranking_engine, sample_ranking_request):
    """Test ranking performance metrics."""
    import time

    start_time = time.time()

    with patch.object(ranking_engine, "_get_candidates") as mock_get_candidates:
        mock_get_candidates.return_value = []

        result = await ranking_engine.rank_content(sample_ranking_request)

        end_time = time.time()
        processing_time = (end_time - start_time) * 1000

        assert result.processing_time_ms >= 0
        assert processing_time < 1000  # Should be under 1 second
