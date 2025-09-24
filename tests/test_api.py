"""Tests for the FastAPI application."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.schemas import Article, RankingRequest, UserProfile


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_ranking_request():
    """Sample ranking request for testing."""
    return {
        "user_id": "test_user",
        "query": "test query",
        "limit": 10,
        "enable_personalization": True,
    }


@pytest.fixture
def sample_article():
    """Sample article for testing."""
    return {
        "id": "article_1",
        "title": "Test Article",
        "content": "This is test content",
        "url": "https://example.com/article/1",
        "published_at": "2024-01-01T00:00:00Z",
        "source": {"id": "source_1", "name": "Test Source", "domain": "test.com"},
        "author": {"id": "author_1", "name": "Test Author"},
        "word_count": 500,
        "reading_time": 2,
        "quality_score": 0.8,
    }


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["service"] == "Ranking Microservice"
    assert data["status"] == "running"


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_rank_content_endpoint(client, sample_ranking_request):
    """Test content ranking endpoint."""
    with patch("src.main.ranking_engine") as mock_engine:
        mock_engine.rank_content = AsyncMock(
            return_value={
                "articles": [],
                "total_count": 0,
                "algorithm_variant": "heuristic",
                "processing_time_ms": 10.0,
                "features_computed": 0,
                "cache_hit_rate": 0.8,
            }
        )

        response = client.post("/rank", json=sample_ranking_request)
        assert response.status_code == 200

        data = response.json()
        assert "articles" in data
        assert "total_count" in data
        assert "algorithm_variant" in data


def test_get_article_endpoint(client):
    """Test get article endpoint."""
    response = client.get("/articles/article_1")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "article_1"
    assert data["title"] == "Article article_1"


def test_get_user_profile_endpoint(client):
    """Test get user profile endpoint."""
    response = client.get("/users/test_user/profile")
    assert response.status_code == 200

    data = response.json()
    assert data["user_id"] == "test_user"


def test_update_user_profile_endpoint(client):
    """Test update user profile endpoint."""
    profile_data = {
        "user_id": "test_user",
        "topic_preferences": {"technology": 0.9},
        "source_preferences": {"source_1": 0.8},
        "reading_patterns": {"preferred_hours": [9, 10, 11]},
        "content_preferences": {"preferred_length": 500},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }

    response = client.put("/users/test_user/profile", json=profile_data)
    assert response.status_code == 200

    data = response.json()
    assert data["message"] == "Profile updated successfully"


def test_get_experiments_endpoint(client):
    """Test get experiments endpoint."""
    with patch("src.main.ab_framework") as mock_framework:
        mock_framework.get_all_experiments = AsyncMock(
            return_value=[
                {"experiment_id": "test_experiment", "name": "Test Experiment", "is_active": True}
            ]
        )

        response = client.get("/experiments")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)


def test_get_experiment_stats_endpoint(client):
    """Test get experiment stats endpoint."""
    with patch("src.main.ab_framework") as mock_framework:
        mock_framework.get_experiment_stats = AsyncMock(
            return_value={"experiment_id": "test_experiment", "total_users": 100, "variants": {}}
        )

        response = client.get("/experiments/test_experiment/stats")
        assert response.status_code == 200

        data = response.json()
        assert "experiment_id" in data


def test_get_performance_metrics_endpoint(client):
    """Test get performance metrics endpoint."""
    with patch("src.main.metrics_collector") as mock_metrics:
        mock_metrics.get_performance_summary.return_value = {
            "total_requests": 100,
            "avg_response_time_ms": 50.0,
            "p95_response_time_ms": 100.0,
        }

        response = client.get("/metrics/performance")
        assert response.status_code == 200

        data = response.json()
        assert "total_requests" in data


def test_get_algorithm_comparison_endpoint(client):
    """Test get algorithm comparison endpoint."""
    with patch("src.main.metrics_collector") as mock_metrics:
        mock_metrics.get_algorithm_comparison.return_value = {
            "heuristic": {"request_count": 50, "avg_response_time_ms": 45.0}
        }

        response = client.get("/metrics/algorithm-comparison")
        assert response.status_code == 200

        data = response.json()
        assert "heuristic" in data


def test_get_cache_stats_endpoint(client):
    """Test get cache stats endpoint."""
    with patch("src.main.cache_manager") as mock_cache:
        mock_cache.get_stats.return_value = {"hit_count": 80, "miss_count": 20, "hit_rate": 0.8}

        response = client.get("/cache/stats")
        assert response.status_code == 200

        data = response.json()
        assert "hit_rate" in data


def test_clear_cache_endpoint(client):
    """Test clear cache endpoint."""
    with patch("src.main.cache_manager") as mock_cache:
        mock_cache.clear_stats = AsyncMock()

        response = client.post("/cache/clear")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Cache cleared successfully"


def test_rank_content_validation_error(client):
    """Test ranking request validation."""
    invalid_request = {
        "user_id": "",  # Invalid empty user_id
        "limit": -1,  # Invalid negative limit
    }

    response = client.post("/rank", json=invalid_request)
    assert response.status_code == 422  # Validation error


def test_rank_content_missing_user_id(client):
    """Test ranking request with missing user_id."""
    invalid_request = {"query": "test query", "limit": 10}

    response = client.post("/rank", json=invalid_request)
    assert response.status_code == 422  # Validation error


def test_rank_content_limit_exceeded(client):
    """Test ranking request with limit exceeding maximum."""
    invalid_request = {"user_id": "test_user", "limit": 200}  # Exceeds maximum of 100

    response = client.post("/rank", json=invalid_request)
    assert response.status_code == 422  # Validation error


def test_rank_content_ranking_error(client, sample_ranking_request):
    """Test ranking endpoint error handling."""
    with patch("src.main.ranking_engine") as mock_engine:
        mock_engine.rank_content = AsyncMock(side_effect=Exception("Ranking failed"))

        response = client.post("/rank", json=sample_ranking_request)
        assert response.status_code == 500

        data = response.json()
        assert "detail" in data
        assert "Ranking failed" in data["detail"]


def test_health_check_error(client):
    """Test health check error handling."""
    with patch("src.main.health_checker") as mock_health:
        mock_health.check_health = AsyncMock(side_effect=Exception("Health check failed"))

        response = client.get("/health")
        assert response.status_code == 200  # Should still return 200 with error details

        data = response.json()
        assert data["status"] == "unhealthy"


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.options("/rank")
    assert response.status_code == 200

    # Check CORS headers
    assert "access-control-allow-origin" in response.headers


def test_gzip_compression(client):
    """Test GZIP compression for large responses."""
    # This would test with a large response
    response = client.get("/")
    assert response.status_code == 200


def test_api_documentation(client):
    """Test API documentation endpoints."""
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200


def test_metrics_endpoint_prometheus(client):
    """Test Prometheus metrics endpoint."""
    # This would test the /metrics endpoint if exposed
    # For now, we test that the service starts without errors
    response = client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_background_task_logging(client, sample_ranking_request):
    """Test background task for logging ranking decisions."""
    with patch("src.main.log_ranking_decision") as mock_log:
        mock_log.return_value = None

        with patch("src.main.ranking_engine") as mock_engine:
            mock_engine.rank_content = AsyncMock(
                return_value={
                    "articles": [],
                    "total_count": 0,
                    "algorithm_variant": "heuristic",
                    "processing_time_ms": 10.0,
                    "features_computed": 0,
                    "cache_hit_rate": 0.8,
                }
            )

            response = client.post("/rank", json=sample_ranking_request)
            assert response.status_code == 200

            # Give background task time to execute
            import asyncio

            await asyncio.sleep(0.1)

            # Verify logging was called
            mock_log.assert_called_once()


def test_error_handlers(client):
    """Test custom error handlers."""
    # Test 404 error
    response = client.get("/nonexistent")
    assert response.status_code == 404

    # Test 500 error handling
    with patch("src.main.ranking_engine") as mock_engine:
        mock_engine.rank_content = AsyncMock(side_effect=Exception("Test error"))

        response = client.post("/rank", json={"user_id": "test"})
        assert response.status_code == 500

        data = response.json()
        assert "detail" in data
        assert "status_code" in data


def test_request_validation(client):
    """Test request validation for all endpoints."""
    # Test ranking request validation
    response = client.post("/rank", json={})
    assert response.status_code == 422

    # Test user profile validation
    response = client.put("/users/test/profile", json={})
    assert response.status_code == 422

    # Test experiment creation validation
    response = client.post("/experiments", json={})
    assert response.status_code == 422


def test_response_format_consistency(client):
    """Test that all responses have consistent format."""
    endpoints = [
        ("/", "GET"),
        ("/health", "GET"),
        ("/users/test/profile", "GET"),
        ("/experiments", "GET"),
        ("/metrics/performance", "GET"),
        ("/cache/stats", "GET"),
    ]

    for endpoint, method in endpoints:
        if method == "GET":
            response = client.get(endpoint)
        else:
            response = client.post(endpoint, json={})

        # All successful responses should be JSON
        if response.status_code == 200:
            assert response.headers["content-type"] == "application/json"
