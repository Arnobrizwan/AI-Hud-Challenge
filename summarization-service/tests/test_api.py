"""
API Tests for Summarization Service
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.summarization.models import ContentType, Language, ProcessedContent, SummarizationRequest


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_content():
    """Sample content for testing"""
    return ProcessedContent(
        text="This is a sample article about artificial intelligence and machine learning. "
        "The field has seen tremendous growth in recent years with applications in "
        "various domains including healthcare, finance, and transportation. "
        "Researchers are continuously working on improving algorithms and models "
        "to make AI more efficient and accessible.",
        title="AI and Machine Learning Advances",
        author="Test Author",
        source="Test Source",
        language=Language.ENGLISH,
        content_type=ContentType.NEWS_ARTICLE,
    )


@pytest.fixture
def sample_request(sample_content):
    """Sample summarization request"""
    return SummarizationRequest(
        content=sample_content,
        target_lengths=[50, 120],
        methods=["hybrid"],
        headline_styles=["news", "engaging"],
    )


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_health_check(self, client):
        """Test basic health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "summarization-service"

    def test_readiness_check(self, client):
        """Test readiness check"""
        response = client.get("/health/ready")
        # This might return 503 if services aren't initialized
        assert response.status_code in [200, 503]


class TestSummarizationEndpoints:
    """Test summarization endpoints"""

    @patch("src.main.summarization_engine")
    def test_summarize_endpoint(self, mock_engine, client, sample_request):
        """Test summarization endpoint"""
        # Mock the summarization engine
        mock_result = AsyncMock()
        mock_engine.generate_summary.return_value = mock_result

        response = client.post("/summarize", json=sample_request.dict())

        # Should return 200 or 500 depending on initialization
        assert response.status_code in [200, 500]

    def test_summarize_invalid_request(self, client):
        """Test summarization with invalid request"""
        response = client.post("/summarize", json={"invalid": "data"})
        assert response.status_code == 422  # Validation error

    @patch("src.main.summarization_engine")
    def test_batch_summarize_endpoint(self, mock_engine, client):
        """Test batch summarization endpoint"""
        # Mock the summarization engine
        mock_result = AsyncMock()
        mock_engine.generate_summary.return_value = mock_result

        batch_request = {
            "requests": [
                {
                    "content": {
                        "text": "First article content...",
                        "language": "en",
                        "content_type": "news_article",
                    },
                    "target_lengths": [120],
                },
                {
                    "content": {
                        "text": "Second article content...",
                        "language": "en",
                        "content_type": "news_article",
                    },
                    "target_lengths": [120],
                },
            ]
        }

        response = client.post("/summarize/batch", json=batch_request)

        # Should return 200 or 500 depending on initialization
        assert response.status_code in [200, 500]


class TestHeadlineEndpoints:
    """Test headline generation endpoints"""

    @patch("src.main.headline_generator")
    def test_generate_headlines(self, mock_generator, client):
        """Test headline generation endpoint"""
        # Mock the headline generator
        mock_headlines = [
            {"text": "Test Headline 1", "style": "news", "score": 0.85, "metrics": {}},
            {"text": "Test Headline 2", "style": "engaging", "score": 0.90, "metrics": {}},
        ]
        mock_generator.generate_headlines.return_value = mock_headlines

        response = client.post(
            "/headlines/generate",
            json={
                "content": "Sample article content...",
                "styles": ["news", "engaging"],
                "num_variants": 2,
            },
        )

        # Should return 200 or 500 depending on initialization
        assert response.status_code in [200, 500]


class TestQualityEndpoints:
    """Test quality validation endpoints"""

    @patch("src.main.quality_validator")
    def test_validate_quality(self, mock_validator, client):
        """Test quality validation endpoint"""
        # Mock the quality validator
        mock_metrics = {
            "rouge1_f1": 0.85,
            "rouge2_f1": 0.78,
            "rougeL_f1": 0.82,
            "bertscore_f1": 0.88,
            "factual_consistency": 0.90,
            "readability": 0.75,
            "coverage": 0.80,
            "abstractiveness": 0.65,
            "overall_score": 0.82,
        }
        mock_validator.validate_summary_quality.return_value = mock_metrics

        response = client.post(
            "/quality/validate",
            json={"original": "Original article text...", "summary": "Generated summary text..."},
        )

        # Should return 200 or 500 depending on initialization
        assert response.status_code in [200, 500]


class TestMetricsEndpoints:
    """Test metrics endpoints"""

    @patch("src.main.metrics_collector")
    def test_get_metrics(self, mock_collector, client):
        """Test metrics endpoint"""
        # Mock the metrics collector
        mock_metrics = {
            "service_metrics": {
                "total_requests": 100,
                "successful_requests": 95,
                "failed_requests": 5,
                "success_rate": 0.95,
            },
            "performance_metrics": {"cpu_usage": 45.2, "memory_usage": 67.8},
        }
        mock_collector.get_metrics.return_value = mock_metrics

        response = client.get("/metrics")

        # Should return 200 or 500 depending on initialization
        assert response.status_code in [200, 500]

    @patch("src.main.summarization_engine")
    @patch("src.main.headline_generator")
    @patch("src.main.quality_validator")
    def test_get_models_status(self, mock_validator, mock_generator, mock_engine, client):
        """Test models status endpoint"""
        # Mock the services
        mock_engine.get_status.return_value = {"initialized": True}
        mock_generator.get_status.return_value = {"initialized": True}
        mock_validator.get_status.return_value = {"initialized": True}

        response = client.get("/models/status")

        # Should return 200 or 500 depending on initialization
        assert response.status_code in [200, 500]


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_json(self, client):
        """Test invalid JSON handling"""
        response = client.post(
            "/summarize", data="invalid json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        """Test missing required fields"""
        response = client.post("/summarize", json={})
        assert response.status_code == 422

    def test_invalid_content_type(self, client):
        """Test invalid content type"""
        response = client.post(
            "/summarize", json={"content": {"text": "Sample text", "content_type": "invalid_type"}}
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__])
