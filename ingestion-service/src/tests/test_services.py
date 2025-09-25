"""
Unit tests for core services.
"""

from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.content import NormalizedArticle, ProcessingStatus, SourceConfig, SourceType
from src.services.ingestion_service import IngestionService


class TestIngestionService:
    """Test cases for IngestionService."""

    @pytest.fixture
    def source_config(self):
        """Create a test source config."""
        return SourceConfig(
            id="test-source",
            name="Test Source",
            type=SourceType.RSS_FEED,
            url="https://example.com/feed.xml",
        )

    @pytest.fixture
    def ingestion_service(self):
        """Create a test ingestion service."""
        with (
            patch("src.services.ingestion_service.HTTPClient"),
            patch("src.services.ingestion_service.ContentParser"),
            patch("src.services.ingestion_service.URLUtils"),
            patch("src.services.ingestion_service.DateUtils"),
        ):
            return IngestionService()

    @pytest.mark.asyncio
    async def test_get_adapter(self, ingestion_service, source_config) -> Dict[str, Any]:
        """Test getting adapter for source."""
        adapter = await ingestion_service._get_adapter(source_config)

        assert adapter is not None
        assert adapter.source_config == source_config
        assert source_config.id in ingestion_service.source_adapters

    @pytest.mark.asyncio
    async def test_get_existing_adapter(self, ingestion_service, source_config) -> Dict[str, Any]:
        """Test getting existing adapter."""
        # Create adapter first
        adapter1 = await ingestion_service._get_adapter(source_config)

        # Get same adapter again
        adapter2 = await ingestion_service._get_adapter(source_config)

        assert adapter1 is adapter2

    @pytest.mark.asyncio
    async def test_process_source(self, ingestion_service, source_config) -> Dict[str, Any]:
    """Test processing a single source."""
        # Mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.fetch_content.return_value = []
        mock_adapter.test_connection.return_value = True

        # Mock the async generator properly
        async def mock_fetch_content():
            yield  # This makes it an async generator

        mock_adapter.fetch_content = mock_fetch_content

        with patch.object(ingestion_service, "_get_adapter", return_value=mock_adapter):
            batch = await ingestion_service.process_source(source_config)

            assert batch is not None
            assert batch.source_id == source_config.id
            assert batch.status == ProcessingStatus.COMPLETED
            assert source_config.id in ingestion_service.active_batches

    @pytest.mark.asyncio
    async def test_process_sources(self, ingestion_service) -> Dict[str, Any]:
    """Test processing multiple sources."""
        source_configs = [
            SourceConfig(
                id="source-1",
                name="Source 1",
                type=SourceType.RSS_FEED,
                url="https://example.com/feed1.xml",
            ),
            SourceConfig(
                id="source-2",
                name="Source 2",
                type=SourceType.RSS_FEED,
                url="https://example.com/feed2.xml",
            ),
        ]

        # Mock adapters
        mock_adapter = AsyncMock()
        mock_adapter.test_connection.return_value = True

        # Mock the async generator properly
        async def mock_fetch_content():
            yield  # This makes it an async generator

        mock_adapter.fetch_content = mock_fetch_content

        with patch.object(ingestion_service, "_get_adapter", return_value=mock_adapter):
            batches = await ingestion_service.process_sources(source_configs)

            assert len(batches) == 2
            assert all(batch.status == ProcessingStatus.COMPLETED for batch in batches)

    @pytest.mark.asyncio
    async def test_process_articles(self, ingestion_service) -> Dict[str, Any]:
    """Test processing articles through normalization and duplicate detection."""
        # Create test articles
        articles = [
            NormalizedArticle(
                id="article-1",
                url="https://example.com/article1",
                title="Test Article 1",
                source="Test Source",
                source_url="https://example.com",
                published_at=datetime.utcnow(),
                content_hash="hash1",
                ingestion_metadata={
                    "source_id": "test-source",
                    "source_type": "rss_feed",
                    "source_url": "https://example.com",
                    "ingestion_time": datetime.utcnow(),
                    "ingestion_method": "test",
                },
            ),
            NormalizedArticle(
                id="article-2",
                url="https://example.com/article2",
                title="Test Article 2",
                source="Test Source",
                source_url="https://example.com",
                published_at=datetime.utcnow(),
                content_hash="hash2",
                ingestion_metadata={
                    "source_id": "test-source",
                    "source_type": "rss_feed",
                    "source_url": "https://example.com",
                    "ingestion_time": datetime.utcnow(),
                    "ingestion_method": "test",
                },
            ),
        ]

        # Mock content normalizer
        mock_normalizer = AsyncMock()
        mock_normalizer.normalize_article.side_effect = lambda article: article

        # Mock duplicate detector
        mock_detector = AsyncMock()
        mock_detector.detect_duplicates.return_value = []

        with (
            patch.object(ingestion_service, "content_normalizer", mock_normalizer),
            patch.object(ingestion_service, "duplicate_detector", mock_detector),
        ):

            processed_articles = await ingestion_service._process_articles(articles)

            assert len(processed_articles) == 2
            assert all(article.processing_status == ProcessingStatus.COMPLETED for article in processed_articles)

    @pytest.mark.asyncio
    async def test_get_source_health(self, ingestion_service, source_config) -> Dict[str, Any]:
    """Test getting source health status."""
        # Create mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.health_check.return_value = {
            "status": "healthy",
            "is_connected": True,
            "error_count": 0,
        }

        ingestion_service.source_adapters[source_config.id] = mock_adapter

        health = await ingestion_service.get_source_health(source_config.id)

        assert health["status"] == "healthy"
        assert health["is_connected"] is True

    @pytest.mark.asyncio
    async def test_get_source_health_not_found(self, ingestion_service) -> Dict[str, Any]:
    """Test getting health for non-existent source."""
        health = await ingestion_service.get_source_health("non-existent")

        assert health["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_get_all_source_health(self, ingestion_service, source_config) -> Dict[str, Any]:
    """Test getting health for all sources."""
        # Create mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.health_check.return_value = {"status": "healthy", "is_connected": True}

        ingestion_service.source_adapters[source_config.id] = mock_adapter

        health_status = await ingestion_service.get_all_source_health()

        assert source_config.id in health_status
        assert health_status[source_config.id]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_processing_metrics(self, ingestion_service, source_config) -> Dict[str, Any]:
    """Test getting processing metrics."""
        # Add some test metrics
        from src.models.content import ContentMetrics

        metrics = ContentMetrics(
            source_id=source_config.id,
            date=datetime.utcnow().date(),
            total_articles=100,
            successful_articles=95,
            failed_articles=5,
            duplicate_articles=10,
        )

        ingestion_service.processing_metrics[source_config.id] = metrics

        # Test getting metrics for specific source
        source_metrics = await ingestion_service.get_processing_metrics(source_config.id)
        assert source_metrics.total_articles == 100

        # Test getting all metrics
        all_metrics = await ingestion_service.get_processing_metrics()
        assert source_config.id in all_metrics

    @pytest.mark.asyncio
    async def test_get_batch_status(self, ingestion_service) -> Dict[str, Any]:
    """Test getting batch status."""
        from src.models.content import ProcessingBatch

        batch = ProcessingBatch(
            batch_id="test-batch",
            source_id="test-source",
            articles=[],
            total_count=10,
            processed_count=8,
            failed_count=2,
        )

        ingestion_service.active_batches["test-batch"] = batch

        retrieved_batch = await ingestion_service.get_batch_status("test-batch")
        assert retrieved_batch == batch

        # Test non-existent batch
        non_existent = await ingestion_service.get_batch_status("non-existent")
        assert non_existent is None

    @pytest.mark.asyncio
    async def test_get_active_batches(self, ingestion_service) -> Dict[str, Any]:
    """Test getting all active batches."""
        from src.models.content import ProcessingBatch

        batch1 = ProcessingBatch(batch_id="batch-1", source_id="source-1", articles=[], total_count=5)

        batch2 = ProcessingBatch(batch_id="batch-2", source_id="source-2", articles=[], total_count=3)

        ingestion_service.active_batches["batch-1"] = batch1
        ingestion_service.active_batches["batch-2"] = batch2

        active_batches = await ingestion_service.get_active_batches()

        assert len(active_batches) == 2
        assert any(batch.batch_id == "batch-1" for batch in active_batches)
        assert any(batch.batch_id == "batch-2" for batch in active_batches)

    @pytest.mark.asyncio
    async def test_cleanup_completed_batches(self, ingestion_service) -> Dict[str, Any]:
    """Test cleaning up completed batches."""
        from src.models.content import ProcessingBatch

        # Create old completed batch
        old_batch = ProcessingBatch(
            batch_id="old-batch",
            source_id="test-source",
            articles=[],
            total_count=5,
            completed_at=datetime.utcnow() - timedelta(hours=2),
        )

        # Create recent completed batch
        recent_batch = ProcessingBatch(
            batch_id="recent-batch",
            source_id="test-source",
            articles=[],
            total_count=3,
            completed_at=datetime.utcnow(),
        )

        ingestion_service.active_batches["old-batch"] = old_batch
        ingestion_service.active_batches["recent-batch"] = recent_batch

        # Clean up batches older than 1 hour
        await ingestion_service.cleanup_completed_batches(max_age_hours=1)

        # Only recent batch should remain
        assert "old-batch" not in ingestion_service.active_batches
        assert "recent-batch" in ingestion_service.active_batches

    @pytest.mark.asyncio
    async def test_test_source_connection(self, ingestion_service, source_config) -> Dict[str, Any]:
    """Test testing source connection."""
        # Mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.test_connection.return_value = True

        with patch.object(ingestion_service, "_get_adapter", return_value=mock_adapter):
            result = await ingestion_service.test_source_connection(source_config)
            assert result is True

    @pytest.mark.asyncio
    async def test_test_source_connection_failure(self, ingestion_service, source_config) -> Dict[str, Any]:
    """Test source connection failure."""
        with patch.object(ingestion_service, "_get_adapter", side_effect=Exception("Connection failed")):
            result = await ingestion_service.test_source_connection(source_config)
            assert result is False

    @pytest.mark.asyncio
    async def test_get_source_info(self, ingestion_service, source_config) -> Dict[str, Any]:
    """Test getting source information."""
        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.get_source_info.return_value = {
            "type": "RSS/Atom Feed",
            "url": source_config.url,
            "enabled": True,
        }

        ingestion_service.source_adapters[source_config.id] = mock_adapter

        info = await ingestion_service.get_source_info(source_config.id)

        assert info["type"] == "RSS/Atom Feed"
        assert info["url"] == source_config.url

    @pytest.mark.asyncio
    async def test_get_source_info_not_found(self, ingestion_service) -> Dict[str, Any]:
    """Test getting info for non-existent source."""
        info = await ingestion_service.get_source_info("non-existent")
        assert info is None

    @pytest.mark.asyncio
    async def test_shutdown(self, ingestion_service) -> Dict[str, Any]:
    """Test service shutdown."""
        # Mock HTTP client
        mock_http_client = AsyncMock()
        ingestion_service.http_client = mock_http_client

        await ingestion_service.shutdown()

        # Verify HTTP client was closed
        mock_http_client.close.assert_called_once()

        # Verify state was cleared
        assert len(ingestion_service.source_adapters) == 0
        assert len(ingestion_service.active_batches) == 0
        assert len(ingestion_service.processing_metrics) == 0
