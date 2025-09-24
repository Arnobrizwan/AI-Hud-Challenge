"""
Unit tests for data models.
"""

from datetime import datetime

import pytest

from src.models.content import (
    ContentType,
    IngestionMetadata,
    NormalizedArticle,
    ProcessingBatch,
    ProcessingStatus,
    SourceConfig,
    SourceType,
)


class TestNormalizedArticle:
    """Test cases for NormalizedArticle model."""

    def test_create_article(self):
        """Test creating a valid article."""
        article = NormalizedArticle(
            id="test-article-1",
            url="https://example.com/article",
            title="Test Article",
            source="Test Source",
            source_url="https://example.com",
            published_at=datetime.utcnow(),
            content_hash="abc123",
        )

        assert article.id == "test-article-1"
        assert article.title == "Test Article"
        assert article.content_type == ContentType.ARTICLE
        assert article.processing_status == ProcessingStatus.PENDING

    def test_article_validation(self):
        """Test article validation rules."""
        # Test invalid URL
        with pytest.raises(ValueError):
            NormalizedArticle(
                id="test-article-1",
                url="invalid-url",
                title="Test Article",
                source="Test Source",
                source_url="https://example.com",
                published_at=datetime.utcnow(),
                content_hash="abc123",
            )

        # Test invalid language code
        with pytest.raises(ValueError):
            NormalizedArticle(
                id="test-article-1",
                url="https://example.com/article",
                title="Test Article",
                source="Test Source",
                source_url="https://example.com",
                published_at=datetime.utcnow(),
                content_hash="abc123",
                language="invalid",
            )

    def test_content_hash_calculation(self):
        """Test content hash calculation."""
        article = NormalizedArticle(
            id="test-article-1",
            url="https://example.com/article",
            title="Test Article",
            source="Test Source",
            source_url="https://example.com",
            published_at=datetime.utcnow(),
            content_hash="",
        )

        hash_value = article.calculate_content_hash()
        assert len(hash_value) == 64  # SHA256 hash length
        assert hash_value != ""

    def test_reading_time_calculation(self):
        """Test reading time calculation."""
        article = NormalizedArticle(
            id="test-article-1",
            url="https://example.com/article",
            title="Test Article",
            source="Test Source",
            source_url="https://example.com",
            published_at=datetime.utcnow(),
            content_hash="abc123",
            word_count=400,
        )

        reading_time = article.calculate_reading_time()
        assert reading_time == 2  # 400 words / 200 words per minute

    def test_duplicate_detection(self):
        """Test duplicate detection logic."""
        article1 = NormalizedArticle(
            id="test-article-1",
            url="https://example.com/article",
            title="Test Article",
            source="Test Source",
            source_url="https://example.com",
            published_at=datetime.utcnow(),
            content_hash="abc123",
        )

        article2 = NormalizedArticle(
            id="test-article-2",
            url="https://example.com/article",
            title="Test Article",
            source="Test Source",
            source_url="https://example.com",
            published_at=datetime.utcnow(),
            content_hash="abc123",
        )

        # Same content hash should be duplicate
        assert article1.is_duplicate_of(article2, threshold=0.8)

        # Different content hash should not be duplicate
        article2.content_hash = "def456"
        assert not article1.is_duplicate_of(article2, threshold=0.8)


class TestSourceConfig:
    """Test cases for SourceConfig model."""

    def test_create_source_config(self):
        """Test creating a valid source config."""
        config = SourceConfig(
            id="test-source",
            name="Test Source",
            type=SourceType.RSS_FEED,
            url="https://example.com/feed.xml",
        )

        assert config.id == "test-source"
        assert config.name == "Test Source"
        assert config.type == SourceType.RSS_FEED
        assert config.enabled is True
        assert config.priority == 1

    def test_source_config_validation(self):
        """Test source config validation rules."""
        # Test invalid priority
        with pytest.raises(ValueError):
            SourceConfig(
                id="test-source",
                name="Test Source",
                type=SourceType.RSS_FEED,
                url="https://example.com/feed.xml",
                priority=0,
            )

        # Test invalid rate limit
        with pytest.raises(ValueError):
            SourceConfig(
                id="test-source",
                name="Test Source",
                type=SourceType.RSS_FEED,
                url="https://example.com/feed.xml",
                rate_limit=0,
            )


class TestProcessingBatch:
    """Test cases for ProcessingBatch model."""

    def test_create_processing_batch(self):
        """Test creating a valid processing batch."""
        batch = ProcessingBatch(batch_id="test-batch-1", source_id="test-source", articles=[], total_count=0)

        assert batch.batch_id == "test-batch-1"
        assert batch.source_id == "test-source"
        assert batch.status == ProcessingStatus.PENDING
        assert batch.success_rate == 0.0

    def test_batch_validation(self):
        """Test batch validation rules."""
        # Test invalid total count
        with pytest.raises(ValueError):
            ProcessingBatch(batch_id="test-batch-1", source_id="test-source", articles=[], total_count=-1)

    def test_processing_time_calculation(self):
        """Test processing time calculation."""
        batch = ProcessingBatch(
            batch_id="test-batch-1",
            source_id="test-source",
            articles=[],
            total_count=0,
            started_at=datetime(2023, 1, 1, 10, 0, 0),
            completed_at=datetime(2023, 1, 1, 10, 5, 0),
        )

        assert batch.processing_time_seconds == 300.0  # 5 minutes


class TestIngestionMetadata:
    """Test cases for IngestionMetadata model."""

    def test_create_ingestion_metadata(self):
        """Test creating valid ingestion metadata."""
        metadata = IngestionMetadata(
            source_id="test-source",
            source_type=SourceType.RSS_FEED,
            source_url="https://example.com/feed.xml",
        )

        assert metadata.source_id == "test-source"
        assert metadata.source_type == SourceType.RSS_FEED
        assert metadata.retry_count == 0
        assert metadata.robots_txt_respected is True
