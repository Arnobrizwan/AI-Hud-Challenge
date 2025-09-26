"""
Unit tests for data models.
"""

import pytest
from pydantic import ValidationError

from models.content import (
    MetadataInfo,
    ContentType,
    ExtractedContent,
    ImageInfo,
    QualityAnalysis,
    ProcessingStatus,
    CacheStats,
    ProcessingMetrics,
)


class TestMetadataInfo:
    """Test MetadataInfo model."""

    def test_valid_metadata(self):
        """Test valid metadata creation."""
        metadata = MetadataInfo(
            author="Test Author",
            og_title="Test Title",
            og_description="Test Description",
            og_image="https://example.com/image.jpg",
            tags=["tag1", "tag2"],
            categories=["category1"],
        )

        assert metadata.author == "Test Author"
        assert metadata.og_title == "Test Title"
        assert metadata.og_description == "Test Description"
        assert metadata.og_image == "https://example.com/image.jpg"
        assert metadata.tags == ["tag1", "tag2"]
        assert metadata.categories == ["category1"]

    def test_empty_metadata(self):
        """Test empty metadata creation."""
        metadata = MetadataInfo()
        assert metadata.og_title is None
        assert metadata.og_description is None
        assert metadata.tags == []
        assert metadata.categories == []


class TestQualityAnalysis:
    """Test QualityAnalysis model."""

    def test_valid_quality_analysis(self):
        """Test valid quality analysis creation."""
        analysis = QualityAnalysis(
            content_id="test-123",
            quality_score=0.85,
            readability_score=0.75,
            word_count=100,
            sentence_count=10,
            paragraph_count=5,
            avg_sentence_length=10.0,
            flesch_kincaid_grade=8.0,
            sentiment_score=0.1,
            language_confidence=0.99,
        )

        assert analysis.content_id == "test-123"
        assert analysis.quality_score == 0.85
        assert analysis.readability_score == 0.75
        assert analysis.word_count == 100
        assert analysis.sentence_count == 10
        assert analysis.paragraph_count == 5
        assert analysis.avg_sentence_length == 10.0
        assert analysis.flesch_kincaid_grade == 8.0
        assert analysis.sentiment_score == 0.1
        assert analysis.language_confidence == 0.99

    def test_invalid_scores(self):
        """Test invalid score validation."""
        with pytest.raises(ValidationError):
            QualityAnalysis(
                content_id="test-123",
                quality_score=1.5,  # Invalid: > 1.0
                readability_score=0.75,
                word_count=100,
                sentence_count=10,
                paragraph_count=5,
                avg_sentence_length=10.0,
                flesch_kincaid_grade=8.0,
                sentiment_score=0.1,
                language_confidence=0.99,
            )


class TestProcessingMetrics:
    """Test ProcessingMetrics model."""

    def test_valid_processing_metrics(self):
        """Test valid processing metrics creation."""
        metrics = ProcessingMetrics(
            total_extractions=100,
            successful_extractions=95,
            failed_extractions=5,
            avg_processing_time_ms=500.0,
            cache_hit_rate=0.8,
            quality_scores={"high": 0.9, "medium": 0.7, "low": 0.5},
            content_types={"article": 50, "blog": 30, "news": 20},
            languages={"en": 80, "es": 15, "fr": 5},
        )

        assert metrics.total_extractions == 100
        assert metrics.successful_extractions == 95
        assert metrics.failed_extractions == 5
        assert metrics.avg_processing_time_ms == 500.0
        assert metrics.cache_hit_rate == 0.8
        assert metrics.quality_scores == {"high": 0.9, "medium": 0.7, "low": 0.5}
        assert metrics.content_types == {"article": 50, "blog": 30, "news": 20}
        assert metrics.languages == {"en": 80, "es": 15, "fr": 5}

    def test_invalid_metrics(self):
        """Test invalid metrics validation."""
        with pytest.raises(ValidationError):
            ProcessingMetrics(
                total_extractions=-1,  # Invalid: negative
                successful_extractions=95,
                failed_extractions=5,
                avg_processing_time_ms=500.0,
                cache_hit_rate=0.8,
                quality_scores={},
                content_types={},
                languages={},
            )


class TestImageInfo:
    """Test ImageInfo model."""

    def test_valid_image_info(self):
        """Test valid image info creation."""
        image = ImageInfo(
            url="https://example.com/image.jpg",
            width=800,
            height=600,
            file_size=1024000,
            format="JPEG",
            alt_text="Test image",
            caption="Test caption",
        )

        assert image.url == "https://example.com/image.jpg"
        assert image.width == 800
        assert image.height == 600
        assert image.file_size == 1024000
        assert image.format == "JPEG"
        assert image.alt_text == "Test image"
        assert image.caption == "Test caption"

    def test_minimal_image_info(self):
        """Test minimal image info creation."""
        image = ImageInfo(url="https://example.com/image.jpg")
        assert image.url == "https://example.com/image.jpg"
        assert image.alt_text is None
        assert image.caption is None


class TestExtractedContent:
    """Test ExtractedContent model."""

    def test_valid_extracted_content(self):
        """Test valid extracted content creation."""
        metadata = MetadataInfo(og_title="Test Title")
        
        content = ExtractedContent(
            content_id="test-123",
            url="https://example.com/article",
            title="Test Article",
            content="This is test content.",
            summary="Test summary",
            language="en",
            content_type=ContentType.ARTICLE,
            word_count=500,
            reading_time=3,
            quality_score=0.85,
            metadata=metadata,
            status=ProcessingStatus.COMPLETED,
        )

        assert content.content_id == "test-123"
        assert content.url == "https://example.com/article"
        assert content.title == "Test Article"
        assert content.word_count == 500
        assert content.reading_time == 3

    def test_content_validation(self):
        """Test content validation."""
        with pytest.raises(ValidationError):
            ExtractedContent(
                content_id="test-123",
                title="Test Article",
                content="This is test content.",
                language="en",
                content_type=ContentType.ARTICLE,
                word_count=500,
                reading_time=3,
                quality_score=1.5,  # Invalid: > 1.0
                status=ProcessingStatus.COMPLETED,
            )

    def test_content_type_enum(self):
        """Test content type enum values."""
        content = ExtractedContent(
            content_id="test-123",
            title="Test Article",
            content="This is test content.",
            language="en",
            content_type=ContentType.NEWS,
            word_count=100,
            reading_time=1,
            quality_score=0.8,
            status=ProcessingStatus.COMPLETED,
        )
        assert content.content_type == ContentType.NEWS

    def test_processing_status_enum(self):
        """Test processing status enum values."""
        content = ExtractedContent(
            content_id="test-123",
            title="Test Article",
            content="This is test content.",
            language="en",
            content_type=ContentType.ARTICLE,
            word_count=100,
            reading_time=1,
            quality_score=0.8,
            status=ProcessingStatus.PENDING,
        )
        assert content.status == ProcessingStatus.PENDING


class TestCacheStats:
    """Test CacheStats model."""

    def test_valid_cache_stats(self):
        """Test valid cache stats creation."""
        stats = CacheStats(
            hit_count=100,
            miss_count=20,
            total_requests=120,
            hit_rate=0.833,
            cache_size=1024,
            memory_usage=512,
        )

        assert stats.hit_count == 100
        assert stats.miss_count == 20
        assert stats.total_requests == 120
        assert stats.hit_rate == 0.833
        assert stats.cache_size == 1024
        assert stats.memory_usage == 512
