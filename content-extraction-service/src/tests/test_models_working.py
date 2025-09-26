"""
Working unit tests for data models.
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
            published_date="2023-01-01T00:00:00Z",
            tags=["tag1", "tag2"],
            categories=["category1"],
            keywords=["keyword1"],
            description="Test Description",
            canonical_url="https://example.com/canonical"
        )
        assert metadata.author == "Test Author"
        assert "tag1" in metadata.tags

    def test_metadata_with_defaults(self):
        """Test metadata with default values."""
        metadata = MetadataInfo()
        assert metadata.tags == []
        assert metadata.categories == []


class TestQualityAnalysis:
    """Test QualityAnalysis model."""

    def test_valid_quality_analysis(self):
        """Test valid quality analysis creation."""
        analysis = QualityAnalysis(
            content_id="123",
            quality_score=0.85,
            readability_score=70.5,
            word_count=100,
            sentence_count=10,
            paragraph_count=5,
            avg_sentence_length=10.0,
            flesch_kincaid_grade=8.0,
            sentiment_score=0.1,
            language_confidence=0.99
        )
        assert analysis.quality_score == 0.85
        assert analysis.word_count == 100

    def test_quality_analysis_validation(self):
        """Test quality analysis validation."""
        # Test that valid values work
        analysis = QualityAnalysis(
            content_id="123",
            quality_score=0.85,
            readability_score=70.5,
            word_count=100,
            sentence_count=10,
            paragraph_count=5,
            avg_sentence_length=10.0,
            flesch_kincaid_grade=8.0,
            sentiment_score=0.1,
            language_confidence=0.99
        )
        assert analysis.quality_score == 0.85


class TestExtractedContent:
    """Test ExtractedContent model."""

    def test_valid_content(self):
        """Test valid content creation."""
        content = ExtractedContent(
            content_id="123",
            title="Test Title",
            content="This is some test content.",
            language="en",
            content_type=ContentType.ARTICLE,
            word_count=5,
            reading_time=1,
            quality_score=0.9,
            status=ProcessingStatus.COMPLETED
        )
        assert content.title == "Test Title"
        assert content.content_type == ContentType.ARTICLE

    def test_content_validation(self):
        """Test content validation."""
        with pytest.raises(ValidationError):
            ExtractedContent(
                content_id="123",
                title="Title",
                content="Content",
                language="en",
                content_type="INVALID",  # Invalid content type
                word_count=1,
                reading_time=1,
                quality_score=0.5,
                status=ProcessingStatus.PENDING
            )


class TestImageInfo:
    """Test ImageInfo model."""

    def test_valid_image_info(self):
        """Test valid image info creation."""
        image = ImageInfo(
            url="http://example.com/image.jpg",
            alt_text="A test image",
            caption="Test caption",
            width=800,
            height=600
        )
        assert image.url == "http://example.com/image.jpg"
        assert image.alt_text == "A test image"

    def test_image_info_with_defaults(self):
        """Test image info with default values."""
        image = ImageInfo(url="http://example.com/image.jpg")
        assert image.url == "http://example.com/image.jpg"
        assert image.alt_text is None


class TestProcessingStatus:
    """Test ProcessingStatus enum."""

    def test_processing_status_values(self):
        """Test processing status enum values."""
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.PROCESSING == "processing"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"
        assert ProcessingStatus.CACHED == "cached"


class TestContentType:
    """Test ContentType enum."""

    def test_content_type_values(self):
        """Test content type enum values."""
        assert ContentType.ARTICLE == "article"
        assert ContentType.BLOG_POST == "blog_post"
        assert ContentType.NEWS == "news"
        assert ContentType.PRESS_RELEASE == "press_release"
        assert ContentType.OPINION == "opinion"
        assert ContentType.REVIEW == "review"
        assert ContentType.TUTORIAL == "tutorial"
        assert ContentType.DOCUMENTATION == "documentation"
        assert ContentType.OTHER == "other"
