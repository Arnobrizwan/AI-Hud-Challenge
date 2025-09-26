"""
Simple unit tests for data models.
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
)


class TestMetadataInfo:
    """Test MetadataInfo model."""

    def test_valid_metadata(self):
        """Test valid metadata creation."""
        metadata = MetadataInfo(
            author="Test Author",
            published_date="2023-01-01T00:00:00Z",
            tags=["test", "example"],
            description="Test description"
        )
        assert metadata.author == "Test Author"
        assert metadata.tags == ["test", "example"]
        assert metadata.description == "Test description"

    def test_metadata_with_defaults(self):
        """Test metadata with default values."""
        metadata = MetadataInfo()
        assert metadata.tags == []
        assert metadata.categories == []
        assert metadata.keywords == []


class TestExtractedContent:
    """Test ExtractedContent model."""

    def test_valid_content(self):
        """Test valid content creation."""
        content = ExtractedContent(
            content_id="test-123",
            title="Test Title",
            content="Test content here",
            language="en",
            content_type=ContentType.ARTICLE,
            word_count=3,
            reading_time=1,
            quality_score=0.8,
            status=ProcessingStatus.COMPLETED
        )
        assert content.content_id == "test-123"
        assert content.title == "Test Title"
        assert content.content_type == ContentType.ARTICLE
        assert content.status == ProcessingStatus.COMPLETED

    def test_content_validation(self):
        """Test content validation."""
        with pytest.raises(ValidationError):
            ExtractedContent(
                content_id="test-123",
                title="Test Title",
                content="Test content here",
                language="en",
                content_type=ContentType.ARTICLE,
                word_count=3,
                reading_time=1,
                quality_score=0.8,
                # Missing required field: status
            )


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
            flesch_kincaid_grade=8.5,
            sentiment_score=0.2,
            language_confidence=0.95
        )
        assert analysis.content_id == "test-123"
        assert analysis.quality_score == 0.85
        assert analysis.word_count == 100


class TestImageInfo:
    """Test ImageInfo model."""

    def test_valid_image_info(self):
        """Test valid image info creation."""
        image = ImageInfo(
            url="https://example.com/image.jpg",
            alt_text="Test image",
            width=800,
            height=600
        )
        assert image.url == "https://example.com/image.jpg"
        assert image.alt_text == "Test image"
        assert image.width == 800
        assert image.height == 600
