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
    """Test ContentMetadata model."""

    def test_valid_metadata(self):
        """Test valid metadata creation."""
        metadata = MetadataInfo(
            og_title="Test Title",
            og_description="Test Description",
            og_image="https://example.com/image.jpg",
            site_name="Test Site",
        )

        assert metadata.og_title == "Test Title"
        assert metadata.og_description == "Test Description"
        assert metadata.og_image == "https://example.com/image.jpg"
        assert metadata.site_name == "Test Site"

    def test_empty_metadata(self):
        """Test empty metadata creation."""
        metadata = MetadataInfo()
        assert metadata.og_title is None
        assert metadata.og_description is None
        assert metadata.json_ld == {}


class TestQualityMetrics:
    """Test QualityMetrics model."""

    def test_valid_quality_metrics(self):
        """Test valid quality metrics creation."""
        metrics = QualityAnalysis(
            readability_score=75.0,
            word_count=500,
            character_count=2500,
            sentence_count=25,
            paragraph_count=5,
            average_sentence_length=20.0,
            average_word_length=5.0,
            image_to_text_ratio=0.1,
            link_density=0.05,
            spam_score=10.0,
            duplicate_score=5.0,
            content_freshness=80.0,
            overall_quality=70.0,
        )

        assert metrics.readability_score == 75.0
        assert metrics.word_count == 500
        assert metrics.overall_quality == 70.0

    def test_invalid_scores(self):
        """Test invalid score validation."""
        with pytest.raises(ValidationError):
            QualityMetrics(
                readability_score=150.0,  # Invalid: > 100
                word_count=500,
                character_count=2500,
                sentence_count=25,
                paragraph_count=5,
                average_sentence_length=20.0,
                average_word_length=5.0,
                image_to_text_ratio=0.1,
                link_density=0.05,
                spam_score=10.0,
                duplicate_score=5.0,
                content_freshness=80.0,
                overall_quality=70.0,
            )


class TestLanguageInfo:
    """Test LanguageInfo model."""

    def test_valid_language_info(self):
        """Test valid language info creation."""
        lang_info = LanguageInfo(detected_language="en", confidence=0.95, charset="utf-8", is_reliable=True)

        assert lang_info.detected_language == "en"
        assert lang_info.confidence == 0.95
        assert lang_info.is_reliable is True

    def test_invalid_confidence(self):
        """Test invalid confidence validation."""
        with pytest.raises(ValidationError):
            LanguageInfo(detected_language="en", confidence=1.5, charset="utf-8")  # Invalid: > 1.0


class TestProcessedImage:
    """Test ProcessedImage model."""

    def test_valid_processed_image(self):
        """Test valid processed image creation."""
        image = ProcessedImage(
            url="https://example.com/image.jpg",
            width=800,
            height=600,
            file_size=1024000,
            format="JPEG",
            alt_text="Test image",
            is_optimized=True,
            quality_score=0.85,
        )

        assert image.url == "https://example.com/image.jpg"
        assert image.width == 800
        assert image.height == 600
        assert image.quality_score == 0.85

    def test_invalid_quality_score(self):
        """Test invalid quality score validation."""
        with pytest.raises(ValidationError):
            ProcessedImage(
                url="https://example.com/image.jpg",
                width=800,
                height=600,
                file_size=1024000,
                format="JPEG",
                quality_score=1.5,  # Invalid: > 1.0
            )


class TestExtractedContent:
    """Test ExtractedContent model."""

    def test_valid_extracted_content(self):
        """Test valid extracted content creation."""
        metadata = MetadataInfo(og_title="Test Title")
        quality_metrics = QualityMetrics(
            readability_score=75.0,
            word_count=500,
            character_count=2500,
            sentence_count=25,
            paragraph_count=5,
            average_sentence_length=20.0,
            average_word_length=5.0,
            image_to_text_ratio=0.1,
            link_density=0.05,
            spam_score=10.0,
            duplicate_score=5.0,
            content_freshness=80.0,
            overall_quality=70.0,
        )
        language_info = LanguageInfo(detected_language="en", confidence=0.95)
        extraction_stats = ExtractionStats(
            extraction_time_ms=5000,
            html_fetch_time_ms=1000,
            content_extraction_time_ms=2000,
            image_processing_time_ms=1000,
            quality_analysis_time_ms=1000,
            total_processing_time_ms=5000,
            bytes_processed=50000,
            images_processed=3,
            links_found=10,
            scripts_removed=5,
            styles_removed=2,
            ads_removed=1,
        )

        content = ExtractedContent(
            url="https://example.com/article",
            title="Test Article",
            content="This is test content.",
            metadata=metadata,
            quality_metrics=quality_metrics,
            language_info=language_info,
            word_count=500,
            reading_time=3,
            extraction_method=ExtractionMethod.READABILITY,
            content_type=ContentType.HTML,
            content_hash="abc123",
            extraction_stats=extraction_stats,
        )

        assert content.url == "https://example.com/article"
        assert content.title == "Test Article"
        assert content.word_count == 500
        assert content.reading_time == 3

    def test_content_hash_calculation(self):
        """Test content hash calculation."""
        metadata = MetadataInfo(og_title="Test Title")
        quality_metrics = QualityMetrics(
            readability_score=75.0,
            word_count=500,
            character_count=2500,
            sentence_count=25,
            paragraph_count=5,
            average_sentence_length=20.0,
            average_word_length=5.0,
            image_to_text_ratio=0.1,
            link_density=0.05,
            spam_score=10.0,
            duplicate_score=5.0,
            content_freshness=80.0,
            overall_quality=70.0,
        )
        language_info = LanguageInfo(detected_language="en", confidence=0.95)
        extraction_stats = ExtractionStats(
            extraction_time_ms=5000,
            html_fetch_time_ms=1000,
            content_extraction_time_ms=2000,
            image_processing_time_ms=1000,
            quality_analysis_time_ms=1000,
            total_processing_time_ms=5000,
            bytes_processed=50000,
            images_processed=3,
            links_found=10,
            scripts_removed=5,
            styles_removed=2,
            ads_removed=1,
        )

        content = ExtractedContent(
            url="https://example.com/article",
            title="Test Article",
            content="This is test content.",
            metadata=metadata,
            quality_metrics=quality_metrics,
            language_info=language_info,
            word_count=500,
            reading_time=3,
            extraction_method=ExtractionMethod.READABILITY,
            content_type=ContentType.HTML,
            content_hash="",
            extraction_stats=extraction_stats,
        )

        calculated_hash = content.calculate_content_hash()
        assert calculated_hash is not None
        assert len(calculated_hash) == 64  # SHA256 hash length

    def test_reading_time_calculation(self):
        """Test reading time calculation."""
        metadata = MetadataInfo()
        quality_metrics = QualityMetrics(
            readability_score=75.0,
            word_count=400,  # 400 words = 2 minutes at 200 wpm
            character_count=2000,
            sentence_count=20,
            paragraph_count=4,
            average_sentence_length=20.0,
            average_word_length=5.0,
            image_to_text_ratio=0.1,
            link_density=0.05,
            spam_score=10.0,
            duplicate_score=5.0,
            content_freshness=80.0,
            overall_quality=70.0,
        )
        language_info = LanguageInfo(detected_language="en", confidence=0.95)
        extraction_stats = ExtractionStats(
            extraction_time_ms=5000,
            html_fetch_time_ms=1000,
            content_extraction_time_ms=2000,
            image_processing_time_ms=1000,
            quality_analysis_time_ms=1000,
            total_processing_time_ms=5000,
            bytes_processed=50000,
            images_processed=3,
            links_found=10,
            scripts_removed=5,
            styles_removed=2,
            ads_removed=1,
        )

        content = ExtractedContent(
            url="https://example.com/article",
            title="Test Article",
            content="This is test content.",
            metadata=metadata,
            quality_metrics=quality_metrics,
            language_info=language_info,
            word_count=400,
            reading_time=0,  # Will be calculated
            extraction_method=ExtractionMethod.READABILITY,
            content_type=ContentType.HTML,
            content_hash="abc123",
            extraction_stats=extraction_stats,
        )

        calculated_reading_time = content.calculate_reading_time()
        assert calculated_reading_time == 2  # 400 words / 200 wpm = 2 minutes

    def test_high_quality_check(self):
        """Test high quality content check."""
        metadata = MetadataInfo()
        quality_metrics = QualityMetrics(
            readability_score=80.0,
            word_count=500,
            character_count=2500,
            sentence_count=25,
            paragraph_count=5,
            average_sentence_length=20.0,
            average_word_length=5.0,
            image_to_text_ratio=0.1,
            link_density=0.05,
            spam_score=20.0,  # Low spam score
            duplicate_score=10.0,  # Low duplicate score
            content_freshness=80.0,
            overall_quality=85.0,  # High quality
        )
        language_info = LanguageInfo(detected_language="en", confidence=0.95)
        extraction_stats = ExtractionStats(
            extraction_time_ms=5000,
            html_fetch_time_ms=1000,
            content_extraction_time_ms=2000,
            image_processing_time_ms=1000,
            quality_analysis_time_ms=1000,
            total_processing_time_ms=5000,
            bytes_processed=50000,
            images_processed=3,
            links_found=10,
            scripts_removed=5,
            styles_removed=2,
            ads_removed=1,
        )

        content = ExtractedContent(
            url="https://example.com/article",
            title="Test Article",
            content="This is test content.",
            metadata=metadata,
            quality_metrics=quality_metrics,
            language_info=language_info,
            word_count=500,
            reading_time=3,
            extraction_method=ExtractionMethod.READABILITY,
            content_type=ContentType.HTML,
            content_hash="abc123",
            extraction_stats=extraction_stats,
        )

        assert content.is_high_quality() is True
        assert content.is_high_quality(min_quality_score=0.8) is True
        assert content.is_high_quality(min_quality_score=0.9) is False

    def test_duplicate_detection(self):
        """Test duplicate content detection."""
        metadata = MetadataInfo()
        quality_metrics = QualityMetrics(
            readability_score=75.0,
            word_count=500,
            character_count=2500,
            sentence_count=25,
            paragraph_count=5,
            average_sentence_length=20.0,
            average_word_length=5.0,
            image_to_text_ratio=0.1,
            link_density=0.05,
            spam_score=10.0,
            duplicate_score=5.0,
            content_freshness=80.0,
            overall_quality=70.0,
        )
        language_info = LanguageInfo(detected_language="en", confidence=0.95)
        extraction_stats = ExtractionStats(
            extraction_time_ms=5000,
            html_fetch_time_ms=1000,
            content_extraction_time_ms=2000,
            image_processing_time_ms=1000,
            quality_analysis_time_ms=1000,
            total_processing_time_ms=5000,
            bytes_processed=50000,
            images_processed=3,
            links_found=10,
            scripts_removed=5,
            styles_removed=2,
            ads_removed=1,
        )

        content1 = ExtractedContent(
            url="https://example.com/article1",
            title="Test Article",
            content="This is test content for article 1.",
            metadata=metadata,
            quality_metrics=quality_metrics,
            language_info=language_info,
            word_count=500,
            reading_time=3,
            extraction_method=ExtractionMethod.READABILITY,
            content_type=ContentType.HTML,
            content_hash="abc123",
            extraction_stats=extraction_stats,
        )

        content2 = ExtractedContent(
            url="https://example.com/article2",
            title="Test Article",
            content="This is test content for article 2.",
            metadata=metadata,
            quality_metrics=quality_metrics,
            language_info=language_info,
            word_count=500,
            reading_time=3,
            extraction_method=ExtractionMethod.READABILITY,
            content_type=ContentType.HTML,
            content_hash="def456",
            extraction_stats=extraction_stats,
        )

        # Same content should be detected as duplicate
        content1.content_hash = "same_hash"
        content2.content_hash = "same_hash"
        assert content1.is_duplicate_of(content2) is True

        # Different content should not be detected as duplicate
        content1.content_hash = "hash1"
        content2.content_hash = "hash2"
        assert content1.is_duplicate_of(content2) is False
