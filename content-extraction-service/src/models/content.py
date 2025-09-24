"""
Content models for extraction and cleanup.
"""

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ContentType(str, Enum):
    """Content type enumeration."""

    HTML = "html"
    PDF = "pdf"
    DOC = "doc"
    DOCX = "docx"
    TXT = "txt"
    RTF = "rtf"
    ODT = "odt"
    XLS = "xls"
    XLSX = "xlsx"
    PPT = "ppt"
    PPTX = "pptx"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    """Extraction method enumeration."""

    PLAYWRIGHT = "playwright"
    BEAUTIFULSOUP = "beautifulsoup"
    READABILITY = "readability"
    NEWSPAPER = "newspaper"
    DOCUMENT_AI = "document_ai"
    PYPDF2 = "pypdf2"
    PDFPLUMBER = "pdfplumber"
    PYTHON_DOCX = "python_docx"
    MANUAL = "manual"


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""

    PENDING = "pending"
    EXTRACTING = "extracting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
    SKIPPED = "skipped"


class LanguageInfo(BaseModel):
    """Language detection and analysis information."""

    detected_language: str = Field(..., description="Detected language code")
    confidence: float = Field(..., description="Detection confidence score")
    alternative_languages: List[Dict[str, float]] = Field(
        default_factory=list, description="Alternative language candidates"
    )
    charset: str = Field(default="utf-8", description="Detected character encoding")
    is_reliable: bool = Field(default=True, description="Whether detection is reliable")

    @validator("confidence")
    def validate_confidence(cls, v):
        """Validate confidence score."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class ProcessedImage(BaseModel):
    """Processed image information."""

    url: str = Field(..., description="Original image URL")
    local_path: Optional[str] = Field(None, description="Local cached path")
    optimized_url: Optional[str] = Field(None, description="Optimized image URL")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Image format (JPEG, PNG, WebP)")
    alt_text: Optional[str] = Field(None, description="Alt text")
    caption: Optional[str] = Field(None, description="Image caption")
    is_optimized: bool = Field(default=False, description="Whether image is optimized")
    quality_score: float = Field(default=0.0, description="Image quality score")

    @validator("quality_score")
    def validate_quality_score(cls, v):
        """Validate quality score."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")
        return v


class VideoMetadata(BaseModel):
    """Video metadata information."""

    url: str = Field(..., description="Video URL")
    title: Optional[str] = Field(None, description="Video title")
    duration: Optional[int] = Field(None, description="Duration in seconds")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    width: Optional[int] = Field(None, description="Video width")
    height: Optional[int] = Field(None, description="Video height")
    format: Optional[str] = Field(None, description="Video format")
    file_size: Optional[int] = Field(None, description="File size in bytes")


class ContentMetadata(BaseModel):
    """Comprehensive content metadata."""

    og_title: Optional[str] = Field(None, description="Open Graph title")
    og_description: Optional[str] = Field(None, description="Open Graph description")
    og_image: Optional[str] = Field(None, description="Open Graph image URL")
    og_type: Optional[str] = Field(None, description="Open Graph type")
    og_url: Optional[str] = Field(None, description="Open Graph URL")
    twitter_card: Optional[str] = Field(None, description="Twitter card type")
    twitter_title: Optional[str] = Field(None, description="Twitter title")
    twitter_description: Optional[str] = Field(None, description="Twitter description")
    twitter_image: Optional[str] = Field(None, description="Twitter image URL")
    json_ld: Dict[str, Any] = Field(default_factory=dict, description="JSON-LD structured data")
    canonical_link: Optional[str] = Field(None, description="Canonical URL")
    amp_url: Optional[str] = Field(None, description="AMP URL")
    rss_feed: Optional[str] = Field(None, description="RSS feed URL")
    favicon: Optional[str] = Field(None, description="Favicon URL")
    site_name: Optional[str] = Field(None, description="Site name")
    robots: Optional[str] = Field(None, description="Robots meta tag")
    viewport: Optional[str] = Field(None, description="Viewport meta tag")
    charset: Optional[str] = Field(None, description="Character encoding")
    generator: Optional[str] = Field(None, description="Site generator")
    author: Optional[str] = Field(None, description="Content author")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    categories: List[str] = Field(default_factory=list, description="Categories")
    tags: List[str] = Field(default_factory=list, description="Tags")


class QualityMetrics(BaseModel):
    """Content quality metrics."""

    readability_score: float = Field(..., description="Flesch-Kincaid readability score")
    word_count: int = Field(..., description="Word count")
    character_count: int = Field(..., description="Character count")
    sentence_count: int = Field(..., description="Sentence count")
    paragraph_count: int = Field(..., description="Paragraph count")
    average_sentence_length: float = Field(..., description="Average sentence length")
    average_word_length: float = Field(..., description="Average word length")
    image_to_text_ratio: float = Field(..., description="Image to text ratio")
    link_density: float = Field(..., description="Link density")
    spam_score: float = Field(..., description="Spam detection score")
    duplicate_score: float = Field(..., description="Duplicate content score")
    content_freshness: float = Field(..., description="Content freshness score")
    overall_quality: float = Field(..., description="Overall quality score")

    @validator("readability_score", "spam_score", "duplicate_score", "content_freshness", "overall_quality")
    def validate_scores(cls, v):
        """Validate score values."""
        if not 0.0 <= v <= 100.0:
            raise ValueError("Scores must be between 0.0 and 100.0")
        return v

    @validator("image_to_text_ratio", "link_density")
    def validate_ratios(cls, v):
        """Validate ratio values."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Ratios must be between 0.0 and 1.0")
        return v


class ExtractionStats(BaseModel):
    """Extraction process statistics."""

    extraction_time_ms: int = Field(..., description="Total extraction time in milliseconds")
    html_fetch_time_ms: int = Field(..., description="HTML fetch time in milliseconds")
    content_extraction_time_ms: int = Field(..., description="Content extraction time in milliseconds")
    image_processing_time_ms: int = Field(..., description="Image processing time in milliseconds")
    quality_analysis_time_ms: int = Field(..., description="Quality analysis time in milliseconds")
    total_processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    bytes_processed: int = Field(..., description="Bytes processed")
    images_processed: int = Field(..., description="Number of images processed")
    links_found: int = Field(..., description="Number of links found")
    scripts_removed: int = Field(..., description="Number of scripts removed")
    styles_removed: int = Field(..., description="Number of style elements removed")
    ads_removed: int = Field(..., description="Number of ads removed")

    @validator(
        "extraction_time_ms",
        "html_fetch_time_ms",
        "content_extraction_time_ms",
        "image_processing_time_ms",
        "quality_analysis_time_ms",
        "total_processing_time_ms",
    )
    def validate_times(cls, v):
        """Validate time values."""
        if v < 0:
            raise ValueError("Time values must be non-negative")
        return v


class ExtractedContent(BaseModel):
    """Clean, extracted content with metadata."""

    url: str = Field(..., description="Original URL")
    canonical_url: Optional[str] = Field(None, description="Canonical URL")
    title: str = Field(..., description="Content title")
    content: str = Field(..., description="Clean content text")
    summary: Optional[str] = Field(None, description="Content summary")
    author: Optional[str] = Field(None, description="Content author")
    publish_date: Optional[datetime] = Field(None, description="Publication date")
    images: List[ProcessedImage] = Field(default_factory=list, description="Processed images")
    videos: List[VideoMetadata] = Field(default_factory=list, description="Video metadata")
    metadata: ContentMetadata = Field(..., description="Content metadata")
    quality_metrics: QualityMetrics = Field(..., description="Quality metrics")
    language_info: LanguageInfo = Field(..., description="Language information")
    word_count: int = Field(..., description="Word count")
    reading_time: int = Field(..., description="Estimated reading time in minutes")
    extraction_method: ExtractionMethod = Field(..., description="Extraction method used")
    content_type: ContentType = Field(..., description="Content type")
    content_hash: str = Field(..., description="Content hash for deduplication")
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Extraction timestamp")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Processing status")
    extraction_stats: ExtractionStats = Field(..., description="Extraction statistics")
    raw_html: Optional[str] = Field(None, description="Raw HTML content")
    cleaned_html: Optional[str] = Field(None, description="Cleaned HTML content")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    @validator("url", "canonical_url")
    def validate_urls(cls, v):
        """Validate URL format."""
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @validator("word_count", "reading_time")
    def validate_positive_numbers(cls, v):
        """Validate positive numbers."""
        if v < 0:
            raise ValueError("Must be non-negative")
        return v

    def calculate_content_hash(self) -> str:
        """Calculate content hash for deduplication."""
        content_for_hash = f"{self.title}|{self.content}|{self.url}"
        return hashlib.sha256(content_for_hash.encode("utf-8")).hexdigest()

    def calculate_reading_time(self) -> int:
        """Calculate estimated reading time in minutes."""
        if not self.word_count:
            return 0
        # Average reading speed: 200 words per minute
        return max(1, self.word_count // 200)

    def is_high_quality(self, min_quality_score: float = 0.7) -> bool:
        """Check if content meets quality standards."""
        return (
            self.quality_metrics.overall_quality >= min_quality_score
            and self.quality_metrics.spam_score < 0.3
            and self.quality_metrics.readability_score >= 30.0
            and self.word_count >= 100
        )

    def is_duplicate_of(self, other: "ExtractedContent", threshold: float = 0.8) -> bool:
        """Check if this content is a duplicate of another."""
        if self.content_hash == other.content_hash:
            return True

        # Calculate similarity based on title and content
        title_similarity = self._calculate_similarity(self.title, other.title)
        content_similarity = self._calculate_similarity(self.content, other.content)

        return (title_similarity + content_similarity) / 2 >= threshold

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple word overlap."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0


class ExtractionRequest(BaseModel):
    """Request for content extraction."""

    url: str = Field(..., description="URL to extract content from")
    content_type: Optional[ContentType] = Field(None, description="Expected content type")
    extraction_method: Optional[ExtractionMethod] = Field(None, description="Preferred extraction method")
    force_refresh: bool = Field(default=False, description="Force refresh even if cached")
    include_images: bool = Field(default=True, description="Include image processing")
    include_videos: bool = Field(default=True, description="Include video metadata")
    quality_threshold: float = Field(default=0.3, description="Minimum quality threshold")
    language_hint: Optional[str] = Field(None, description="Language hint for detection")
    custom_selectors: Dict[str, str] = Field(default_factory=dict, description="Custom CSS selectors")
    timeout: Optional[int] = Field(None, description="Request timeout in seconds")

    @validator("url")
    def validate_url(cls, v):
        """Validate URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @validator("quality_threshold")
    def validate_quality_threshold(cls, v):
        """Validate quality threshold."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Quality threshold must be between 0.0 and 1.0")
        return v


class ExtractionResponse(BaseModel):
    """Response from content extraction."""

    success: bool = Field(..., description="Whether extraction was successful")
    content: Optional[ExtractedContent] = Field(None, description="Extracted content")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    cache_hit: bool = Field(default=False, description="Whether result was from cache")
    extraction_id: str = Field(..., description="Unique extraction ID")


class BatchExtractionRequest(BaseModel):
    """Request for batch content extraction."""

    urls: List[str] = Field(..., description="URLs to extract content from")
    content_types: List[ContentType] = Field(default_factory=list, description="Expected content types")
    extraction_methods: List[ExtractionMethod] = Field(default_factory=list, description="Preferred extraction methods")
    force_refresh: bool = Field(default=False, description="Force refresh even if cached")
    include_images: bool = Field(default=True, description="Include image processing")
    quality_threshold: float = Field(default=0.3, description="Minimum quality threshold")
    max_concurrent: int = Field(default=10, description="Maximum concurrent extractions")

    @validator("urls")
    def validate_urls(cls, v):
        """Validate URLs."""
        if not v:
            raise ValueError("URLs list cannot be empty")
        if len(v) > 100:
            raise ValueError("Maximum 100 URLs per batch")
        return v

    @validator("max_concurrent")
    def validate_max_concurrent(cls, v):
        """Validate max concurrent value."""
        if v < 1 or v > 50:
            raise ValueError("Max concurrent must be between 1 and 50")
        return v


class BatchExtractionResponse(BaseModel):
    """Response from batch content extraction."""

    batch_id: str = Field(..., description="Batch ID")
    total_urls: int = Field(..., description="Total URLs processed")
    successful_extractions: int = Field(..., description="Number of successful extractions")
    failed_extractions: int = Field(..., description="Number of failed extractions")
    cached_extractions: int = Field(..., description="Number of cached extractions")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    results: List[ExtractionResponse] = Field(..., description="Individual extraction results")
    errors: List[str] = Field(default_factory=list, description="Error messages")
