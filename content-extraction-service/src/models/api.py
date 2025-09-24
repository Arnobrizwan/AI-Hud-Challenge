"""
API models for the content extraction service.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .content import ContentType, ProcessingStatus


class APIResponse(BaseModel):
    """Base API response model."""

    success: bool = Field(..., description="Whether request was successful")
    message: Optional[str] = Field(None, description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    errors: Optional[List[str]] = Field(None, description="Error messages")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime")
    dependencies: Dict[str, str] = Field(..., description="Dependency status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")


class ExtractionStatusResponse(BaseModel):
    """Response for extraction status check."""

    extraction_id: str = Field(..., description="Extraction ID")
    status: ProcessingStatus = Field(..., description="Current status")
    progress_percentage: float = Field(..., description="Progress percentage")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class ContentSearchRequest(BaseModel):
    """Request to search extracted content."""

    query: Optional[str] = Field(None, description="Search query")
    content_types: Optional[List[ContentType]] = Field(None, description="Filter by content types")
    languages: Optional[List[str]] = Field(None, description="Filter by languages")
    quality_threshold: Optional[float] = Field(None, description="Minimum quality threshold")
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")
    limit: int = Field(default=20, description="Maximum results")
    offset: int = Field(default=0, description="Result offset")
    sort_by: str = Field(default="extraction_timestamp", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order")

    @validator("limit")
    def validate_limit(cls, v):
        """Validate limit."""
        if v < 1 or v > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        return v

    @validator("offset")
    def validate_offset(cls, v):
        """Validate offset."""
        if v < 0:
            raise ValueError("Offset must be non-negative")
        return v

    @validator("sort_order")
    def validate_sort_order(cls, v):
        """Validate sort order."""
        if v not in ["asc", "desc"]:
            raise ValueError("Sort order must be 'asc' or 'desc'")
        return v


class ContentSearchResponse(BaseModel):
    """Response for content search."""

    results: List[Dict[str, Any]] = Field(..., description="Matching content")
    total_count: int = Field(..., description="Total matching results")
    limit: int = Field(..., description="Result limit")
    offset: int = Field(..., description="Result offset")
    query_time_ms: int = Field(..., description="Query execution time")


class QualityAnalysisRequest(BaseModel):
    """Request for content quality analysis."""

    content: str = Field(..., description="Content to analyze")
    title: Optional[str] = Field(None, description="Content title")
    url: Optional[str] = Field(None, description="Content URL")
    language_hint: Optional[str] = Field(None, description="Language hint")

    @validator("content")
    def validate_content(cls, v):
        """Validate content."""
        if not v or len(v.strip()) < 10:
            raise ValueError("Content must be at least 10 characters")
        return v


class QualityAnalysisResponse(BaseModel):
    """Response for content quality analysis."""

    quality_score: float = Field(..., description="Overall quality score")
    readability_score: float = Field(..., description="Readability score")
    spam_score: float = Field(..., description="Spam detection score")
    word_count: int = Field(..., description="Word count")
    sentence_count: int = Field(..., description="Sentence count")
    average_sentence_length: float = Field(..., description="Average sentence length")
    language: str = Field(..., description="Detected language")
    language_confidence: float = Field(..., description="Language detection confidence")
    recommendations: List[str] = Field(..., description="Quality improvement recommendations")
    analysis_time_ms: int = Field(..., description="Analysis time in milliseconds")


class ImageProcessingRequest(BaseModel):
    """Request for image processing."""

    image_url: str = Field(..., description="Image URL to process")
    optimize: bool = Field(default=True, description="Whether to optimize image")
    max_width: Optional[int] = Field(None, description="Maximum width")
    max_height: Optional[int] = Field(None, description="Maximum height")
    quality: int = Field(default=85, description="JPEG quality (1-100)")
    format: Optional[str] = Field(None, description="Output format")

    @validator("image_url")
    def validate_image_url(cls, v):
        """Validate image URL."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Image URL must start with http:// or https://")
        return v

    @validator("quality")
    def validate_quality(cls, v):
        """Validate quality value."""
        if not 1 <= v <= 100:
            raise ValueError("Quality must be between 1 and 100")
        return v


class ImageProcessingResponse(BaseModel):
    """Response for image processing."""

    success: bool = Field(..., description="Whether processing was successful")
    original_url: str = Field(..., description="Original image URL")
    processed_url: Optional[str] = Field(None, description="Processed image URL")
    width: int = Field(..., description="Image width")
    height: int = Field(..., description="Image height")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Image format")
    quality_score: float = Field(..., description="Image quality score")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class BatchProcessingRequest(BaseModel):
    """Request for batch processing."""

    urls: List[str] = Field(..., description="URLs to process")
    processing_type: str = Field(..., description="Type of processing")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")
    priority: int = Field(default=1, description="Processing priority (1-10)")
    callback_url: Optional[str] = Field(None, description="Callback URL for completion notification")

    @validator("urls")
    def validate_urls(cls, v):
        """Validate URLs."""
        if not v:
            raise ValueError("URLs list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 URLs per batch")
        return v

    @validator("priority")
    def validate_priority(cls, v):
        """Validate priority."""
        if not 1 <= v <= 10:
            raise ValueError("Priority must be between 1 and 10")
        return v


class BatchProcessingResponse(BaseModel):
    """Response for batch processing."""

    batch_id: str = Field(..., description="Batch ID")
    status: str = Field(..., description="Batch status")
    total_items: int = Field(..., description="Total items to process")
    queued_items: int = Field(..., description="Items queued for processing")
    processing_items: int = Field(..., description="Items currently processing")
    completed_items: int = Field(..., description="Items completed")
    failed_items: int = Field(..., description="Items failed")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Batch creation time")


class MetricsResponse(BaseModel):
    """Response for metrics data."""

    period: str = Field(..., description="Metrics period")
    total_extractions: int = Field(..., description="Total extractions performed")
    successful_extractions: int = Field(..., description="Successful extractions")
    failed_extractions: int = Field(..., description="Failed extractions")
    cached_extractions: int = Field(..., description="Cached extractions")
    success_rate: float = Field(..., description="Success rate")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    throughput_per_minute: float = Field(..., description="Extractions per minute")
    error_rate: float = Field(..., description="Error rate")
    content_type_distribution: Dict[str, int] = Field(..., description="Content type distribution")
    language_distribution: Dict[str, int] = Field(..., description="Language distribution")
    quality_score_distribution: Dict[str, int] = Field(..., description="Quality score distribution")


class CacheStatsResponse(BaseModel):
    """Response for cache statistics."""

    total_entries: int = Field(..., description="Total cache entries")
    hit_rate: float = Field(..., description="Cache hit rate")
    miss_rate: float = Field(..., description="Cache miss rate")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    disk_usage_mb: float = Field(..., description="Disk usage in MB")
    oldest_entry: Optional[datetime] = Field(None, description="Oldest cache entry")
    newest_entry: Optional[datetime] = Field(None, description="Newest cache entry")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    error_code: Optional[str] = Field(None, description="Error code")
