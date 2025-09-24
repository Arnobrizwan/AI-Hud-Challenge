"""
API models for the ingestion service.
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


class IngestionRequest(BaseModel):
    """Request to ingest content from a source."""

    source_id: str = Field(..., description="Source identifier")
    url: Optional[str] = Field(None, description="Specific URL to ingest")
    force_refresh: bool = Field(default=False, description="Force refresh even if recently processed")
    batch_size: Optional[int] = Field(None, description="Batch size for processing")
    priority: Optional[int] = Field(None, description="Processing priority")

    @validator("batch_size")
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if v is not None and (v < 1 or v > 1000):
            raise ValueError("Batch size must be between 1 and 1000")
        return v


class IngestionResponse(BaseModel):
    """Response from ingestion request."""

    batch_id: str = Field(..., description="Batch identifier")
    source_id: str = Field(..., description="Source identifier")
    articles_processed: int = Field(..., description="Number of articles processed")
    articles_successful: int = Field(..., description="Number of successful articles")
    articles_failed: int = Field(..., description="Number of failed articles")
    articles_duplicates: int = Field(..., description="Number of duplicate articles")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    status: ProcessingStatus = Field(..., description="Processing status")
    errors: List[str] = Field(default_factory=list, description="Processing errors")


class SourceListResponse(BaseModel):
    """Response for listing sources."""

    sources: List[Dict[str, Any]] = Field(..., description="List of sources")
    total_count: int = Field(..., description="Total number of sources")
    enabled_count: int = Field(..., description="Number of enabled sources")
    disabled_count: int = Field(..., description="Number of disabled sources")


class SourceStatsResponse(BaseModel):
    """Response for source statistics."""

    source_id: str = Field(..., description="Source identifier")
    period: str = Field(..., description="Statistics period")
    total_articles: int = Field(..., description="Total articles processed")
    successful_articles: int = Field(..., description="Successful articles")
    failed_articles: int = Field(..., description="Failed articles")
    duplicate_articles: int = Field(..., description="Duplicate articles")
    success_rate: float = Field(..., description="Success rate")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    last_processed: Optional[datetime] = Field(None, description="Last processing time")
    error_rate: float = Field(..., description="Error rate")


class ContentSearchRequest(BaseModel):
    """Request to search content."""

    query: Optional[str] = Field(None, description="Search query")
    source_ids: Optional[List[str]] = Field(None, description="Filter by source IDs")
    content_types: Optional[List[ContentType]] = Field(None, description="Filter by content types")
    languages: Optional[List[str]] = Field(None, description="Filter by languages")
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")
    limit: int = Field(default=20, description="Maximum results")
    offset: int = Field(default=0, description="Result offset")
    sort_by: str = Field(default="published_at", description="Sort field")
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

    articles: List[Dict[str, Any]] = Field(..., description="Matching articles")
    total_count: int = Field(..., description="Total matching articles")
    limit: int = Field(..., description="Result limit")
    offset: int = Field(..., description="Result offset")
    query_time_ms: int = Field(..., description="Query execution time")


class DuplicateDetectionRequest(BaseModel):
    """Request to detect duplicates."""

    article_id: str = Field(..., description="Article identifier")
    similarity_threshold: float = Field(default=0.8, description="Similarity threshold")
    check_period_days: int = Field(default=7, description="Check period in days")

    @validator("similarity_threshold")
    def validate_threshold(cls, v):
        """Validate similarity threshold."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        return v

    @validator("check_period_days")
    def validate_period(cls, v):
        """Validate check period."""
        if v < 1 or v > 365:
            raise ValueError("Check period must be between 1 and 365 days")
        return v


class DuplicateDetectionResponse(BaseModel):
    """Response for duplicate detection."""

    article_id: str = Field(..., description="Article identifier")
    duplicates: List[Dict[str, Any]] = Field(..., description="Found duplicates")
    total_duplicates: int = Field(..., description="Total number of duplicates")
    detection_time_ms: int = Field(..., description="Detection time in milliseconds")


class MetricsResponse(BaseModel):
    """Response for metrics data."""

    period: str = Field(..., description="Metrics period")
    total_articles: int = Field(..., description="Total articles processed")
    successful_articles: int = Field(..., description="Successful articles")
    failed_articles: int = Field(..., description="Failed articles")
    duplicate_articles: int = Field(..., description="Duplicate articles")
    success_rate: float = Field(..., description="Success rate")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    throughput_per_hour: float = Field(..., description="Articles per hour")
    error_rate: float = Field(..., description="Error rate")
    language_distribution: Dict[str, int] = Field(..., description="Language distribution")
    source_distribution: Dict[str, int] = Field(..., description="Source distribution")
    content_type_distribution: Dict[str, int] = Field(..., description="Content type distribution")


class BatchStatusResponse(BaseModel):
    """Response for batch status."""

    batch_id: str = Field(..., description="Batch identifier")
    status: ProcessingStatus = Field(..., description="Batch status")
    progress_percentage: float = Field(..., description="Progress percentage")
    total_items: int = Field(..., description="Total items")
    processed_items: int = Field(..., description="Processed items")
    failed_items: int = Field(..., description="Failed items")
    duplicate_items: int = Field(..., description="Duplicate items")
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion")
    errors: List[str] = Field(default_factory=list, description="Processing errors")


class SourceConfigRequest(BaseModel):
    """Request to update source configuration."""

    source_id: str = Field(..., description="Source identifier")
    enabled: Optional[bool] = Field(None, description="Enable/disable source")
    rate_limit: Optional[int] = Field(None, description="Rate limit per minute")
    timeout: Optional[int] = Field(None, description="Request timeout")
    retry_attempts: Optional[int] = Field(None, description="Retry attempts")
    priority: Optional[int] = Field(None, description="Processing priority")
    filters: Optional[Dict[str, Any]] = Field(None, description="Content filters")

    @validator("rate_limit")
    def validate_rate_limit(cls, v):
        """Validate rate limit."""
        if v is not None and v < 1:
            raise ValueError("Rate limit must be positive")
        return v

    @validator("timeout")
    def validate_timeout(cls, v):
        """Validate timeout."""
        if v is not None and v < 1:
            raise ValueError("Timeout must be positive")
        return v

    @validator("retry_attempts")
    def validate_retry_attempts(cls, v):
        """Validate retry attempts."""
        if v is not None and v < 0:
            raise ValueError("Retry attempts must be non-negative")
        return v

    @validator("priority")
    def validate_priority(cls, v):
        """Validate priority."""
        if v is not None and (v < 1 or v > 10):
            raise ValueError("Priority must be between 1 and 10")
        return v


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
