"""
Request validation models for news aggregation pipeline.
"""

import re
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, validator


class ContentType(str, Enum):
    """Content types for news aggregation."""

    ARTICLE = "article"
    VIDEO = "video"
    PODCAST = "podcast"
    IMAGE = "image"
    LIVE_STREAM = "live_stream"


class Priority(str, Enum):
    """Priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Status(str, Enum):
    """Processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NewsSourceRequest(BaseModel):
    """Request model for news source configuration."""

    name: str = Field(min_length=1, max_length=100, description="Source name")
    url: HttpUrl = Field(description="Source URL")
    description: Optional[str] = Field(
        default=None, max_length=500, description="Source description"
    )
    category: str = Field(min_length=1, max_length=50, description="News category")
    language: str = Field(min_length=2, max_length=5, description="Language code (ISO 639-1)")
    country: Optional[str] = Field(
        default=None, min_length=2, max_length=2, description="Country code (ISO 3166-1)"
    )
    enabled: bool = Field(default=True, description="Source enabled status")
    scraping_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Scraping configuration"
    )

    @validator("language")
    def validate_language(cls, v):
        """Validate language code format."""
        if not re.match(r"^[a-z]{2}(-[A-Z]{2})?$", v):
            raise ValueError("Invalid language code format")
        return v

    @validator("country")
    def validate_country(cls, v):
        """Validate country code format."""
        if v and not re.match(r"^[A-Z]{2}$", v):
            raise ValueError("Invalid country code format")
        return v


class ContentFilterRequest(BaseModel):
    """Request model for content filtering."""

    keywords: Optional[List[str]] = Field(default=None, description="Keywords to filter")
    exclude_keywords: Optional[List[str]] = Field(default=None, description="Keywords to exclude")
    categories: Optional[List[str]] = Field(default=None, description="Content categories")
    content_types: Optional[List[ContentType]] = Field(default=None, description="Content types")
    min_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum quality score"
    )
    languages: Optional[List[str]] = Field(default=None, description="Allowed languages")
    date_from: Optional[date] = Field(default=None, description="Filter from date")
    date_to: Optional[date] = Field(default=None, description="Filter to date")

    @validator("keywords", "exclude_keywords")
    def validate_keywords(cls, v):
        """Validate keywords list."""
        if v and len(v) > 50:
            raise ValueError("Maximum 50 keywords allowed")
        return v

    @validator("date_to")
    def validate_date_range(cls, v, values):
        """Validate date range."""
        if v and "date_from" in values and values["date_from"] and v < values["date_from"]:
            raise ValueError("date_to must be after date_from")
        return v


class ProcessingJobRequest(BaseModel):
    """Request model for processing job creation."""

    job_type: str = Field(min_length=1, max_length=50, description="Job type")
    priority: Priority = Field(default=Priority.NORMAL, description="Job priority")
    source_urls: List[HttpUrl] = Field(min_items=1, max_items=100, description="URLs to process")
    filters: Optional[ContentFilterRequest] = Field(default=None, description="Content filters")
    callback_url: Optional[HttpUrl] = Field(default=None, description="Callback URL for completion")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    scheduled_for: Optional[datetime] = Field(default=None, description="Scheduled execution time")

    @validator("scheduled_for")
    def validate_scheduled_time(cls, v):
        """Validate scheduled time is in the future."""
        if v and v <= datetime.utcnow():
            raise ValueError("Scheduled time must be in the future")
        return v


class BulkProcessingRequest(BaseModel):
    """Request model for bulk processing operations."""

    jobs: List[ProcessingJobRequest] = Field(
        min_items=1, max_items=10, description="Processing jobs"
    )
    batch_size: Optional[int] = Field(default=5, ge=1, le=10, description="Batch processing size")
    parallel_execution: bool = Field(default=True, description="Enable parallel execution")
    fail_on_error: bool = Field(default=False, description="Fail entire batch on single error")


class WebhookRequest(BaseModel):
    """Request model for webhook configuration."""

    url: HttpUrl = Field(description="Webhook URL")
    events: List[str] = Field(min_items=1, description="Event types to trigger webhook")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Custom headers")
    secret: Optional[str] = Field(
        default=None, min_length=8, description="Webhook secret for signing"
    )
    retry_policy: Optional[Dict[str, Any]] = Field(default=None, description="Retry configuration")
    enabled: bool = Field(default=True, description="Webhook enabled status")


class SearchRequest(BaseModel):
    """Request model for content search."""

    query: str = Field(min_length=1, max_length=200, description="Search query")
    filters: Optional[ContentFilterRequest] = Field(default=None, description="Search filters")
    sort_by: Optional[str] = Field(default="relevance", description="Sort field")
    sort_order: Optional[str] = Field(
        default="desc", regex="^(asc|desc)$", description="Sort order"
    )
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    include_snippets: bool = Field(default=True, description="Include content snippets")
    highlight: bool = Field(default=True, description="Highlight search terms")


class AnalyticsRequest(BaseModel):
    """Request model for analytics queries."""

    metric: str = Field(min_length=1, max_length=50, description="Metric name")
    dimensions: Optional[List[str]] = Field(default=None, description="Grouping dimensions")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filter criteria")
    date_from: date = Field(description="Start date")
    date_to: date = Field(description="End date")
    granularity: str = Field(
        default="day", regex="^(hour|day|week|month)$", description="Time granularity"
    )

    @validator("date_to")
    def validate_date_range(cls, v, values):
        """Validate date range."""
        if "date_from" in values and values["date_from"] and v < values["date_from"]:
            raise ValueError("date_to must be after date_from")
        return v


class ConfigurationUpdateRequest(BaseModel):
    """Request model for configuration updates."""

    section: str = Field(min_length=1, max_length=50, description="Configuration section")
    key: str = Field(min_length=1, max_length=100, description="Configuration key")
    value: Any = Field(description="Configuration value")
    description: Optional[str] = Field(
        default=None, max_length=200, description="Change description"
    )
    environment: Optional[str] = Field(default=None, description="Target environment")

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment value."""
        if v and v not in ["development", "staging", "production"]:
            raise ValueError("Invalid environment")
        return v
