"""
API models for the Content Extraction & Cleanup microservice.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Content type enumeration."""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    NEWS = "news"
    PRESS_RELEASE = "press_release"
    OPINION = "opinion"
    REVIEW = "review"
    TUTORIAL = "tutorial"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


class ImageInfo(BaseModel):
    """Image information model."""
    url: str
    alt_text: Optional[str] = None
    caption: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    format: Optional[str] = None


class MetadataInfo(BaseModel):
    """Metadata information model."""
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    canonical_url: Optional[str] = None
    og_title: Optional[str] = None
    og_description: Optional[str] = None
    og_image: Optional[str] = None
    twitter_card: Optional[str] = None
    twitter_title: Optional[str] = None
    twitter_description: Optional[str] = None
    twitter_image: Optional[str] = None


class ContentExtractionRequest(BaseModel):
    """Content extraction request model."""
    url: Optional[str] = Field(None, description="URL to extract content from")
    html_content: Optional[str] = Field(None, description="Raw HTML content to process")
    text_content: Optional[str] = Field(None, description="Plain text content to process")
    content_type: Optional[ContentType] = Field(None, description="Expected content type")
    language: Optional[str] = Field("en", description="Content language code")
    extract_images: bool = Field(True, description="Whether to extract images")
    extract_metadata: bool = Field(True, description="Whether to extract metadata")
    quality_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum quality score")


class ContentExtractionResponse(BaseModel):
    """Content extraction response model."""
    content_id: str
    title: str
    content: str
    summary: Optional[str] = None
    language: str
    content_type: ContentType
    word_count: int
    reading_time: int
    quality_score: float
    images: List[ImageInfo] = Field(default_factory=list)
    metadata: Optional[MetadataInfo] = None
    processing_time_ms: int
    status: ProcessingStatus


class ExtractionItem(BaseModel):
    """Individual extraction item for batch processing."""
    item_id: str
    url: Optional[str] = None
    html_content: Optional[str] = None
    text_content: Optional[str] = None
    content_type: Optional[ContentType] = None
    language: Optional[str] = "en"
    extract_images: bool = True
    extract_metadata: bool = True
    quality_threshold: float = 0.5


class ExtractionBatchRequest(BaseModel):
    """Batch extraction request model."""
    items: List[ExtractionItem]
    batch_id: Optional[str] = Field(None, description="Custom batch ID")
    parallel_workers: int = Field(4, ge=1, le=10, description="Number of parallel workers")
    quality_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum quality score")


class ExtractionBatchResponse(BaseModel):
    """Batch extraction response model."""
    batch_id: str
    total_items: int
    processed_items: int
    failed_items: int
    processing_time_ms: int
    status: ProcessingStatus
    results: List[ContentExtractionResponse] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


class QualityAnalysisRequest(BaseModel):
    """Quality analysis request model."""
    content: str = Field(..., description="Content to analyze")
    title: Optional[str] = Field(None, description="Content title")
    language: Optional[str] = Field("en", description="Content language")
    content_type: Optional[ContentType] = Field(None, description="Content type")


class QualityAnalysisResponse(BaseModel):
    """Quality analysis response model."""
    content_id: str
    quality_score: float
    readability_score: float
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float
    flesch_kincaid_grade: float
    sentiment_score: float
    language_confidence: float
    processing_time_ms: int


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    uptime_seconds: float
    dependencies: Dict[str, str]


class MetricsResponse(BaseModel):
    """Metrics response model."""
    total_extractions: int
    successful_extractions: int
    failed_extractions: int
    avg_processing_time_ms: float
    cache_hit_rate: float
    quality_scores: Dict[str, float]
    content_types: Dict[str, int]
    languages: Dict[str, int]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)