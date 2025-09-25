"""
Content models for the Content Extraction & Cleanup microservice.
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
    extracted_at: datetime = Field(default_factory=datetime.utcnow)


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
    extracted_at: datetime = Field(default_factory=datetime.utcnow)


class ExtractedContent(BaseModel):
    """Extracted content model."""
    content_id: str
    url: Optional[str] = None
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
    status: ProcessingStatus
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None


class QualityAnalysis(BaseModel):
    """Quality analysis model."""
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
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class ExtractionBatch(BaseModel):
    """Extraction batch model."""
    batch_id: str
    total_items: int
    processed_items: int
    failed_items: int
    status: ProcessingStatus
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None


class CacheStats(BaseModel):
    """Cache statistics model."""
    hit_count: int
    miss_count: int
    total_requests: int
    hit_rate: float
    cache_size: int
    memory_usage: int
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class ProcessingMetrics(BaseModel):
    """Processing metrics model."""
    total_extractions: int
    successful_extractions: int
    failed_extractions: int
    avg_processing_time_ms: float
    cache_hit_rate: float
    quality_scores: Dict[str, float]
    content_types: Dict[str, int]
    languages: Dict[str, int]
    last_updated: datetime = Field(default_factory=datetime.utcnow)