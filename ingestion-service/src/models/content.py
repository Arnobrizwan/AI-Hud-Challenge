"""
Content models for news ingestion and normalization.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, HttpUrl
import hashlib


class ContentType(str, Enum):
    """Content type enumeration."""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    NEWS_ITEM = "news_item"
    PRESS_RELEASE = "press_release"
    OPINION = "opinion"
    ANALYSIS = "analysis"
    INTERVIEW = "interview"
    REVIEW = "review"


class SourceType(str, Enum):
    """Source type enumeration."""
    RSS_FEED = "rss_feed"
    ATOM_FEED = "atom_feed"
    JSON_FEED = "json_feed"
    API = "api"
    WEB_SCRAPING = "web_scraping"
    SOCIAL_MEDIA = "social_media"
    NEWS_WIRE = "news_wire"
    EMAIL = "email"


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    INGESTING = "ingesting"
    NORMALIZING = "normalizing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    DUPLICATE = "duplicate"


class IngestionMetadata(BaseModel):
    """Metadata about the ingestion process."""
    source_id: str = Field(..., description="Source identifier")
    source_type: SourceType = Field(..., description="Type of source")
    source_url: str = Field(..., description="Source URL")
    ingested_at: datetime = Field(default_factory=datetime.utcnow, description="Ingestion timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    http_status_code: Optional[int] = Field(None, description="HTTP status code")
    content_length: Optional[int] = Field(None, description="Content length in bytes")
    etag: Optional[str] = Field(None, description="HTTP ETag")
    last_modified: Optional[datetime] = Field(None, description="Last-Modified header")
    user_agent: Optional[str] = Field(None, description="User agent used")
    referrer: Optional[str] = Field(None, description="Referrer URL")
    ip_address: Optional[str] = Field(None, description="IP address of source")
    robots_txt_respected: bool = Field(default=True, description="Whether robots.txt was respected")
    rate_limit_delay: Optional[float] = Field(None, description="Rate limit delay applied")


class NormalizedArticle(BaseModel):
    """Normalized article model."""
    id: str = Field(..., description="Unique content identifier")
    url: str = Field(..., description="Original article URL")
    canonical_url: Optional[str] = Field(None, description="Canonical URL if different")
    title: str = Field(..., description="Article title")
    summary: Optional[str] = Field(None, description="Article summary/description")
    content: Optional[str] = Field(None, description="Full article content")
    author: Optional[str] = Field(None, description="Article author")
    byline: Optional[str] = Field(None, description="Author byline")
    source: str = Field(..., description="Source publication name")
    source_url: str = Field(..., description="Source domain URL")
    published_at: datetime = Field(..., description="Publication timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    language: str = Field(default="en", description="Detected language code")
    image_url: Optional[str] = Field(None, description="Main article image")
    tags: List[str] = Field(default_factory=list, description="Article tags/categories")
    word_count: int = Field(default=0, description="Article word count")
    reading_time: int = Field(default=0, description="Estimated reading time in minutes")
    content_hash: str = Field(..., description="SHA256 hash of normalized content")
    content_type: ContentType = Field(default=ContentType.ARTICLE, description="Type of content")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Original feed data")
    ingestion_metadata: IngestionMetadata = Field(..., description="Processing metadata")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Processing status")
    
    @validator("url", "canonical_url")
    def validate_urls(cls, v):
        """Validate URL format."""
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v
    
    @validator("language")
    def validate_language(cls, v):
        """Validate language code."""
        if len(v) != 2:
            raise ValueError("Language code must be 2 characters")
        return v.lower()
    
    @validator("word_count", "reading_time")
    def validate_positive_numbers(cls, v):
        """Validate positive numbers."""
        if v < 0:
            raise ValueError("Must be non-negative")
        return v
    
    def calculate_content_hash(self) -> str:
        """Calculate content hash for duplicate detection."""
        content_for_hash = f"{self.title}|{self.content or ''}|{self.url}"
        return hashlib.sha256(content_for_hash.encode('utf-8')).hexdigest()
    
    def calculate_reading_time(self) -> int:
        """Calculate estimated reading time in minutes."""
        if not self.word_count:
            return 0
        # Average reading speed: 200 words per minute
        return max(1, self.word_count // 200)
    
    def is_duplicate_of(self, other: "NormalizedArticle", threshold: float = 0.8) -> bool:
        """Check if this article is a duplicate of another."""
        if self.content_hash == other.content_hash:
            return True
        
        # Simple similarity check based on title and content
        title_similarity = self._calculate_similarity(self.title, other.title)
        content_similarity = self._calculate_similarity(
            self.content or "", 
            other.content or ""
        )
        
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


class SourceConfig(BaseModel):
    """Configuration for a content source."""
    id: str = Field(..., description="Unique source identifier")
    name: str = Field(..., description="Source name")
    type: SourceType = Field(..., description="Source type")
    url: str = Field(..., description="Source URL")
    enabled: bool = Field(default=True, description="Whether source is enabled")
    priority: int = Field(default=1, description="Processing priority (1=highest)")
    rate_limit: int = Field(default=60, description="Requests per minute")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    backoff_factor: float = Field(default=2.0, description="Exponential backoff factor")
    user_agent: Optional[str] = Field(None, description="Custom user agent")
    headers: Dict[str, str] = Field(default_factory=dict, description="Custom headers")
    auth: Optional[Dict[str, str]] = Field(None, description="Authentication credentials")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Content filters")
    last_checked: Optional[datetime] = Field(None, description="Last check timestamp")
    last_success: Optional[datetime] = Field(None, description="Last successful check")
    error_count: int = Field(default=0, description="Consecutive error count")
    success_count: int = Field(default=0, description="Consecutive success count")
    
    @validator("priority")
    def validate_priority(cls, v):
        """Validate priority value."""
        if v < 1 or v > 10:
            raise ValueError("Priority must be between 1 and 10")
        return v
    
    @validator("rate_limit")
    def validate_rate_limit(cls, v):
        """Validate rate limit value."""
        if v < 1:
            raise ValueError("Rate limit must be positive")
        return v


class ProcessingBatch(BaseModel):
    """Batch of articles being processed."""
    batch_id: str = Field(..., description="Unique batch identifier")
    source_id: str = Field(..., description="Source identifier")
    articles: List[NormalizedArticle] = Field(..., description="Articles in batch")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Batch creation time")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Batch status")
    total_count: int = Field(..., description="Total articles in batch")
    processed_count: int = Field(default=0, description="Processed articles count")
    failed_count: int = Field(default=0, description="Failed articles count")
    duplicate_count: int = Field(default=0, description="Duplicate articles count")
    error_message: Optional[str] = Field(None, description="Batch error message")
    
    @validator("total_count")
    def validate_total_count(cls, v):
        """Validate total count matches articles length."""
        if v < 0:
            raise ValueError("Total count must be non-negative")
        return v
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_count == 0:
            return 0.0
        return self.processed_count / self.total_count
    
    @property
    def processing_time_seconds(self) -> Optional[float]:
        """Calculate processing time in seconds."""
        if not self.started_at or not self.completed_at:
            return None
        return (self.completed_at - self.started_at).total_seconds()


class DuplicateDetection(BaseModel):
    """Duplicate detection result."""
    article_id: str = Field(..., description="Article identifier")
    duplicate_of: str = Field(..., description="ID of duplicate article")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    detection_method: str = Field(..., description="Detection method used")
    detected_at: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")
    
    @validator("similarity_score")
    def validate_similarity_score(cls, v):
        """Validate similarity score."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Similarity score must be between 0.0 and 1.0")
        return v


class ContentMetrics(BaseModel):
    """Content processing metrics."""
    source_id: str = Field(..., description="Source identifier")
    date: datetime = Field(..., description="Metrics date")
    total_articles: int = Field(default=0, description="Total articles processed")
    successful_articles: int = Field(default=0, description="Successful articles")
    failed_articles: int = Field(default=0, description="Failed articles")
    duplicate_articles: int = Field(default=0, description="Duplicate articles")
    average_processing_time_ms: float = Field(default=0.0, description="Average processing time")
    average_word_count: float = Field(default=0.0, description="Average word count")
    language_distribution: Dict[str, int] = Field(default_factory=dict, description="Language distribution")
    content_type_distribution: Dict[str, int] = Field(default_factory=dict, description="Content type distribution")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_articles == 0:
            return 0.0
        return self.successful_articles / self.total_articles
    
    @property
    def duplicate_rate(self) -> float:
        """Calculate duplicate rate."""
        if self.total_articles == 0:
            return 0.0
        return self.duplicate_articles / self.total_articles
