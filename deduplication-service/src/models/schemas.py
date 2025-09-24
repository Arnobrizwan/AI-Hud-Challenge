"""Pydantic schemas for the deduplication service."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


class Entity(BaseModel):
    """Entity model."""

    text: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    start: int
    end: int


class Location(BaseModel):
    """Location model."""

    name: str
    country: Optional[str] = None
    region: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    confidence: float = Field(ge=0.0, le=1.0)


class Topic(BaseModel):
    """Topic model."""

    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    category: Optional[str] = None


class NormalizedArticle(BaseModel):
    """Normalized article model."""

    id: UUID
    title: str
    content: str
    summary: Optional[str] = None
    url: str
    source: str
    published_at: datetime
    quality_score: float = Field(ge=0.0, le=1.0, default=0.0)
    content_hash: Optional[str] = None
    title_hash: Optional[str] = None
    entities: List[Entity] = Field(default_factory=list)
    topics: List[Topic] = Field(default_factory=list)
    locations: List[Location] = Field(default_factory=list)
    language: str = "en"
    word_count: int = 0
    reading_time: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @validator("word_count", pre=True, always=True)
    def calculate_word_count(cls, v, values):
        if v == 0 and "content" in values:
            return len(values["content"].split())
        return v

    @validator("reading_time", pre=True, always=True)
    def calculate_reading_time(cls, v, values):
        if v == 0 and "word_count" in values:
            # Average reading speed: 200 words per minute
            return max(1, values["word_count"] // 200)
        return v


class DuplicateResult(BaseModel):
    """Duplicate detection result."""

    article_id: UUID
    is_duplicate: bool
    duplicate_of: Optional[UUID] = None
    similarity_scores: List[tuple[UUID, float]] = Field(default_factory=list)
    cluster_id: Optional[UUID] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    detection_method: str = "combined"


class SimilarityScore(BaseModel):
    """Similarity score between two articles."""

    article_id: UUID
    similarity: float = Field(ge=0.0, le=1.0)
    similarity_type: str  # 'lsh', 'semantic', 'content', 'title'
    confidence: float = Field(ge=0.0, le=1.0)


class Cluster(BaseModel):
    """News event cluster."""

    id: UUID
    representative_article_id: UUID
    article_count: int = 0
    quality_score: float = Field(ge=0.0, le=1.0, default=0.0)
    topics: List[Topic] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    locations: List[Location] = Field(default_factory=list)
    time_span: Optional[int] = None  # in seconds
    is_active: bool = True
    created_at: datetime
    updated_at: datetime


class NewsEvent(BaseModel):
    """News event with articles."""

    id: UUID
    cluster: Cluster
    articles: List[NormalizedArticle]
    representative_article: NormalizedArticle
    event_summary: Optional[str] = None
    event_keywords: List[str] = Field(default_factory=list)
    temporal_coherence: float = Field(ge=0.0, le=1.0, default=0.0)
    topical_coherence: float = Field(ge=0.0, le=1.0, default=0.0)


class ProcessingStatus(BaseModel):
    """Processing status for an article."""

    article_id: UUID
    status: str  # 'pending', 'processing', 'completed', 'failed'
    priority: int = 0
    retry_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None


class DeduplicationRequest(BaseModel):
    """Request for deduplication."""

    articles: List[NormalizedArticle]
    batch_id: Optional[str] = None
    force_reprocess: bool = False
    similarity_threshold: Optional[float] = None
    clustering_enabled: bool = True


class DeduplicationResponse(BaseModel):
    """Response from deduplication."""

    batch_id: Optional[str] = None
    results: List[DuplicateResult]
    clusters: List[Cluster] = Field(default_factory=list)
    processing_time: float
    articles_processed: int
    duplicates_found: int
    clusters_created: int


class EventGroupingRequest(BaseModel):
    """Request for event grouping."""

    articles: List[NormalizedArticle]
    time_window_hours: int = 24
    min_cluster_size: int = 2
    max_cluster_size: int = 100
    clustering_eps: Optional[float] = None
    min_samples: Optional[int] = None


class EventGroupingResponse(BaseModel):
    """Response from event grouping."""

    events: List[NewsEvent]
    unclustered_articles: List[NormalizedArticle]
    processing_time: float
    events_created: int
    articles_clustered: int


class ClusterMetrics(BaseModel):
    """Cluster quality metrics."""

    cluster_id: UUID
    silhouette_score: float
    cohesion: float
    separation: float
    temporal_coherence: float
    topical_coherence: float
    article_count: int
    quality_score: float


class SystemMetrics(BaseModel):
    """System performance metrics."""

    articles_processed: int
    duplicates_detected: int
    clusters_created: int
    processing_latency_avg: float
    processing_latency_p95: float
    processing_latency_p99: float
    memory_usage_mb: float
    cpu_usage_percent: float
    redis_memory_usage_mb: float
    database_connections: int
    active_processing_tasks: int


class HealthCheck(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    version: str
    uptime: float
    dependencies: Dict[str, str]
    metrics: Optional[SystemMetrics] = None


class APIError(BaseModel):
    """API error response."""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    request_id: Optional[str] = None


class PaginationParams(BaseModel):
    """Pagination parameters."""

    page: int = Field(ge=1, default=1)
    size: int = Field(ge=1, le=1000, default=100)
    sort_by: str = "created_at"
    sort_order: str = "desc"


class PaginatedResponse(BaseModel):
    """Paginated response."""

    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    has_next: bool
    has_prev: bool


class SearchRequest(BaseModel):
    """Search request."""

    query: str
    filters: Optional[Dict[str, Any]] = None
    pagination: PaginationParams = Field(default_factory=PaginationParams)


class SearchResponse(BaseModel):
    """Search response."""

    results: PaginatedResponse
    search_time: float
    total_matches: int
