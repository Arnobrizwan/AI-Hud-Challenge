"""
API models for the Realtime Interfaces microservice.
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


class FeedbackType(str, Enum):
    """Feedback type enumeration."""
    LIKE = "like"
    DISLIKE = "dislike"
    SAVE = "save"
    SHARE = "share"
    CLICK = "click"
    VIEW = "view"
    NOT_INTERESTED = "not_interested"
    SHOW_MORE_LIKE_THIS = "show_more_like_this"
    MUTE = "mute"
    REPORT = "report"


class WebSocketMessageType(str, Enum):
    """WebSocket message type enumeration."""
    PING = "ping"
    PONG = "pong"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    UPDATE = "update"
    ERROR = "error"


class ArticleItem(BaseModel):
    """Article item model."""
    id: str
    title: str
    summary: str
    url: str
    source: str
    published_at: datetime
    content_type: ContentType
    topics: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    sentiment: Optional[float] = None
    quality_score: float
    image_url: Optional[str] = None
    reading_time: int
    word_count: int


class ClusterItem(BaseModel):
    """Cluster item model."""
    cluster_id: str
    title: str
    summary: str
    article_count: int
    key_entities: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    sentiment: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    representative_article: Optional[ArticleItem] = None


class FeedItem(BaseModel):
    """Feed item model."""
    item_id: str
    item_type: str  # "article" or "cluster"
    article: Optional[ArticleItem] = None
    cluster: Optional[ClusterItem] = None
    rank: int
    score: float
    personalized_score: Optional[float] = None
    explanation: Optional[str] = None


class FeedRequest(BaseModel):
    """Feed request model."""
    user_id: str
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)
    topics: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    content_types: Optional[List[ContentType]] = None
    time_range: str = Field("24h")
    personalization: bool = True


class FeedResponse(BaseModel):
    """Feed response model."""
    user_id: str
    items: List[FeedItem]
    total_count: int
    limit: int
    offset: int
    processing_time_ms: int
    personalization_enabled: bool
    cache_hit: bool
    etag: Optional[str] = None


class ClusterDetailResponse(BaseModel):
    """Cluster detail response model."""
    cluster_id: str
    title: str
    summary: str
    articles: List[ArticleItem]
    related_clusters: List[ClusterItem] = Field(default_factory=list)
    event_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    sentiment: Optional[float] = None
    processing_time_ms: int
    cache_hit: bool


class FeedbackRequest(BaseModel):
    """Feedback request model."""
    user_id: str
    item_id: str
    feedback_type: FeedbackType
    rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    comment: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class FeedbackResponse(BaseModel):
    """Feedback response model."""
    feedback_id: str
    user_id: str
    item_id: str
    feedback_type: FeedbackType
    processed_at: datetime
    processing_time_ms: int
    success: bool


class LiveUpdateResponse(BaseModel):
    """Live update response model."""
    update_id: str
    update_type: str
    timestamp: datetime
    data: Dict[str, Any]
    user_id: Optional[str] = None


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: WebSocketMessageType
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    uptime_seconds: float
    dependencies: Dict[str, str]


class MetricsResponse(BaseModel):
    """Metrics response model."""
    total_requests: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_rate: float
    active_connections: int
    cache_hit_rate: float
    feed_requests: int
    cluster_requests: int
    feedback_requests: int
