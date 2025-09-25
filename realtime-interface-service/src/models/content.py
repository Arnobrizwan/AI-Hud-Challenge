"""
Content models for the Realtime Interfaces microservice.
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


class Article(BaseModel):
    """Article model."""
    id: str
    title: str
    summary: str
    content: Optional[str] = None
    url: str
    source: str
    author: Optional[str] = None
    published_at: datetime
    updated_at: Optional[datetime] = None
    content_type: ContentType
    topics: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    sentiment: Optional[float] = None
    quality_score: float
    image_url: Optional[str] = None
    reading_time: int
    word_count: int
    language: str = "en"
    country: Optional[str] = None
    region: Optional[str] = None


class Cluster(BaseModel):
    """Cluster model."""
    cluster_id: str
    title: str
    summary: str
    article_count: int
    key_entities: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    sentiment: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    representative_article_id: Optional[str] = None
    event_timeline: List[Dict[str, Any]] = Field(default_factory=list)


class FeedItem(BaseModel):
    """Feed item model."""
    item_id: str
    item_type: str  # "article" or "cluster"
    article: Optional[Article] = None
    cluster: Optional[Cluster] = None
    rank: int
    score: float
    personalized_score: Optional[float] = None
    explanation: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserFeedback(BaseModel):
    """User feedback model."""
    feedback_id: str
    user_id: str
    item_id: str
    feedback_type: FeedbackType
    rating: Optional[float] = None
    comment: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None


class LiveUpdate(BaseModel):
    """Live update model."""
    update_id: str
    update_type: str
    timestamp: datetime
    data: Dict[str, Any]
    user_id: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    clusters: List[str] = Field(default_factory=list)


class WebSocketConnection(BaseModel):
    """WebSocket connection model."""
    user_id: str
    websocket: Any  # WebSocket object
    subscribed_topics: List[str] = Field(default_factory=list)
    subscribed_clusters: List[str] = Field(default_factory=list)
    connected_at: datetime = Field(default_factory=datetime.utcnow)
    last_ping: Optional[datetime] = None


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
    last_updated: datetime = Field(default_factory=datetime.utcnow)
