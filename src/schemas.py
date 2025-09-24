"""Data models and schemas for the ranking microservice."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Content type enumeration."""

    ARTICLE = "article"
    VIDEO = "video"
    PODCAST = "podcast"
    IMAGE = "image"


class Sentiment(BaseModel):
    """Sentiment analysis result."""

    polarity: float = Field(..., ge=-1.0, le=1.0)
    subjectivity: float = Field(..., ge=0.0, le=1.0)


class Entity(BaseModel):
    """Named entity."""

    text: str
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class Topic(BaseModel):
    """Content topic."""

    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    category: Optional[str] = None


class Author(BaseModel):
    """Content author."""

    id: str
    name: str
    bio: Optional[str] = None
    authority_score: Optional[float] = None


class Source(BaseModel):
    """Content source."""

    id: str
    name: str
    domain: str
    authority_score: Optional[float] = None
    reliability_score: Optional[float] = None
    popularity_score: Optional[float] = None


class Article(BaseModel):
    """Article content model."""

    id: str
    title: str
    content: Optional[str] = None
    summary: Optional[str] = None
    url: str
    published_at: datetime
    updated_at: Optional[datetime] = None

    # Content metadata
    content_type: ContentType = ContentType.ARTICLE
    word_count: int = 0
    reading_time: int = 0  # in minutes
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)

    # Media
    image_url: Optional[str] = None
    videos: Optional[List[str]] = None

    # Analysis results
    sentiment: Optional[Sentiment] = None
    entities: List[Entity] = []
    topics: List[Topic] = []

    # Authors and sources
    author: Optional[Author] = None
    source: Source

    # Engagement metrics
    view_count: int = 0
    like_count: int = 0
    share_count: int = 0
    comment_count: int = 0

    # Geographic and temporal
    language: str = "en"
    country: Optional[str] = None
    region: Optional[str] = None


class UserProfile(BaseModel):
    """User profile for personalization."""

    user_id: str
    topic_preferences: Dict[str, float] = Field(default_factory=dict)
    source_preferences: Dict[str, float] = Field(default_factory=dict)
    reading_patterns: Dict[str, Any] = Field(default_factory=dict)
    content_preferences: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class PersonalizedScore(BaseModel):
    """Personalized ranking score."""

    article_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    explanation: Optional[str] = None
    feature_breakdown: Optional[Dict[str, float]] = None


class RankingRequest(BaseModel):
    """Request for content ranking."""

    user_id: str
    query: Optional[str] = None
    content_types: List[ContentType] = Field(default_factory=lambda: [ContentType.ARTICLE])
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    # Context
    location: Optional[Dict[str, float]] = None  # lat, lng
    timezone: Optional[str] = None
    device_type: Optional[str] = None

    # Personalization settings
    enable_personalization: bool = True
    personalization_weights: Optional[Dict[str, float]] = None

    # Filtering
    topics: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    date_range: Optional[Dict[str, datetime]] = None  # start, end


class RankedArticle(BaseModel):
    """Ranked article result."""

    article: Article
    rank: int
    score: float
    personalized_score: Optional[float] = None
    explanation: Optional[str] = None
    feature_scores: Optional[Dict[str, float]] = None


class RankedResults(BaseModel):
    """Ranked content results."""

    articles: List[RankedArticle]
    total_count: int
    algorithm_variant: str
    processing_time_ms: float
    features_computed: int
    cache_hit_rate: float


class FeatureVector(BaseModel):
    """Feature vector for ML models."""

    article_id: str
    features: List[float]
    feature_names: List[str]
    computed_at: datetime


class ABTestVariant(BaseModel):
    """A/B test variant."""

    variant_id: str
    name: str
    weight: float = Field(..., ge=0.0, le=1.0)
    config: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class ABTestExperiment(BaseModel):
    """A/B test experiment."""

    experiment_id: str
    name: str
    variants: List[ABTestVariant]
    start_date: datetime
    end_date: Optional[datetime] = None
    is_active: bool = True


class RankingMetrics(BaseModel):
    """Ranking performance metrics."""

    total_requests: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    cache_hit_rate: float
    feature_computation_time_ms: float
    model_inference_time_ms: float
    error_rate: float


class TrendingScore(BaseModel):
    """Trending content score."""

    article_id: str
    trending_score: float
    velocity: float
    acceleration: float
    time_window_hours: int
    baseline_comparison: float


class AuthorityScore(BaseModel):
    """Source authority score."""

    source_id: str
    authority_score: float
    reliability_score: float
    popularity_score: float
    recency_score: float
    computed_at: datetime
