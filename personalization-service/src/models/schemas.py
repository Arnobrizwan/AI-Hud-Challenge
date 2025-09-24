"""Pydantic schemas for the personalization service."""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class InteractionType(str, Enum):
    """Types of user interactions."""
    CLICK = "click"
    VIEW = "view"
    SHARE = "share"
    SAVE = "save"
    LIKE = "like"
    DISLIKE = "dislike"
    READ = "read"
    SKIP = "skip"


class AlgorithmType(str, Enum):
    """Personalization algorithm types."""
    HYBRID = "hybrid"
    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    BANDIT = "bandit"
    POPULARITY = "popularity"


class PersonalizationRequest(BaseModel):
    """Request for content personalization."""
    user_id: str
    candidates: List['ContentItem']
    context: Optional['UserContext'] = None
    diversity_params: Optional['DiversityParams'] = None
    max_results: int = Field(default=10, ge=1, le=100)
    include_explanation: bool = Field(default=False)


class ContentItem(BaseModel):
    """Content item representation."""
    id: str
    title: str
    content: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    source: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    content_features: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class UserContext(BaseModel):
    """User context information."""
    session_id: Optional[str] = None
    device_type: Optional[str] = None
    location: Optional[str] = None
    time_of_day: Optional[str] = None
    day_of_week: Optional[str] = None
    referrer: Optional[str] = None
    custom_context: Dict[str, Any] = Field(default_factory=dict)


class DiversityParams(BaseModel):
    """Diversity optimization parameters."""
    enable_diversity: bool = Field(default=True)
    topic_diversity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    source_diversity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_results: int = Field(default=10, ge=1, le=100)
    serendipity_weight: float = Field(default=0.2, ge=0.0, le=1.0)


class Recommendation(BaseModel):
    """Individual recommendation."""
    item_id: str
    score: float = Field(ge=0.0, le=1.0)
    method: str
    features: Dict[str, Any] = Field(default_factory=dict)
    explanation: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    source: Optional[str] = None


class BanditRecommendation(BaseModel):
    """Bandit-based recommendation."""
    item_id: str
    expected_reward: float
    uncertainty: float
    features: List[float]
    confidence: float = Field(ge=0.0, le=1.0)


class PersonalizedResponse(BaseModel):
    """Personalized content response."""
    user_id: str
    recommendations: List[Recommendation]
    algorithm_used: str
    personalization_strength: float = Field(ge=0.0, le=1.0)
    explanation: Optional[str] = None
    request_id: Optional[str] = None
    processing_time_ms: Optional[float] = None


class UserProfile(BaseModel):
    """User profile for personalization."""
    user_id: str
    created_at: datetime
    updated_at: datetime
    total_interactions: int = 0
    last_interaction_at: Optional[datetime] = None
    topic_preferences: Dict[str, float] = Field(default_factory=dict)
    source_preferences: Dict[str, float] = Field(default_factory=dict)
    reading_patterns: Dict[str, Any] = Field(default_factory=dict)
    collaborative_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    content_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    diversity_preference: float = Field(default=0.3, ge=0.0, le=1.0)
    serendipity_preference: float = Field(default=0.2, ge=0.0, le=1.0)
    demographic_data: Dict[str, Any] = Field(default_factory=dict)
    privacy_settings: Dict[str, Any] = Field(default_factory=dict)


class UserInteraction(BaseModel):
    """User interaction record."""
    user_id: str
    item_id: str
    interaction_type: InteractionType
    rating: Optional[float] = Field(None, ge=0.0, le=5.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    device_type: Optional[str] = None
    location: Optional[str] = None


class SimilarUser(BaseModel):
    """Similar user for collaborative filtering."""
    user_id: str
    similarity: float = Field(ge=0.0, le=1.0)
    common_interactions: int = 0


class ABExperiment(BaseModel):
    """A/B testing experiment."""
    experiment_id: str
    name: str
    description: Optional[str] = None
    variants: Dict[str, Any]
    traffic_allocation: Dict[str, float]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: str = "active"


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    model_name: str
    metric_name: str
    metric_value: float
    evaluation_date: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ColdStartProfile(BaseModel):
    """Cold start profile template."""
    profile_type: str
    template_profile: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class PrivacySettings(BaseModel):
    """Privacy settings for user data."""
    allow_personalization: bool = True
    allow_data_collection: bool = True
    allow_analytics: bool = True
    data_retention_days: int = 365
    anonymize_data: bool = False


# Update forward references
PersonalizationRequest.model_rebuild()
