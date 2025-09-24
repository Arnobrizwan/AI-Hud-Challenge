"""
Pydantic schemas for notification decisioning service.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from pydantic import BaseModel, Field


class NotificationType(str, Enum):
    """Types of notifications."""
    BREAKING_NEWS = "breaking_news"
    PERSONALIZED = "personalized"
    TRENDING = "trending"
    DIGEST = "digest"
    URGENT = "urgent"
    MARKETING = "marketing"


class DeliveryChannel(str, Enum):
    """Delivery channels for notifications."""
    PUSH = "push"
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"


class Priority(str, Enum):
    """Notification priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class NotificationContent(BaseModel):
    """Notification content model."""
    title: str = Field(..., max_length=100)
    body: str = Field(..., max_length=500)
    action_url: Optional[str] = None
    image_url: Optional[str] = None
    category: str
    priority: Priority = Priority.MEDIUM
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NewsItem(BaseModel):
    """News item model."""
    id: str
    title: str
    content: str
    url: str
    image_url: Optional[str] = None
    published_at: datetime
    category: str
    topics: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    source: str
    is_breaking: bool = False
    urgency_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NotificationCandidate(BaseModel):
    """Notification candidate for decisioning."""
    user_id: str
    content: NewsItem
    notification_type: NotificationType
    urgency_score: float = Field(ge=0.0, le=1.0)
    priority: Priority = Priority.MEDIUM
    bypass_fatigue: bool = False
    immediate_delivery: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NotificationDecision(BaseModel):
    """Notification decision result."""
    should_send: bool
    user_id: str
    reason: Optional[str] = None
    delivery_time: Optional[datetime] = None
    delivery_channel: Optional[DeliveryChannel] = None
    content: Optional[NotificationContent] = None
    priority: Optional[Priority] = None
    strategy_variant: Optional[str] = None
    score: Optional[float] = None
    threshold: Optional[float] = None
    next_eligible_time: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeliveryResult(BaseModel):
    """Notification delivery result."""
    success: bool
    delivery_id: str
    channel: DeliveryChannel
    delivered_at: datetime
    error_message: Optional[str] = None
    retry_count: int = 0
    was_engaged: bool = False
    engagement_time: Optional[datetime] = None
    original_features: Optional[Dict[str, Any]] = None


class OptimalTiming(BaseModel):
    """Optimal timing prediction result."""
    scheduled_time: datetime
    predicted_engagement: float = Field(ge=0.0, le=1.0)
    alternative_times: List[datetime] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class TimingPrediction(BaseModel):
    """Timing prediction for a specific time window."""
    scheduled_time: datetime
    engagement_probability: float = Field(ge=0.0, le=1.0)
    features: Dict[str, Any] = Field(default_factory=dict)


class FatigueCheck(BaseModel):
    """Notification fatigue check result."""
    is_fatigued: bool
    hourly_count: int
    daily_count: int
    next_eligible_time: Optional[datetime] = None
    fatigue_score: float = Field(ge=0.0, le=1.0)


class NotificationPreferences(BaseModel):
    """User notification preferences."""
    user_id: str
    enabled_types: List[NotificationType] = Field(default_factory=list)
    delivery_channels: List[DeliveryChannel] = Field(default_factory=list)
    quiet_hours_start: Optional[int] = None  # Hour in 24h format
    quiet_hours_end: Optional[int] = None
    timezone: str = "UTC"
    allow_emojis: bool = True
    max_daily_notifications: int = 50
    max_hourly_notifications: int = 10
    relevance_thresholds: Dict[NotificationType, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_threshold(self, notification_type: NotificationType) -> float:
        """Get relevance threshold for notification type."""
        return self.relevance_thresholds.get(notification_type, 0.3)
    
    def is_notification_type_enabled(self, notification_type: NotificationType) -> bool:
        """Check if notification type is enabled."""
        return notification_type in self.enabled_types


class UserProfile(BaseModel):
    """User profile for personalization."""
    user_id: str
    topic_preferences: List[str] = Field(default_factory=list)
    source_preferences: List[str] = Field(default_factory=list)
    location_preferences: List[str] = Field(default_factory=list)
    engagement_history: List[Dict[str, Any]] = Field(default_factory=list)
    device_info: Dict[str, Any] = Field(default_factory=dict)
    timezone: str = "UTC"
    created_at: datetime
    updated_at: datetime


class ABTestVariant(BaseModel):
    """A/B test variant."""
    experiment_name: str
    variant_name: str
    user_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class NotificationAnalytics(BaseModel):
    """Notification analytics data."""
    notification_id: str
    user_id: str
    notification_type: NotificationType
    channel: DeliveryChannel
    sent_at: datetime
    delivered_at: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    clicked_at: Optional[datetime] = None
    engagement_score: Optional[float] = None
    delivery_duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchNotificationRequest(BaseModel):
    """Request for batch notification processing."""
    candidates: List[NotificationCandidate]
    batch_id: Optional[str] = None
    priority: Priority = Priority.MEDIUM
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchNotificationResponse(BaseModel):
    """Response for batch notification processing."""
    batch_id: str
    total_candidates: int
    decisions_made: int
    notifications_sent: int
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
