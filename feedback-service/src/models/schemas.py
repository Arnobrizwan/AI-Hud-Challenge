"""
Pydantic schemas for API requests and responses
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


# Enums
class FeedbackType(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    CROWDSOURCED = "crowdsourced"
    EDITORIAL = "editorial"


class SignalType(str, Enum):
    CLICK = "click"
    DWELL_TIME = "dwell_time"
    SHARE = "share"
    LIKE = "like"
    DISLIKE = "dislike"
    RATING = "rating"
    COMMENT = "comment"
    COMPLAINT = "complaint"
    REPORT = "report"


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    OVERDUE = "overdue"


class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    ESCALATE = "escalate"


class AnnotationType(str, Enum):
    SENTIMENT = "sentiment"
    TOPIC = "topic"
    QUALITY = "quality"
    BIAS = "bias"
    FACTUAL = "factual"
    COMPLETENESS = "completeness"


class CampaignStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class QualityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXPERT = "expert"


# Base schemas
class BaseSchema(BaseModel):
    """Base schema with common fields"""

    class Config:
        from_attributes = True
        use_enum_values = True


# User schemas
class UserBase(BaseSchema):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    full_name: Optional[str] = None
    role: str = "annotator"


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserUpdate(BaseSchema):
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class User(UserBase):
    id: UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime


# Content schemas
class ContentItemBase(BaseSchema):
    external_id: Optional[str] = None
    title: Optional[str] = None
    content: str
    content_type: str = "text"
    category: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ContentItemCreate(ContentItemBase):
    pass


class ContentItemUpdate(BaseSchema):
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ContentItem(ContentItemBase):
    id: UUID
    created_at: datetime
    updated_at: datetime


# Feedback schemas
class UserFeedback(BaseSchema):
    id: UUID
    user_id: Optional[UUID] = None
    content_id: UUID
    feedback_type: FeedbackType
    signal_type: Optional[SignalType] = None
    rating: Optional[float] = Field(None, ge=0, le=5)
    comment: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime


class FeedbackBase(BaseSchema):
    content_id: UUID
    feedback_type: FeedbackType
    signal_type: Optional[SignalType] = None
    rating: Optional[float] = Field(None, ge=0, le=5)
    comment: Optional[str] = None
    metadata: Dict[str, Any] = {}


class FeedbackCreate(FeedbackBase):
    user_id: Optional[UUID] = None


class FeedbackUpdate(BaseSchema):
    rating: Optional[float] = Field(None, ge=0, le=5)
    comment: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Feedback(FeedbackBase):
    id: UUID
    user_id: Optional[UUID]
    processed_at: Optional[datetime]
    created_at: datetime


class ProcessingResult(BaseSchema):
    feedback_id: UUID
    processed_at: datetime
    quality_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    confidence_score: Optional[float] = None
    actions_taken: List[str] = []
    requires_immediate_attention: bool = False


# Editorial workflow schemas
class ReviewTaskBase(BaseSchema):
    content_id: UUID
    task_type: str
    priority: TaskPriority = TaskPriority.NORMAL
    assigned_to: Optional[UUID] = None
    due_date: Optional[datetime] = None


class ReviewTaskCreate(ReviewTaskBase):
    created_by: Optional[UUID] = None


class ReviewTaskUpdate(BaseSchema):
    assigned_to: Optional[UUID] = None
    status: Optional[TaskStatus] = None
    due_date: Optional[datetime] = None


class ReviewTask(ReviewTaskBase):
    id: UUID
    created_by: Optional[UUID]
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


class ReviewResultBase(BaseSchema):
    decision: ReviewDecision
    comments: Optional[str] = None
    changes_requested: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ReviewResultCreate(ReviewResultBase):
    task_id: UUID
    reviewer_id: UUID


class ReviewResult(ReviewResultBase):
    id: UUID
    task_id: UUID
    reviewer_id: UUID
    created_at: datetime


# Annotation schemas
class AnnotationTaskBase(BaseSchema):
    content_id: UUID
    annotation_type: AnnotationType
    guidelines: Optional[str] = None
    deadline: Optional[datetime] = None


class AnnotationTaskCreate(AnnotationTaskBase):
    annotator_ids: List[UUID] = []


class AnnotationTaskUpdate(BaseSchema):
    status: Optional[TaskStatus] = None
    deadline: Optional[datetime] = None


class AnnotationTask(AnnotationTaskBase):
    id: UUID
    status: TaskStatus
    created_at: datetime
    updated_at: datetime


class AnnotationBase(BaseSchema):
    annotation_data: Dict[str, Any]
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    time_spent_seconds: Optional[int] = None


class AnnotationCreate(AnnotationBase):
    task_id: UUID
    annotator_id: UUID


class Annotation(AnnotationBase):
    id: UUID
    task_id: UUID
    annotator_id: UUID
    created_at: datetime


# Quality assurance schemas
class QualityAssessment(BaseSchema):
    content_id: UUID
    overall_quality_score: float = Field(..., ge=0, le=1)
    factual_accuracy: Optional[float] = Field(None, ge=0, le=1)
    bias_score: Optional[float] = Field(None, ge=0, le=1)
    readability_score: Optional[float] = Field(None, ge=0, le=1)
    completeness_score: Optional[float] = Field(None, ge=0, le=1)
    spam_likelihood: Optional[float] = Field(None, ge=0, le=1)
    needs_human_review: bool = False
    assessment_metadata: Dict[str, Any] = {}


class ModerationAction(BaseSchema):
    content_id: UUID
    action: str
    reason: str
    severity: str
    moderator_id: Optional[UUID] = None


# Crowdsourcing schemas
class CampaignBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    task_type: str
    target_annotations: Optional[int] = None
    reward_per_task: Optional[float] = Field(None, ge=0)
    quality_threshold: float = Field(0.7, ge=0, le=1)


class CampaignCreate(CampaignBase):
    pass


class CampaignUpdate(BaseSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[CampaignStatus] = None


class Campaign(CampaignBase):
    id: str
    status: CampaignStatus
    created_by: Optional[str]
    created_at: datetime
    updated_at: datetime


class CampaignTask(BaseSchema):
    id: str
    campaign_id: str
    content_id: str
    task_data: Dict[str, Any]
    status: TaskStatus
    created_at: datetime


class CampaignSubmission(BaseSchema):
    id: str
    task_id: str
    worker_id: str
    submission_data: Dict[str, Any]
    quality_score: Optional[float] = None
    created_at: datetime


# Analytics schemas
class FeedbackInsights(BaseSchema):
    summary: Dict[str, Any]
    trends: Dict[str, Any]
    insights: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime


class PerformanceMetrics(BaseSchema):
    metric_name: str
    metric_value: float
    model_name: Optional[str] = None
    timestamp: datetime
    metadata: Dict[str, Any] = {}


# WebSocket schemas
class WebSocketMessage(BaseSchema):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RealtimeFeedback(BaseSchema):
    user_id: UUID
    content_id: UUID
    feedback_type: FeedbackType
    signal_type: Optional[SignalType] = None
    rating: Optional[float] = Field(None, ge=0, le=5)
    comment: Optional[str] = None
    metadata: Dict[str, Any] = {}


# Error schemas
class ErrorResponse(BaseSchema):
    error: str
    status_code: int
    details: Optional[Dict[str, Any]] = None


# Response schemas
class SuccessResponse(BaseSchema):
    message: str
    data: Optional[Dict[str, Any]] = None


class PaginatedResponse(BaseSchema):
    items: List[Dict[str, Any]]
    total: int
    page: int
    size: int
    pages: int


# Training and model schemas
class TrainingBatch(BaseSchema):
    id: str
    model_name: str
    batch_data: Dict[str, Any]
    example_count: int
    status: str
    created_at: datetime


class ModelUpdate(BaseSchema):
    id: str
    model_name: str
    training_batch_id: str
    performance_metrics: Dict[str, Any]
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime


class UncertainPrediction(BaseSchema):
    prediction_id: str
    model_name: str
    content_id: str
    uncertainty_score: float
    prediction_confidence: float
    requires_review: bool
