"""
SQLAlchemy database models
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
)
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import INET, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database.connection import Base
from .schemas import (
    AnnotationType,
    CampaignStatus,
    FeedbackType,
    QualityLevel,
    ReviewDecision,
    SignalType,
    TaskPriority,
    TaskStatus,
)


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255))
    role = Column(String(50), nullable=False, default="annotator")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    feedback = relationship("Feedback", back_populates="user")
    review_tasks_assigned = relationship(
        "ReviewTask", foreign_keys="ReviewTask.assigned_to", back_populates="assigned_user"
    )
    review_tasks_created = relationship(
        "ReviewTask", foreign_keys="ReviewTask.created_by", back_populates="created_by_user"
    )
    annotations = relationship("Annotation", back_populates="annotator")
    campaigns_created = relationship("Campaign", back_populates="creator")


class ContentItem(Base):
    __tablename__ = "content_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    external_id = Column(String(255), index=True)
    title = Column(Text)
    content = Column(Text, nullable=False)
    content_type = Column(String(50), default="text")
    category = Column(String(100), index=True)
    source = Column(String(100))
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    feedback = relationship("Feedback", back_populates="content")
    review_tasks = relationship("ReviewTask", back_populates="content")
    annotation_tasks = relationship("AnnotationTask", back_populates="content")
    quality_assessments = relationship("QualityAssessment", back_populates="content")
    campaign_tasks = relationship("CampaignTask", back_populates="content")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    content_id = Column(
        UUID(as_uuid=True), ForeignKey("content_items.id"), nullable=False, index=True
    )
    feedback_type = Column(SQLEnum(FeedbackType), nullable=False, index=True)
    signal_type = Column(SQLEnum(SignalType), index=True)
    rating = Column(Float, index=True)
    comment = Column(Text)
    metadata = Column(JSON, default={})
    processed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    user = relationship("User", back_populates="feedback")
    content = relationship("ContentItem", back_populates="feedback")
    processing_results = relationship("FeedbackProcessingResult", back_populates="feedback")


class FeedbackProcessingResult(Base):
    __tablename__ = "feedback_processing_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    feedback_id = Column(
        UUID(as_uuid=True), ForeignKey("feedback.id", ondelete="CASCADE"), nullable=False
    )
    quality_score = Column(Float)
    sentiment_score = Column(Float)
    confidence_score = Column(Float)
    actions_taken = Column(JSON, default=[])
    processing_metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    feedback = relationship("Feedback", back_populates="processing_results")


class ReviewTask(Base):
    __tablename__ = "review_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    content_id = Column(
        UUID(as_uuid=True), ForeignKey("content_items.id"), nullable=False, index=True
    )
    task_type = Column(String(100), nullable=False)
    priority = Column(SQLEnum(TaskPriority), default=TaskPriority.NORMAL, index=True)
    assigned_to = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, index=True)
    due_date = Column(DateTime(timezone=True), index=True)
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    content = relationship("ContentItem", back_populates="review_tasks")
    assigned_user = relationship(
        "User", foreign_keys=[assigned_to], back_populates="review_tasks_assigned"
    )
    created_by_user = relationship(
        "User", foreign_keys=[created_by], back_populates="review_tasks_created"
    )
    review_results = relationship("ReviewResult", back_populates="task")


class ReviewResult(Base):
    __tablename__ = "review_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    task_id = Column(
        UUID(as_uuid=True), ForeignKey("review_tasks.id", ondelete="CASCADE"), nullable=False
    )
    reviewer_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    decision = Column(SQLEnum(ReviewDecision), nullable=False)
    comments = Column(Text)
    changes_requested = Column(Text)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    task = relationship("ReviewTask", back_populates="review_results")
    reviewer = relationship("User")


class AnnotationTask(Base):
    __tablename__ = "annotation_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    content_id = Column(
        UUID(as_uuid=True), ForeignKey("content_items.id"), nullable=False, index=True
    )
    annotation_type = Column(SQLEnum(AnnotationType), nullable=False)
    guidelines = Column(Text)
    deadline = Column(DateTime(timezone=True))
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    content = relationship("ContentItem", back_populates="annotation_tasks")
    assignments = relationship("AnnotationAssignment", back_populates="task")
    annotations = relationship("Annotation", back_populates="task")


class AnnotationAssignment(Base):
    __tablename__ = "annotation_assignments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    task_id = Column(
        UUID(as_uuid=True), ForeignKey("annotation_tasks.id", ondelete="CASCADE"), nullable=False
    )
    annotator_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    assigned_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))

    # Relationships
    task = relationship("AnnotationTask", back_populates="assignments")
    annotator = relationship("User")


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("annotation_tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    annotator_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    annotation_data = Column(JSON, nullable=False)
    confidence_score = Column(Float)
    time_spent_seconds = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    task = relationship("AnnotationTask", back_populates="annotations")
    annotator = relationship("User", back_populates="annotations")


class QualityAssessment(Base):
    __tablename__ = "quality_assessments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    content_id = Column(
        UUID(as_uuid=True), ForeignKey("content_items.id"), nullable=False, index=True
    )
    overall_quality_score = Column(Float, nullable=False)
    factual_accuracy = Column(Float)
    bias_score = Column(Float)
    readability_score = Column(Float)
    completeness_score = Column(Float)
    spam_likelihood = Column(Float)
    needs_human_review = Column(Boolean, default=False)
    assessment_metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    content = relationship("ContentItem", back_populates="quality_assessments")


class ModerationAction(Base):
    __tablename__ = "moderation_actions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    content_id = Column(
        UUID(as_uuid=True), ForeignKey("content_items.id"), nullable=False, index=True
    )
    action = Column(String(50), nullable=False)
    reason = Column(Text)
    severity = Column(String(20))
    moderator_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    content = relationship("ContentItem")
    moderator = relationship("User")


class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    task_type = Column(String(100), nullable=False)
    target_annotations = Column(Integer)
    reward_per_task = Column(Float)
    quality_threshold = Column(Float, default=0.7)
    status = Column(SQLEnum(CampaignStatus), default=CampaignStatus.DRAFT, index=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    creator = relationship("User", back_populates="campaigns_created")
    tasks = relationship("CampaignTask", back_populates="campaign")


class CampaignTask(Base):
    __tablename__ = "campaign_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    campaign_id = Column(
        UUID(as_uuid=True),
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content_id = Column(
        UUID(as_uuid=True), ForeignKey("content_items.id"), nullable=False, index=True
    )
    task_data = Column(JSON, nullable=False)
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    campaign = relationship("Campaign", back_populates="tasks")
    content = relationship("ContentItem", back_populates="campaign_tasks")
    submissions = relationship("CampaignSubmission", back_populates="task")


class CampaignSubmission(Base):
    __tablename__ = "campaign_submissions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("campaign_tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    worker_id = Column(String(255), nullable=False)
    submission_data = Column(JSON, nullable=False)
    quality_score = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    task = relationship("CampaignTask", back_populates="submissions")


class TrainingBatch(Base):
    __tablename__ = "training_batches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_name = Column(String(100), nullable=False, index=True)
    batch_data = Column(JSON, nullable=False)
    example_count = Column(Integer)
    status = Column(String(50), default="pending", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelUpdate(Base):
    __tablename__ = "model_updates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_name = Column(String(100), nullable=False, index=True)
    training_batch_id = Column(UUID(as_uuid=True), ForeignKey("training_batches.id"))
    performance_metrics = Column(JSON, default={})
    status = Column(String(50), default="pending", index=True)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    training_batch = relationship("TrainingBatch")


class FeedbackInsight(Base):
    __tablename__ = "feedback_insights"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    insight_type = Column(String(100), nullable=False)
    insight_data = Column(JSON, nullable=False)
    time_window_start = Column(DateTime(timezone=True))
    time_window_end = Column(DateTime(timezone=True))
    generated_at = Column(DateTime(timezone=True), server_default=func.now())


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    model_name = Column(String(100), index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    metadata = Column(JSON, default={})


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(UUID(as_uuid=True))
    old_values = Column(JSON)
    new_values = Column(JSON)
    ip_address = Column(INET)
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    user = relationship("User")


# Indexes for performance
Index("idx_feedback_user_created", Feedback.user_id, Feedback.created_at)
Index("idx_feedback_content_created", Feedback.content_id, Feedback.created_at)
Index("idx_review_tasks_status_due", ReviewTask.status, ReviewTask.due_date)
Index("idx_annotations_task_annotator", Annotation.task_id, Annotation.annotator_id)
Index("idx_quality_assessments_score", QualityAssessment.overall_quality_score)
Index("idx_campaign_tasks_campaign_status", CampaignTask.campaign_id, CampaignTask.status)
Index(
    "idx_performance_metrics_model_timestamp",
    PerformanceMetric.model_name,
    PerformanceMetric.timestamp,
)
