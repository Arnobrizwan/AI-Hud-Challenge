"""
FastAPI routes for feedback service
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..analytics.aggregator import DataAggregator
from ..annotation.manager import AnnotationManager
from ..crowdsourcing.manager import CrowdsourcingManager
from ..database.connection import get_db
from ..editorial_workflow.engine import EditorialWorkflowEngine
from ..feedback_collection.engine import FeedbackCollectionEngine
from ..models.schemas import (
    AnnotationCreate,
    AnnotationTask,
    AnnotationTaskCreate,
    Campaign,
    CampaignCreate,
    ContentItem,
    Feedback,
    FeedbackCreate,
    FeedbackInsights,
    PaginatedResponse,
    ProcessingResult,
    ReviewResultCreate,
    ReviewTask,
    ReviewTaskCreate,
    SuccessResponse,
    User,
)

logger = structlog.get_logger(__name__)

# Create routers
feedback_router = APIRouter()
editorial_router = APIRouter()
annotation_router = APIRouter()
analytics_router = APIRouter()


# Feedback routes
@feedback_router.post("/", response_model=ProcessingResult)
async def submit_feedback(
    feedback: FeedbackCreate, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)
):
    """Submit user feedback"""

    try:
        engine = FeedbackCollectionEngine(db)

        # Convert to UserFeedback for processing
        user_feedback = UserFeedback(
            id=UUID(),
            user_id=feedback.user_id,
            content_id=feedback.content_id,
            feedback_type=feedback.feedback_type,
            signal_type=feedback.signal_type,
            rating=feedback.rating,
            comment=feedback.comment,
            metadata=feedback.metadata,
            created_at=datetime.utcnow(),
        )

        result = await engine.process_user_feedback(user_feedback)

        # Add background task for additional processing
        background_tasks.add_task(process_feedback_background, user_feedback)

        return result

    except Exception as e:
        logger.error("Error submitting feedback", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@feedback_router.get("/", response_model=PaginatedResponse)
async def get_feedback(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    feedback_type: Optional[str] = None,
    user_id: Optional[UUID] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get feedback with pagination and filtering"""

    try:
        # This would implement actual pagination and filtering
        # For now, return mock data
        return PaginatedResponse(items=[], total=0, page=page, size=size, pages=0)

    except Exception as e:
        logger.error("Error getting feedback", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@feedback_router.get("/stats")
async def get_feedback_stats(hours: int = Query(24, ge=1, le=168), db: AsyncSession = Depends(get_db)):
    """Get feedback processing statistics"""

    try:
        engine = FeedbackCollectionEngine(db)
        time_window = timedelta(hours=hours)
        stats = await engine.get_feedback_stats(time_window)

        return stats

    except Exception as e:
        logger.error("Error getting feedback stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Editorial workflow routes
@editorial_router.post("/tasks", response_model=ReviewTask)
async def create_review_task(task: ReviewTaskCreate, db: AsyncSession = Depends(get_db)):
    """Create editorial review task"""

    try:
        engine = EditorialWorkflowEngine(db)
        result = await engine.create_review_task(
            content_id=task.content_id,
            task_type=task.task_type,
            priority=task.priority.value,
            created_by=task.created_by,
        )

        return result

    except Exception as e:
        logger.error("Error creating review task", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@editorial_router.post("/tasks/{task_id}/complete")
async def complete_review_task(task_id: UUID, review_result: ReviewResultCreate, db: AsyncSession = Depends(get_db)):
    """Complete editorial review task"""

    try:
        engine = EditorialWorkflowEngine(db)
        result = await engine.process_review_completion(task_id, review_result)

        return result

    except Exception as e:
        logger.error("Error completing review task", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@editorial_router.get("/tasks", response_model=List[ReviewTask])
async def get_review_tasks(
    user_id: Optional[UUID] = None, status: Optional[str] = None, db: AsyncSession = Depends(get_db)
):
    """Get review tasks"""

    try:
        engine = EditorialWorkflowEngine(db)

        if user_id:
            tasks = await engine.get_tasks_for_user(user_id, status)
        else:
            tasks = []

        return tasks

    except Exception as e:
        logger.error("Error getting review tasks", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@editorial_router.get("/tasks/overdue", response_model=List[ReviewTask])
async def get_overdue_tasks(db: AsyncSession = Depends(get_db)):
    """Get overdue review tasks"""

    try:
        engine = EditorialWorkflowEngine(db)
        tasks = await engine.get_overdue_tasks()

        return tasks

    except Exception as e:
        logger.error("Error getting overdue tasks", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Annotation routes
@annotation_router.post("/tasks", response_model=AnnotationTask)
async def create_annotation_task(task: AnnotationTaskCreate, db: AsyncSession = Depends(get_db)):
    """Create annotation task"""

    try:
        manager = AnnotationManager(db)
        result = await manager.create_annotation_task(
            content_id=task.content_id,
            annotation_type=task.annotation_type,
            annotator_ids=task.annotator_ids,
            guidelines=task.guidelines,
            deadline=task.deadline,
        )

        return result

    except Exception as e:
        logger.error("Error creating annotation task", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@annotation_router.post("/tasks/{task_id}/submit")
async def submit_annotation(task_id: UUID, annotation: AnnotationCreate, db: AsyncSession = Depends(get_db)):
    """Submit annotation"""

    try:
        manager = AnnotationManager(db)
        result = await manager.submit_annotation(
            task_id=task_id,
            annotator_id=annotation.annotator_id,
            annotation_data=annotation.annotation_data,
            confidence_score=annotation.confidence_score,
            time_spent_seconds=annotation.time_spent_seconds,
        )

        return result

    except Exception as e:
        logger.error("Error submitting annotation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@annotation_router.get("/tasks", response_model=List[AnnotationTask])
async def get_annotation_tasks(
    annotator_id: Optional[UUID] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get annotation tasks"""

    try:
        manager = AnnotationManager(db)

        if annotator_id:
            tasks = await manager.get_tasks_for_annotator(annotator_id, status)
        else:
            tasks = []

        return tasks

    except Exception as e:
        logger.error("Error getting annotation tasks", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Analytics routes
@analytics_router.get("/insights", response_model=FeedbackInsights)
async def get_feedback_insights(hours: int = Query(24, ge=1, le=168), db: AsyncSession = Depends(get_db)):
    """Get feedback insights and analytics"""

    try:
        aggregator = DataAggregator(db)
        time_window = timedelta(hours=hours)

        # Get feedback summary
        summary = await aggregator.aggregate_feedback(
            start_time=datetime.utcnow() - time_window, end_time=datetime.utcnow()
        )

        # Generate insights
        insights = await aggregator.generate_insights(summary)

        return FeedbackInsights(
            summary=summary,
            trends={},
            insights=insights,
            recommendations=[],
            generated_at=datetime.utcnow(),
        )

    except Exception as e:
        logger.error("Error getting feedback insights", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/metrics")
async def get_metrics(
    metric_name: Optional[str] = None,
    model_name: Optional[str] = None,
    hours: int = Query(24, ge=1, le=168),
    db: AsyncSession = Depends(get_db),
):
    """Get performance metrics"""

    try:
        aggregator = DataAggregator(db)
        time_window = timedelta(hours=hours)

        metrics = await aggregator.get_metrics(metric_name=metric_name, model_name=model_name, time_window=time_window)

        return metrics

    except Exception as e:
        logger.error("Error getting metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def process_feedback_background(feedback: UserFeedback) -> Dict[str, Any]:
    """Background task for additional feedback processing"""
    try:
        # This would implement additional background processing
        # such as updating ML models, sending notifications, etc.
        logger.info("Background feedback processing completed", feedback_id=str(feedback.id))

    except Exception as e:
        logger.error("Error in background feedback processing", error=str(e))
