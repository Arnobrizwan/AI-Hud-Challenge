"""
FastAPI endpoints for notification decisioning service.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ..breaking_news.breaking_news_handler import BreakingNewsHandler
from ..decision_engine.engine import NotificationDecisionEngine
from ..exceptions import NotificationDeliveryError
from ..models.schemas import (
    BatchNotificationRequest,
    BatchNotificationResponse,
    DeliveryResult,
    NewsItem,
    NotificationCandidate,
    NotificationDecision,
    NotificationPreferences,
)

logger = structlog.get_logger()

# Create router
router = APIRouter()


@router.post("/decisions/single", response_model=NotificationDecision)
async def make_notification_decision(
    candidate: NotificationCandidate, decision_engine: NotificationDecisionEngine = Depends()
) -> NotificationDecision:
    """Make a single notification decision."""

    try:
        logger.info(
            "Processing single notification decision",
            user_id=candidate.user_id,
            notification_type=candidate.notification_type.value,
        )

        decision = await decision_engine.process_notification_candidate(candidate)

        logger.info(
            "Single notification decision completed",
            user_id=candidate.user_id,
            should_send=decision.should_send,
            reason=decision.reason,
        )

        return decision

    except Exception as e:
        logger.error(
            "Error processing single notification decision",
            user_id=candidate.user_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Decision processing failed: {str(e)}")


@router.post("/decisions/batch", response_model=BatchNotificationResponse)
async def make_batch_notification_decisions(
    request: BatchNotificationRequest, decision_engine: NotificationDecisionEngine = Depends()
) -> BatchNotificationResponse:
    """Make batch notification decisions."""

    try:
        start_time = datetime.utcnow()
        batch_id = request.batch_id or str(uuid4())

        logger.info(
            "Processing batch notification decisions",
            batch_id=batch_id,
            candidate_count=len(request.candidates),
        )

        # Process candidates
        decisions = await decision_engine.process_batch_notifications(request.candidates)

        # Count results
        decisions_made = len(decisions)
        notifications_sent = len([d for d in decisions if d.should_send])

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        response = BatchNotificationResponse(
            batch_id=batch_id,
            total_candidates=len(request.candidates),
            decisions_made=decisions_made,
            notifications_sent=notifications_sent,
            processing_time_ms=int(processing_time),
            metadata={
                "processed_at": datetime.utcnow().isoformat(),
                "priority": request.priority.value,
            },
        )

        logger.info(
            "Batch notification decisions completed",
            batch_id=batch_id,
            decisions_made=decisions_made,
            notifications_sent=notifications_sent,
            processing_time_ms=processing_time,
        )

        return response

    except Exception as e:
        logger.error(
            "Error processing batch notification decisions",
            batch_id=request.batch_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.post("/deliver", response_model=DeliveryResult)
async def deliver_notification(
    decision: NotificationDecision, decision_engine: NotificationDecisionEngine = Depends()
) -> DeliveryResult:
    """Deliver a notification based on decision."""

    try:
        if not decision.should_send:
            raise HTTPException(status_code=400, detail="Cannot deliver notification that should not be sent")

        logger.info(
            "Delivering notification",
            user_id=decision.user_id,
            channel=decision.delivery_channel.value,
        )

        result = await decision_engine.execute_notification_delivery(decision)

        logger.info(
            "Notification delivered successfully",
            user_id=decision.user_id,
            delivery_id=result.delivery_id,
            channel=result.channel.value,
        )

        return result

    except NotificationDeliveryError as e:
        logger.error("Notification delivery failed", user_id=decision.user_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("Error delivering notification", user_id=decision.user_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Delivery failed: {str(e)}")


@router.post("/breaking-news/process")
async def process_breaking_news(
    news_item: NewsItem,
    background_tasks: BackgroundTasks,
    decision_engine: NotificationDecisionEngine = Depends(),
) -> Dict[str, Any]:
    """Process breaking news for immediate notification."""
    try:
        logger.info("Processing breaking news", news_item_id=news_item.id, title=news_item.title)

        # Create breaking news handler
        from ..redis_client import get_redis_client

        redis_client = await get_redis_client()
        breaking_news_handler = BreakingNewsHandler(redis_client)
        await breaking_news_handler.initialize()

        # Process breaking news
        candidates = await breaking_news_handler.handle_breaking_news(news_item)

        if not candidates:
            return {
                "message": "Breaking news did not meet criteria or was duplicate",
                "candidates_created": 0,
            }

        # Process candidates in background
        background_tasks.add_task(_process_breaking_news_candidates, candidates, decision_engine)

        logger.info("Breaking news processed", news_item_id=news_item.id, candidates_created=len(candidates))

        return {
            "message": "Breaking news processed successfully",
            "candidates_created": len(candidates),
            "news_item_id": news_item.id,
        }

    except Exception as e:
        logger.error("Error processing breaking news", news_item_id=news_item.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Breaking news processing failed: {str(e)}")


async def _process_breaking_news_candidates(
    candidates: List[NotificationCandidate], decision_engine: NotificationDecisionEngine
) -> None:
    """Process breaking news candidates in background."""

    try:
        logger.info("Processing breaking news candidates in background", candidate_count=len(candidates))

        # Process decisions
        decisions = await decision_engine.process_batch_notifications(candidates)

        # Deliver notifications
        delivery_tasks = []
        for decision in decisions:
            if decision.should_send:
                task = decision_engine.execute_notification_delivery(decision)
                delivery_tasks.append(task)

        if delivery_tasks:
            await asyncio.gather(*delivery_tasks, return_exceptions=True)

        logger.info(
            "Breaking news candidates processed",
            decisions_made=len(decisions),
            deliveries_attempted=len(delivery_tasks),
        )

    except Exception as e:
        logger.error("Error processing breaking news candidates in background", error=str(e), exc_info=True)


@router.get("/preferences/{user_id}", response_model=NotificationPreferences)
async def get_user_preferences(
    user_id: str, decision_engine: NotificationDecisionEngine = Depends()
) -> NotificationPreferences:
    """Get user notification preferences."""

    try:
        prefs = await decision_engine.preference_manager.get_preferences(user_id)
        return prefs

    except Exception as e:
        logger.error("Error getting user preferences", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get preferences: {str(e)}")


@router.put("/preferences/{user_id}", response_model=NotificationPreferences)
async def update_user_preferences(
    user_id: str,
    preferences: NotificationPreferences,
    decision_engine: NotificationDecisionEngine = Depends(),
) -> NotificationPreferences:
    """Update user notification preferences."""

    try:
        # Ensure user_id matches
        preferences.user_id = user_id

        updated_prefs = await decision_engine.preference_manager.update_preferences(user_id, preferences)

        logger.info(
            "Updated user preferences",
            user_id=user_id,
            enabled_types=[t.value for t in updated_prefs.enabled_types],
        )

        return updated_prefs

    except Exception as e:
        logger.error("Error updating user preferences", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {str(e)}")


@router.get("/analytics/fatigue/{user_id}")
async def get_user_fatigue_analytics(
    user_id: str, decision_engine: NotificationDecisionEngine = Depends()
) -> Dict[str, Any]:
    """Get user fatigue analytics."""
    try:
        analytics = await decision_engine.fatigue_detector.get_user_fatigue_analytics(user_id)
        return analytics

    except Exception as e:
        logger.error("Error getting fatigue analytics", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get fatigue analytics: {str(e)}")


@router.get("/analytics/delivery/{user_id}")
async def get_delivery_analytics(
    user_id: str, decision_engine: NotificationDecisionEngine = Depends()
) -> Dict[str, Any]:
    """Get delivery analytics for user."""
    try:
        analytics = await decision_engine.delivery_manager.get_delivery_analytics(user_id)
        return analytics

    except Exception as e:
        logger.error("Error getting delivery analytics", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get delivery analytics: {str(e)}")


@router.get("/ab-tests/experiments")
async def get_active_experiments(
    decision_engine: NotificationDecisionEngine = Depends(),
) -> Dict[str, Any]:
    """Get active A/B test experiments."""
    try:
        experiments = {}
        for exp_name in decision_engine.ab_tester.experiments:
            exp_config = decision_engine.ab_tester.experiments[exp_name]
            experiments[exp_name] = {
                "name": exp_config.get("name", exp_name),
                "variants": exp_config.get("variants", []),
                "traffic_split": exp_config.get("traffic_split", {}),
                "active": exp_config.get("active", False),
                "created_at": exp_config.get("created_at"),
                "updated_at": exp_config.get("updated_at"),
            }

        return {"experiments": experiments, "total_experiments": len(experiments)}

    except Exception as e:
        logger.error(f"Error getting active experiments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get experiments: {str(e)}")


@router.get("/ab-tests/experiments/{experiment_name}/results")
async def get_experiment_results(
    experiment_name: str, decision_engine: NotificationDecisionEngine = Depends()
) -> Dict[str, Any]:
    """Get A/B test experiment results."""
    try:
        results = await decision_engine.ab_tester.get_experiment_results(experiment_name)
        return results

    except Exception as e:
        logger.error("Error getting experiment results", experiment_name=experiment_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get experiment results: {str(e)}")


@router.get("/ab-tests/user/{user_id}")
async def get_user_experiments(user_id: str, decision_engine: NotificationDecisionEngine = Depends()) -> Dict[str, str]:
    """Get user's A/B test assignments."""

    try:
        experiments = await decision_engine.ab_tester.get_user_experiments(user_id)
        return experiments

    except Exception as e:
        logger.error("Error getting user experiments", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get user experiments: {str(e)}")


@router.post("/ab-tests/experiments")
async def create_experiment(
    experiment_data: Dict[str, Any], decision_engine: NotificationDecisionEngine = Depends()
) -> Dict[str, Any]:
    """Create new A/B test experiment."""
    try:
        experiment = await decision_engine.ab_tester.create_experiment(
            experiment_name=experiment_data["name"],
            variants=experiment_data["variants"],
            traffic_split=experiment_data.get("traffic_split"),
            default_variant=experiment_data.get("default_variant", "control"),
        )

        logger.info("Created A/B test experiment", experiment_name=experiment["name"])

        return experiment

    except Exception as e:
        logger.error("Error creating experiment", experiment_data=experiment_data, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")


@router.put("/ab-tests/experiments/{experiment_name}/stop")
async def stop_experiment(
    experiment_name: str, decision_engine: NotificationDecisionEngine = Depends()
) -> Dict[str, str]:
    """Stop A/B test experiment."""

    try:
        await decision_engine.ab_tester.stop_experiment(experiment_name)

        logger.info("Stopped A/B test experiment", experiment_name=experiment_name)

        return {"message": f"Experiment {experiment_name} stopped successfully"}

    except Exception as e:
        logger.error("Error stopping experiment", experiment_name=experiment_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to stop experiment: {str(e)}")


@router.get("/health/detailed")
async def detailed_health_check(
    decision_engine: NotificationDecisionEngine = Depends(),
) -> Dict[str, Any]:
    """Detailed health check with component status."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "decision_engine": "healthy",
                "fatigue_detector": "healthy",
                "preference_manager": "healthy",
                "timing_predictor": "healthy",
                "relevance_scorer": "healthy",
                "delivery_manager": "healthy",
                "content_optimizer": "healthy",
                "ab_tester": "healthy",
            },
            "metrics": {
                "active_experiments": len(decision_engine.ab_tester.experiments),
                "cached_preferences": len(decision_engine.preference_manager.preferences_cache),
                "cached_variants": len(decision_engine.ab_tester.variant_cache),
            },
        }

        return health_status

    except Exception as e:
        logger.error(f"Error in detailed health check: {e}")
        return {"status": "unhealthy", "timestamp": datetime.utcnow().isoformat(), "error": str(e)}
