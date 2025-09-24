"""
Online Evaluation Router - A/B testing and online evaluation endpoints
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from ..core import EvaluationEngine
from ..dependencies import get_evaluation_engine

logger = logging.getLogger(__name__)

online_evaluation_router = APIRouter()


@online_evaluation_router.post("/experiments")
async def create_experiment(
    experiment_config: Dict[str, Any],
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """Create a new A/B test experiment"""

    try:
        logger.info("Creating A/B test experiment")

        experiment = await evaluation_engine.ab_tester.create_experiment(experiment_config)

        return {
            "status": "success",
            "experiment_id": experiment.id,
            "experiment_name": experiment.name,
            "status": experiment.status.value,
            "message": "Experiment created successfully",
        }

    except Exception as e:
        logger.error(f"Error creating experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@online_evaluation_router.post("/experiments/{experiment_id}/start")
async def start_experiment(
    experiment_id: str, evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine)
):
    """Start an A/B test experiment"""

    try:
        logger.info(f"Starting experiment {experiment_id}")

        success = await evaluation_engine.ab_tester.start_experiment(experiment_id)

        if not success:
            raise HTTPException(status_code=404, detail="Experiment not found or cannot be started")

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "message": "Experiment started successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@online_evaluation_router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: str, evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine)
):
    """Stop an A/B test experiment"""

    try:
        logger.info(f"Stopping experiment {experiment_id}")

        success = await evaluation_engine.ab_tester.stop_experiment(experiment_id)

        if not success:
            raise HTTPException(status_code=404, detail="Experiment not found or cannot be stopped")

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "message": "Experiment stopped successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@online_evaluation_router.post("/experiments/{experiment_id}/analyze")
async def analyze_experiment(
    experiment_id: str,
    analysis_type: str = "frequentist",
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """Analyze an A/B test experiment"""

    try:
        logger.info(f"Analyzing experiment {experiment_id} with {analysis_type} analysis")

        analysis = await evaluation_engine.ab_tester.analyze_experiment(
            experiment_id, analysis_type
        )

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "analysis_type": analysis_type,
            "analysis": analysis.dict(),
            "message": "Experiment analysis completed",
        }

    except Exception as e:
        logger.error(f"Error analyzing experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@online_evaluation_router.post("/experiments/{experiment_id}/events")
async def record_event(
    experiment_id: str,
    user_id: str,
    variant: str,
    event_type: str,
    value: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """Record an event for an experiment"""

    try:
        logger.info(f"Recording event for experiment {experiment_id}")

        success = await evaluation_engine.ab_tester.record_event(
            experiment_id, user_id, variant, event_type, value, metadata
        )

        if not success:
            raise HTTPException(
                status_code=404, detail="Experiment not found or event recording failed"
            )

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "user_id": user_id,
            "variant": variant,
            "event_type": event_type,
            "message": "Event recorded successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@online_evaluation_router.get("/experiments")
async def list_experiments(
    status: Optional[str] = None,
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """List A/B test experiments"""

    try:
        logger.info("Listing experiments")

        experiments = await evaluation_engine.ab_tester.list_experiments(status)

        return {
            "status": "success",
            "experiments": [exp.dict() for exp in experiments],
            "total_count": len(experiments),
            "message": "Experiments retrieved successfully",
        }

    except Exception as e:
        logger.error(f"Error listing experiments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@online_evaluation_router.get("/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: str, evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine)
):
    """Get experiment details"""

    try:
        logger.info(f"Getting experiment {experiment_id}")

        experiment = await evaluation_engine.ab_tester.get_experiment(experiment_id)

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        return {
            "status": "success",
            "experiment": experiment.dict(),
            "message": "Experiment retrieved successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@online_evaluation_router.post("/bandits")
async def create_bandit(
    bandit_config: Dict[str, Any],
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """Create a new multi-armed bandit"""

    try:
        logger.info("Creating multi-armed bandit")

        bandit_id = await evaluation_engine.online_evaluator.bandit_tester.create_bandit(
            bandit_config
        )

        return {
            "status": "success",
            "bandit_id": bandit_id,
            "message": "Multi-armed bandit created successfully",
        }

    except Exception as e:
        logger.error(f"Error creating bandit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@online_evaluation_router.post("/bandits/{bandit_id}/select")
async def select_arm(
    bandit_id: str,
    context: Optional[Dict[str, Any]] = None,
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """Select an arm from a multi-armed bandit"""

    try:
        logger.info(f"Selecting arm for bandit {bandit_id}")

        arm = await evaluation_engine.online_evaluator.bandit_tester.select_arm(bandit_id, context)

        return {
            "status": "success",
            "bandit_id": bandit_id,
            "selected_arm": arm,
            "message": "Arm selected successfully",
        }

    except Exception as e:
        logger.error(f"Error selecting arm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@online_evaluation_router.post("/bandits/{bandit_id}/reward")
async def update_reward(
    bandit_id: str,
    arm: str,
    reward: float,
    context: Optional[Dict[str, Any]] = None,
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """Update reward for a bandit arm"""

    try:
        logger.info(f"Updating reward for bandit {bandit_id}, arm {arm}")

        success = await evaluation_engine.online_evaluator.bandit_tester.update_reward(
            bandit_id, arm, reward, context
        )

        if not success:
            raise HTTPException(status_code=404, detail="Bandit not found or reward update failed")

        return {
            "status": "success",
            "bandit_id": bandit_id,
            "arm": arm,
            "reward": reward,
            "message": "Reward updated successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating reward: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
