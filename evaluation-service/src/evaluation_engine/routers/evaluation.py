"""
Main Evaluation Router - Core evaluation endpoints
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ..core import EvaluationEngine
from ..dependencies import get_evaluation_engine
from ..models import (
    CreateEvaluationRequest,
    CreateEvaluationResponse,
    EvaluationStatus,
    GetEvaluationResponse,
    ListEvaluationsResponse,
)

logger = logging.getLogger(__name__)

evaluation_router = APIRouter()


@evaluation_router.post("/comprehensive", response_model=CreateEvaluationResponse)
async def create_comprehensive_evaluation(
    request: CreateEvaluationRequest,
    background_tasks: BackgroundTasks,
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """Create a comprehensive evaluation"""

    try:
        logger.info("Creating comprehensive evaluation")

        # Run evaluation in background
        evaluation = await evaluation_engine.run_comprehensive_evaluation(request.config)

        return CreateEvaluationResponse(
            evaluation_id=evaluation.evaluation_id,
            status=evaluation.status,
            message="Comprehensive evaluation created successfully",
        )

    except Exception as e:
        logger.error(f"Error creating comprehensive evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@evaluation_router.post("/offline", response_model=CreateEvaluationResponse)
async def create_offline_evaluation(
    models: List[dict],
    datasets: List[dict],
    metrics: dict,
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """Create an offline evaluation"""

    try:
        logger.info("Creating offline evaluation")

        evaluation = await evaluation_engine.run_offline_evaluation(models, datasets, metrics)

        return CreateEvaluationResponse(
            evaluation_id=evaluation.evaluation_id,
            status=evaluation.status,
            message="Offline evaluation created successfully",
        )

    except Exception as e:
        logger.error(f"Error creating offline evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@evaluation_router.post("/online", response_model=CreateEvaluationResponse)
async def create_online_evaluation(
    experiments: List[dict],
    evaluation_period: dict,
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """Create an online evaluation"""

    try:
        logger.info("Creating online evaluation")

        evaluation = await evaluation_engine.run_online_evaluation(experiments, evaluation_period)

        return CreateEvaluationResponse(
            evaluation_id=evaluation.evaluation_id,
            status=evaluation.status,
            message="Online evaluation created successfully",
        )

    except Exception as e:
        logger.error(f"Error creating online evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@evaluation_router.get("/{evaluation_id}", response_model=GetEvaluationResponse)
async def get_evaluation(
    evaluation_id: str, evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine)
):
    """Get evaluation details"""

    try:
        evaluation = await evaluation_engine.get_evaluation_status(evaluation_id)

        if not evaluation:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        return GetEvaluationResponse(evaluation=evaluation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@evaluation_router.get("/", response_model=ListEvaluationsResponse)
async def list_evaluations(
    status: Optional[EvaluationStatus] = None,
    limit: int = 100,
    offset: int = 0,
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """List evaluations with optional filtering"""

    try:
        evaluations = await evaluation_engine.list_evaluations(status, limit, offset)

        return ListEvaluationsResponse(
            evaluations=evaluations, total_count=len(evaluations), limit=limit, offset=offset
        )

    except Exception as e:
        logger.error(f"Error listing evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@evaluation_router.delete("/{evaluation_id}")
async def cancel_evaluation(
    evaluation_id: str, evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine)
):
    """Cancel a running evaluation"""

    try:
        success = await evaluation_engine.cancel_evaluation(evaluation_id)

        if not success:
            raise HTTPException(status_code=404, detail="Evaluation not found or not cancellable")

        return {"message": "Evaluation cancelled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
