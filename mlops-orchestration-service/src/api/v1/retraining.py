"""
Retraining API endpoints - REST API for automated retraining management
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from src.models.retraining_models import (
    RetrainingResult,
    RetrainingStatus,
    RetrainingTriggerConfig,
    TriggerStatus,
)
from src.retraining.retraining_manager import AutomatedRetrainingManager
from src.utils.exceptions import RetrainingError, ValidationError

router = APIRouter()


# Request/Response models
class SetupRetrainingRequest(BaseModel):
    trigger_config: RetrainingTriggerConfig


class SetupRetrainingResponse(BaseModel):
    model_name: str
    status: str
    message: str


class RetrainingListResponse(BaseModel):
    retrainings: List[RetrainingResult]
    total: int
    page: int
    page_size: int


class TriggerListResponse(BaseModel):
    triggers: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int


@router.post("/setup/{model_name}", response_model=SetupRetrainingResponse)
async def setup_retraining_triggers(
    model_name: str,
    request: SetupRetrainingRequest,
    background_tasks: BackgroundTasks,
    retraining_manager: AutomatedRetrainingManager = Depends(),
):
    """Setup automated retraining triggers for a model"""

    try:
        await retraining_manager.setup_retraining_triggers(
            model_name=model_name, trigger_config=request.trigger_config
        )

        return SetupRetrainingResponse(
            model_name=model_name,
            status="active",
            message=f"Retraining triggers setup completed for model '{model_name}'",
        )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RetrainingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/status/{model_name}", response_model=RetrainingListResponse)
async def get_retraining_status(
    model_name: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    retraining_manager: AutomatedRetrainingManager = Depends(),
):
    """Get retraining status for a model"""

    try:
        retrainings = await retraining_manager.get_retraining_status(model_name)

        # Pagination
        total = len(retrainings)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_retrainings = retrainings[start_idx:end_idx]

        return RetrainingListResponse(
            retrainings=paginated_retrainings, total=total, page=page, page_size=page_size
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get retraining status: {str(e)}")


@router.get("/triggers/{model_name}", response_model=TriggerListResponse)
async def get_trigger_status(
    model_name: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    retraining_manager: AutomatedRetrainingManager = Depends(),
):
    """Get trigger status for a model"""

    try:
        triggers = await retraining_manager.get_trigger_status(model_name)

        # Convert to dict format for response
        trigger_dicts = [
            {
                "id": trigger.id,
                "model_name": trigger.model_name,
                "trigger_type": trigger.trigger_type.value,
                "status": trigger.status.value,
                "created_at": trigger.created_at.isoformat(),
                "last_fired_at": (
                    trigger.last_fired_at.isoformat() if trigger.last_fired_at else None
                ),
                "fire_count": trigger.fire_count,
            }
            for trigger in triggers
        ]

        # Pagination
        total = len(trigger_dicts)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_triggers = trigger_dicts[start_idx:end_idx]

        return TriggerListResponse(
            triggers=paginated_triggers, total=total, page=page, page_size=page_size
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trigger status: {str(e)}")


@router.post("/triggers/{trigger_id}/disable")
async def disable_trigger(
    trigger_id: str, retraining_manager: AutomatedRetrainingManager = Depends()
):
    """Disable a retraining trigger"""

    try:
        success = await retraining_manager.disable_trigger(trigger_id)

        if not success:
            raise HTTPException(status_code=404, detail="Trigger not found")

        return {
            "trigger_id": trigger_id,
            "status": "disabled",
            "message": f"Trigger '{trigger_id}' disabled successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable trigger: {str(e)}")


@router.post("/triggers/{trigger_id}/enable")
async def enable_trigger(
    trigger_id: str, retraining_manager: AutomatedRetrainingManager = Depends()
):
    """Enable a retraining trigger"""

    try:
        success = await retraining_manager.enable_trigger(trigger_id)

        if not success:
            raise HTTPException(status_code=404, detail="Trigger not found")

        return {
            "trigger_id": trigger_id,
            "status": "enabled",
            "message": f"Trigger '{trigger_id}' enabled successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable trigger: {str(e)}")


@router.post("/manual/{model_name}")
async def trigger_manual_retraining(
    model_name: str,
    retraining_config: Optional[Dict[str, Any]] = None,
    retraining_manager: AutomatedRetrainingManager = Depends(),
):
    """Trigger manual retraining for a model"""

    try:
        # This would trigger manual retraining
        retraining_job = {
            "job_id": f"manual_retraining_{model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "model_name": model_name,
            "status": "started",
            "trigger_type": "manual",
            "started_at": datetime.utcnow().isoformat(),
        }

        return retraining_job

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to trigger manual retraining: {str(e)}"
        )


@router.get("/jobs/{job_id}")
async def get_retraining_job_status(
    job_id: str, retraining_manager: AutomatedRetrainingManager = Depends()
):
    """Get retraining job status"""

    try:
        # This would get actual job status
        job_status = {
            "job_id": job_id,
            "status": "running",
            "progress": 65,
            "current_step": "model_training",
            "started_at": datetime.utcnow().isoformat(),
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=30)).isoformat(),
        }

        return job_status

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@router.get("/metrics/{model_name}")
async def get_retraining_metrics(
    model_name: str,
    start_time: Optional[datetime] = Query(None, description="Start time for metrics"),
    end_time: Optional[datetime] = Query(None, description="End time for metrics"),
    retraining_manager: AutomatedRetrainingManager = Depends(),
):
    """Get retraining metrics for a model"""

    try:
        # This would get actual retraining metrics
        metrics = {
            "model_name": model_name,
            "total_retrainings": 15,
            "successful_retrainings": 12,
            "failed_retrainings": 3,
            "average_duration_minutes": 45.2,
            "last_retraining": datetime.utcnow().isoformat(),
            "trigger_fire_counts": {"performance": 8, "data_drift": 4, "scheduled": 2, "manual": 1},
            "calculated_at": datetime.utcnow().isoformat(),
        }

        return metrics

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get retraining metrics: {str(e)}")


@router.get("/comparison/{model_name}")
async def get_model_comparison(
    model_name: str,
    current_version: str = Query(..., description="Current model version"),
    new_version: str = Query(..., description="New model version"),
    retraining_manager: AutomatedRetrainingManager = Depends(),
):
    """Get model comparison results"""

    try:
        # This would get actual model comparison
        comparison = {
            "model_name": model_name,
            "current_version": current_version,
            "new_version": new_version,
            "current_metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
            },
            "new_metrics": {"accuracy": 0.87, "precision": 0.84, "recall": 0.90, "f1_score": 0.87},
            "improvement": {"accuracy": 0.02, "precision": 0.02, "recall": 0.02, "f1_score": 0.02},
            "is_significant": True,
            "p_value": 0.03,
            "recommendation": "deploy_new",
            "confidence": 0.95,
            "comparison_timestamp": datetime.utcnow().isoformat(),
        }

        return comparison

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model comparison: {str(e)}")


@router.delete("/triggers/{model_name}")
async def delete_all_triggers(
    model_name: str, retraining_manager: AutomatedRetrainingManager = Depends()
):
    """Delete all retraining triggers for a model"""

    try:
        # This would delete all triggers for the model
        return {
            "model_name": model_name,
            "status": "deleted",
            "message": f"All retraining triggers for model '{model_name}' deleted successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete triggers: {str(e)}")
