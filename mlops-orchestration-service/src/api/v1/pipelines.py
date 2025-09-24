"""
Pipeline API endpoints - REST API for ML pipeline management
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from src.models.pipeline_models import (
    MLPipeline,
    MLPipelineConfig,
    PipelineExecution,
    PipelineStatus,
    PipelineType,
)
from src.orchestration.pipeline_orchestrator import MLOpsPipelineOrchestrator
from src.utils.exceptions import PipelineError, ValidationError

router = APIRouter()


# Request/Response models
class CreatePipelineRequest(BaseModel):
    config: MLPipelineConfig


class CreatePipelineResponse(BaseModel):
    pipeline_id: str
    status: str
    message: str


class TriggerExecutionRequest(BaseModel):
    execution_params: Dict[str, Any] = {}


class TriggerExecutionResponse(BaseModel):
    execution_id: str
    status: str
    message: str


class PipelineListResponse(BaseModel):
    pipelines: List[MLPipeline]
    total: int
    page: int
    page_size: int


class ExecutionListResponse(BaseModel):
    executions: List[PipelineExecution]
    total: int
    page: int
    page_size: int


@router.post("/", response_model=CreatePipelineResponse)
async def create_pipeline(
    request: CreatePipelineRequest,
    background_tasks: BackgroundTasks,
    orchestrator: MLOpsPipelineOrchestrator = Depends(),
):
    """Create a new ML pipeline"""

    try:
        pipeline = await orchestrator.create_ml_pipeline(request.config)

        return CreatePipelineResponse(
            pipeline_id=pipeline.id,
            status=pipeline.status.value,
            message=f"Pipeline '{pipeline.name}' created successfully",
        )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PipelineError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{pipeline_id}", response_model=MLPipeline)
async def get_pipeline(pipeline_id: str, orchestrator: MLOpsPipelineOrchestrator = Depends()):
    """Get pipeline by ID"""

    pipeline = await orchestrator.get_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    return pipeline


@router.get("/", response_model=PipelineListResponse)
async def list_pipelines(
    pipeline_type: Optional[PipelineType] = Query(None, description="Filter by pipeline type"),
    status: Optional[PipelineStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    orchestrator: MLOpsPipelineOrchestrator = Depends(),
):
    """List pipelines with optional filtering"""

    try:
        pipelines = await orchestrator.list_pipelines(pipeline_type=pipeline_type, status=status)

        # Pagination
        total = len(pipelines)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_pipelines = pipelines[start_idx:end_idx]

        return PipelineListResponse(
            pipelines=paginated_pipelines, total=total, page=page, page_size=page_size
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list pipelines: {str(e)}")


@router.post("/{pipeline_id}/execute", response_model=TriggerExecutionResponse)
async def trigger_pipeline_execution(
    pipeline_id: str,
    request: TriggerExecutionRequest,
    background_tasks: BackgroundTasks,
    orchestrator: MLOpsPipelineOrchestrator = Depends(),
):
    """Trigger pipeline execution"""

    try:
        # Check if pipeline exists
        pipeline = await orchestrator.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        # Trigger execution
        execution = await orchestrator.trigger_pipeline_execution(
            pipeline_id=pipeline_id, execution_params=request.execution_params
        )

        return TriggerExecutionResponse(
            execution_id=execution.id,
            status=execution.status.value,
            message=f"Pipeline execution '{execution.id}' triggered successfully",
        )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PipelineError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger execution: {str(e)}")


@router.get("/{pipeline_id}/executions", response_model=ExecutionListResponse)
async def list_pipeline_executions(
    pipeline_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    orchestrator: MLOpsPipelineOrchestrator = Depends(),
):
    """List executions for a pipeline"""

    try:
        # Check if pipeline exists
        pipeline = await orchestrator.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        # Get executions (this would need to be implemented in orchestrator)
        executions = []  # Placeholder - would get from orchestrator

        # Pagination
        total = len(executions)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_executions = executions[start_idx:end_idx]

        return ExecutionListResponse(
            executions=paginated_executions, total=total, page=page, page_size=page_size
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list executions: {str(e)}")


@router.get("/executions/{execution_id}", response_model=PipelineExecution)
async def get_pipeline_execution(
    execution_id: str, orchestrator: MLOpsPipelineOrchestrator = Depends()
):
    """Get pipeline execution by ID"""

    execution = await orchestrator.get_pipeline_execution(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    return execution


@router.post("/{pipeline_id}/stop")
async def stop_pipeline_execution(
    pipeline_id: str, execution_id: str, orchestrator: MLOpsPipelineOrchestrator = Depends()
):
    """Stop running pipeline execution"""

    try:
        success = await orchestrator.stop_pipeline_execution(execution_id)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to stop execution")

        return {"message": f"Pipeline execution '{execution_id}' stopped successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop execution: {str(e)}")


@router.put("/{pipeline_id}/status")
async def update_pipeline_status(
    pipeline_id: str, status: PipelineStatus, orchestrator: MLOpsPipelineOrchestrator = Depends()
):
    """Update pipeline status"""

    try:
        await orchestrator.update_pipeline_status(pipeline_id, status)

        return {"message": f"Pipeline status updated to '{status.value}'"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update status: {str(e)}")


@router.delete("/{pipeline_id}")
async def delete_pipeline(pipeline_id: str, orchestrator: MLOpsPipelineOrchestrator = Depends()):
    """Delete pipeline"""

    try:
        success = await orchestrator.delete_pipeline(pipeline_id)

        if not success:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        return {"message": f"Pipeline '{pipeline_id}' deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete pipeline: {str(e)}")


@router.get("/{pipeline_id}/logs")
async def get_pipeline_logs(
    pipeline_id: str,
    execution_id: Optional[str] = Query(None, description="Specific execution ID"),
    lines: int = Query(100, ge=1, le=1000, description="Number of log lines"),
    orchestrator: MLOpsPipelineOrchestrator = Depends(),
):
    """Get pipeline logs"""

    try:
        # Check if pipeline exists
        pipeline = await orchestrator.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        # This would integrate with actual logging system
        logs = {
            "pipeline_id": pipeline_id,
            "execution_id": execution_id,
            "logs": [
                f"[{datetime.utcnow().isoformat()}] Pipeline {pipeline_id} is running",
                f"[{datetime.utcnow().isoformat()}] Execution {execution_id} started",
                f"[{datetime.utcnow().isoformat()}] Data validation completed",
                f"[{datetime.utcnow().isoformat()}] Feature engineering completed",
                f"[{datetime.utcnow().isoformat()}] Model training completed",
                f"[{datetime.utcnow().isoformat()}] Model evaluation completed",
                f"[{datetime.utcnow().isoformat()}] Pipeline execution completed successfully",
            ],
        }

        return logs

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


@router.get("/{pipeline_id}/metrics")
async def get_pipeline_metrics(
    pipeline_id: str,
    start_time: Optional[datetime] = Query(None, description="Start time for metrics"),
    end_time: Optional[datetime] = Query(None, description="End time for metrics"),
    orchestrator: MLOpsPipelineOrchestrator = Depends(),
):
    """Get pipeline metrics"""

    try:
        # Check if pipeline exists
        pipeline = await orchestrator.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        # This would integrate with actual metrics system
        metrics = {
            "pipeline_id": pipeline_id,
            "execution_count": 10,
            "success_rate": 0.95,
            "average_duration_minutes": 45.2,
            "last_execution": datetime.utcnow().isoformat(),
            "resource_usage": {"cpu_hours": 120.5, "memory_gb_hours": 480.2, "storage_gb": 25.8},
        }

        return metrics

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.post("/{pipeline_id}/validate")
async def validate_pipeline_config(
    pipeline_id: str, config: MLPipelineConfig, orchestrator: MLOpsPipelineOrchestrator = Depends()
):
    """Validate pipeline configuration"""

    try:
        # This would implement actual validation logic
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [
                "Consider enabling hyperparameter tuning for better performance",
                "Data validation timeout might be too low for large datasets",
            ],
            "suggestions": [
                "Add more monitoring metrics",
                "Consider using canary deployment strategy",
            ],
        }

        return validation_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
