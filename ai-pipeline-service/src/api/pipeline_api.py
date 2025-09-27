"""
AI Pipeline API - FastAPI endpoints for pipeline management
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ..core.pipeline_manager import PipelineManager, ComponentType, PipelineStatus

logger = logging.getLogger(__name__)

# Initialize pipeline manager
pipeline_manager = PipelineManager()

# Pydantic models for API
class ComponentCreateRequest(BaseModel):
    name: str = Field(..., description="Component name")
    component_type: ComponentType = Field(..., description="Component type")
    config: Dict[str, Any] = Field(default_factory=dict, description="Component configuration")
    dependencies: List[str] = Field(default_factory=list, description="Component dependencies")

class PipelineCreateRequest(BaseModel):
    name: str = Field(..., description="Pipeline name")
    description: str = Field(default="", description="Pipeline description")

class PipelineResponse(BaseModel):
    pipeline_id: str
    name: str
    description: str
    components: List[Dict[str, Any]]
    status: str
    created_at: datetime
    last_execution: Optional[datetime] = None

class ExecutionResponse(BaseModel):
    execution_id: str
    pipeline_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    components_executed: List[str]
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

# Create router
router = APIRouter(prefix="/api/v1/pipeline", tags=["AI Pipeline"])

@router.post("/pipelines", response_model=Dict[str, str])
async def create_pipeline(request: PipelineCreateRequest):
    """Create a new AI pipeline"""
    try:
        pipeline_id = await pipeline_manager.create_pipeline(
            name=request.name,
            description=request.description
        )
        return {
            "pipeline_id": pipeline_id,
            "message": "Pipeline created successfully"
        }
    except Exception as e:
        logger.error(f"Failed to create pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pipelines/{pipeline_id}/components", response_model=Dict[str, str])
async def add_component(
    pipeline_id: str,
    request: ComponentCreateRequest
):
    """Add a component to a pipeline"""
    try:
        component_id = await pipeline_manager.add_component(
            pipeline_id=pipeline_id,
            name=request.name,
            component_type=request.component_type,
            config=request.config,
            dependencies=request.dependencies
        )
        return {
            "component_id": component_id,
            "message": "Component added successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to add component: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pipelines/{pipeline_id}/execute", response_model=Dict[str, str])
async def execute_pipeline(
    pipeline_id: str,
    background_tasks: BackgroundTasks
):
    """Execute a pipeline"""
    try:
        execution_id = await pipeline_manager.execute_pipeline(pipeline_id)
        return {
            "execution_id": execution_id,
            "message": "Pipeline execution started"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipelines/{pipeline_id}/status", response_model=Dict[str, Any])
async def get_pipeline_status(pipeline_id: str):
    """Get pipeline status and metrics"""
    try:
        status = await pipeline_manager.get_pipeline_status(pipeline_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/executions/{execution_id}", response_model=ExecutionResponse)
async def get_execution_status(execution_id: str):
    """Get execution status"""
    if execution_id not in pipeline_manager.executions:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = pipeline_manager.executions[execution_id]
    return ExecutionResponse(**execution.__dict__)

@router.get("/analytics", response_model=Dict[str, Any])
async def get_analytics():
    """Get pipeline analytics"""
    try:
        analytics = await pipeline_manager.get_analytics()
        return analytics
    except Exception as e:
        logger.error(f"Failed to get analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipelines", response_model=List[Dict[str, Any]])
async def list_pipelines():
    """List all pipelines"""
    try:
        pipelines = []
        for pipeline_id, components in pipeline_manager.pipelines.items():
            # Get recent executions for this pipeline
            recent_executions = [
                exec for exec in pipeline_manager.executions.values() 
                if exec.pipeline_id == pipeline_id
            ]
            
            last_execution = None
            if recent_executions:
                last_execution = max(recent_executions, key=lambda x: x.start_time).start_time
            
            pipelines.append({
                "pipeline_id": pipeline_id,
                "name": f"Pipeline {pipeline_id[:8]}",
                "description": "AI/ML Pipeline",
                "components": [comp.__dict__ for comp in components],
                "status": "active" if components else "empty",
                "created_at": datetime.utcnow().isoformat(),
                "last_execution": last_execution.isoformat() if last_execution else None
            })
        
        return pipelines
    except Exception as e:
        logger.error(f"Failed to list pipelines: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a pipeline"""
    try:
        if pipeline_id not in pipeline_manager.pipelines:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        del pipeline_manager.pipelines[pipeline_id]
        return {"message": "Pipeline deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "pipelines_count": len(pipeline_manager.pipelines),
        "executions_count": len(pipeline_manager.executions)
    }
