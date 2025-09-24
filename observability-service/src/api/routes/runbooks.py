"""
Runbooks API routes
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel

from monitoring.observability_engine import observability_engine
from monitoring.runbooks import Runbook, RunbookEngine, RunbookExecution, RunbookStatus

logger = logging.getLogger(__name__)

router = APIRouter()


class RunbookRequest(BaseModel):
    """Runbook creation request"""

    name: str
    description: str
    trigger_conditions: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]]
    requires_approval: bool = False
    timeout_minutes: int = 60
    tags: List[str] = []


class RunbookResponse(BaseModel):
    """Runbook response"""

    id: str
    name: str
    description: str
    version: str
    trigger_conditions: List[Dict[str, Any]]
    steps: List[Dict[str, Any]]
    requires_approval: bool
    timeout_minutes: int
    created_at: str
    updated_at: str
    enabled: bool
    tags: List[str]


class RunbookExecutionRequest(BaseModel):
    """Runbook execution request"""

    runbook_id: str
    incident_id: Optional[str] = None
    execution_params: Dict[str, Any] = {}
    executed_by: str = "system"


class RunbookExecutionResponse(BaseModel):
    """Runbook execution response"""

    id: str
    runbook_id: str
    incident_id: Optional[str]
    status: str
    started_at: str
    completed_at: Optional[str]
    executed_by: str
    step_results: List[Dict[str, Any]]
    error_message: Optional[str] = None


class StepResultResponse(BaseModel):
    """Step result response"""

    step_id: str
    success: bool
    output: str
    error_message: Optional[str]
    execution_time: float
    retry_count: int
    metadata: Dict[str, Any]


@router.post("/", response_model=RunbookResponse)
async def create_runbook(runbook_request: RunbookRequest):
    """Create new runbook"""

    try:
        if not observability_engine or not observability_engine.runbook_engine:
            raise HTTPException(status_code=503, detail="Runbook engine not available")

        # Create runbook definition
        runbook_definition = {
            "id": f"runbook_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "name": runbook_request.name,
            "description": runbook_request.description,
            "trigger_conditions": runbook_request.trigger_conditions,
            "steps": runbook_request.steps,
            "requires_approval": runbook_request.requires_approval,
            "timeout_minutes": runbook_request.timeout_minutes,
            "tags": runbook_request.tags,
        }

        runbook = await observability_engine.runbook_engine.create_runbook(runbook_definition)

        return RunbookResponse(
            id=runbook.id,
            name=runbook.name,
            description=runbook.description,
            version=runbook.version,
            trigger_conditions=runbook.trigger_conditions,
            steps=[step.__dict__ for step in runbook.steps],
            requires_approval=runbook.requires_approval,
            timeout_minutes=runbook.timeout_minutes,
            created_at=runbook.created_at.isoformat(),
            updated_at=runbook.updated_at.isoformat(),
            enabled=runbook.enabled,
            tags=runbook.tags,
        )

    except Exception as e:
        logger.error(f"Failed to create runbook: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create runbook: {str(e)}")


@router.get("/", response_model=List[RunbookResponse])
async def get_runbooks(
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    enabled_only: bool = Query(True, description="Show only enabled runbooks"),
):
    """Get all runbooks"""

    try:
        if not observability_engine or not observability_engine.runbook_engine:
            raise HTTPException(status_code=503, detail="Runbook engine not available")

        # Get runbooks from registry
        runbook_registry = observability_engine.runbook_engine.runbook_registry
        runbooks = await runbook_registry.list_runbooks(tags=tags)

        # Filter by enabled status
        if enabled_only:
            runbooks = [r for r in runbooks if r.enabled]

        # Convert to response format
        runbook_responses = []
        for runbook in runbooks:
            runbook_responses.append(
                RunbookResponse(
                    id=runbook.id,
                    name=runbook.name,
                    description=runbook.description,
                    version=runbook.version,
                    trigger_conditions=runbook.trigger_conditions,
                    steps=[step.__dict__ for step in runbook.steps],
                    requires_approval=runbook.requires_approval,
                    timeout_minutes=runbook.timeout_minutes,
                    created_at=runbook.created_at.isoformat(),
                    updated_at=runbook.updated_at.isoformat(),
                    enabled=runbook.enabled,
                    tags=runbook.tags,
                )
            )

        return runbook_responses

    except Exception as e:
        logger.error(f"Failed to get runbooks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get runbooks: {str(e)}")


@router.get("/{runbook_id}", response_model=RunbookResponse)
async def get_runbook(runbook_id: str):
    """Get specific runbook by ID"""

    try:
        if not observability_engine or not observability_engine.runbook_engine:
            raise HTTPException(status_code=503, detail="Runbook engine not available")

        # Get runbook from registry
        runbook_registry = observability_engine.runbook_engine.runbook_registry
        runbook = await runbook_registry.get_runbook(runbook_id)

        if not runbook:
            raise HTTPException(status_code=404, detail=f"Runbook {runbook_id} not found")

        return RunbookResponse(
            id=runbook.id,
            name=runbook.name,
            description=runbook.description,
            version=runbook.version,
            trigger_conditions=runbook.trigger_conditions,
            steps=[step.__dict__ for step in runbook.steps],
            requires_approval=runbook.requires_approval,
            timeout_minutes=runbook.timeout_minutes,
            created_at=runbook.created_at.isoformat(),
            updated_at=runbook.updated_at.isoformat(),
            enabled=runbook.enabled,
            tags=runbook.tags,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get runbook {runbook_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get runbook {runbook_id}: {str(e)}")


@router.post("/{runbook_id}/execute", response_model=RunbookExecutionResponse)
async def execute_runbook(runbook_id: str, execution_request: RunbookExecutionRequest):
    """Execute runbook"""

    try:
        if not observability_engine or not observability_engine.runbook_engine:
            raise HTTPException(status_code=503, detail="Runbook engine not available")

        # Get incident if provided
        incident = None
        if execution_request.incident_id:
            if observability_engine.incident_manager:
                incident = await observability_engine.incident_manager.get_incident(
                    execution_request.incident_id
                )

        # Execute runbook
        execution = await observability_engine.runbook_engine.execute_runbook(
            runbook_id=runbook_id,
            incident=incident,
            execution_params=execution_request.execution_params,
        )

        return RunbookExecutionResponse(
            id=execution.id,
            runbook_id=execution.runbook_id,
            incident_id=execution.incident_id,
            status=execution.status.value,
            started_at=execution.started_at.isoformat(),
            completed_at=execution.completed_at.isoformat() if execution.completed_at else None,
            executed_by=execution.executed_by,
            step_results=[result.__dict__ for result in execution.step_results],
            error_message=execution.error_message,
        )

    except Exception as e:
        logger.error(f"Failed to execute runbook {runbook_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to execute runbook {runbook_id}: {str(e)}"
        )


@router.get("/{runbook_id}/executions", response_model=List[RunbookExecutionResponse])
async def get_runbook_executions(
    runbook_id: str, limit: int = Query(50, description="Maximum number of executions to return")
):
    """Get runbook execution history"""

    try:
        if not observability_engine or not observability_engine.runbook_engine:
            raise HTTPException(status_code=503, detail="Runbook engine not available")

        # Get executions from execution engine
        execution_engine = observability_engine.runbook_engine.execution_engine
        executions = list(execution_engine.active_executions.values())

        # Filter by runbook ID
        runbook_executions = [e for e in executions if e.runbook_id == runbook_id]

        # Limit results
        runbook_executions = runbook_executions[:limit]

        # Convert to response format
        execution_responses = []
        for execution in runbook_executions:
            execution_responses.append(
                RunbookExecutionResponse(
                    id=execution.id,
                    runbook_id=execution.runbook_id,
                    incident_id=execution.incident_id,
                    status=execution.status.value,
                    started_at=execution.started_at.isoformat(),
                    completed_at=(
                        execution.completed_at.isoformat() if execution.completed_at else None
                    ),
                    executed_by=execution.executed_by,
                    step_results=[result.__dict__ for result in execution.step_results],
                    error_message=execution.error_message,
                )
            )

        return execution_responses

    except Exception as e:
        logger.error(f"Failed to get executions for runbook {runbook_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get executions for runbook {runbook_id}: {str(e)}"
        )


@router.get("/executions/{execution_id}", response_model=RunbookExecutionResponse)
async def get_runbook_execution(execution_id: str):
    """Get specific runbook execution by ID"""

    try:
        if not observability_engine or not observability_engine.runbook_engine:
            raise HTTPException(status_code=503, detail="Runbook engine not available")

        # Get execution from execution engine
        execution_engine = observability_engine.runbook_engine.execution_engine
        execution = execution_engine.active_executions.get(execution_id)

        if not execution:
            raise HTTPException(
                status_code=404, detail=f"Runbook execution {execution_id} not found"
            )

        return RunbookExecutionResponse(
            id=execution.id,
            runbook_id=execution.runbook_id,
            incident_id=execution.incident_id,
            status=execution.status.value,
            started_at=execution.started_at.isoformat(),
            completed_at=execution.completed_at.isoformat() if execution.completed_at else None,
            executed_by=execution.executed_by,
            step_results=[result.__dict__ for result in execution.step_results],
            error_message=execution.error_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get runbook execution {execution_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get runbook execution {execution_id}: {str(e)}"
        )


@router.put("/{runbook_id}/enable")
async def enable_runbook(runbook_id: str):
    """Enable runbook"""

    try:
        if not observability_engine or not observability_engine.runbook_engine:
            raise HTTPException(status_code=503, detail="Runbook engine not available")

        # Get runbook
        runbook_registry = observability_engine.runbook_engine.runbook_registry
        runbook = await runbook_registry.get_runbook(runbook_id)

        if not runbook:
            raise HTTPException(status_code=404, detail=f"Runbook {runbook_id} not found")

        # Enable runbook
        runbook.enabled = True

        return {
            "message": f"Runbook {runbook_id} enabled",
            "runbook_id": runbook_id,
            "enabled": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable runbook {runbook_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to enable runbook {runbook_id}: {str(e)}"
        )


@router.put("/{runbook_id}/disable")
async def disable_runbook(runbook_id: str):
    """Disable runbook"""

    try:
        if not observability_engine or not observability_engine.runbook_engine:
            raise HTTPException(status_code=503, detail="Runbook engine not available")

        # Get runbook
        runbook_registry = observability_engine.runbook_engine.runbook_registry
        runbook = await runbook_registry.get_runbook(runbook_id)

        if not runbook:
            raise HTTPException(status_code=404, detail=f"Runbook {runbook_id} not found")

        # Disable runbook
        runbook.enabled = False

        return {
            "message": f"Runbook {runbook_id} disabled",
            "runbook_id": runbook_id,
            "enabled": False,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable runbook {runbook_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to disable runbook {runbook_id}: {str(e)}"
        )


@router.get("/search")
async def search_runbooks(
    query: str = Query(..., description="Search query"),
    limit: int = Query(20, description="Maximum number of results"),
):
    """Search runbooks by name or description"""

    try:
        if not observability_engine or not observability_engine.runbook_engine:
            raise HTTPException(status_code=503, detail="Runbook engine not available")

        # Search runbooks
        runbook_registry = observability_engine.runbook_engine.runbook_registry
        runbooks = await runbook_registry.search_runbooks(query)

        # Limit results
        runbooks = runbooks[:limit]

        # Convert to response format
        runbook_responses = []
        for runbook in runbooks:
            runbook_responses.append(
                RunbookResponse(
                    id=runbook.id,
                    name=runbook.name,
                    description=runbook.description,
                    version=runbook.version,
                    trigger_conditions=runbook.trigger_conditions,
                    steps=[step.__dict__ for step in runbook.steps],
                    requires_approval=runbook.requires_approval,
                    timeout_minutes=runbook.timeout_minutes,
                    created_at=runbook.created_at.isoformat(),
                    updated_at=runbook.updated_at.isoformat(),
                    enabled=runbook.enabled,
                    tags=runbook.tags,
                )
            )

        return {
            "query": query,
            "results": runbook_responses,
            "total_results": len(runbook_responses),
        }

    except Exception as e:
        logger.error(f"Failed to search runbooks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search runbooks: {str(e)}")


@router.get("/templates")
async def get_runbook_templates():
    """Get available runbook templates"""

    try:
        # Return predefined runbook templates
        templates = [
            {
                "id": "service_restart",
                "name": "Service Restart",
                "description": "Restart a failed service",
                "category": "recovery",
                "steps": [
                    {
                        "id": "check_service_status",
                        "name": "Check Service Status",
                        "type": "command",
                        "command": "systemctl status {service_name}",
                        "critical": True,
                    },
                    {
                        "id": "restart_service",
                        "name": "Restart Service",
                        "type": "command",
                        "command": "systemctl restart {service_name}",
                        "critical": True,
                    },
                    {
                        "id": "verify_restart",
                        "name": "Verify Restart",
                        "type": "command",
                        "command": "systemctl is-active {service_name}",
                        "critical": True,
                    },
                ],
            },
            {
                "id": "database_connection_reset",
                "name": "Database Connection Reset",
                "description": "Reset database connections",
                "category": "database",
                "steps": [
                    {
                        "id": "check_db_connections",
                        "name": "Check Database Connections",
                        "type": "database_query",
                        "db_config": {"query": "SELECT COUNT(*) FROM pg_stat_activity"},
                        "critical": True,
                    },
                    {
                        "id": "reset_connections",
                        "name": "Reset Connections",
                        "type": "command",
                        "command": "pkill -f postgres",
                        "critical": True,
                    },
                ],
            },
            {
                "id": "cache_clear",
                "name": "Cache Clear",
                "description": "Clear application cache",
                "category": "cache",
                "steps": [
                    {
                        "id": "check_cache_status",
                        "name": "Check Cache Status",
                        "type": "command",
                        "command": "redis-cli ping",
                        "critical": True,
                    },
                    {
                        "id": "clear_cache",
                        "name": "Clear Cache",
                        "type": "command",
                        "command": "redis-cli FLUSHALL",
                        "critical": True,
                    },
                ],
            },
        ]

        return {"templates": templates, "total_templates": len(templates)}

    except Exception as e:
        logger.error(f"Failed to get runbook templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get runbook templates: {str(e)}")


@router.post("/{runbook_id}/validate")
async def validate_runbook(runbook_id: str):
    """Validate runbook definition"""

    try:
        if not observability_engine or not observability_engine.runbook_engine:
            raise HTTPException(status_code=503, detail="Runbook engine not available")

        # Get runbook
        runbook_registry = observability_engine.runbook_engine.runbook_registry
        runbook = await runbook_registry.get_runbook(runbook_id)

        if not runbook:
            raise HTTPException(status_code=404, detail=f"Runbook {runbook_id} not found")

        # Validate runbook
        validation_result = await observability_engine.runbook_engine.validate_runbook(runbook)

        return {
            "runbook_id": runbook_id,
            "is_valid": validation_result["is_valid"],
            "errors": validation_result["errors"],
            "validated_at": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate runbook {runbook_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to validate runbook {runbook_id}: {str(e)}"
        )
