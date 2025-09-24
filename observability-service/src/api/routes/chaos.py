"""
Chaos engineering API routes
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel

from monitoring.chaos import ChaosEngine, ChaosExperiment, ExperimentExecution, ExperimentStatus, ExperimentSeverity, ChaosExperimentType
from monitoring.observability_engine import observability_engine

logger = logging.getLogger(__name__)

router = APIRouter()


class ChaosExperimentRequest(BaseModel):
    """Chaos experiment creation request"""
    name: str
    description: str
    experiment_type: str
    severity: str
    target_services: List[str]
    parameters: Dict[str, Any]
    duration_minutes: int
    schedule: Optional[str] = None
    enabled: bool = True


class ChaosExperimentResponse(BaseModel):
    """Chaos experiment response"""
    id: str
    name: str
    description: str
    experiment_type: str
    severity: str
    target_services: List[str]
    parameters: Dict[str, Any]
    duration_minutes: int
    schedule: Optional[str] = None
    enabled: bool
    created_at: str
    created_by: str


class ExperimentExecutionResponse(BaseModel):
    """Experiment execution response"""
    id: str
    experiment_id: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    executed_by: str
    results: Dict[str, Any]
    error_message: Optional[str] = None
    metrics_before: Dict[str, Any]
    metrics_during: Dict[str, Any]
    metrics_after: Dict[str, Any]


class ReliabilityReportResponse(BaseModel):
    """Reliability report response"""
    id: str
    experiment_id: str
    execution_id: str
    overall_reliability_score: float
    service_impact: Dict[str, Any]
    recovery_time: float
    recommendations: List[str]
    generated_at: str


@router.post("/experiments", response_model=ChaosExperimentResponse)
async def create_chaos_experiment(experiment_request: ChaosExperimentRequest):
    """Create new chaos experiment"""
    
    try:
        if not observability_engine or not observability_engine.chaos_engine:
            raise HTTPException(status_code=503, detail="Chaos engine not available")
        
        # Create experiment data
        experiment_data = {
            "id": f"chaos_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "name": experiment_request.name,
            "description": experiment_request.description,
            "type": experiment_request.experiment_type,
            "severity": experiment_request.severity,
            "target_services": experiment_request.target_services,
            "parameters": experiment_request.parameters,
            "duration_minutes": experiment_request.duration_minutes,
            "schedule": experiment_request.schedule,
            "enabled": experiment_request.enabled
        }
        
        # Create experiment
        experiment = await observability_engine.chaos_engine.create_experiment(experiment_data)
        
        return ChaosExperimentResponse(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            experiment_type=experiment.experiment_type.value,
            severity=experiment.severity.value,
            target_services=experiment.target_services,
            parameters=experiment.parameters,
            duration_minutes=experiment.duration_minutes,
            schedule=experiment.schedule,
            enabled=experiment.enabled,
            created_at=experiment.created_at.isoformat(),
            created_by=experiment.created_by
        )
        
    except Exception as e:
        logger.error(f"Failed to create chaos experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create chaos experiment: {str(e)}")


@router.get("/experiments", response_model=List[ChaosExperimentResponse])
async def get_chaos_experiments(
    enabled_only: bool = Query(True, description="Show only enabled experiments")
):
    """Get all chaos experiments"""
    
    try:
        if not observability_engine or not observability_engine.chaos_engine:
            raise HTTPException(status_code=503, detail="Chaos engine not available")
        
        # Get experiments
        experiments = list(observability_engine.chaos_engine.experiments.values())
        
        # Filter by enabled status
        if enabled_only:
            experiments = [e for e in experiments if e.enabled]
        
        # Convert to response format
        experiment_responses = []
        for experiment in experiments:
            experiment_responses.append(ChaosExperimentResponse(
                id=experiment.id,
                name=experiment.name,
                description=experiment.description,
                experiment_type=experiment.experiment_type.value,
                severity=experiment.severity.value,
                target_services=experiment.target_services,
                parameters=experiment.parameters,
                duration_minutes=experiment.duration_minutes,
                schedule=experiment.schedule,
                enabled=experiment.enabled,
                created_at=experiment.created_at.isoformat(),
                created_by=experiment.created_by
            ))
        
        return experiment_responses
        
    except Exception as e:
        logger.error(f"Failed to get chaos experiments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chaos experiments: {str(e)}")


@router.get("/experiments/{experiment_id}", response_model=ChaosExperimentResponse)
async def get_chaos_experiment(experiment_id: str):
    """Get specific chaos experiment by ID"""
    
    try:
        if not observability_engine or not observability_engine.chaos_engine:
            raise HTTPException(status_code=503, detail="Chaos engine not available")
        
        # Get experiment
        experiment = observability_engine.chaos_engine.experiments.get(experiment_id)
        
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Chaos experiment {experiment_id} not found")
        
        return ChaosExperimentResponse(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            experiment_type=experiment.experiment_type.value,
            severity=experiment.severity.value,
            target_services=experiment.target_services,
            parameters=experiment.parameters,
            duration_minutes=experiment.duration_minutes,
            schedule=experiment.schedule,
            enabled=experiment.enabled,
            created_at=experiment.created_at.isoformat(),
            created_by=experiment.created_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chaos experiment {experiment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chaos experiment {experiment_id}: {str(e)}")


@router.post("/experiments/{experiment_id}/execute", response_model=ExperimentExecutionResponse)
async def execute_chaos_experiment(experiment_id: str):
    """Execute chaos experiment"""
    
    try:
        if not observability_engine or not observability_engine.chaos_engine:
            raise HTTPException(status_code=503, detail="Chaos engine not available")
        
        # Get experiment
        experiment = observability_engine.chaos_engine.experiments.get(experiment_id)
        
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Chaos experiment {experiment_id} not found")
        
        # Execute experiment
        execution = await observability_engine.chaos_engine.execute_experiment(experiment)
        
        return ExperimentExecutionResponse(
            id=execution.id,
            experiment_id=execution.experiment_id,
            status=execution.status.value,
            started_at=execution.started_at.isoformat(),
            completed_at=execution.completed_at.isoformat() if execution.completed_at else None,
            executed_by=execution.executed_by,
            results=execution.results,
            error_message=execution.error_message,
            metrics_before=execution.metrics_before,
            metrics_during=execution.metrics_during,
            metrics_after=execution.metrics_after
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute chaos experiment {experiment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute chaos experiment {experiment_id}: {str(e)}")


@router.get("/experiments/{experiment_id}/executions", response_model=List[ExperimentExecutionResponse])
async def get_experiment_executions(experiment_id: str):
    """Get experiment execution history"""
    
    try:
        if not observability_engine or not observability_engine.chaos_engine:
            raise HTTPException(status_code=503, detail="Chaos engine not available")
        
        # Get executions for experiment
        executions = await observability_engine.chaos_engine.get_experiment_results(experiment_id)
        
        # Convert to response format
        execution_responses = []
        for execution in executions:
            execution_responses.append(ExperimentExecutionResponse(
                id=execution.id,
                experiment_id=execution.experiment_id,
                status=execution.status.value,
                started_at=execution.started_at.isoformat(),
                completed_at=execution.completed_at.isoformat() if execution.completed_at else None,
                executed_by=execution.executed_by,
                results=execution.results,
                error_message=execution.error_message,
                metrics_before=execution.metrics_before,
                metrics_during=execution.metrics_during,
                metrics_after=execution.metrics_after
            ))
        
        return execution_responses
        
    except Exception as e:
        logger.error(f"Failed to get executions for experiment {experiment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get executions for experiment {experiment_id}: {str(e)}")


@router.get("/executions/{execution_id}", response_model=ExperimentExecutionResponse)
async def get_experiment_execution(execution_id: str):
    """Get specific experiment execution by ID"""
    
    try:
        if not observability_engine or not observability_engine.chaos_engine:
            raise HTTPException(status_code=503, detail="Chaos engine not available")
        
        # Get execution
        execution = observability_engine.chaos_engine.executions.get(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail=f"Experiment execution {execution_id} not found")
        
        return ExperimentExecutionResponse(
            id=execution.id,
            experiment_id=execution.experiment_id,
            status=execution.status.value,
            started_at=execution.started_at.isoformat(),
            completed_at=execution.completed_at.isoformat() if execution.completed_at else None,
            executed_by=execution.executed_by,
            results=execution.results,
            error_message=execution.error_message,
            metrics_before=execution.metrics_before,
            metrics_during=execution.metrics_during,
            metrics_after=execution.metrics_after
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment execution {execution_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get experiment execution {execution_id}: {str(e)}")


@router.get("/reliability", response_model=Dict[str, Any])
async def get_reliability_summary():
    """Get overall reliability summary"""
    
    try:
        if not observability_engine or not observability_engine.chaos_engine:
            raise HTTPException(status_code=503, detail="Chaos engine not available")
        
        # Get reliability summary
        summary = await observability_engine.chaos_engine.get_reliability_summary()
        
        return {
            "reliability_summary": summary,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get reliability summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get reliability summary: {str(e)}")


@router.get("/templates")
async def get_chaos_experiment_templates():
    """Get available chaos experiment templates"""
    
    try:
        # Return predefined chaos experiment templates
        templates = [
            {
                "id": "network_latency",
                "name": "Network Latency Injection",
                "description": "Inject network latency to test service resilience",
                "category": "network",
                "experiment_type": "network_latency",
                "severity": "medium",
                "parameters": {
                    "latency_ms": 1000,
                    "duration_minutes": 5
                },
                "target_services": ["all"]
            },
            {
                "id": "cpu_stress",
                "name": "CPU Stress Test",
                "description": "Generate CPU load to test resource limits",
                "category": "resource",
                "experiment_type": "cpu_stress",
                "severity": "high",
                "parameters": {
                    "cpu_percentage": 80,
                    "duration_minutes": 10
                },
                "target_services": ["all"]
            },
            {
                "id": "service_failure",
                "name": "Service Failure Simulation",
                "description": "Simulate service failures to test failover",
                "category": "service",
                "experiment_type": "service_failure",
                "severity": "critical",
                "parameters": {
                    "failure_duration_minutes": 2,
                    "services_to_fail": ["random"]
                },
                "target_services": ["ingestion-service", "content-extraction-service"]
            },
            {
                "id": "database_failure",
                "name": "Database Failure Simulation",
                "description": "Simulate database failures to test data resilience",
                "category": "database",
                "experiment_type": "database_failure",
                "severity": "critical",
                "parameters": {
                    "failure_duration_minutes": 1,
                    "database_type": "primary"
                },
                "target_services": ["all"]
            },
            {
                "id": "random_kill",
                "name": "Random Pod Kill",
                "description": "Randomly kill pods to test Kubernetes resilience",
                "category": "kubernetes",
                "experiment_type": "random_kill",
                "severity": "medium",
                "parameters": {
                    "kill_percentage": 50,
                    "max_pods_to_kill": 3
                },
                "target_services": ["all"]
            }
        ]
        
        return {
            "templates": templates,
            "total_templates": len(templates),
            "categories": list(set(t["category"] for t in templates))
        }
        
    except Exception as e:
        logger.error(f"Failed to get chaos experiment templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chaos experiment templates: {str(e)}")


@router.post("/experiments/{experiment_id}/enable")
async def enable_chaos_experiment(experiment_id: str):
    """Enable chaos experiment"""
    
    try:
        if not observability_engine or not observability_engine.chaos_engine:
            raise HTTPException(status_code=503, detail="Chaos engine not available")
        
        # Get experiment
        experiment = observability_engine.chaos_engine.experiments.get(experiment_id)
        
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Chaos experiment {experiment_id} not found")
        
        # Enable experiment
        experiment.enabled = True
        
        return {
            "message": f"Chaos experiment {experiment_id} enabled",
            "experiment_id": experiment_id,
            "enabled": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable chaos experiment {experiment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to enable chaos experiment {experiment_id}: {str(e)}")


@router.post("/experiments/{experiment_id}/disable")
async def disable_chaos_experiment(experiment_id: str):
    """Disable chaos experiment"""
    
    try:
        if not observability_engine or not observability_engine.chaos_engine:
            raise HTTPException(status_code=503, detail="Chaos engine not available")
        
        # Get experiment
        experiment = observability_engine.chaos_engine.experiments.get(experiment_id)
        
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Chaos experiment {experiment_id} not found")
        
        # Disable experiment
        experiment.enabled = False
        
        return {
            "message": f"Chaos experiment {experiment_id} disabled",
            "experiment_id": experiment_id,
            "enabled": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable chaos experiment {experiment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to disable chaos experiment {experiment_id}: {str(e)}")


@router.get("/active")
async def get_active_experiments():
    """Get currently active experiments"""
    
    try:
        if not observability_engine or not observability_engine.chaos_engine:
            raise HTTPException(status_code=503, detail="Chaos engine not available")
        
        # Get active experiments
        active_experiments = list(observability_engine.chaos_engine.active_experiments)
        
        # Get experiment details
        experiments = []
        for experiment_id in active_experiments:
            experiment = observability_engine.chaos_engine.experiments.get(experiment_id)
            if experiment:
                experiments.append({
                    "id": experiment.id,
                    "name": experiment.name,
                    "experiment_type": experiment.experiment_type.value,
                    "severity": experiment.severity.value,
                    "target_services": experiment.target_services,
                    "duration_minutes": experiment.duration_minutes
                })
        
        return {
            "active_experiments": experiments,
            "total_active": len(experiments),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get active experiments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active experiments: {str(e)}")


@router.post("/run")
async def run_chaos_experiments():
    """Run scheduled chaos experiments"""
    
    try:
        if not observability_engine or not observability_engine.chaos_engine:
            raise HTTPException(status_code=503, detail="Chaos engine not available")
        
        # Run chaos experiments
        await observability_engine.chaos_engine.run_chaos_experiments()
        
        return {
            "message": "Chaos experiments execution triggered",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to run chaos experiments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to run chaos experiments: {str(e)}")


@router.get("/stats")
async def get_chaos_statistics():
    """Get chaos engineering statistics"""
    
    try:
        if not observability_engine or not observability_engine.chaos_engine:
            raise HTTPException(status_code=503, detail="Chaos engine not available")
        
        # Get experiments and executions
        experiments = list(observability_engine.chaos_engine.experiments.values())
        executions = list(observability_engine.chaos_engine.executions.values())
        
        # Calculate statistics
        total_experiments = len(experiments)
        enabled_experiments = len([e for e in experiments if e.enabled])
        total_executions = len(executions)
        completed_executions = len([e for e in executions if e.status == ExperimentStatus.COMPLETED])
        failed_executions = len([e for e in executions if e.status == ExperimentStatus.FAILED])
        
        # Group by experiment type
        type_counts = {}
        for experiment in experiments:
            exp_type = experiment.experiment_type.value
            type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
        
        # Group by severity
        severity_counts = {}
        for experiment in experiments:
            severity = experiment.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_experiments": total_experiments,
            "enabled_experiments": enabled_experiments,
            "total_executions": total_executions,
            "completed_executions": completed_executions,
            "failed_executions": failed_executions,
            "success_rate": (completed_executions / total_executions * 100) if total_executions > 0 else 0,
            "experiment_types": type_counts,
            "severity_breakdown": severity_counts,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get chaos statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chaos statistics: {str(e)}")
