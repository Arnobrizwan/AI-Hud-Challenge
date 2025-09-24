"""
SLO API routes
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel

from monitoring.slo import SLOMonitor, SLOStatus, SLIType
from monitoring.observability_engine import observability_engine

logger = logging.getLogger(__name__)

router = APIRouter()


class SLODefinitionRequest(BaseModel):
    """SLO definition creation request"""
    name: str
    description: str
    target_percentage: float
    evaluation_window: int
    sli_definitions: List[Dict[str, Any]]
    error_budget_policy: Dict[str, Any] = {}
    alerting_thresholds: Dict[str, float] = {}


class SLODefinitionResponse(BaseModel):
    """SLO definition response"""
    id: str
    name: str
    description: str
    target_percentage: float
    evaluation_window: int
    sli_definitions: List[Dict[str, Any]]
    error_budget_policy: Dict[str, Any]
    alerting_thresholds: Dict[str, float]
    created_at: str
    enabled: bool


class SLOStatusResponse(BaseModel):
    """SLO status response"""
    slo_id: str
    overall_status: str
    sli_results: List[Dict[str, Any]]
    error_budget_remaining: float
    error_budget_consumed: float
    burn_rate: float
    time_to_breach: Optional[str]
    calculated_at: str
    recommendations: List[str]


class SLIResultResponse(BaseModel):
    """SLI result response"""
    sli_id: str
    value: float
    target: float
    status: str
    evaluation_window: int
    calculated_at: str
    metadata: Dict[str, Any]


@router.post("/definitions", response_model=SLODefinitionResponse)
async def create_slo_definition(slo_request: SLODefinitionRequest):
    """Create new SLO definition"""
    
    try:
        if not observability_engine or not observability_engine.slo_monitor:
            raise HTTPException(status_code=503, detail="SLO monitor not available")
        
        # Create SLO definition
        slo_definition = {
            "id": f"slo_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "name": slo_request.name,
            "description": slo_request.description,
            "target_percentage": slo_request.target_percentage,
            "evaluation_window": slo_request.evaluation_window,
            "sli_definitions": slo_request.sli_definitions,
            "error_budget_policy": slo_request.error_budget_policy,
            "alerting_thresholds": slo_request.alerting_thresholds
        }
        
        # Add to SLO monitor
        await observability_engine.slo_monitor.load_slo_definitions([slo_definition])
        
        return SLODefinitionResponse(
            id=slo_definition["id"],
            name=slo_definition["name"],
            description=slo_definition["description"],
            target_percentage=slo_definition["target_percentage"],
            evaluation_window=slo_definition["evaluation_window"],
            sli_definitions=slo_definition["sli_definitions"],
            error_budget_policy=slo_definition["error_budget_policy"],
            alerting_thresholds=slo_definition["alerting_thresholds"],
            created_at=datetime.utcnow().isoformat(),
            enabled=True
        )
        
    except Exception as e:
        logger.error(f"Failed to create SLO definition: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create SLO definition: {str(e)}")


@router.get("/definitions", response_model=List[SLODefinitionResponse])
async def get_slo_definitions():
    """Get all SLO definitions"""
    
    try:
        if not observability_engine or not observability_engine.slo_monitor:
            raise HTTPException(status_code=503, detail="SLO monitor not available")
        
        # Get SLO definitions
        slo_definitions = list(observability_engine.slo_monitor.slo_definitions.values())
        
        # Convert to response format
        definitions = []
        for slo in slo_definitions:
            definitions.append(SLODefinitionResponse(
                id=slo.id,
                name=slo.name,
                description=slo.description,
                target_percentage=slo.target_percentage,
                evaluation_window=slo.evaluation_window,
                sli_definitions=[sli.__dict__ for sli in slo.sli_definitions],
                error_budget_policy=slo.error_budget_policy,
                alerting_thresholds=slo.alerting_thresholds,
                created_at=slo.created_at.isoformat(),
                enabled=slo.enabled
            ))
        
        return definitions
        
    except Exception as e:
        logger.error(f"Failed to get SLO definitions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get SLO definitions: {str(e)}")


@router.get("/definitions/{slo_id}", response_model=SLODefinitionResponse)
async def get_slo_definition(slo_id: str):
    """Get specific SLO definition by ID"""
    
    try:
        if not observability_engine or not observability_engine.slo_monitor:
            raise HTTPException(status_code=503, detail="SLO monitor not available")
        
        # Get SLO definition
        slo = observability_engine.slo_monitor.slo_definitions.get(slo_id)
        
        if not slo:
            raise HTTPException(status_code=404, detail=f"SLO definition {slo_id} not found")
        
        return SLODefinitionResponse(
            id=slo.id,
            name=slo.name,
            description=slo.description,
            target_percentage=slo.target_percentage,
            evaluation_window=slo.evaluation_window,
            sli_definitions=[sli.__dict__ for sli in slo.sli_definitions],
            error_budget_policy=slo.error_budget_policy,
            alerting_thresholds=slo.alerting_thresholds,
            created_at=slo.created_at.isoformat(),
            enabled=slo.enabled
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get SLO definition {slo_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get SLO definition {slo_id}: {str(e)}")


@router.get("/status", response_model=List[SLOStatusResponse])
async def get_slo_status(
    time_window: int = Query(3600, description="Time window in seconds")
):
    """Get SLO status for all SLOs"""
    
    try:
        if not observability_engine or not observability_engine.slo_monitor:
            raise HTTPException(status_code=503, detail="SLO monitor not available")
        
        # Get SLO status for all SLOs
        slo_definitions = list(observability_engine.slo_monitor.slo_definitions.keys())
        time_window_delta = timedelta(seconds=time_window)
        
        status_responses = []
        for slo_id in slo_definitions:
            try:
                status = await observability_engine.slo_monitor.calculate_slo_status(slo_id, time_window_delta)
                
                status_responses.append(SLOStatusResponse(
                    slo_id=status.slo_id,
                    overall_status=status.overall_status.value,
                    sli_results=[result.__dict__ for result in status.sli_results],
                    error_budget_remaining=status.error_budget_remaining,
                    error_budget_consumed=status.error_budget_consumed,
                    burn_rate=status.burn_rate,
                    time_to_breach=status.time_to_breach.isoformat() if status.time_to_breach else None,
                    calculated_at=status.calculated_at.isoformat(),
                    recommendations=status.recommendations
                ))
            except Exception as e:
                logger.error(f"Failed to get status for SLO {slo_id}: {str(e)}")
        
        return status_responses
        
    except Exception as e:
        logger.error(f"Failed to get SLO status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get SLO status: {str(e)}")


@router.get("/status/{slo_id}", response_model=SLOStatusResponse)
async def get_slo_status_by_id(
    slo_id: str,
    time_window: int = Query(3600, description="Time window in seconds")
):
    """Get SLO status for specific SLO"""
    
    try:
        if not observability_engine or not observability_engine.slo_monitor:
            raise HTTPException(status_code=503, detail="SLO monitor not available")
        
        # Get SLO status
        time_window_delta = timedelta(seconds=time_window)
        status = await observability_engine.slo_monitor.calculate_slo_status(slo_id, time_window_delta)
        
        return SLOStatusResponse(
            slo_id=status.slo_id,
            overall_status=status.overall_status.value,
            sli_results=[result.__dict__ for result in status.sli_results],
            error_budget_remaining=status.error_budget_remaining,
            error_budget_consumed=status.error_budget_consumed,
            burn_rate=status.burn_rate,
            time_to_breach=status.time_to_breach.isoformat() if status.time_to_breach else None,
            calculated_at=status.calculated_at.isoformat(),
            recommendations=status.recommendations
        )
        
    except Exception as e:
        logger.error(f"Failed to get SLO status for {slo_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get SLO status for {slo_id}: {str(e)}")


@router.get("/overview")
async def get_slo_overview():
    """Get SLO overview and dashboard data"""
    
    try:
        if not observability_engine or not observability_engine.slo_monitor:
            raise HTTPException(status_code=503, detail="SLO monitor not available")
        
        # Get overall SLO status
        overall_status = await observability_engine.slo_monitor.get_overall_slo_status()
        
        return {
            "slo_overview": overall_status,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get SLO overview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get SLO overview: {str(e)}")


@router.get("/error-budgets")
async def get_error_budgets():
    """Get error budget information for all SLOs"""
    
    try:
        if not observability_engine or not observability_engine.slo_monitor:
            raise HTTPException(status_code=503, detail="SLO monitor not available")
        
        # Get error budgets
        error_budgets = list(observability_engine.slo_monitor.error_budget_manager.error_budgets.values())
        
        # Convert to response format
        budget_responses = []
        for budget in error_budgets:
            budget_responses.append({
                "slo_id": budget.slo_id,
                "total_budget": budget.total_budget,
                "consumed_budget": budget.consumed_budget,
                "remaining_budget": budget.remaining_budget,
                "burn_rate": budget.burn_rate,
                "time_to_exhaustion": budget.time_to_exhaustion.isoformat() if budget.time_to_exhaustion else None,
                "last_updated": budget.last_updated.isoformat()
            })
        
        return {
            "error_budgets": budget_responses,
            "total_slos": len(budget_responses),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get error budgets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get error budgets: {str(e)}")


@router.get("/error-budgets/{slo_id}")
async def get_error_budget_by_slo(slo_id: str):
    """Get error budget for specific SLO"""
    
    try:
        if not observability_engine or not observability_engine.slo_monitor:
            raise HTTPException(status_code=503, detail="SLO monitor not available")
        
        # Get error budget
        error_budget = observability_engine.slo_monitor.error_budget_manager.error_budgets.get(slo_id)
        
        if not error_budget:
            raise HTTPException(status_code=404, detail=f"Error budget for SLO {slo_id} not found")
        
        return {
            "slo_id": error_budget.slo_id,
            "total_budget": error_budget.total_budget,
            "consumed_budget": error_budget.consumed_budget,
            "remaining_budget": error_budget.remaining_budget,
            "burn_rate": error_budget.burn_rate,
            "time_to_exhaustion": error_budget.time_to_exhaustion.isoformat() if error_budget.time_to_exhaustion else None,
            "last_updated": error_budget.last_updated.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get error budget for SLO {slo_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get error budget for SLO {slo_id}: {str(e)}")


@router.get("/sli/{sli_id}")
async def get_sli_status(
    sli_id: str,
    time_window: int = Query(3600, description="Time window in seconds")
):
    """Get SLI status for specific SLI"""
    
    try:
        if not observability_engine or not observability_engine.slo_monitor:
            raise HTTPException(status_code=503, detail="SLO monitor not available")
        
        # Find SLI definition
        sli_definition = None
        for slo in observability_engine.slo_monitor.slo_definitions.values():
            for sli in slo.sli_definitions:
                if sli.id == sli_id:
                    sli_definition = sli
                    break
            if sli_definition:
                break
        
        if not sli_definition:
            raise HTTPException(status_code=404, detail=f"SLI {sli_id} not found")
        
        # Calculate SLI
        time_window_delta = timedelta(seconds=time_window)
        sli_result = await observability_engine.slo_monitor.sli_calculator.calculate_sli(sli_definition, time_window_delta)
        
        return SLIResultResponse(
            sli_id=sli_result.sli_id,
            value=sli_result.value,
            target=sli_result.target,
            status=sli_result.status.value,
            evaluation_window=sli_result.evaluation_window,
            calculated_at=sli_result.calculated_at.isoformat(),
            metadata=sli_result.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get SLI status for {sli_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get SLI status for {sli_id}: {str(e)}")


@router.post("/{slo_id}/monitor")
async def trigger_slo_monitoring(slo_id: str):
    """Trigger immediate SLO monitoring for specific SLO"""
    
    try:
        if not observability_engine or not observability_engine.slo_monitor:
            raise HTTPException(status_code=503, detail="SLO monitor not available")
        
        # Get SLO definition
        slo = observability_engine.slo_monitor.slo_definitions.get(slo_id)
        
        if not slo:
            raise HTTPException(status_code=404, detail=f"SLO {slo_id} not found")
        
        # Trigger monitoring
        await observability_engine.slo_monitor.monitor_slo(slo)
        
        return {
            "message": f"SLO monitoring triggered for {slo_id}",
            "slo_id": slo_id,
            "slo_name": slo.name,
            "triggered_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger SLO monitoring for {slo_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger SLO monitoring for {slo_id}: {str(e)}")


@router.get("/trends")
async def get_slo_trends(
    time_window: int = Query(86400, description="Time window in seconds (default: 24 hours)")
):
    """Get SLO trends over time"""
    
    try:
        if not observability_engine or not observability_engine.slo_monitor:
            raise HTTPException(status_code=503, detail="SLO monitor not available")
        
        # This would query historical SLO data
        # For now, return mock trends
        
        trends = {
            "time_window_seconds": time_window,
            "trends": {
                "availability": {
                    "current": 99.9,
                    "trend": "stable",
                    "change_percent": 0.1
                },
                "latency": {
                    "current": 150.5,
                    "trend": "improving",
                    "change_percent": -5.2
                },
                "error_rate": {
                    "current": 0.1,
                    "trend": "stable",
                    "change_percent": 0.0
                }
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return trends
        
    except Exception as e:
        logger.error(f"Failed to get SLO trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get SLO trends: {str(e)}")


@router.get("/alerts")
async def get_slo_alerts():
    """Get SLO-related alerts"""
    
    try:
        if not observability_engine or not observability_engine.slo_monitor:
            raise HTTPException(status_code=503, detail="SLO monitor not available")
        
        # This would get SLO-related alerts
        # For now, return mock alerts
        
        alerts = [
            {
                "slo_id": "slo_availability",
                "alert_type": "error_budget_burn_rate",
                "severity": "warning",
                "message": "Error budget burn rate is high: 15%/hour",
                "threshold": 10.0,
                "current_value": 15.0,
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        
        return {
            "slo_alerts": alerts,
            "total_alerts": len(alerts),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get SLO alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get SLO alerts: {str(e)}")


@router.get("/recommendations")
async def get_slo_recommendations():
    """Get SLO improvement recommendations"""
    
    try:
        if not observability_engine or not observability_engine.slo_monitor:
            raise HTTPException(status_code=503, detail="SLO monitor not available")
        
        # Get overall SLO status for recommendations
        overall_status = await observability_engine.slo_monitor.get_overall_slo_status()
        
        # Generate recommendations based on status
        recommendations = []
        
        if overall_status.get("health_percentage", 0) < 80:
            recommendations.append({
                "type": "health_improvement",
                "priority": "high",
                "title": "Improve overall SLO health",
                "description": f"Current health is {overall_status.get('health_percentage', 0)}%. Focus on improving failing SLOs.",
                "action_items": [
                    "Review failing SLO definitions",
                    "Investigate root causes of SLO breaches",
                    "Implement better monitoring and alerting"
                ]
            })
        
        # Add more recommendations based on specific SLO statuses
        slo_statuses = overall_status.get("slo_statuses", [])
        for status in slo_statuses:
            if status.get("overall_status") == "breached":
                recommendations.append({
                    "type": "slo_breach",
                    "priority": "critical",
                    "title": f"Fix breached SLO: {status.get('slo_id')}",
                    "description": "SLO is currently breached and needs immediate attention.",
                    "action_items": [
                        "Investigate root cause",
                        "Implement immediate fixes",
                        "Review SLO definition if needed"
                    ]
                })
        
        return {
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get SLO recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get SLO recommendations: {str(e)}")
