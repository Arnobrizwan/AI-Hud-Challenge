"""
Cost monitoring API routes
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel

from monitoring.cost import CostAlert, CostBudget, CostCategory, CostData, CostMonitor
from monitoring.observability_engine import observability_engine

logger = logging.getLogger(__name__)

router = APIRouter()


class CostDataResponse(BaseModel):
    """Cost data response"""

    timestamp: str
    category: str
    service: str
    amount: float
    currency: str
    region: Optional[str] = None
    resource_id: Optional[str] = None
    metadata: Dict[str, Any]


class CostSummaryResponse(BaseModel):
    """Cost summary response"""

    total_cost: float
    daily_average: float
    category_breakdown: Dict[str, float]
    service_breakdown: Dict[str, float]
    time_window_days: int
    generated_at: str


class CostAlertResponse(BaseModel):
    """Cost alert response"""

    id: str
    alert_type: str
    category: str
    service: str
    current_cost: float
    threshold: float
    message: str
    severity: str
    created_at: str
    resolved: bool
    recommendations: List[str]


class CostBudgetRequest(BaseModel):
    """Cost budget creation request"""

    name: str
    category: str
    service: str
    monthly_limit: float
    daily_limit: Optional[float] = None
    currency: str = "USD"
    alert_thresholds: List[float] = []


class CostBudgetResponse(BaseModel):
    """Cost budget response"""

    id: str
    name: str
    category: str
    service: str
    monthly_limit: float
    daily_limit: Optional[float] = None
    currency: str
    alert_thresholds: List[float]
    created_at: str
    enabled: bool


@router.get("/", response_model=List[CostDataResponse])
async def get_cost_data(
    category: Optional[str] = Query(None, description="Filter by cost category"),
    service: Optional[str] = Query(None, description="Filter by service"),
    time_window: int = Query(3600, description="Time window in seconds"),
    limit: int = Query(1000, description="Maximum number of records"),
):
    """Get cost data with optional filtering"""

    try:
        if not observability_engine or not observability_engine.cost_monitor:
            raise HTTPException(status_code=503, detail="Cost monitor not available")

        # Collect cost metrics
        cost_data = await observability_engine.cost_monitor.collect_cost_metrics()

        # Apply filters
        filtered_data = cost_data

        if category:
            filtered_data = [d for d in filtered_data if d.category.value == category]

        if service:
            filtered_data = [d for d in filtered_data if d.service == service]

        # Filter by time window
        cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
        filtered_data = [d for d in filtered_data if d.timestamp >= cutoff_time]

        # Limit results
        filtered_data = filtered_data[:limit]

        # Convert to response format
        cost_responses = []
        for data in filtered_data:
            cost_responses.append(
                CostDataResponse(
                    timestamp=data.timestamp.isoformat(),
                    category=data.category.value,
                    service=data.service,
                    amount=data.amount,
                    currency=data.currency,
                    region=data.region,
                    resource_id=data.resource_id,
                    metadata=data.metadata,
                )
            )

        return cost_responses

    except Exception as e:
        logger.error(f"Failed to get cost data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cost data: {str(e)}")


@router.get("/summary", response_model=CostSummaryResponse)
async def get_cost_summary(time_window: int = Query(604800, description="Time window in seconds (default: 7 days)")):
    """Get cost summary"""

    try:
        if not observability_engine or not observability_engine.cost_monitor:
            raise HTTPException(status_code=503, detail="Cost monitor not available")

        # Get cost summary
        time_window_delta = timedelta(seconds=time_window)
        summary = await observability_engine.cost_monitor.get_cost_summary(time_window_delta)

        return CostSummaryResponse(
            total_cost=summary["total_cost"],
            daily_average=summary["daily_average"],
            category_breakdown=summary["category_breakdown"],
            service_breakdown=summary["service_breakdown"],
            time_window_days=summary["time_window_days"],
            generated_at=summary["generated_at"],
        )

    except Exception as e:
        logger.error(f"Failed to get cost summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cost summary: {str(e)}")


@router.get("/analysis")
async def get_cost_analysis() -> Dict[str, Any]:
    """Get cost analysis and insights"""
    try:
        if not observability_engine or not observability_engine.cost_monitor:
            raise HTTPException(status_code=503, detail="Cost monitor not available")

        # Get cost analysis
        analysis = await observability_engine.cost_monitor.analyze_costs()

        return {"cost_analysis": analysis, "generated_at": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Failed to get cost analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cost analysis: {str(e)}")


@router.get("/alerts", response_model=List[CostAlertResponse])
async def get_cost_alerts() -> Dict[str, Any]:
    """Get active cost alerts"""
    try:
        if not observability_engine or not observability_engine.cost_monitor:
            raise HTTPException(status_code=503, detail="Cost monitor not available")

        # Get active alerts
        alerts = await observability_engine.cost_monitor.get_active_alerts()

        # Convert to response format
        alert_responses = []
        for alert in alerts:
            alert_responses.append(
                CostAlertResponse(
                    id=alert.id,
                    alert_type=alert.alert_type.value,
                    category=alert.category.value,
                    service=alert.service,
                    current_cost=alert.current_cost,
                    threshold=alert.threshold,
                    message=alert.message,
                    severity=alert.severity,
                    created_at=alert.created_at.isoformat(),
                    resolved=alert.resolved,
                    recommendations=alert.recommendations,
                )
            )

        return alert_responses

    except Exception as e:
        logger.error(f"Failed to get cost alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cost alerts: {str(e)}")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_cost_alert(
    alert_id: str, user: str = Body(..., embed=True), notes: Optional[str] = Body(None, embed=True)
):
    """Resolve cost alert"""

    try:
        if not observability_engine or not observability_engine.cost_monitor:
            raise HTTPException(status_code=503, detail="Cost monitor not available")

        # Resolve alert
        await observability_engine.cost_monitor.resolve_alert(alert_id, user, notes)

        return {
            "message": f"Cost alert {alert_id} resolved by {user}",
            "alert_id": alert_id,
            "resolved_by": user,
            "resolved_at": datetime.utcnow().isoformat(),
            "notes": notes,
        }

    except Exception as e:
        logger.error(f"Failed to resolve cost alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve cost alert {alert_id}: {str(e)}")


@router.post("/budgets", response_model=CostBudgetResponse)
async def create_cost_budget(budget_request: CostBudgetRequest) -> Dict[str, Any]:
    """Create cost budget"""
    try:
        if not observability_engine or not observability_engine.cost_monitor:
            raise HTTPException(status_code=503, detail="Cost monitor not available")

        # Create budget data
        budget_data = {
            "id": f"budget_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "name": budget_request.name,
            "category": budget_request.category,
            "service": budget_request.service,
            "monthly_limit": budget_request.monthly_limit,
            "daily_limit": budget_request.daily_limit,
            "currency": budget_request.currency,
            "alert_thresholds": budget_request.alert_thresholds,
        }

        # Create budget
        budget = await observability_engine.cost_monitor.create_budget(budget_data)

        return CostBudgetResponse(
            id=budget.id,
            name=budget.name,
            category=budget.category.value,
            service=budget.service,
            monthly_limit=budget.monthly_limit,
            daily_limit=budget.daily_limit,
            currency=budget.currency,
            alert_thresholds=budget.alert_thresholds,
            created_at=budget.created_at.isoformat(),
            enabled=budget.enabled,
        )

    except Exception as e:
        logger.error(f"Failed to create cost budget: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create cost budget: {str(e)}")


@router.get("/budgets", response_model=List[CostBudgetResponse])
async def get_cost_budgets() -> Dict[str, Any]:
    """Get all cost budgets"""
    try:
        if not observability_engine or not observability_engine.cost_monitor:
            raise HTTPException(status_code=503, detail="Cost monitor not available")

        # Get budgets from alert manager
        budgets = list(observability_engine.cost_monitor.alert_manager.budgets.values())

        # Convert to response format
        budget_responses = []
        for budget in budgets:
            budget_responses.append(
                CostBudgetResponse(
                    id=budget.id,
                    name=budget.name,
                    category=budget.category.value,
                    service=budget.service,
                    monthly_limit=budget.monthly_limit,
                    daily_limit=budget.daily_limit,
                    currency=budget.currency,
                    alert_thresholds=budget.alert_thresholds,
                    created_at=budget.created_at.isoformat(),
                    enabled=budget.enabled,
                )
            )

        return budget_responses

    except Exception as e:
        logger.error(f"Failed to get cost budgets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cost budgets: {str(e)}")


@router.get("/budgets/{budget_id}", response_model=CostBudgetResponse)
async def get_cost_budget(budget_id: str) -> Dict[str, Any]:
    """Get specific cost budget by ID"""
    try:
        if not observability_engine or not observability_engine.cost_monitor:
            raise HTTPException(status_code=503, detail="Cost monitor not available")

        # Get budget
        budget = observability_engine.cost_monitor.alert_manager.budgets.get(budget_id)

        if not budget:
            raise HTTPException(status_code=404, detail=f"Cost budget {budget_id} not found")

        return CostBudgetResponse(
            id=budget.id,
            name=budget.name,
            category=budget.category.value,
            service=budget.service,
            monthly_limit=budget.monthly_limit,
            daily_limit=budget.daily_limit,
            currency=budget.currency,
            alert_thresholds=budget.alert_thresholds,
            created_at=budget.created_at.isoformat(),
            enabled=budget.enabled,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cost budget {budget_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cost budget {budget_id}: {str(e)}")


@router.get("/trends")
async def get_cost_trends(time_window: int = Query(2592000, description="Time window in seconds (default: 30 days)")):
    """Get cost trends over time"""

    try:
        if not observability_engine or not observability_engine.cost_monitor:
            raise HTTPException(status_code=503, detail="Cost monitor not available")

        # This would query historical cost data
        # For now, return mock trends

        trends = {
            "time_window_seconds": time_window,
            "trends": {
                "total_cost": {
                    "current": 1500.50,
                    "trend": "increasing",
                    "change_percent": 12.5,
                    "daily_average": 50.02,
                },
                "compute_costs": {"current": 800.25, "trend": "stable", "change_percent": 2.1},
                "storage_costs": {"current": 200.75, "trend": "increasing", "change_percent": 25.3},
                "third_party_costs": {
                    "current": 499.50,
                    "trend": "decreasing",
                    "change_percent": -5.2,
                },
            },
            "generated_at": datetime.utcnow().isoformat(),
        }

        return trends

    except Exception as e:
        logger.error(f"Failed to get cost trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cost trends: {str(e)}")


@router.get("/optimization")
async def get_cost_optimization_recommendations() -> Dict[str, Any]:
    """Get cost optimization recommendations"""
    try:
        if not observability_engine or not observability_engine.cost_monitor:
            raise HTTPException(status_code=503, detail="Cost monitor not available")

        # Get cost analysis for optimization recommendations
        analysis = await observability_engine.cost_monitor.analyze_costs()

        # Extract optimization recommendations
        optimizations = analysis.get("optimizations", [])

        # Convert to response format
        recommendations = []
        for opt in optimizations:
            recommendations.append(
                {
                    "id": opt.id,
                    "category": opt.category.value,
                    "service": opt.service,
                    "current_cost": opt.current_cost,
                    "potential_savings": opt.potential_savings,
                    "recommendation": opt.recommendation,
                    "implementation_effort": opt.implementation_effort,
                    "priority": opt.priority,
                    "created_at": opt.created_at.isoformat(),
                }
            )

        return {
            "optimization_recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "potential_total_savings": sum(opt.potential_savings for opt in optimizations),
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get cost optimization recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cost optimization recommendations: {str(e)}")


@router.get("/breakdown")
async def get_cost_breakdown(
    group_by: str = Query("category", description="Group by: category, service, region"),
    time_window: int = Query(604800, description="Time window in seconds (default: 7 days)"),
):
    """Get cost breakdown by category, service, or region"""

    try:
        if not observability_engine or not observability_engine.cost_monitor:
            raise HTTPException(status_code=503, detail="Cost monitor not available")

        # Collect cost data
        cost_data = await observability_engine.cost_monitor.collect_cost_metrics()

        # Filter by time window
        cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
        recent_data = [d for d in cost_data if d.timestamp >= cutoff_time]

        # Group by specified field
        breakdown = {}
        for data in recent_data:
            if group_by == "category":
                key = data.category.value
            elif group_by == "service":
                key = data.service
            elif group_by == "region":
                key = data.region or "unknown"
            else:
                key = "unknown"

            if key not in breakdown:
                breakdown[key] = 0.0
            breakdown[key] += data.amount

        # Calculate percentages
        total_cost = sum(breakdown.values())
        breakdown_with_percentages = {}
        for key, amount in breakdown.items():
            percentage = (amount / total_cost * 100) if total_cost > 0 else 0
            breakdown_with_percentages[key] = {"amount": amount, "percentage": percentage}

        return {
            "group_by": group_by,
            "time_window_seconds": time_window,
            "total_cost": total_cost,
            "breakdown": breakdown_with_percentages,
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get cost breakdown: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cost breakdown: {str(e)}")


@router.post("/collect")
async def trigger_cost_collection() -> Dict[str, Any]:
    """Trigger immediate cost data collection"""
    try:
        if not observability_engine or not observability_engine.cost_monitor:
            raise HTTPException(status_code=503, detail="Cost monitor not available")

        # Trigger cost collection
        cost_data = await observability_engine.cost_monitor.collect_cost_metrics()

        return {
            "message": "Cost data collection triggered successfully",
            "records_collected": len(cost_data),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to trigger cost collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger cost collection: {str(e)}")
