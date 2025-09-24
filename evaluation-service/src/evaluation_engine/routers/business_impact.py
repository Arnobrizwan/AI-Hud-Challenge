"""
Business Impact Router - Business impact analysis endpoints
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from ..core import EvaluationEngine
from ..dependencies import get_evaluation_engine

logger = logging.getLogger(__name__)

business_impact_router = APIRouter()


@business_impact_router.post("/analyze")
async def analyze_business_impact(
    business_metrics: List[str],
    evaluation_period: Dict[str, Any],
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """Analyze business impact of ML model changes"""

    try:
        logger.info(f"Analyzing business impact for metrics: {business_metrics}")

        analysis = await evaluation_engine.business_impact_analyzer.analyze(
            business_metrics, evaluation_period
        )

        return {
            "status": "success",
            "analysis": analysis.dict(),
            "message": "Business impact analysis completed successfully",
        }

    except Exception as e:
        logger.error(f"Error analyzing business impact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@business_impact_router.post("/revenue")
async def analyze_revenue_impact(
    pre_period_data: Dict[str, Any],
    post_period_data: Dict[str, Any],
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """Analyze revenue impact specifically"""

    try:
        logger.info("Analyzing revenue impact")

        # Mock revenue impact analysis
        revenue_impact = await evaluation_engine.business_impact_analyzer.revenue_calculator.calculate_revenue_impact(
            pre_period_data, post_period_data
        )

        return {
            "status": "success",
            "revenue_impact": revenue_impact,
            "message": "Revenue impact analysis completed",
        }

    except Exception as e:
        logger.error(f"Error analyzing revenue impact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@business_impact_router.post("/engagement")
async def analyze_engagement_impact(
    pre_period_data: Dict[str, Any],
    post_period_data: Dict[str, Any],
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """Analyze user engagement impact"""

    try:
        logger.info("Analyzing engagement impact")

        # Mock engagement impact analysis
        engagement_impact = await evaluation_engine.business_impact_analyzer.engagement_analyzer.analyze_engagement_impact(
            pre_period_data, post_period_data
        )

        return {
            "status": "success",
            "engagement_impact": engagement_impact,
            "message": "Engagement impact analysis completed",
        }

    except Exception as e:
        logger.error(f"Error analyzing engagement impact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@business_impact_router.post("/retention")
async def analyze_retention_impact(
    pre_period_data: Dict[str, Any],
    post_period_data: Dict[str, Any],
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """Analyze user retention impact"""

    try:
        logger.info("Analyzing retention impact")

        # Mock retention impact analysis
        retention_impact = await evaluation_engine.business_impact_analyzer.retention_analyzer.analyze_retention_impact(
            pre_period_data, post_period_data
        )

        return {
            "status": "success",
            "retention_impact": retention_impact,
            "message": "Retention impact analysis completed",
        }

    except Exception as e:
        logger.error(f"Error analyzing retention impact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@business_impact_router.post("/roi")
async def calculate_roi(
    benefits: Dict[str, float],
    costs: Dict[str, float],
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
):
    """Calculate return on investment (ROI)"""

    try:
        logger.info("Calculating ROI")

        total_benefits = sum(benefits.values())
        total_costs = sum(costs.values())

        if total_costs == 0:
            roi = float("inf") if total_benefits > 0 else 0
        else:
            roi = (total_benefits - total_costs) / total_costs

        return {
            "status": "success",
            "roi": roi,
            "total_benefits": total_benefits,
            "total_costs": total_costs,
            "net_benefit": total_benefits - total_costs,
            "message": "ROI calculation completed",
        }

    except Exception as e:
        logger.error(f"Error calculating ROI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@business_impact_router.get("/metrics")
async def get_business_metrics():
    """Get available business metrics"""

    try:
        metrics = {
            "revenue": {
                "description": "Revenue impact analysis",
                "metrics": ["revenue_change", "revenue_change_percent", "revenue_per_user"],
            },
            "engagement": {
                "description": "User engagement impact analysis",
                "metrics": ["engagement_rate", "session_duration", "page_views"],
            },
            "retention": {
                "description": "User retention impact analysis",
                "metrics": ["retention_rate", "churn_rate", "lifetime_value"],
            },
            "content_consumption": {
                "description": "Content consumption impact analysis",
                "metrics": ["articles_read", "time_spent", "sharing_rate"],
            },
        }

        return {
            "status": "success",
            "business_metrics": metrics,
            "message": "Available business metrics retrieved",
        }

    except Exception as e:
        logger.error(f"Error getting business metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
