"""
Dashboard API - Enterprise-grade dashboard backend
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import httpx
import asyncio

logger = logging.getLogger(__name__)

# Pydantic models
class DashboardMetrics(BaseModel):
    total_pipelines: int
    active_executions: int
    success_rate: float
    avg_execution_time: float
    system_health: str
    last_updated: datetime

class PipelineSummary(BaseModel):
    pipeline_id: str
    name: str
    status: str
    last_execution: Optional[datetime]
    components_count: int
    success_rate: float

class ExecutionSummary(BaseModel):
    execution_id: str
    pipeline_id: str
    status: str
    start_time: datetime
    duration: Optional[float]
    components_completed: int
    total_components: int

# Create router
router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard"])

# Configuration
PIPELINE_SERVICE_URL = "http://ai-pipeline-service:8000"
NEWS_SERVICE_URL = "http://localhost:8000"

class DashboardService:
    """Dashboard service for aggregating data from multiple services"""
    
    def __init__(self):
        self.pipeline_client = httpx.AsyncClient(base_url=PIPELINE_SERVICE_URL)
        self.news_client = httpx.AsyncClient(base_url=NEWS_SERVICE_URL)
    
    async def get_overview_metrics(self) -> DashboardMetrics:
        """Get overview metrics for dashboard"""
        try:
            # Get pipeline analytics
            pipeline_response = await self.pipeline_client.get("/api/v1/pipeline/analytics")
            pipeline_data = pipeline_response.json()
            
            # Get news service health
            try:
                news_response = await self.news_client.get("/health")
                news_health = "healthy" if news_response.status_code == 200 else "unhealthy"
            except:
                news_health = "unreachable"
            
            # Calculate system health
            system_health = "healthy"
            if pipeline_data.get("success_rate", 0) < 0.8:
                system_health = "warning"
            if news_health != "healthy":
                system_health = "degraded"
            
            return DashboardMetrics(
                total_pipelines=pipeline_data.get("total_pipelines", 0),
                active_executions=pipeline_data.get("running_executions", 0),
                success_rate=pipeline_data.get("success_rate", 0),
                avg_execution_time=pipeline_data.get("avg_execution_time", 0),
                system_health=system_health,
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to get overview metrics: {str(e)}")
            return DashboardMetrics(
                total_pipelines=0,
                active_executions=0,
                success_rate=0,
                avg_execution_time=0,
                system_health="error",
                last_updated=datetime.utcnow()
            )
    
    async def get_pipeline_summaries(self) -> List[PipelineSummary]:
        """Get pipeline summaries for dashboard"""
        try:
            response = await self.pipeline_client.get("/api/v1/pipeline/pipelines")
            pipelines_data = response.json()
            
            summaries = []
            for pipeline in pipelines_data:
                # Calculate success rate for this pipeline
                executions = pipeline.get("recent_executions", [])
                successful = sum(1 for exec in executions if exec.get("status") == "completed")
                success_rate = successful / len(executions) if executions else 0
                
                summaries.append(PipelineSummary(
                    pipeline_id=pipeline["pipeline_id"],
                    name=pipeline["name"],
                    status=pipeline["status"],
                    last_execution=datetime.fromisoformat(pipeline["last_execution"]) if pipeline.get("last_execution") else None,
                    components_count=len(pipeline.get("components", [])),
                    success_rate=success_rate
                ))
            
            return summaries
            
        except Exception as e:
            logger.error(f"Failed to get pipeline summaries: {str(e)}")
            return []
    
    async def get_recent_executions(self, limit: int = 10) -> List[ExecutionSummary]:
        """Get recent executions for dashboard"""
        try:
            # Get all pipelines first
            pipelines_response = await self.pipeline_client.get("/api/v1/pipeline/pipelines")
            pipelines_data = pipelines_response.json()
            
            all_executions = []
            for pipeline in pipelines_data:
                executions = pipeline.get("recent_executions", [])
                for exec_data in executions:
                    start_time = datetime.fromisoformat(exec_data["start_time"])
                    end_time = datetime.fromisoformat(exec_data["end_time"]) if exec_data.get("end_time") else None
                    duration = (end_time - start_time).total_seconds() if end_time else None
                    
                    all_executions.append(ExecutionSummary(
                        execution_id=exec_data["id"],
                        pipeline_id=exec_data["pipeline_id"],
                        status=exec_data["status"],
                        start_time=start_time,
                        duration=duration,
                        components_completed=len(exec_data.get("components_executed", [])),
                        total_components=len(pipeline.get("components", []))
                    ))
            
            # Sort by start time and limit
            all_executions.sort(key=lambda x: x.start_time, reverse=True)
            return all_executions[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get recent executions: {str(e)}")
            return []
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health"""
        try:
            health_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "services": {}
            }
            
            # Check pipeline service
            try:
                pipeline_health = await self.pipeline_client.get("/api/v1/pipeline/health")
                health_data["services"]["pipeline_service"] = {
                    "status": "healthy" if pipeline_health.status_code == 200 else "unhealthy",
                    "response_time": pipeline_health.elapsed.total_seconds()
                }
            except Exception as e:
                health_data["services"]["pipeline_service"] = {
                    "status": "unreachable",
                    "error": str(e)
                }
            
            # Check news service
            try:
                news_health = await self.news_client.get("/health")
                health_data["services"]["news_service"] = {
                    "status": "healthy" if news_health.status_code == 200 else "unhealthy",
                    "response_time": news_health.elapsed.total_seconds()
                }
            except Exception as e:
                health_data["services"]["news_service"] = {
                    "status": "unreachable",
                    "error": str(e)
                }
            
            # Overall health
            all_healthy = all(
                service["status"] == "healthy" 
                for service in health_data["services"].values()
            )
            health_data["overall_status"] = "healthy" if all_healthy else "degraded"
            
            return health_data
            
        except Exception as e:
            logger.error(f"Failed to get system health: {str(e)}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }

# Initialize dashboard service
dashboard_service = DashboardService()

@router.get("/overview", response_model=DashboardMetrics)
async def get_overview():
    """Get dashboard overview metrics"""
    return await dashboard_service.get_overview_metrics()

@router.get("/pipelines", response_model=List[PipelineSummary])
async def get_pipelines():
    """Get pipeline summaries"""
    return await dashboard_service.get_pipeline_summaries()

@router.get("/executions", response_model=List[ExecutionSummary])
async def get_recent_executions(limit: int = Query(10, ge=1, le=100)):
    """Get recent executions"""
    return await dashboard_service.get_recent_executions(limit)

@router.get("/health", response_model=Dict[str, Any])
async def get_system_health():
    """Get system health status"""
    return await dashboard_service.get_system_health()

@router.get("/metrics/trends")
async def get_metrics_trends(days: int = Query(7, ge=1, le=30)):
    """Get metrics trends over time"""
    try:
        # This would typically query a time-series database
        # For now, return mock data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        trends = {
            "execution_count": [],
            "success_rate": [],
            "avg_duration": [],
            "dates": []
        }
        
        # Generate mock trend data
        for i in range(days):
            date = start_date + timedelta(days=i)
            trends["dates"].append(date.isoformat())
            trends["execution_count"].append(10 + (i % 5) * 2)
            trends["success_rate"].append(0.85 + (i % 3) * 0.05)
            trends["avg_duration"].append(120 + (i % 4) * 10)
        
        return trends
        
    except Exception as e:
        logger.error(f"Failed to get metrics trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_alerts():
    """Get system alerts"""
    try:
        # Get current metrics
        metrics = await dashboard_service.get_overview_metrics()
        
        alerts = []
        
        # Check for low success rate
        if metrics.success_rate < 0.8:
            alerts.append({
                "id": "low_success_rate",
                "type": "warning",
                "title": "Low Success Rate",
                "message": f"Pipeline success rate is {metrics.success_rate:.1%}",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Check for system health
        if metrics.system_health != "healthy":
            alerts.append({
                "id": "system_health",
                "type": "error" if metrics.system_health == "error" else "warning",
                "title": "System Health Issue",
                "message": f"System health is {metrics.system_health}",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return alerts
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
