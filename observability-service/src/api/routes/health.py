"""
Health check API routes
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from monitoring.health import HealthChecker, HealthStatus
from monitoring.observability_engine import observability_engine

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: str
    service: str
    version: str
    uptime: float
    health_score: float
    checks: List[Dict[str, Any]]


class ServiceHealthResponse(BaseModel):
    """Service health response"""

    service_name: str
    status: str
    health_score: float
    response_time: float
    last_checked: str
    details: Dict[str, Any]


class SystemHealthResponse(BaseModel):
    """System health response"""

    overall_health_score: float
    service_health: Dict[str, Any]
    infrastructure_health: Dict[str, Any]
    pipeline_health: Dict[str, Any]
    ml_model_health: Dict[str, Any]
    dependency_health: Dict[str, Any]
    health_check_timestamp: str
    recommendations: List[str]


@router.get("/", response_model=HealthResponse)
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""

    try:
        # Get basic service info
        uptime = 0.0  # This would be calculated from service start time
        health_score = 1.0  # This would be calculated from actual health checks

        # Basic health checks
        checks = [
            {
                "name": "service_running",
                "status": "healthy",
                "message": "Service is running",
                "timestamp": datetime.utcnow().isoformat(),
            }
        ]

        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            service="observability-service",
            version="1.0.0",
            uptime=uptime,
            health_score=health_score,
            checks=checks,
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.get("/detailed", response_model=SystemHealthResponse)
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with system status"""

    try:
        if not observability_engine:
            raise HTTPException(status_code=503,
                                detail="Observability engine not initialized")

        # Get comprehensive health report
        health_report = await observability_engine.collect_system_health()

        return SystemHealthResponse(
            overall_health_score=health_report.overall_health_score,
            service_health=health_report.service_health,
            infrastructure_health=health_report.infrastructure_health,
            pipeline_health=health_report.pipeline_health,
            ml_model_health=health_report.ml_model_health,
            dependency_health=health_report.dependency_health,
            health_check_timestamp=health_report.health_check_timestamp.isoformat(),
            recommendations=health_report.recommendations or [],
        )

    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Health check failed: {str(e)}")


@router.get("/services", response_model=List[ServiceHealthResponse])
async def get_services_health() -> Dict[str, Any]:
    """Get health status of all services"""

    try:
        if not observability_engine:
            raise HTTPException(status_code=503,
                                detail="Observability engine not initialized")

        # Get service health
        service_health = await observability_engine.check_service_health()

        responses = []
        for service_name, health_data in service_health.items():
            if isinstance(health_data, dict):
                responses.append(
                    ServiceHealthResponse(
                        service_name=service_name, status=health_data.get(
                            "status", "unknown"), health_score=health_data.get(
                            "health_score", 0.0), response_time=health_data.get(
                            "response_time", 0.0), last_checked=health_data.get(
                            "last_checked", datetime.utcnow().isoformat()), details=health_data.get(
                            "details", {}), ))

        return responses

    except Exception as e:
        logger.error(f"Failed to get services health: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get services health: {str(e)}")


@router.get("/services/{service_name}", response_model=ServiceHealthResponse)
async def get_service_health(service_name: str) -> Dict[str, Any]:
    """Get health status of specific service"""

    try:
        if not observability_engine:
            raise HTTPException(status_code=503,
                                detail="Observability engine not initialized")

        # Get specific service health
        service_health = await observability_engine.check_service_health()

        if service_name not in service_health:
            raise HTTPException(status_code=404,
                                detail=f"Service {service_name} not found")

        health_data = service_health[service_name]

        if not isinstance(health_data, dict):
            raise HTTPException(
                status_code=500,
                detail=f"Invalid health data for {service_name}")

        return ServiceHealthResponse(
            service_name=service_name,
            status=health_data.get("status", "unknown"),
            health_score=health_data.get("health_score", 0.0),
            response_time=health_data.get("response_time", 0.0),
            last_checked=health_data.get("last_checked", datetime.utcnow().isoformat()),
            details=health_data.get("details", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to get health for service {service_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get health for service {service_name}: {str(e)}")


@router.get("/infrastructure")
async def get_infrastructure_health() -> Dict[str, Any]:
    """Get infrastructure health status"""

    try:
        if not observability_engine:
            raise HTTPException(status_code=503,
                                detail="Observability engine not initialized")

        # Get infrastructure health
        infra_health = await observability_engine.check_infrastructure_health()

        return {"infrastructure_health": infra_health,
                "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Failed to get infrastructure health: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get infrastructure health: {str(e)}")


@router.get("/pipeline")
async def get_pipeline_health() -> Dict[str, Any]:
    """Get data pipeline health status"""

    try:
        if not observability_engine:
            raise HTTPException(status_code=503,
                                detail="Observability engine not initialized")

        # Get pipeline health
        pipeline_health = await observability_engine.check_data_pipeline_health()

        return {"pipeline_health": pipeline_health,
                "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Failed to get pipeline health: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get pipeline health: {str(e)}")


@router.get("/ml-models")
async def get_ml_models_health() -> Dict[str, Any]:
    """Get ML models health status"""

    try:
        if not observability_engine:
            raise HTTPException(status_code=503,
                                detail="Observability engine not initialized")

        # Get ML models health
        ml_health = await observability_engine.check_ml_model_health()

        return {"ml_models_health": ml_health,
                "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Failed to get ML models health: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get ML models health: {str(e)}")


@router.get("/dependencies")
async def get_dependencies_health() -> Dict[str, Any]:
    """Get external dependencies health status"""

    try:
        if not observability_engine:
            raise HTTPException(status_code=503,
                                detail="Observability engine not initialized")

        # Get dependencies health
        deps_health = await observability_engine.check_external_dependencies_health()

        return {"dependencies_health": deps_health,
                "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Failed to get dependencies health: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dependencies health: {str(e)}")


@router.post("/services/{service_name}/check")
async def trigger_service_health_check(service_name: str) -> Dict[str, Any]:
    """Trigger immediate health check for specific service"""

    try:
        if not observability_engine:
            raise HTTPException(status_code=503,
                                detail="Observability engine not initialized")

        # Trigger health check
        health_data = await observability_engine.check_service_health()

        if service_name not in health_data:
            raise HTTPException(status_code=404,
                                detail=f"Service {service_name} not found")

        return {
            "message": f"Health check triggered for {service_name}",
            "service_name": service_name,
            "timestamp": datetime.utcnow().isoformat(),
            "health_data": health_data[service_name],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to trigger health check for {service_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger health check for {service_name}: {str(e)}")


@router.get("/metrics/summary")
async def get_health_metrics_summary() -> Dict[str, Any]:
    """Get health metrics summary"""

    try:
        if not observability_engine:
            raise HTTPException(status_code=503,
                                detail="Observability engine not initialized")

        # Get system status
        system_status = await observability_engine.get_system_status()

        return {
            "health_metrics": {
                "overall_health_score": system_status.get("health_score", 0.0),
                "active_incidents": system_status.get("active_incidents", 0),
                "slo_status": system_status.get("slo_status", {}),
                "metrics_summary": system_status.get("metrics_summary", {}),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get health metrics summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get health metrics summary: {str(e)}")
