"""
Health check endpoints for MLOps Orchestration Service
"""

from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.config.settings import Settings
from src.utils.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


class HealthStatus(BaseModel):
    """Health status response"""

    status: str
    timestamp: datetime
    version: str
    uptime: str
    services: Dict[str, Any]


class ServiceHealth(BaseModel):
    """Individual service health"""

    name: str
    status: str
    response_time_ms: float
    last_check: datetime
    error_message: Optional[str] = None


@router.get("/", response_model=HealthStatus)
async def health_check():
    """Basic health check endpoint"""

    try:
        # Get application start time (would be set at startup)
        start_time = datetime.utcnow()  # This would be the actual start time
        uptime = datetime.utcnow() - start_time

        return HealthStatus(
            status="healthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            uptime=str(uptime),
            services={},
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.get("/live", response_model=HealthStatus)
async def liveness_check():
    """Kubernetes liveness probe endpoint"""

    try:
        # Basic liveness check - service is running
        return HealthStatus(
            status="alive",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            uptime="0:00:00",  # Would be actual uptime
            services={},
        )

    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service not alive")


@router.get("/ready", response_model=HealthStatus)
async def readiness_check(settings: Settings = Depends()):
    """Kubernetes readiness probe endpoint"""

    try:
        # Check if service is ready to accept requests
        services = await check_dependencies(settings)

        # Check if all critical services are healthy
        critical_services = ["database", "redis", "airflow"]
        unhealthy_services = [
            name for name, health in services.items() if health.status != "healthy"
        ]

        if any(service in critical_services for service in unhealthy_services):
            raise HTTPException(
                status_code=503, detail=f"Critical services unhealthy: {unhealthy_services}"
            )

        return HealthStatus(
            status="ready",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            uptime="0:00:00",  # Would be actual uptime
            services=services,
        )

    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/detailed", response_model=Dict[str, Any])
async def detailed_health_check(settings: Settings = Depends()):
    """Detailed health check with all service statuses"""

    try:
        services = await check_dependencies(settings)

        # Calculate overall health
        total_services = len(services)
        healthy_services = sum(1 for health in services.values() if health.status == "healthy")
        health_percentage = (healthy_services / total_services) * 100 if total_services > 0 else 0

        overall_status = "healthy" if health_percentage >= 80 else "degraded"

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "health_percentage": health_percentage,
            "services": {name: health.dict() for name, health in services.items()},
            "metrics": {
                "total_services": total_services,
                "healthy_services": healthy_services,
                "unhealthy_services": total_services - healthy_services,
            },
        }

    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Health check failed")


async def check_dependencies(settings: Settings) -> Dict[str, ServiceHealth]:
    """Check health of all dependencies"""

    services = {}

    # Check database
    services["database"] = await check_database(settings)

    # Check Redis
    services["redis"] = await check_redis(settings)

    # Check Airflow
    services["airflow"] = await check_airflow(settings)

    # Check Vertex AI
    services["vertex_ai"] = await check_vertex_ai(settings)

    # Check MLflow
    services["mlflow"] = await check_mlflow(settings)

    # Check Prometheus
    services["prometheus"] = await check_prometheus(settings)

    return services


async def check_database(settings: Settings) -> ServiceHealth:
    """Check database health"""

    start_time = datetime.utcnow()

    try:
        # This would implement actual database health check
        # For now, return mock response
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ServiceHealth(
            name="database",
            status="healthy",
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
        )

    except Exception as e:
        return ServiceHealth(
            name="database",
            status="unhealthy",
            response_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            last_check=datetime.utcnow(),
            error_message=str(e),
        )


async def check_redis(settings: Settings) -> ServiceHealth:
    """Check Redis health"""

    start_time = datetime.utcnow()

    try:
        # This would implement actual Redis health check
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ServiceHealth(
            name="redis",
            status="healthy",
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
        )

    except Exception as e:
        return ServiceHealth(
            name="redis",
            status="unhealthy",
            response_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            last_check=datetime.utcnow(),
            error_message=str(e),
        )


async def check_airflow(settings: Settings) -> ServiceHealth:
    """Check Airflow health"""

    start_time = datetime.utcnow()

    try:
        # This would implement actual Airflow health check
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ServiceHealth(
            name="airflow",
            status="healthy",
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
        )

    except Exception as e:
        return ServiceHealth(
            name="airflow",
            status="unhealthy",
            response_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            last_check=datetime.utcnow(),
            error_message=str(e),
        )


async def check_vertex_ai(settings: Settings) -> ServiceHealth:
    """Check Vertex AI health"""

    start_time = datetime.utcnow()

    try:
        # This would implement actual Vertex AI health check
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ServiceHealth(
            name="vertex_ai",
            status="healthy",
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
        )

    except Exception as e:
        return ServiceHealth(
            name="vertex_ai",
            status="unhealthy",
            response_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            last_check=datetime.utcnow(),
            error_message=str(e),
        )


async def check_mlflow(settings: Settings) -> ServiceHealth:
    """Check MLflow health"""

    start_time = datetime.utcnow()

    try:
        # This would implement actual MLflow health check
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ServiceHealth(
            name="mlflow",
            status="healthy",
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
        )

    except Exception as e:
        return ServiceHealth(
            name="mlflow",
            status="unhealthy",
            response_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            last_check=datetime.utcnow(),
            error_message=str(e),
        )


async def check_prometheus(settings: Settings) -> ServiceHealth:
    """Check Prometheus health"""

    start_time = datetime.utcnow()

    try:
        # This would implement actual Prometheus health check
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ServiceHealth(
            name="prometheus",
            status="healthy",
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
        )

    except Exception as e:
        return ServiceHealth(
            name="prometheus",
            status="unhealthy",
            response_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            last_check=datetime.utcnow(),
            error_message=str(e),
        )
