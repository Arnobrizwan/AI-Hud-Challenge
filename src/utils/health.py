"""
Health check utilities for monitoring service and dependency health.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import aioredis
import httpx

from src.config.settings import settings
from src.models.common import DependencyStatus, HealthCheckResponse, HealthCheckStatus
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HealthCheck:
    """Health check configuration."""

    name: str
    check_function: callable
    timeout: float = 5.0
    critical: bool = True


class HealthChecker:
    """Service health checker with dependency monitoring."""

    def __init__(self):
        self.start_time = time.time()
        self.checks: List[HealthCheck] = []
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks."""
        if settings.HEALTH_CHECK_DEPENDENCIES:
            self.checks.extend(
                [
                    HealthCheck(
                        "redis", self._check_redis, timeout=3.0, critical=True), HealthCheck(
                        "memory", self._check_memory, timeout=1.0, critical=False), HealthCheck(
                        "disk", self._check_disk, timeout=1.0, critical=False), ])

    def register_check(
            self,
            name: str,
            check_function: callable,
            timeout: float = 5.0,
            critical: bool = True):
        """Register a custom health check."""
        self.checks.append(
            HealthCheck(
                name,
                check_function,
                timeout,
                critical))

    async def check_health(self) -> HealthCheckResponse:
        """Perform comprehensive health check."""
        start_time = time.time()
        dependencies = []
        overall_status = HealthCheckStatus.HEALTHY

        # Check all registered dependencies
        for check in self.checks:
            dependency_status = await self._run_health_check(check)
            dependencies.append(dependency_status)

            # Update overall status based on dependency health
            if dependency_status.status == HealthCheckStatus.UNHEALTHY:
                if check.critical:
                    overall_status = HealthCheckStatus.UNHEALTHY
                elif overall_status == HealthCheckStatus.HEALTHY:
                    overall_status = HealthCheckStatus.DEGRADED
            elif dependency_status.status == HealthCheckStatus.DEGRADED:
                if overall_status == HealthCheckStatus.HEALTHY:
                    overall_status = HealthCheckStatus.DEGRADED

        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version=settings.APP_VERSION,
            uptime=time.time() - self.start_time,
            dependencies=dependencies,
        )

    async def _run_health_check(self, check: HealthCheck) -> DependencyStatus:
        """Run a single health check with timeout."""
        start_time = time.time()

        try:
            # Run check with timeout
            result = await asyncio.wait_for(check.check_function(), timeout=check.timeout)

            response_time = time.time() - start_time

            if result is True:
                return DependencyStatus(
                    name=check.name,
                    status=HealthCheckStatus.HEALTHY,
                    response_time=response_time)
            elif isinstance(result, dict) and result.get("status") == "degraded":
                return DependencyStatus(
                    name=check.name,
                    status=HealthCheckStatus.DEGRADED,
                    response_time=response_time,
                    error=result.get("message"),
                )
            else:
                return DependencyStatus(
                    name=check.name,
                    status=HealthCheckStatus.UNHEALTHY,
                    response_time=response_time,
                    error=str(result) if result is not True else "Check failed",
                )

        except asyncio.TimeoutError:
            return DependencyStatus(
                name=check.name,
                status=HealthCheckStatus.UNHEALTHY,
                response_time=check.timeout,
                error="Health check timeout",
            )

        except Exception as e:
            logger.error(f"Health check failed for {check.name}", error=str(e))
            return DependencyStatus(
                name=check.name,
                status=HealthCheckStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                error=str(e),
            )

    async def _check_redis(self) -> bool:
        """Check Redis connectivity."""
        try:
            redis = aioredis.from_url(settings.REDIS_URL)
            await redis.ping()
            await redis.close()
            return True
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return False

    async def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil

            memory = psutil.virtual_memory()

            # Consider degraded if memory usage > 80%
            if memory.percent > 80:
                return {
                    "status": "degraded",
                    "message": f"High memory usage: {memory.percent}%"}

            return True

        except ImportError:
            return {
                "status": "degraded",
                "message": "psutil not available for memory monitoring"}
        except Exception as e:
            return False

    async def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage."""
        try:
            import psutil

            disk = psutil.disk_usage("/")

            # Consider degraded if disk usage > 85%
            usage_percent = (disk.used / disk.total) * 100
            if usage_percent > 85:
                return {
                    "status": "degraded",
                    "message": f"High disk usage: {usage_percent:.1f}%"}

            return True

        except ImportError:
            return {
                "status": "degraded",
                "message": "psutil not available for disk monitoring"}
        except Exception as e:
            return False

    async def check_external_service(
            self,
            name: str,
            url: str,
            timeout: float = 5.0,
            expected_status: int = 200) -> DependencyStatus:
        """Check external service health via HTTP."""
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                response_time = time.time() - start_time

                if response.status_code == expected_status:
                    return DependencyStatus(
                        name=name,
                        status=HealthCheckStatus.HEALTHY,
                        response_time=response_time)
                else:
                    return DependencyStatus(
                        name=name,
                        status=HealthCheckStatus.UNHEALTHY,
                        response_time=response_time,
                        error=f"HTTP {response.status_code}",
                    )

        except httpx.TimeoutException:
            return DependencyStatus(
                name=name,
                status=HealthCheckStatus.UNHEALTHY,
                response_time=timeout,
                error="Request timeout",
            )

        except Exception as e:
            return DependencyStatus(
                name=name,
                status=HealthCheckStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                error=str(e),
            )

    def is_healthy(self, status: HealthCheckStatus) -> bool:
        """Check if status indicates healthy service."""
        return status == HealthCheckStatus.HEALTHY

    def is_ready(self, status: HealthCheckStatus) -> bool:
        """Check if service is ready to handle requests."""
        return status in [
            HealthCheckStatus.HEALTHY,
            HealthCheckStatus.DEGRADED]


# Global health checker instance
health_checker = HealthChecker()


async def get_health_status() -> HealthCheckResponse:
    """Get current health status."""
    return await health_checker.check_health()


async def check_readiness() -> bool:
    """Check if service is ready to handle requests."""
    health = await health_checker.check_health()
    return health_checker.is_ready(health.status)


async def check_liveness() -> bool:
    """Check if service is alive (basic health check)."""
    # Basic liveness check - service is running
    return True


def register_health_check(
        name: str,
        check_function: callable,
        timeout: float = 5.0,
        critical: bool = True):
    """Register a custom health check."""
    health_checker.register_check(name, check_function, timeout, critical)


# Kubernetes-style health check functions
async def liveness_probe() -> Dict[str, Any]:
    """Kubernetes liveness probe."""
    is_alive = await check_liveness()
    return {
        "status": "healthy" if is_alive else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
    }


async def readiness_probe() -> Dict[str, Any]:
    """Kubernetes readiness probe."""
    is_ready = await check_readiness()
    return {
        "status": "ready" if is_ready else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
    }


async def startup_probe() -> Dict[str, Any]:
    """Kubernetes startup probe."""
    health = await health_checker.check_health()
    return {
        "status": "started" if health.status != HealthCheckStatus.UNHEALTHY else "starting",
        "uptime": health.uptime,
        "timestamp": datetime.utcnow().isoformat(),
    }
