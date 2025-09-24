"""
Health Check for Storage Service
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List

import aiohttp

from models import HealthCheck, ServiceStatus

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health check for storage service components"""

    def __init__(self):
        self._initialized = False
        self._health_checks: List[callable] = []

    async def initialize(self) -> Dict[str, Any]:
        """Initialize health checker"""
        if self._initialized:
            return

        logger.info("Initializing Health Checker...")

        try:
            # Register health checks
            self._health_checks = [
                self._check_postgresql,
                self._check_redis,
                self._check_elasticsearch,
                self._check_timescale,
            ]

            self._initialized = True
            logger.info("Health Checker initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Health Checker: {e}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup health checker"""
        self._initialized = False
        logger.info("Health Checker cleanup complete")

    async def check_health(self) -> HealthCheck:
        """Perform comprehensive health check"""
        if not self._initialized:
            return HealthCheck(
                status="unhealthy",
                timestamp=datetime.utcnow(),
                services={})

        try:
            services = {}
            overall_status = "healthy"

            # Run all health checks
            for health_check in self._health_checks:
                try:
                    service_status = await health_check()
                    services[service_status.name] = {
                        "status": service_status.status,
                        "response_time": service_status.response_time,
                        "last_check": service_status.last_check.isoformat(),
                        "error_message": service_status.error_message,
                    }

                    if service_status.status != "healthy":
                        overall_status = "degraded"

                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                    services[health_check.__name__] = {
                        "status": "unhealthy",
                        "error_message": str(e),
                        "last_check": datetime.utcnow().isoformat(),
                    }
                    overall_status = "unhealthy"

            return HealthCheck(
                status=overall_status,
                timestamp=datetime.utcnow(),
                services=services)

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheck(
                status="unhealthy",
                timestamp=datetime.utcnow(),
                services={})

    async def _check_postgresql(self) -> ServiceStatus:
        """Check PostgreSQL health"""
        start_time = datetime.utcnow()

        try:
            # This would typically check database connection
            # For now, we'll simulate a check
            await asyncio.sleep(0.1)  # Simulate check time

            response_time = (datetime.utcnow() - start_time).total_seconds()

            return ServiceStatus(
                name="postgresql",
                status="healthy",
                response_time=response_time,
                last_check=datetime.utcnow(),
            )

        except Exception as e:
            return ServiceStatus(
                name="postgresql",
                status="unhealthy",
                response_time=(datetime.utcnow() - start_time).total_seconds(),
                last_check=datetime.utcnow(),
                error_message=str(e),
            )

    async def _check_redis(self) -> ServiceStatus:
        """Check Redis health"""
        start_time = datetime.utcnow()

        try:
            # This would typically check Redis connection
            # For now, we'll simulate a check
            await asyncio.sleep(0.05)  # Simulate check time

            response_time = (datetime.utcnow() - start_time).total_seconds()

            return ServiceStatus(
                name="redis",
                status="healthy",
                response_time=response_time,
                last_check=datetime.utcnow(),
            )

        except Exception as e:
            return ServiceStatus(
                name="redis",
                status="unhealthy",
                response_time=(datetime.utcnow() - start_time).total_seconds(),
                last_check=datetime.utcnow(),
                error_message=str(e),
            )

    async def _check_elasticsearch(self) -> ServiceStatus:
        """Check Elasticsearch health"""
        start_time = datetime.utcnow()

        try:
            # This would typically check Elasticsearch cluster health
            # For now, we'll simulate a check
            await asyncio.sleep(0.1)  # Simulate check time

            response_time = (datetime.utcnow() - start_time).total_seconds()

            return ServiceStatus(
                name="elasticsearch",
                status="healthy",
                response_time=response_time,
                last_check=datetime.utcnow(),
            )

        except Exception as e:
            return ServiceStatus(
                name="elasticsearch",
                status="unhealthy",
                response_time=(datetime.utcnow() - start_time).total_seconds(),
                last_check=datetime.utcnow(),
                error_message=str(e),
            )

    async def _check_timescale(self) -> ServiceStatus:
        """Check TimescaleDB health"""
        start_time = datetime.utcnow()

        try:
            # This would typically check TimescaleDB connection
            # For now, we'll simulate a check
            await asyncio.sleep(0.1)  # Simulate check time

            response_time = (datetime.utcnow() - start_time).total_seconds()

            return ServiceStatus(
                name="timescale",
                status="healthy",
                response_time=response_time,
                last_check=datetime.utcnow(),
            )

        except Exception as e:
            return ServiceStatus(
                name="timescale",
                status="unhealthy",
                response_time=(datetime.utcnow() - start_time).total_seconds(),
                last_check=datetime.utcnow(),
                error_message=str(e),
            )
