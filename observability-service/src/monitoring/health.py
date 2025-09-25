"""
Health checking system for all services and components
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import psutil
import requests

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result"""

    name: str
    status: HealthStatus
    message: str
    response_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class ServiceHealth:
    """Service health summary"""

    service_name: str
    overall_status: HealthStatus
    health_score: float
    checks: List[HealthCheck]
    last_updated: datetime
    recommendations: List[str] = None


class HealthChecker:
    """Comprehensive health checking system"""

    def __init__(self):
        self.service_endpoints = {}
        self.health_checks = {}
        self.is_initialized = False

    async def initialize(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Initialize health checker"""
        # Load service endpoints
        self.service_endpoints = config.get("service_endpoints", {}) if config else {}

        # Set up default health checks
        self._setup_default_health_checks()

        self.is_initialized = True
        logger.info("Health checker initialized")

    def _setup_default_health_checks(self):
        """Set up default health checks"""

        # System health checks
        self.health_checks["system"] = [
            self._check_cpu_usage,
            self._check_memory_usage,
            self._check_disk_usage,
            self._check_network_connectivity,
        ]

        # Database health checks
        self.health_checks["database"] = [
            self._check_database_connection,
            self._check_database_performance,
        ]

        # Cache health checks
        self.health_checks["cache"] = [self._check_redis_connection, self._check_cache_performance]

        # External service health checks
        self.health_checks["external"] = [self._check_external_apis, self._check_dns_resolution]

    async def check_all_services(self) -> Dict[str, Any]:
        """Check health of all microservices"""
        services = [
            "ingestion-service",
            "content-extraction-service",
            "content-enrichment-service",
            "deduplication-service",
            "personalization-service",
            "summarization-service",
            "notification-service",
            "feedback-service",
            "evaluation-service",
            "safety-service",
        ]

        service_health = {}

        for service in services:
            try:
                health = await self.check_service_health(service)
                service_health[service] = health
            except Exception as e:
                logger.error(f"Failed to check health for {service}: {str(e)}")
                service_health[service] = {
                    "status": HealthStatus.UNKNOWN,
                    "health_score": 0.0,
                    "error": str(e),
                }

        return service_health

    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        endpoint = self.service_endpoints.get(service_name, f"http://{service_name}:8000")
        health_endpoint = f"{endpoint}/health"

        try:
            # Check service endpoint
            start_time = datetime.utcnow()
            response = requests.get(health_endpoint, timeout=10)
            response_time = (datetime.utcnow() - start_time).total_seconds()

            if response.status_code == 200:
                health_data = response.json()
                return {
                    "status": HealthStatus.HEALTHY,
                    "health_score": health_data.get("health_score", 1.0),
                    "response_time": response_time,
                    "details": health_data,
                    "last_checked": datetime.utcnow().isoformat(),
                }
            else:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "health_score": 0.0,
                    "response_time": response_time,
                    "error": f"HTTP {response.status_code}",
                    "last_checked": datetime.utcnow().isoformat(),
                }

        except requests.exceptions.Timeout:
            return {
                "status": HealthStatus.UNHEALTHY,
                "health_score": 0.0,
                "response_time": 10.0,
                "error": "Request timeout",
                "last_checked": datetime.utcnow().isoformat(),
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": HealthStatus.UNHEALTHY,
                "health_score": 0.0,
                "response_time": 0.0,
                "error": "Connection failed",
                "last_checked": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "health_score": 0.0,
                "response_time": 0.0,
                "error": str(e),
                "last_checked": datetime.utcnow().isoformat(),
            }

    async def check_infrastructure(self) -> Dict[str, Any]:
        """Check infrastructure health"""
        infrastructure_checks = {}

        # Kubernetes health
        k8s_health = await self._check_kubernetes_health()
        infrastructure_checks["kubernetes"] = k8s_health

        # Load balancer health
        lb_health = await self._check_load_balancer_health()
        infrastructure_checks["load_balancer"] = lb_health

        # Storage health
        storage_health = await self._check_storage_health()
        infrastructure_checks["storage"] = storage_health

        # Network health
        network_health = await self._check_network_health()
        infrastructure_checks["network"] = network_health

        return infrastructure_checks

    async def check_data_pipeline(self) -> Dict[str, Any]:
        """Check data pipeline health"""
        pipeline_checks = {}

        # Ingestion pipeline
        ingestion_health = await self._check_ingestion_pipeline()
        pipeline_checks["ingestion"] = ingestion_health

        # Processing pipeline
        processing_health = await self._check_processing_pipeline()
        pipeline_checks["processing"] = processing_health

        # Storage pipeline
        storage_health = await self._check_storage_pipeline()
        pipeline_checks["storage"] = storage_health

        return pipeline_checks

    async def check_ml_models(self) -> Dict[str, Any]:
        """Check ML model health"""
        model_checks = {}

        # Model availability
        model_availability = await self._check_model_availability()
        model_checks["availability"] = model_availability

        # Model performance
        model_performance = await self._check_model_performance()
        model_checks["performance"] = model_performance

        # Model drift
        model_drift = await self._check_model_drift()
        model_checks["drift"] = model_drift

        return model_checks

    async def check_external_dependencies(self) -> Dict[str, Any]:
        """Check external dependencies health"""
        dependency_checks = {}

        # External APIs
        api_health = await self._check_external_apis()
        dependency_checks["apis"] = api_health

        # Third-party services
        third_party_health = await self._check_third_party_services()
        dependency_checks["third_party"] = third_party_health

        return dependency_checks

    # Individual health check methods

    async def _check_cpu_usage(self) -> HealthCheck:
        """Check CPU usage"""

        try:
            cpu_percent = psutil.cpu_percent(interval=1)

            if cpu_percent < 70:
                status = HealthStatus.HEALTHY
                message = f"CPU usage: {cpu_percent}%"
            elif cpu_percent < 90:
                status = HealthStatus.DEGRADED
                message = f"High CPU usage: {cpu_percent}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Critical CPU usage: {cpu_percent}%"

            return HealthCheck(
                name="cpu_usage",
                status=status,
                message=message,
                response_time=0.0,
                timestamp=datetime.utcnow(),
                metadata={"cpu_percent": cpu_percent},
            )

        except Exception as e:
            return HealthCheck(
                name="cpu_usage",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check CPU usage: {str(e)}",
                response_time=0.0,
                timestamp=datetime.utcnow(),
            )

    async def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage"""

        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            if memory_percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Memory usage: {memory_percent}%"
            elif memory_percent < 95:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {memory_percent}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage: {memory_percent}%"

            return HealthCheck(
                name="memory_usage",
                status=status,
                message=message,
                response_time=0.0,
                timestamp=datetime.utcnow(),
                metadata={
                    "memory_percent": memory_percent,
                    "available_gb": memory.available / (1024**3),
                },
            )

        except Exception as e:
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check memory usage: {str(e)}",
                response_time=0.0,
                timestamp=datetime.utcnow(),
            )

    async def _check_disk_usage(self) -> HealthCheck:
        """Check disk usage"""

        try:
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100

            if disk_percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Disk usage: {disk_percent:.1f}%"
            elif disk_percent < 95:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {disk_percent:.1f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk usage: {disk_percent:.1f}%"

            return HealthCheck(
                name="disk_usage",
                status=status,
                message=message,
                response_time=0.0,
                timestamp=datetime.utcnow(),
                metadata={"disk_percent": disk_percent, "free_gb": disk.free / (1024**3)},
            )

        except Exception as e:
            return HealthCheck(
                name="disk_usage",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check disk usage: {str(e)}",
                response_time=0.0,
                timestamp=datetime.utcnow(),
            )

    async def _check_network_connectivity(self) -> HealthCheck:
        """Check network connectivity"""

        try:
            # Test basic connectivity
            response = requests.get("https://httpbin.org/status/200", timeout=5)

            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                message = "Network connectivity OK"
            else:
                status = HealthStatus.DEGRADED
                message = f"Network connectivity issues: HTTP {response.status_code}"

            return HealthCheck(
                name="network_connectivity",
                status=status,
                message=message,
                response_time=response.elapsed.total_seconds(),
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            return HealthCheck(
                name="network_connectivity",
                status=HealthStatus.UNHEALTHY,
                message=f"Network connectivity failed: {str(e)}",
                response_time=0.0,
                timestamp=datetime.utcnow(),
            )

    async def _check_database_connection(self) -> HealthCheck:
        """Check database connection"""

        try:
            # This would check actual database connection
            # For now, return mock result

            status = HealthStatus.HEALTHY
            message = "Database connection OK"

            return HealthCheck(
                name="database_connection",
                status=status,
                message=message,
                response_time=0.1,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            return HealthCheck(
                name="database_connection",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                response_time=0.0,
                timestamp=datetime.utcnow(),
            )

    async def _check_database_performance(self) -> HealthCheck:
        """Check database performance"""

        try:
            # This would check actual database performance
            # For now, return mock result

            status = HealthStatus.HEALTHY
            message = "Database performance OK"

            return HealthCheck(
                name="database_performance",
                status=status,
                message=message,
                response_time=0.05,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            return HealthCheck(
                name="database_performance",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check database performance: {str(e)}",
                response_time=0.0,
                timestamp=datetime.utcnow(),
            )

    async def _check_redis_connection(self) -> HealthCheck:
        """Check Redis connection"""

        try:
            # This would check actual Redis connection
            # For now, return mock result

            status = HealthStatus.HEALTHY
            message = "Redis connection OK"

            return HealthCheck(
                name="redis_connection",
                status=status,
                message=message,
                response_time=0.02,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            return HealthCheck(
                name="redis_connection",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}",
                response_time=0.0,
                timestamp=datetime.utcnow(),
            )

    async def _check_cache_performance(self) -> HealthCheck:
        """Check cache performance"""

        try:
            # This would check actual cache performance
            # For now, return mock result

            status = HealthStatus.HEALTHY
            message = "Cache performance OK"

            return HealthCheck(
                name="cache_performance",
                status=status,
                message=message,
                response_time=0.01,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            return HealthCheck(
                name="cache_performance",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check cache performance: {str(e)}",
                response_time=0.0,
                timestamp=datetime.utcnow(),
            )

    async def _check_external_apis(self) -> HealthCheck:
        """Check external API health"""

        try:
            # Check multiple external APIs
            apis = [
                "https://httpbin.org/status/200",
                "https://jsonplaceholder.typicode.com/posts/1",
            ]

            successful_checks = 0
            total_checks = len(apis)

            for api in apis:
                try:
                    response = requests.get(api, timeout=5)
                    if response.status_code == 200:
                        successful_checks += 1
                except BaseException:
                    pass

            success_rate = successful_checks / total_checks

            if success_rate >= 0.8:
                status = HealthStatus.HEALTHY
                message = f"External APIs OK ({successful_checks}/{total_checks})"
            elif success_rate >= 0.5:
                status = HealthStatus.DEGRADED
                message = f"External APIs degraded ({successful_checks}/{total_checks})"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"External APIs unhealthy ({successful_checks}/{total_checks})"

            return HealthCheck(
                name="external_apis",
                status=status,
                message=message,
                response_time=0.0,
                timestamp=datetime.utcnow(),
                metadata={"success_rate": success_rate, "successful_checks": successful_checks},
            )

        except Exception as e:
            return HealthCheck(
                name="external_apis",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check external APIs: {str(e)}",
                response_time=0.0,
                timestamp=datetime.utcnow(),
            )

    async def _check_dns_resolution(self) -> HealthCheck:
        """Check DNS resolution"""

        try:
            # This would check actual DNS resolution
            # For now, return mock result

            status = HealthStatus.HEALTHY
            message = "DNS resolution OK"

            return HealthCheck(
                name="dns_resolution",
                status=status,
                message=message,
                response_time=0.05,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            return HealthCheck(
                name="dns_resolution",
                status=HealthStatus.UNHEALTHY,
                message=f"DNS resolution failed: {str(e)}",
                response_time=0.0,
                timestamp=datetime.utcnow(),
            )

    # Infrastructure health checks

    async def _check_kubernetes_health(self) -> Dict[str, Any]:
        """Check Kubernetes cluster health"""
        try:
            # This would check actual Kubernetes health
            # For now, return mock result

            return {
                "status": HealthStatus.HEALTHY,
                "health_score": 0.95,
                "message": "Kubernetes cluster healthy",
                "details": {
                    "nodes_ready": 3,
                    "pods_running": 15,
                    "pods_pending": 0,
                    "pods_failed": 0,
                },
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "health_score": 0.0,
                "message": f"Failed to check Kubernetes health: {str(e)}",
            }

    async def _check_load_balancer_health(self) -> Dict[str, Any]:
        """Check load balancer health"""
        try:
            # This would check actual load balancer health
            # For now, return mock result

            return {
                "status": HealthStatus.HEALTHY,
                "health_score": 0.98,
                "message": "Load balancer healthy",
                "details": {"backend_servers": 3, "healthy_servers": 3, "response_time": 0.05},
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "health_score": 0.0,
                "message": f"Failed to check load balancer health: {str(e)}",
            }

    async def _check_storage_health(self) -> Dict[str, Any]:
        """Check storage health"""
        try:
            # This would check actual storage health
            # For now, return mock result

            return {
                "status": HealthStatus.HEALTHY,
                "health_score": 0.92,
                "message": "Storage healthy",
                "details": {
                    "total_space_gb": 1000,
                    "used_space_gb": 750,
                    "available_space_gb": 250,
                },
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "health_score": 0.0,
                "message": f"Failed to check storage health: {str(e)}",
            }

    async def _check_network_health(self) -> Dict[str, Any]:
        """Check network health"""
        try:
            # This would check actual network health
            # For now, return mock result

            return {
                "status": HealthStatus.HEALTHY,
                "health_score": 0.97,
                "message": "Network healthy",
                "details": {"latency_ms": 5.2, "packet_loss": 0.001, "bandwidth_mbps": 1000},
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "health_score": 0.0,
                "message": f"Failed to check network health: {str(e)}",
            }

    # Pipeline health checks

    async def _check_ingestion_pipeline(self) -> Dict[str, Any]:
        """Check ingestion pipeline health"""
        try:
            # This would check actual ingestion pipeline
            # For now, return mock result

            return {
                "status": HealthStatus.HEALTHY,
                "health_score": 0.94,
                "message": "Ingestion pipeline healthy",
                "details": {
                    "articles_processed_per_hour": 1000,
                    "processing_lag_minutes": 2,
                    "error_rate": 0.01,
                },
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "health_score": 0.0,
                "message": f"Failed to check ingestion pipeline: {str(e)}",
            }

    async def _check_processing_pipeline(self) -> Dict[str, Any]:
        """Check processing pipeline health"""
        try:
            # This would check actual processing pipeline
            # For now, return mock result

            return {
                "status": HealthStatus.HEALTHY,
                "health_score": 0.91,
                "message": "Processing pipeline healthy",
                "details": {
                    "processing_time_avg_seconds": 2.5,
                    "queue_length": 5,
                    "throughput_per_minute": 400,
                },
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "health_score": 0.0,
                "message": f"Failed to check processing pipeline: {str(e)}",
            }

    async def _check_storage_pipeline(self) -> Dict[str, Any]:
        """Check storage pipeline health"""
        try:
            # This would check actual storage pipeline
            # For now, return mock result

            return {
                "status": HealthStatus.HEALTHY,
                "health_score": 0.96,
                "message": "Storage pipeline healthy",
                "details": {
                    "write_latency_ms": 10,
                    "read_latency_ms": 5,
                    "storage_utilization": 0.75,
                },
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "health_score": 0.0,
                "message": f"Failed to check storage pipeline: {str(e)}",
            }

    # ML model health checks

    async def _check_model_availability(self) -> Dict[str, Any]:
        """Check ML model availability"""
        try:
            # This would check actual model availability
            # For now, return mock result

            return {
                "status": HealthStatus.HEALTHY,
                "health_score": 0.98,
                "message": "ML models available",
                "details": {"models_loaded": 5, "models_loading": 0, "models_failed": 0},
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "health_score": 0.0,
                "message": f"Failed to check model availability: {str(e)}",
            }

    async def _check_model_performance(self) -> Dict[str, Any]:
        """Check ML model performance"""
        try:
            # This would check actual model performance
            # For now, return mock result

            return {
                "status": HealthStatus.HEALTHY,
                "health_score": 0.89,
                "message": "ML model performance OK",
                "details": {
                    "avg_inference_time_ms": 50,
                    "accuracy": 0.92,
                    "throughput_per_second": 100,
                },
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "health_score": 0.0,
                "message": f"Failed to check model performance: {str(e)}",
            }

    async def _check_model_drift(self) -> Dict[str, Any]:
    """Check ML model drift"""
        try:
            # This would check actual model drift
            # For now, return mock result

            return {
                "status": HealthStatus.HEALTHY,
                "health_score": 0.85,
                "message": "No significant model drift detected",
                "details": {
                    "drift_score": 0.15,
                    "last_retrain": "2024-01-15",
                    "drift_threshold": 0.2,
                },
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "health_score": 0.0,
                "message": f"Failed to check model drift: {str(e)}",
            }

    # Third-party service checks

    async def _check_third_party_services(self) -> Dict[str, Any]:
    """Check third-party services"""
        try:
            # This would check actual third-party services
            # For now, return mock result

            return {
                "status": HealthStatus.HEALTHY,
                "health_score": 0.93,
                "message": "Third-party services OK",
                "details": {"services_checked": 3, "services_healthy": 3, "services_degraded": 0},
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "health_score": 0.0,
                "message": f"Failed to check third-party services: {str(e)}",
            }
