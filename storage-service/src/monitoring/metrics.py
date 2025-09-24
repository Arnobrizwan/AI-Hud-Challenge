"""
Metrics Collection for Storage Service
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "storage_requests_total", "Total number of requests", ["method", "endpoint", "status"]
)

REQUEST_DURATION = Histogram(
    "storage_request_duration_seconds", "Request duration in seconds", ["method", "endpoint"]
)

CACHE_HITS = Counter("storage_cache_hits_total", "Total cache hits", ["cache_level"])

CACHE_MISSES = Counter("storage_cache_misses_total", "Total cache misses", ["cache_level"])

VECTOR_SEARCH_DURATION = Histogram(
    "storage_vector_search_duration_seconds", "Vector search duration in seconds"
)

FULLTEXT_SEARCH_DURATION = Histogram(
    "storage_fulltext_search_duration_seconds", "Full-text search duration in seconds"
)

ACTIVE_CONNECTIONS = Gauge("storage_active_connections", "Number of active database connections")

STORAGE_SIZE = Gauge("storage_data_size_bytes", "Total storage size in bytes", ["storage_type"])


class MetricsCollector:
    """Collect and expose metrics for the storage service"""

    def __init__(self):
        self._initialized = False
        self._metrics_task: asyncio.Task = None

    async def initialize(self):
        """Initialize metrics collection"""
        if self._initialized:
            return

        logger.info("Initializing Metrics Collector...")

        try:
            # Start Prometheus HTTP server
            start_http_server(9090)

            # Start background metrics collection
            self._metrics_task = asyncio.create_task(self._collect_metrics_loop())

            self._initialized = True
            logger.info("Metrics Collector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Metrics Collector: {e}")
            raise

    async def cleanup(self):
        """Cleanup metrics collection"""
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass

        self._initialized = False
        logger.info("Metrics Collector cleanup complete")

    async def start_metrics_collection(self):
        """Start background metrics collection"""
        if not self._initialized:
            return

        if self._metrics_task and not self._metrics_task.done():
            return

        self._metrics_task = asyncio.create_task(self._collect_metrics_loop())
        logger.info("Metrics collection started")

    async def _collect_metrics_loop(self):
        """Background loop for collecting metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute

                if not self._initialized:
                    break

                # Collect various metrics
                await self._collect_database_metrics()
                await self._collect_cache_metrics()
                await self._collect_storage_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")

    async def _collect_database_metrics(self):
        """Collect database-related metrics"""
        try:
            # This would typically query database connection pools
            # For now, we'll use placeholder values
            ACTIVE_CONNECTIONS.set(10)  # Placeholder

        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")

    async def _collect_cache_metrics(self):
        """Collect cache-related metrics"""
        try:
            # This would typically query cache statistics
            # For now, we'll use placeholder values
            pass

        except Exception as e:
            logger.error(f"Failed to collect cache metrics: {e}")

    async def _collect_storage_metrics(self):
        """Collect storage-related metrics"""
        try:
            # This would typically query storage backends
            # For now, we'll use placeholder values
            STORAGE_SIZE.labels(storage_type="postgresql").set(1024 * 1024 * 100)  # 100MB
            STORAGE_SIZE.labels(storage_type="elasticsearch").set(1024 * 1024 * 200)  # 200MB
            STORAGE_SIZE.labels(storage_type="redis").set(1024 * 1024 * 50)  # 50MB

        except Exception as e:
            logger.error(f"Failed to collect storage metrics: {e}")

    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record a request metric"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()

        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

    def record_cache_hit(self, cache_level: str):
        """Record a cache hit"""
        CACHE_HITS.labels(cache_level=cache_level).inc()

    def record_cache_miss(self, cache_level: str):
        """Record a cache miss"""
        CACHE_MISSES.labels(cache_level=cache_level).inc()

    def record_vector_search(self, duration: float):
        """Record vector search duration"""
        VECTOR_SEARCH_DURATION.observe(duration)

    def record_fulltext_search(self, duration: float):
        """Record full-text search duration"""
        FULLTEXT_SEARCH_DURATION.observe(duration)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics as dictionary"""
        try:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "request_count": REQUEST_COUNT._value.sum(),
                "cache_hits": CACHE_HITS._value.sum(),
                "cache_misses": CACHE_MISSES._value.sum(),
                "active_connections": ACTIVE_CONNECTIONS._value,
                "storage_size": {
                    "postgresql": STORAGE_SIZE.labels(storage_type="postgresql")._value,
                    "elasticsearch": STORAGE_SIZE.labels(storage_type="elasticsearch")._value,
                    "redis": STORAGE_SIZE.labels(storage_type="redis")._value,
                },
            }

        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}
