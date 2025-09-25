"""Metrics collection and monitoring."""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import redis.asyncio as redis
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from ..config.settings import settings


@dataclass
class MetricValue:
    """Metric value with timestamp."""

    value: float
    timestamp: float
    labels: Dict[str, str] = None


class MetricsCollector:
    """Metrics collector for the deduplication service."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize metrics collector.

        Args:
            redis_client: Redis client for storage
        """
        self.redis = redis_client
        self.registry = CollectorRegistry()

        # Prometheus metrics
        self._init_prometheus_metrics()

        # In-memory metrics storage
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)

        # Metrics history for trend analysis
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))

        # System metrics
        self.start_time = time.time()

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # Counters
        self.articles_processed_counter = Counter(
            "articles_processed_total", "Total number of articles processed", registry=self.registry
        )

        self.duplicates_found_counter = Counter(
            "duplicates_found_total", "Total number of duplicates found", registry=self.registry
        )

        self.clusters_created_counter = Counter(
            "clusters_created_total", "Total number of clusters created", registry=self.registry
        )

        self.events_created_counter = Counter(
            "events_created_total", "Total number of events created", registry=self.registry
        )

        self.processing_errors_counter = Counter(
            "processing_errors_total",
            "Total number of processing errors",
            ["error_type"],
            registry=self.registry,
        )

        self.api_requests_counter = Counter(
            "api_requests_total",
            "Total number of API requests",
            ["method", "endpoint", "status_code"],
            registry=self.registry,
        )

        # Histograms
        self.processing_time_histogram = Histogram(
            "processing_time_seconds",
            "Time taken to process articles",
            ["operation"],
            registry=self.registry,
        )

        self.similarity_score_histogram = Histogram(
            "similarity_score",
            "Distribution of similarity scores",
            ["similarity_type"],
            registry=self.registry,
        )

        self.cluster_size_histogram = Histogram("cluster_size", "Distribution of cluster sizes", registry=self.registry)

        self.api_response_time_histogram = Histogram(
            "api_response_time_seconds",
            "API response time",
            ["method", "endpoint"],
            registry=self.registry,
        )

        # Gauges
        self.active_articles_gauge = Gauge(
            "active_articles", "Number of active articles in the system", registry=self.registry
        )

        self.active_clusters_gauge = Gauge("active_clusters", "Number of active clusters", registry=self.registry)

        self.active_events_gauge = Gauge("active_events", "Number of active events", registry=self.registry)

        self.lsh_index_size_gauge = Gauge("lsh_index_size", "Size of LSH index", registry=self.registry)

        self.redis_memory_usage_gauge = Gauge(
            "redis_memory_usage_bytes", "Redis memory usage in bytes", registry=self.registry
        )

        self.database_connections_gauge = Gauge(
            "database_connections", "Number of active database connections", registry=self.registry
        )

        self.cpu_usage_gauge = Gauge("cpu_usage_percent", "CPU usage percentage", registry=self.registry)

        self.memory_usage_gauge = Gauge("memory_usage_bytes", "Memory usage in bytes", registry=self.registry)

    async def initialize(self) -> Dict[str, Any]:
        """Initialize metrics collector."""
        # Start background tasks
        asyncio.create_task(self._update_system_metrics())
        asyncio.create_task(self._persist_metrics())

    async def increment_counter(
        self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Increment a counter metric.

        Args:
            name: Metric name
            value: Value to increment by
            labels: Optional labels
        """
        # Update in-memory counter
        key = f"{name}:{labels or ''}"
        self.counters[key] += value

        # Update Prometheus counter
        if hasattr(self, f"{name}_counter"):
            counter = getattr(self, f"{name}_counter")
            if labels:
                counter.labels(**labels).inc(value)
            else:
                counter.inc(value)

        # Store in Redis
        await self.redis.hincrby("metrics:counters", key, value)

    async def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Set a gauge metric.

        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
        """
        # Update in-memory gauge
        key = f"{name}:{labels or ''}"
        self.gauges[key] = value

        # Update Prometheus gauge
        if hasattr(self, f"{name}_gauge"):
            gauge = getattr(self, f"{name}_gauge")
            if labels:
                gauge.labels(**labels).set(value)
            else:
                gauge.set(value)

        # Store in Redis
        await self.redis.hset("metrics:gauges", key, value)

    async def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric.

        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
        """
        # Update in-memory histogram
        key = f"{name}:{labels or ''}"
        self.histograms[key].append(value)

        # Update Prometheus histogram
        if hasattr(self, f"{name}_histogram"):
            histogram = getattr(self, f"{name}_histogram")
            if labels:
                histogram.labels(**labels).observe(value)
            else:
                histogram.observe(value)

        # Store in Redis
        await self.redis.lpush(f"metrics:histograms:{key}", value)
        # Keep last 1000 values
        await self.redis.ltrim(f"metrics:histograms:{key}", 0, 999)

    async def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """Get counter value.

        Args:
            name: Metric name
            labels: Optional labels

        Returns:
            Counter value
        """
        key = f"{name}:{labels or ''}"
        return self.counters.get(key, 0)

    async def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value.

        Args:
            name: Metric name
            labels: Optional labels

        Returns:
            Gauge value
        """
        key = f"{name}:{labels or ''}"
        return self.gauges.get(key, 0.0)

    async def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics.

        Args:
            name: Metric name
            labels: Optional labels

        Returns:
            Histogram statistics
        """
        key = f"{name}:{labels or ''}"
        values = self.histograms.get(key, [])

        if not values:
            return {
                "count": 0,
                "sum": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        values.sort()
        count = len(values)
        total = sum(values)

        return {
            "count": count,
            "sum": total,
            "mean": total / count,
            "min": values[0],
            "max": values[-1],
            "p50": values[int(count * 0.5)],
            "p95": values[int(count * 0.95)],
            "p99": values[int(count * 0.99)],
        }

    async def get_system_metrics(self) -> Dict[str, any]:
        """Get comprehensive system metrics.

        Returns:
            Dictionary with system metrics
        """
        # Get basic metrics
        articles_processed = await self.get_counter("articles_processed")
        duplicates_found = await self.get_counter("duplicates_found")
        clusters_created = await self.get_counter("clusters_created")
        events_created = await self.get_counter("events_created")

        # Get processing time statistics
        processing_time_stats = await self.get_histogram_stats("processing_time")

        # Get system resource usage
        memory_usage = await self._get_memory_usage()
        cpu_usage = await self._get_cpu_usage()

        # Get Redis metrics
        redis_memory = await self._get_redis_memory_usage()

        # Get database metrics
        db_connections = await self._get_database_connections()

        return {
            "articles_processed": articles_processed,
            "duplicates_detected": duplicates_found,
            "clusters_created": clusters_created,
            "events_created": events_created,
            "processing_latency_avg": processing_time_stats["mean"],
            "processing_latency_p95": processing_time_stats["p95"],
            "processing_latency_p99": processing_time_stats["p99"],
            "memory_usage_mb": memory_usage / (1024 * 1024),
            "cpu_usage_percent": cpu_usage,
            "redis_memory_usage_mb": redis_memory / (1024 * 1024),
            "database_connections": db_connections,
            "active_processing_tasks": 0,  # Would track actual tasks
            "uptime_seconds": time.time() - self.start_time,
        }

    async def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format.

        Returns:
            Prometheus metrics text
        """
        return generate_latest(self.registry)

    async def _update_system_metrics(self) -> Dict[str, Any]:
        """Update system metrics periodically."""
        while True:
            try:
                # Update memory usage
                memory_usage = await self._get_memory_usage()
                await self.set_gauge("memory_usage", memory_usage)

                # Update CPU usage
                cpu_usage = await self._get_cpu_usage()
                await self.set_gauge("cpu_usage", cpu_usage)

                # Update Redis memory usage
                redis_memory = await self._get_redis_memory_usage()
                await self.set_gauge("redis_memory_usage", redis_memory)

                # Update database connections
                db_connections = await self._get_database_connections()
                await self.set_gauge("database_connections", db_connections)

                # Wait before next update
                await asyncio.sleep(30)

            except Exception as e:
                print(f"Error updating system metrics: {e}")
                await asyncio.sleep(60)

    async def _persist_metrics(self) -> Dict[str, Any]:
        """Persist metrics to Redis periodically."""
        while True:
            try:
                # Persist counters
                for key, value in self.counters.items():
                    await self.redis.hset("metrics:counters", key, value)

                # Persist gauges
                for key, value in self.gauges.items():
                    await self.redis.hset("metrics:gauges", key, value)

                # Persist histogram summaries
                for key, values in self.histograms.items():
                    if values:
                        stats = await self.get_histogram_stats(key.replace(":", ""))
                        await self.redis.hset("metrics:histogram_stats", key, str(stats))

                # Wait before next persistence
                await asyncio.sleep(60)

            except Exception as e:
                print(f"Error persisting metrics: {e}")
                await asyncio.sleep(60)

    async def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0.0

    async def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil

            return psutil.cpu_percent()
        except ImportError:
            return 0.0

    async def _get_redis_memory_usage(self) -> float:
        """Get Redis memory usage in bytes."""
        try:
            info = await self.redis.info("memory")
            return float(info.get("used_memory", 0))
        except Exception:
            return 0.0

    async def _get_database_connections(self) -> int:
        """Get number of active database connections."""
        # This would query the actual database connection pool
        return 0


class AlertManager:
    """Alert manager for monitoring thresholds."""

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager.

        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics = metrics_collector
        self.alert_thresholds = {
            "processing_latency_p95": 5.0,  # seconds
            "memory_usage_mb": 1000,  # MB
            "cpu_usage_percent": 80.0,  # percentage
            "redis_memory_usage_mb": 500,  # MB
            "error_rate": 0.05,  # 5%
        }
        self.active_alerts = set()

    async def check_alerts(self) -> List[Dict[str, any]]:
        """Check for alert conditions.

        Returns:
            List of active alerts
        """
        alerts = []

        try:
            # Get current metrics
            metrics = await self.metrics.get_system_metrics()

            # Check processing latency
            if metrics["processing_latency_p95"] > self.alert_thresholds["processing_latency_p95"]:
                alert = {
                    "type": "high_processing_latency",
                    "severity": "warning",
                    "message": f"Processing latency P95 is {metrics['processing_latency_p95']:.2f}s",
                    "value": metrics["processing_latency_p95"],
                    "threshold": self.alert_thresholds["processing_latency_p95"],
                }
                alerts.append(alert)

            # Check memory usage
            if metrics["memory_usage_mb"] > self.alert_thresholds["memory_usage_mb"]:
                alert = {
                    "type": "high_memory_usage",
                    "severity": "critical",
                    "message": f"Memory usage is {metrics['memory_usage_mb']:.2f}MB",
                    "value": metrics["memory_usage_mb"],
                    "threshold": self.alert_thresholds["memory_usage_mb"],
                }
                alerts.append(alert)

            # Check CPU usage
            if metrics["cpu_usage_percent"] > self.alert_thresholds["cpu_usage_percent"]:
                alert = {
                    "type": "high_cpu_usage",
                    "severity": "warning",
                    "message": f"CPU usage is {metrics['cpu_usage_percent']:.2f}%",
                    "value": metrics["cpu_usage_percent"],
                    "threshold": self.alert_thresholds["cpu_usage_percent"],
                }
                alerts.append(alert)

            # Check Redis memory usage
            if metrics["redis_memory_usage_mb"] > self.alert_thresholds["redis_memory_usage_mb"]:
                alert = {
                    "type": "high_redis_memory",
                    "severity": "warning",
                    "message": f"Redis memory usage is {metrics['redis_memory_usage_mb']:.2f}MB",
                    "value": metrics["redis_memory_usage_mb"],
                    "threshold": self.alert_thresholds["redis_memory_usage_mb"],
                }
                alerts.append(alert)

            # Update active alerts
            self.active_alerts = {alert["type"] for alert in alerts}

        except Exception as e:
            print(f"Error checking alerts: {e}")

        return alerts
