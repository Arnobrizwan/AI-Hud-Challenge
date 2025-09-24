"""Monitoring and performance metrics collection."""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    timestamp: datetime
    response_time_ms: float
    feature_time_ms: float
    ranking_time_ms: float
    cache_hit_rate: float
    article_count: int
    user_id: Optional[str] = None
    algorithm_variant: Optional[str] = None
    error: bool = False


@dataclass
class SystemMetrics:
    """System metrics data structure."""

    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    redis_connections: int
    active_requests: int
    queue_size: int


class MetricsCollector:
    """Base metrics collector."""

    def __init__(self):
        self.metrics: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)

    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        self.metrics.append(
            {"timestamp": datetime.utcnow(), "name": name, "value": value, "labels": labels or {}}
        )

    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter."""
        self.counters[name] += value

    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        self.gauges[name] = value

    def record_histogram(self, name: str, value: float):
        """Record a histogram value."""
        self.histograms[name].append(value)
        # Keep only last 1000 values
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-1000:]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                name: {
                    "count": len(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "avg": sum(values) / len(values) if values else 0,
                    "p50": self._percentile(values, 50) if values else 0,
                    "p95": self._percentile(values, 95) if values else 0,
                    "p99": self._percentile(values, 99) if values else 0,
                }
                for name, values in self.histograms.items()
            },
        }

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class RankingMetricsCollector(MetricsCollector):
    """Ranking-specific metrics collector."""

    def __init__(self):
        super().__init__()
        self.performance_metrics: deque = deque(maxlen=1000)
        self.error_count = 0
        self.total_requests = 0

        # Prometheus metrics
        self._init_prometheus_metrics()

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        self.request_counter = Counter(
            "ranking_requests_total", "Total ranking requests", ["algorithm_variant", "status"]
        )

        self.response_time_histogram = Histogram(
            "ranking_response_time_seconds",
            "Ranking response time",
            ["algorithm_variant"],
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        )

        self.feature_time_histogram = Histogram(
            "ranking_feature_time_seconds",
            "Feature computation time",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
        )

        self.ranking_time_histogram = Histogram(
            "ranking_algorithm_time_seconds",
            "Ranking algorithm time",
            ["algorithm_variant"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
        )

        self.cache_hit_rate_gauge = Gauge("ranking_cache_hit_rate", "Cache hit rate")

        self.article_count_histogram = Histogram(
            "ranking_article_count",
            "Number of articles ranked",
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )

        self.error_counter = Counter("ranking_errors_total", "Total ranking errors", ["error_type"])

    async def record_ranking(
        self,
        response_time_ms: float,
        feature_time_ms: float,
        ranking_time_ms: float,
        cache_hit_rate: float,
        article_count: int,
        user_id: Optional[str] = None,
        algorithm_variant: Optional[str] = None,
    ):
        """Record ranking performance metrics."""
        try:
            # Record performance metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.utcnow(),
                response_time_ms=response_time_ms,
                feature_time_ms=feature_time_ms,
                ranking_time_ms=ranking_time_ms,
                cache_hit_rate=cache_hit_rate,
                article_count=article_count,
                user_id=user_id,
                algorithm_variant=algorithm_variant,
            )

            self.performance_metrics.append(metrics)

            # Update counters and histograms
            self.total_requests += 1
            self.increment_counter("total_requests")
            self.record_histogram("response_time_ms", response_time_ms)
            self.record_histogram("feature_time_ms", feature_time_ms)
            self.record_histogram("ranking_time_ms", ranking_time_ms)
            self.record_histogram("article_count", article_count)
            self.set_gauge("cache_hit_rate", cache_hit_rate)

            # Update Prometheus metrics
            variant = algorithm_variant or "unknown"
            self.request_counter.labels(algorithm_variant=variant, status="success").inc()
            self.response_time_histogram.labels(algorithm_variant=variant).observe(
                response_time_ms / 1000
            )
            self.feature_time_histogram.observe(feature_time_ms / 1000)
            self.ranking_time_histogram.labels(algorithm_variant=variant).observe(
                ranking_time_ms / 1000
            )
            self.cache_hit_rate_gauge.set(cache_hit_rate)
            self.article_count_histogram.observe(article_count)

        except Exception as e:
            logger.error("Failed to record ranking metrics", error=str(e))

    async def record_error(self, error_type: str = "unknown"):
        """Record ranking error."""
        try:
            self.error_count += 1
            self.increment_counter("errors")
            self.increment_counter(f"error_{error_type}")

            # Update Prometheus metrics
            self.error_counter.labels(error_type=error_type).inc()

        except Exception as e:
            logger.error("Failed to record error metrics", error=str(e))

    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for time window."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)

        # Filter metrics by time window
        recent_metrics = [m for m in self.performance_metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {
                "total_requests": 0,
                "avg_response_time_ms": 0,
                "p95_response_time_ms": 0,
                "p99_response_time_ms": 0,
                "avg_feature_time_ms": 0,
                "avg_ranking_time_ms": 0,
                "avg_cache_hit_rate": 0,
                "error_rate": 0,
                "avg_article_count": 0,
            }

        # Calculate statistics
        response_times = [m.response_time_ms for m in recent_metrics]
        feature_times = [m.feature_time_ms for m in recent_metrics]
        ranking_times = [m.ranking_time_ms for m in recent_metrics]
        cache_hit_rates = [m.cache_hit_rate for m in recent_metrics]
        article_counts = [m.article_count for m in recent_metrics]

        return {
            "total_requests": len(recent_metrics),
            "avg_response_time_ms": sum(response_times) / len(response_times),
            "p95_response_time_ms": self._percentile(response_times, 95),
            "p99_response_time_ms": self._percentile(response_times, 99),
            "avg_feature_time_ms": sum(feature_times) / len(feature_times),
            "avg_ranking_time_ms": sum(ranking_times) / len(ranking_times),
            "avg_cache_hit_rate": sum(cache_hit_rates) / len(cache_hit_rates),
            "error_rate": self.error_count / max(self.total_requests, 1),
            "avg_article_count": sum(article_counts) / len(article_counts),
        }

    def get_algorithm_comparison(self) -> Dict[str, Any]:
        """Compare performance across algorithms."""
        algorithm_metrics = defaultdict(list)

        for metrics in self.performance_metrics:
            if metrics.algorithm_variant:
                algorithm_metrics[metrics.algorithm_variant].append(metrics)

        comparison = {}
        for algorithm, metrics_list in algorithm_metrics.items():
            if not metrics_list:
                continue

            response_times = [m.response_time_ms for m in metrics_list]
            cache_hit_rates = [m.cache_hit_rate for m in metrics_list]

            comparison[algorithm] = {
                "request_count": len(metrics_list),
                "avg_response_time_ms": sum(response_times) / len(response_times),
                "p95_response_time_ms": self._percentile(response_times, 95),
                "avg_cache_hit_rate": sum(cache_hit_rates) / len(cache_hit_rates),
            }

        return comparison


class SystemMetricsCollector(MetricsCollector):
    """System metrics collector."""

    def __init__(self):
        super().__init__()
        self.system_metrics: deque = deque(maxlen=1000)

        # Prometheus metrics
        self.cpu_usage_gauge = Gauge("system_cpu_usage_percent", "CPU usage percentage")
        self.memory_usage_gauge = Gauge("system_memory_usage_bytes", "Memory usage in bytes")
        self.redis_connections_gauge = Gauge("redis_connections", "Number of Redis connections")
        self.active_requests_gauge = Gauge("active_requests", "Number of active requests")
        self.queue_size_gauge = Gauge("queue_size", "Queue size")

    async def record_system_metrics(
        self,
        cpu_usage: float,
        memory_usage: float,
        redis_connections: int,
        active_requests: int,
        queue_size: int,
    ):
        """Record system metrics."""
        try:
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                redis_connections=redis_connections,
                active_requests=active_requests,
                queue_size=queue_size,
            )

            self.system_metrics.append(metrics)

            # Update gauges
            self.set_gauge("cpu_usage", cpu_usage)
            self.set_gauge("memory_usage", memory_usage)
            self.set_gauge("redis_connections", redis_connections)
            self.set_gauge("active_requests", active_requests)
            self.set_gauge("queue_size", queue_size)

            # Update Prometheus metrics
            self.cpu_usage_gauge.set(cpu_usage)
            self.memory_usage_gauge.set(memory_usage)
            self.redis_connections_gauge.set(redis_connections)
            self.active_requests_gauge.set(active_requests)
            self.queue_size_gauge.set(queue_size)

        except Exception as e:
            logger.error("Failed to record system metrics", error=str(e))

    def get_system_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get system summary for time window."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)

        # Filter metrics by time window
        recent_metrics = [m for m in self.system_metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {
                "avg_cpu_usage": 0,
                "max_cpu_usage": 0,
                "avg_memory_usage": 0,
                "max_memory_usage": 0,
                "avg_redis_connections": 0,
                "avg_active_requests": 0,
                "avg_queue_size": 0,
            }

        # Calculate statistics
        cpu_usage = [m.cpu_usage for m in recent_metrics]
        memory_usage = [m.memory_usage for m in recent_metrics]
        redis_connections = [m.redis_connections for m in recent_metrics]
        active_requests = [m.active_requests for m in recent_metrics]
        queue_size = [m.queue_size for m in recent_metrics]

        return {
            "avg_cpu_usage": sum(cpu_usage) / len(cpu_usage),
            "max_cpu_usage": max(cpu_usage),
            "avg_memory_usage": sum(memory_usage) / len(memory_usage),
            "max_memory_usage": max(memory_usage),
            "avg_redis_connections": sum(redis_connections) / len(redis_connections),
            "avg_active_requests": sum(active_requests) / len(active_requests),
            "avg_queue_size": sum(queue_size) / len(queue_size),
        }


class HealthChecker:
    """Health check and alerting system."""

    def __init__(
        self, ranking_collector: RankingMetricsCollector, system_collector: SystemMetricsCollector
    ):
        self.ranking_collector = ranking_collector
        self.system_collector = system_collector
        self.alert_thresholds = {
            "response_time_ms": 100,  # 100ms
            "error_rate": 0.05,  # 5%
            "cpu_usage": 80,  # 80%
            "memory_usage": 80,  # 80%
            "cache_hit_rate": 0.5,  # 50%
        }
        self.alerts: List[Dict[str, Any]] = []

    async def check_health(self) -> Dict[str, Any]:
        """Perform health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
        }

        # Check ranking performance
        ranking_summary = self.ranking_collector.get_performance_summary(5)  # Last 5 minutes

        if ranking_summary["avg_response_time_ms"] > self.alert_thresholds["response_time_ms"]:
            health_status["checks"]["response_time"] = {
                "status": "unhealthy",
                "value": ranking_summary["avg_response_time_ms"],
                "threshold": self.alert_thresholds["response_time_ms"],
            }
            health_status["status"] = "unhealthy"
        else:
            health_status["checks"]["response_time"] = {
                "status": "healthy",
                "value": ranking_summary["avg_response_time_ms"],
                "threshold": self.alert_thresholds["response_time_ms"],
            }

        if ranking_summary["error_rate"] > self.alert_thresholds["error_rate"]:
            health_status["checks"]["error_rate"] = {
                "status": "unhealthy",
                "value": ranking_summary["error_rate"],
                "threshold": self.alert_thresholds["error_rate"],
            }
            health_status["status"] = "unhealthy"
        else:
            health_status["checks"]["error_rate"] = {
                "status": "healthy",
                "value": ranking_summary["error_rate"],
                "threshold": self.alert_thresholds["error_rate"],
            }

        # Check system metrics
        system_summary = self.system_collector.get_system_summary(5)  # Last 5 minutes

        if system_summary["avg_cpu_usage"] > self.alert_thresholds["cpu_usage"]:
            health_status["checks"]["cpu_usage"] = {
                "status": "unhealthy",
                "value": system_summary["avg_cpu_usage"],
                "threshold": self.alert_thresholds["cpu_usage"],
            }
            health_status["status"] = "unhealthy"
        else:
            health_status["checks"]["cpu_usage"] = {
                "status": "healthy",
                "value": system_summary["avg_cpu_usage"],
                "threshold": self.alert_thresholds["cpu_usage"],
            }

        if system_summary["avg_memory_usage"] > self.alert_thresholds["memory_usage"]:
            health_status["checks"]["memory_usage"] = {
                "status": "unhealthy",
                "value": system_summary["avg_memory_usage"],
                "threshold": self.alert_thresholds["memory_usage"],
            }
            health_status["status"] = "unhealthy"
        else:
            health_status["checks"]["memory_usage"] = {
                "status": "healthy",
                "value": system_summary["avg_memory_usage"],
                "threshold": self.alert_thresholds["memory_usage"],
            }

        return health_status

    async def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts."""
        return self.alerts


class MetricsExporter:
    """Export metrics to external systems."""

    def __init__(
        self, ranking_collector: RankingMetricsCollector, system_collector: SystemMetricsCollector
    ):
        self.ranking_collector = ranking_collector
        self.system_collector = system_collector
        self.export_interval = 60  # seconds
        self.export_enabled = True

        # Start export task
        asyncio.create_task(self._export_loop())

    async def _export_loop(self):
        """Export metrics periodically."""
        while self.export_enabled:
            try:
                await self.export_metrics()
                await asyncio.sleep(self.export_interval)
            except Exception as e:
                logger.error("Metrics export failed", error=str(e))
                await asyncio.sleep(self.export_interval)

    async def export_metrics(self):
        """Export metrics to external systems."""
        try:
            # Get metrics summaries
            ranking_summary = self.ranking_collector.get_performance_summary()
            system_summary = self.system_collector.get_system_summary()

            # Export to external systems (e.g., data warehouse, monitoring service)
            export_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "ranking_metrics": ranking_summary,
                "system_metrics": system_summary,
            }

            # In production, this would send to external systems
            logger.info("Metrics exported", data=export_data)

        except Exception as e:
            logger.error("Failed to export metrics", error=str(e))

    def stop_export(self):
        """Stop metrics export."""
        self.export_enabled = False
