"""
Comprehensive metrics collection with Prometheus
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil
import requests
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, HistogramMetricFamily

logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Metrics collection configuration"""

    prometheus_port: int = 9090
    collection_interval: int = 30
    custom_metrics_enabled: bool = True
    business_metrics_enabled: bool = True
    system_metrics_enabled: bool = True


class CustomMetricsRegistry:
    """Registry for custom business metrics"""

    def __init__(self):
        self.metrics = {}

    def register_metric(self, name: str, metric_type: str, **kwargs):
        """Register a custom metric"""
        if metric_type == "counter":
            self.metrics[name] = Counter(name, **kwargs)
        elif metric_type == "histogram":
            self.metrics[name] = Histogram(name, **kwargs)
        elif metric_type == "gauge":
            self.metrics[name] = Gauge(name, **kwargs)
        elif metric_type == "summary":
            self.metrics[name] = Summary(name, **kwargs)

    def get_metric(self, name: str):
        """Get a registered metric"""
        return self.metrics.get(name)

    async def collect_all(self) -> Dict[str, float]:
        """Collect all custom metrics"""
        results = {}
        for name, metric in self.metrics.items():
            try:
                # This is a simplified collection - in practice you'd need to
                # properly extract values from Prometheus metrics
                results[name] = 0.0  # Placeholder
            except Exception as e:
                logger.error(f"Failed to collect metric {name}: {str(e)}")
        return results


class BusinessMetricsCollector:
    """Collect business-specific metrics"""

    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        self._register_business_metrics()

    def _register_business_metrics(self):
        """Register business-specific metrics"""

        # Content processing metrics
        self.articles_processed = Counter(
            "articles_processed_total",
            "Total articles processed",
            ["source", "category", "language", "status"],
            registry=self.registry,
        )

        self.articles_processing_duration = Histogram(
            "articles_processing_duration_seconds",
            "Time taken to process articles",
            ["pipeline_stage", "article_source"],
            registry=self.registry,
        )

        # User engagement metrics
        self.user_engagement = Histogram(
            "user_engagement_duration_seconds",
            "User engagement duration",
            ["user_segment", "content_type", "platform"],
            registry=self.registry,
        )

        self.user_actions = Counter(
            "user_actions_total",
            "Total user actions",
            ["action_type", "user_segment", "content_type"],
            registry=self.registry,
        )

        # Content quality metrics
        self.content_quality_score = Gauge(
            "content_quality_score",
            "Content quality score",
            ["content_type", "source"],
            registry=self.registry,
        )

        self.duplicate_detection_accuracy = Gauge(
            "duplicate_detection_accuracy",
            "Duplicate detection accuracy",
            ["detection_method", "model_version"],
            registry=self.registry,
        )

        # Personalization metrics
        self.personalization_effectiveness = Gauge(
            "personalization_effectiveness",
            "Personalization effectiveness score",
            ["user_segment", "algorithm_type"],
            registry=self.registry,
        )

        self.recommendation_click_through_rate = Gauge(
            "recommendation_ctr",
            "Recommendation click-through rate",
            ["algorithm_type", "user_segment"],
            registry=self.registry,
        )


class PerformanceMetricsCollector:
    """Collect performance-related metrics"""

    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        self._register_performance_metrics()

    def _register_performance_metrics(self):
        """Register performance metrics"""

        # ML model performance
        self.model_inference_duration = Histogram(
            "ml_model_inference_duration_seconds",
            "ML model inference duration",
            ["model_name", "model_version", "model_type"],
            registry=self.registry,
        )

        self.model_accuracy = Gauge(
            "ml_model_accuracy",
            "ML model accuracy score",
            ["model_name", "model_version", "metric_type"],
            registry=self.registry,
        )

        self.model_throughput = Gauge(
            "ml_model_throughput",
            "ML model throughput (requests per second)",
            ["model_name", "model_version"],
            registry=self.registry,
        )

        # Database performance
        self.db_query_duration = Histogram(
            "db_query_duration_seconds",
            "Database query duration",
            ["query_type", "table", "database", "operation"],
            registry=self.registry,
        )

        self.db_connection_pool = Gauge(
            "db_connection_pool_size",
            "Database connection pool size",
            ["database", "status"],
            registry=self.registry,
        )

        # Cache performance
        self.cache_hit_ratio = Gauge(
            "cache_hit_ratio",
            "Cache hit ratio",
            ["cache_type", "cache_key_pattern"],
            registry=self.registry,
        )

        self.cache_operation_duration = Histogram(
            "cache_operation_duration_seconds",
            "Cache operation duration",
            ["operation_type", "cache_type"],
            registry=self.registry,
        )


class MetricsCollector:
    """Main metrics collector for monitoring"""

    def __init__(self):
        self.prometheus_registry = CollectorRegistry()
        self.custom_metrics = CustomMetricsRegistry()
        self.business_metrics = BusinessMetricsCollector(self.prometheus_registry)
        self.performance_metrics = PerformanceMetricsCollector(self.prometheus_registry)

        self._register_standard_metrics()
        self._register_sli_metrics()

        self.collection_interval = 30
        self.is_running = False

    def _register_standard_metrics(self):
        """Register standard application metrics"""

        # HTTP request metrics
        self.request_duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint", "status_code", "service"],
            registry=self.prometheus_registry,
        )

        self.request_count = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code", "service"],
            registry=self.prometheus_registry,
        )

        # System resource metrics
        self.cpu_usage = Gauge(
            "system_cpu_usage_percent",
            "System CPU usage percentage",
            ["cpu_core"],
            registry=self.prometheus_registry,
        )

        self.memory_usage = Gauge(
            "system_memory_usage_bytes",
            "System memory usage in bytes",
            ["memory_type"],
            registry=self.prometheus_registry,
        )

        self.disk_usage = Gauge(
            "system_disk_usage_bytes",
            "System disk usage in bytes",
            ["device", "mountpoint"],
            registry=self.prometheus_registry,
        )

        # Service health metrics
        self.service_health = Gauge(
            "service_health_score",
            "Service health score (0-1)",
            ["service_name", "health_type"],
            registry=self.prometheus_registry,
        )

        # Error metrics
        self.error_count = Counter(
            "errors_total",
            "Total errors",
            ["error_type", "service", "severity"],
            registry=self.prometheus_registry,
        )

    def _register_sli_metrics(self):
        """Register SLI (Service Level Indicator) metrics"""

        # Availability SLI
        self.availability_sli = Gauge(
            "sli_availability",
            "Service availability SLI",
            ["service", "slo_target"],
            registry=self.prometheus_registry,
        )

        # Latency SLI
        self.latency_sli = Gauge(
            "sli_latency_p99",
            "99th percentile latency SLI",
            ["service", "endpoint", "slo_target"],
            registry=self.prometheus_registry,
        )

        # Error rate SLI
        self.error_rate_sli = Gauge(
            "sli_error_rate",
            "Error rate SLI",
            ["service", "slo_target"],
            registry=self.prometheus_registry,
        )

        # Throughput SLI
        self.throughput_sli = Gauge(
            "sli_throughput",
            "Throughput SLI (requests per second)",
            ["service", "slo_target"],
            registry=self.prometheus_registry,
        )

    async def initialize(self, config: MetricsConfig) -> None:
        """Initialize metrics collection"""
        self.collection_interval = config.collection_interval

        # Register custom metrics if enabled
        if config.custom_metrics_enabled:
            await self._register_custom_metrics()

        logger.info("Metrics collector initialized")

    async def _register_custom_metrics(self):
        """Register custom metrics"""

        # Custom business metrics
        self.custom_metrics.register_metric(
            "custom_user_satisfaction",
            "gauge",
            description="User satisfaction score",
            labelnames=["user_segment", "time_period"],
        )

        self.custom_metrics.register_metric(
            "custom_content_freshness",
            "gauge",
            description="Content freshness score",
            labelnames=["content_type", "source"],
        )

        self.custom_metrics.register_metric(
            "custom_system_efficiency",
            "gauge",
            description="System efficiency score",
            labelnames=["component", "metric_type"],
        )

    async def start_metrics_collection(self):
        """Start background metrics collection"""
        if self.is_running:
            return

        self.is_running = True
        asyncio.create_task(self._collection_loop())
        logger.info("Started metrics collection")

    async def _collection_loop(self):
        """Background metrics collection loop"""
        while self.is_running:
            try:
                await self.collect_real_time_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {str(e)}")
                await asyncio.sleep(self.collection_interval)

    async def collect_real_time_metrics(self) -> Dict[str, float]:
        """Collect real-time system metrics"""
        metrics = {}

        try:
            # System resource metrics
            metrics.update(await self._collect_system_resources())

            # Application performance metrics
            metrics.update(await self._collect_application_performance())

            # Business metrics
            metrics.update(await self._collect_business_metrics())

            # Custom metrics
            metrics.update(await self.custom_metrics.collect_all())

            # Update Prometheus metrics
            await self._update_prometheus_metrics(metrics)

        except Exception as e:
            logger.error(f"Failed to collect metrics: {str(e)}")

        return metrics

    async def _collect_system_resources(self) -> Dict[str, float]:
        """Collect system resource metrics"""
        metrics = {}

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            for i, cpu in enumerate(cpu_percent):
                metrics[f"cpu_usage_core_{i}"] = cpu

            # Memory usage
            memory = psutil.virtual_memory()
            metrics["memory_usage_total"] = memory.total
            metrics["memory_usage_available"] = memory.available
            metrics["memory_usage_used"] = memory.used
            metrics["memory_usage_percent"] = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            metrics["disk_usage_total"] = disk.total
            metrics["disk_usage_used"] = disk.used
            metrics["disk_usage_free"] = disk.free
            metrics["disk_usage_percent"] = (disk.used / disk.total) * 100

        except Exception as e:
            logger.error(f"Failed to collect system resources: {str(e)}")

        return metrics

    async def _collect_application_performance(self) -> Dict[str, float]:
        """Collect application performance metrics"""
        metrics = {}

        try:
            # This would typically involve querying application-specific metrics
            # For now, we'll use placeholder values

            # Database connection pool status
            metrics["db_connections_active"] = 10  # Placeholder
            metrics["db_connections_idle"] = 5  # Placeholder

            # Cache performance
            metrics["cache_hit_ratio"] = 0.85  # Placeholder
            metrics["cache_memory_usage"] = 1024  # Placeholder

            # Queue lengths
            metrics["queue_length_processing"] = 0  # Placeholder
            metrics["queue_length_pending"] = 5  # Placeholder

        except Exception as e:
            logger.error(f"Failed to collect application performance: {str(e)}")

        return metrics

    async def _collect_business_metrics(self) -> Dict[str, float]:
        """Collect business-specific metrics"""
        metrics = {}

        try:
            # This would typically involve querying business data
            # For now, we'll use placeholder values

            # Content metrics
            metrics["articles_processed_today"] = 1000  # Placeholder
            metrics["users_active_today"] = 500  # Placeholder
            metrics["content_quality_score"] = 0.92  # Placeholder

            # Engagement metrics
            metrics["avg_engagement_time"] = 120.5  # Placeholder
            metrics["click_through_rate"] = 0.15  # Placeholder

        except Exception as e:
            logger.error(f"Failed to collect business metrics: {str(e)}")

        return metrics

    async def _update_prometheus_metrics(self, metrics: Dict[str, float]):
        """Update Prometheus metrics with collected values"""
        try:
            # Update system metrics
            if "cpu_usage_percent" in metrics:
                self.cpu_usage.labels(cpu_core="total").set(metrics["cpu_usage_percent"])

            if "memory_usage_percent" in metrics:
                self.memory_usage.labels(memory_type="percent").set(metrics["memory_usage_percent"])

            if "disk_usage_percent" in metrics:
                self.disk_usage.labels(device="root", mountpoint="/").set(
                    metrics["disk_usage_percent"]
                )

            # Update business metrics
            if "content_quality_score" in metrics:
                self.business_metrics.content_quality_score.labels(
                    content_type="article", source="all"
                ).set(metrics["content_quality_score"])

            if "cache_hit_ratio" in metrics:
                self.performance_metrics.cache_hit_ratio.labels(
                    cache_type="redis", cache_key_pattern="*"
                ).set(metrics["cache_hit_ratio"])

        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {str(e)}")

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics"""
        try:
            metrics = await self.collect_real_time_metrics()

            return {
                "system_metrics": {
                    "cpu_usage": metrics.get("cpu_usage_percent", 0),
                    "memory_usage": metrics.get("memory_usage_percent", 0),
                    "disk_usage": metrics.get("disk_usage_percent", 0),
                },
                "application_metrics": {
                    "cache_hit_ratio": metrics.get("cache_hit_ratio", 0),
                    "db_connections_active": metrics.get("db_connections_active", 0),
                },
                "business_metrics": {
                    "articles_processed_today": metrics.get("articles_processed_today", 0),
                    "content_quality_score": metrics.get("content_quality_score", 0),
                    "click_through_rate": metrics.get("click_through_rate", 0),
                },
                "collection_timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get metrics summary: {str(e)}")
            return {"error": str(e)}

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.prometheus_registry).decode("utf-8")

    def get_metrics_content_type(self) -> str:
        """Get content type for Prometheus metrics"""
        return CONTENT_TYPE_LATEST

    async def cleanup(self):
        """Cleanup metrics collector"""
        self.is_running = False
        logger.info("Metrics collector cleaned up")
