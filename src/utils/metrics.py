"""
Prometheus metrics collection and instrumentation.
"""

import asyncio
import time
from functools import wraps
from typing import Any, Dict, List, Optional

import psutil
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator, metrics

from src.config.settings import settings

# Application metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total number of HTTP requests", [
        "method", "endpoint", "status_code"])

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds", "HTTP request duration in seconds", [
        "method", "endpoint"])

ACTIVE_CONNECTIONS = Gauge(
    "http_active_connections",
    "Number of active HTTP connections")

# Authentication metrics
AUTH_ATTEMPTS = Counter(
    "auth_attempts_total", "Total authentication attempts", [
        "provider", "status"])

AUTH_DURATION = Histogram(
    "auth_duration_seconds", "Authentication duration in seconds", ["provider"]
)

# Rate limiting metrics
RATE_LIMIT_HITS = Counter(
    "rate_limit_hits_total", "Total rate limit hits", ["limit_type", "user_id"]
)

RATE_LIMIT_BLOCKS = Counter(
    "rate_limit_blocks_total",
    "Total rate limit blocks",
    ["limit_type"])

# System metrics
MEMORY_USAGE = Gauge("memory_usage_bytes", "Memory usage in bytes", ["type"])

CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage percentage")

# Error metrics
ERROR_COUNT = Counter(
    "errors_total", "Total number of errors", [
        "error_type", "endpoint"])

# Business metrics
CONTENT_PROCESSED = Counter(
    "content_processed_total", "Total content items processed", [
        "content_type", "status"])

PROCESSING_DURATION = Histogram(
    "processing_duration_seconds",
    "Content processing duration in seconds",
    ["content_type"])

# Circuit breaker metrics
CIRCUIT_BREAKER_STATE = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    ["service"])

CIRCUIT_BREAKER_FAILURES = Counter(
    "circuit_breaker_failures_total", "Circuit breaker failures", ["service"]
)

# Redis metrics
REDIS_OPERATIONS = Counter(
    "redis_operations_total", "Total Redis operations", ["operation", "status"]
)

REDIS_DURATION = Histogram(
    "redis_operation_duration_seconds",
    "Redis operation duration in seconds",
    ["operation"])

# Application info
APP_INFO = Info("app_info", "Application information")

# Set application info
APP_INFO.info(
    {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }
)


class MetricsCollector:
    """Centralized metrics collection."""

    def __init__(self):
        self.start_time = time.time()

    def record_request(
            self,
            method: str,
            endpoint: str,
            status_code: int,
            duration: float):
        """Record HTTP request metrics."""
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code).inc()

        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint).observe(duration)

    def record_auth_attempt(
            self,
            provider: str,
            success: bool,
            duration: float):
        """Record authentication attempt metrics."""
        status = "success" if success else "failure"
        AUTH_ATTEMPTS.labels(provider=provider, status=status).inc()
        AUTH_DURATION.labels(provider=provider).observe(duration)

    def record_rate_limit_hit(self, limit_type: str, user_id: str = None):
        """Record rate limit hit."""
        RATE_LIMIT_HITS.labels(
            limit_type=limit_type,
            user_id=user_id or "anonymous").inc()

    def record_rate_limit_block(self, limit_type: str):
        """Record rate limit block."""
        RATE_LIMIT_BLOCKS.labels(limit_type=limit_type).inc()

    def record_error(self, error_type: str, endpoint: str):
        """Record error occurrence."""
        ERROR_COUNT.labels(error_type=error_type, endpoint=endpoint).inc()

    def record_content_processing(
            self,
            content_type: str,
            status: str,
            duration: float):
        """Record content processing metrics."""
        CONTENT_PROCESSED.labels(
            content_type=content_type,
            status=status).inc()
        PROCESSING_DURATION.labels(content_type=content_type).observe(duration)

    def update_system_metrics(self):
        """Update system resource metrics."""
        # Memory metrics
        memory = psutil.virtual_memory()
        MEMORY_USAGE.labels(type="used").set(memory.used)
        MEMORY_USAGE.labels(type="available").set(memory.available)
        MEMORY_USAGE.labels(type="percent").set(memory.percent)

        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        CPU_USAGE.set(cpu_percent)

    def record_redis_operation(
            self,
            operation: str,
            success: bool,
            duration: float):
        """Record Redis operation metrics."""
        status = "success" if success else "failure"
        REDIS_OPERATIONS.labels(operation=operation, status=status).inc()
        REDIS_DURATION.labels(operation=operation).observe(duration)

    def set_circuit_breaker_state(self, service: str, state: int):
        """Set circuit breaker state (0=closed, 1=open, 2=half-open)."""
        CIRCUIT_BREAKER_STATE.labels(service=service).set(state)

    def record_circuit_breaker_failure(self, service: str):
        """Record circuit breaker failure."""
        CIRCUIT_BREAKER_FAILURES.labels(service=service).inc()

    def get_uptime(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self.start_time


# Global metrics collector instance
metrics_collector = MetricsCollector()


def time_function(metric_name: str = None, labels: Dict[str, str] = None):
    """Decorator to time function execution."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Dict[str, Any]:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if metric_name:
                    # Record custom metric if provided
                    pass  # Would need to create the metric dynamically

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if metric_name:
                    # Record custom metric if provided
                    pass  # Would need to create the metric dynamically

        return async_wrapper if asyncio.iscoroutinefunction(
            func) else sync_wrapper

    return decorator


def setup_metrics_instrumentation(app):
    """Set up FastAPI metrics instrumentation."""
    if not settings.ENABLE_METRICS:
        return

    # Create instrumentator
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="http_requests_inprogress",
        inprogress_labels=True,
    )

    # Add custom metrics
    instrumentator.add(
        metrics.request_size(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace="http",
            metric_subsystem="requests",
        )
    ).add(
        metrics.response_size(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace="http",
            metric_subsystem="responses",
        )
    )

    # Instrument the app
    instrumentator.instrument(app).expose(app, endpoint=settings.METRICS_PATH)

    return instrumentator


async def collect_periodic_metrics() -> Dict[str, Any]:
    """Collect periodic system metrics."""
    while True:
        try:
            metrics_collector.update_system_metrics()
        except Exception:
            pass  # Ignore errors in metrics collection

        await asyncio.sleep(30)  # Collect every 30 seconds


def get_metrics_data() -> bytes:
    """Get current metrics in Prometheus format."""
    return generate_latest().encode("utf-8")


def get_metrics_content_type() -> str:
    """Get metrics content type."""
    return CONTENT_TYPE_LATEST
