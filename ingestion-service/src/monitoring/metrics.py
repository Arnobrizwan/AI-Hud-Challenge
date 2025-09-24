"""
Prometheus metrics for the ingestion service.
"""

from typing import Any, Dict

from prometheus_client import Counter, Gauge, Histogram, Info

# Application info
app_info = Info("ingestion_service_info", "Information about the ingestion service")
app_info.info({"version": "1.0.0", "name": "News Ingestion & Normalization Service"})

# Request metrics
request_count = Counter(
    "ingestion_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
)

request_duration = Histogram(
    "ingestion_request_duration_seconds", "HTTP request duration in seconds", ["method", "endpoint"]
)

request_size = Histogram("ingestion_request_size_bytes", "HTTP request size in bytes", ["method", "endpoint"])

response_size = Histogram("ingestion_response_size_bytes", "HTTP response size in bytes", ["method", "endpoint"])

# Content processing metrics
articles_processed_total = Counter(
    "ingestion_articles_processed_total",
    "Total number of articles processed",
    ["source_id", "source_type", "status"],
)

articles_processing_duration = Histogram(
    "ingestion_articles_processing_duration_seconds",
    "Time spent processing articles in seconds",
    ["source_id", "source_type"],
)

articles_word_count = Histogram(
    "ingestion_articles_word_count",
    "Word count distribution of processed articles",
    ["source_id", "content_type"],
)

articles_reading_time = Histogram(
    "ingestion_articles_reading_time_minutes",
    "Reading time distribution of processed articles",
    ["source_id", "content_type"],
)

# Source health metrics
source_health_status = Gauge(
    "ingestion_source_health_status",
    "Health status of content sources (1=healthy, 0=unhealthy)",
    ["source_id", "source_type"],
)

source_last_success = Gauge(
    "ingestion_source_last_success_timestamp",
    "Timestamp of last successful content fetch",
    ["source_id", "source_type"],
)

source_error_count = Counter(
    "ingestion_source_errors_total",
    "Total number of source errors",
    ["source_id", "source_type", "error_type"],
)

source_success_count = Counter(
    "ingestion_source_successes_total",
    "Total number of successful source operations",
    ["source_id", "source_type"],
)

# Rate limiting metrics
rate_limit_exceeded = Counter(
    "ingestion_rate_limit_exceeded_total",
    "Total number of rate limit exceeded events",
    ["source_id", "limit_type"],
)

rate_limit_delay = Histogram(
    "ingestion_rate_limit_delay_seconds",
    "Time spent waiting due to rate limiting",
    ["source_id", "limit_type"],
)

# Duplicate detection metrics
duplicates_detected = Counter(
    "ingestion_duplicates_detected_total",
    "Total number of duplicate articles detected",
    ["source_id", "detection_method"],
)

duplicate_similarity = Histogram(
    "ingestion_duplicate_similarity_score",
    "Similarity scores for detected duplicates",
    ["source_id", "detection_method"],
)

# Content quality metrics
content_quality_score = Histogram(
    "ingestion_content_quality_score", "Content quality scores", ["source_id", "content_type"]
)

content_validation_failures = Counter(
    "ingestion_content_validation_failures_total",
    "Total number of content validation failures",
    ["source_id", "validation_type"],
)

# Language detection metrics
language_detection_accuracy = Gauge(
    "ingestion_language_detection_accuracy",
    "Accuracy of language detection",
    ["source_id", "language"],
)

language_distribution = Counter(
    "ingestion_language_distribution_total",
    "Distribution of detected languages",
    ["source_id", "language"],
)

# Content type distribution
content_type_distribution = Counter(
    "ingestion_content_type_distribution_total",
    "Distribution of content types",
    ["source_id", "content_type"],
)

# Processing batch metrics
batch_processing_duration = Histogram(
    "ingestion_batch_processing_duration_seconds",
    "Time spent processing batches",
    ["source_id", "batch_size"],
)

batch_size = Histogram("ingestion_batch_size", "Size of processing batches", ["source_id"])

batch_success_rate = Gauge("ingestion_batch_success_rate", "Success rate of processing batches", ["source_id"])

# Memory and resource metrics
memory_usage_bytes = Gauge("ingestion_memory_usage_bytes", "Memory usage in bytes", ["component"])

cpu_usage_percent = Gauge("ingestion_cpu_usage_percent", "CPU usage percentage", ["component"])

# Database metrics
firestore_operations = Counter(
    "ingestion_firestore_operations_total",
    "Total number of Firestore operations",
    ["operation", "collection", "status"],
)

firestore_operation_duration = Histogram(
    "ingestion_firestore_operation_duration_seconds",
    "Duration of Firestore operations",
    ["operation", "collection"],
)

# Pub/Sub metrics
pubsub_messages_published = Counter(
    "ingestion_pubsub_messages_published_total",
    "Total number of Pub/Sub messages published",
    ["topic", "message_type"],
)

pubsub_messages_consumed = Counter(
    "ingestion_pubsub_messages_consumed_total",
    "Total number of Pub/Sub messages consumed",
    ["subscription", "message_type"],
)

pubsub_publish_duration = Histogram(
    "ingestion_pubsub_publish_duration_seconds", "Duration of Pub/Sub publish operations", ["topic"]
)

# Error metrics
error_count = Counter("ingestion_errors_total", "Total number of errors", ["error_type", "component", "severity"])

# Custom metrics for specific use cases
custom_metrics: Dict[str, Any] = {}


def create_custom_counter(name: str, description: str, labels: list = None) -> Counter:
    """Create a custom counter metric."""
    if labels is None:
        labels = []

    counter = Counter(f"ingestion_custom_{name}", description, labels)

    custom_metrics[name] = counter
    return counter


def create_custom_histogram(name: str, description: str, labels: list = None) -> Histogram:
    """Create a custom histogram metric."""
    if labels is None:
        labels = []

    histogram = Histogram(f"ingestion_custom_{name}", description, labels)

    custom_metrics[name] = histogram
    return histogram


def create_custom_gauge(name: str, description: str, labels: list = None) -> Gauge:
    """Create a custom gauge metric."""
    if labels is None:
        labels = []

    gauge = Gauge(f"ingestion_custom_{name}", description, labels)

    custom_metrics[name] = gauge
    return gauge


# Utility functions for common metric operations
def record_request_metrics(
    method: str,
    endpoint: str,
    status_code: int,
    duration: float,
    request_size_bytes: int = 0,
    response_size_bytes: int = 0,
):
    """Record HTTP request metrics."""
    request_count.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    request_duration.labels(method=method, endpoint=endpoint).observe(duration)

    if request_size_bytes > 0:
        request_size.labels(method=method, endpoint=endpoint).observe(request_size_bytes)

    if response_size_bytes > 0:
        response_size.labels(method=method, endpoint=endpoint).observe(response_size_bytes)


def record_article_processing(
    source_id: str,
    source_type: str,
    status: str,
    duration: float,
    word_count: int = 0,
    reading_time: int = 0,
    content_type: str = "article",
):
    """Record article processing metrics."""
    articles_processed_total.labels(source_id=source_id, source_type=source_type, status=status).inc()
    articles_processing_duration.labels(source_id=source_id, source_type=source_type).observe(duration)

    if word_count > 0:
        articles_word_count.labels(source_id=source_id, content_type=content_type).observe(word_count)

    if reading_time > 0:
        articles_reading_time.labels(source_id=source_id, content_type=content_type).observe(reading_time)


def record_source_health(source_id: str, source_type: str, is_healthy: bool, last_success: float = None):
    """Record source health metrics."""
    source_health_status.labels(source_id=source_id, source_type=source_type).set(1 if is_healthy else 0)

    if last_success:
        source_last_success.labels(source_id=source_id, source_type=source_type).set(last_success)


def record_source_error(source_id: str, source_type: str, error_type: str):
    """Record source error metrics."""
    source_error_count.labels(source_id=source_id, source_type=source_type, error_type=error_type).inc()


def record_source_success(source_id: str, source_type: str):
    """Record source success metrics."""
    source_success_count.labels(source_id=source_id, source_type=source_type).inc()


def record_duplicate_detection(source_id: str, detection_method: str, similarity_score: float):
    """Record duplicate detection metrics."""
    duplicates_detected.labels(source_id=source_id, detection_method=detection_method).inc()
    duplicate_similarity.labels(source_id=source_id, detection_method=detection_method).observe(similarity_score)


def record_language_detection(source_id: str, language: str, is_correct: bool = None):
    """Record language detection metrics."""
    language_distribution.labels(source_id=source_id, language=language).inc()

    if is_correct is not None:
        # Update accuracy metric (simplified)
        current_accuracy = language_detection_accuracy.labels(source_id=source_id, language=language)._value._value
        new_accuracy = (current_accuracy + (1 if is_correct else 0)) / 2
        language_detection_accuracy.labels(source_id=source_id, language=language).set(new_accuracy)


def record_content_type(source_id: str, content_type: str):
    """Record content type distribution."""
    content_type_distribution.labels(source_id=source_id, content_type=content_type).inc()


def record_batch_processing(source_id: str, batch_size: int, duration: float, success_rate: float):
    """Record batch processing metrics."""
    batch_processing_duration.labels(source_id=source_id, batch_size=str(batch_size)).observe(duration)
    batch_size.labels(source_id=source_id).observe(batch_size)
    batch_success_rate.labels(source_id=source_id).set(success_rate)


def record_firestore_operation(operation: str, collection: str, status: str, duration: float):
    """Record Firestore operation metrics."""
    firestore_operations.labels(operation=operation, collection=collection, status=status).inc()
    firestore_operation_duration.labels(operation=operation, collection=collection).observe(duration)


def record_pubsub_publish(topic: str, message_type: str, duration: float):
    """Record Pub/Sub publish metrics."""
    pubsub_messages_published.labels(topic=topic, message_type=message_type).inc()
    pubsub_publish_duration.labels(topic=topic).observe(duration)


def record_pubsub_consume(subscription: str, message_type: str):
    """Record Pub/Sub consume metrics."""
    pubsub_messages_consumed.labels(subscription=subscription, message_type=message_type).inc()


def record_error(error_type: str, component: str, severity: str = "error"):
    """Record error metrics."""
    error_count.labels(error_type=error_type, component=component, severity=severity).inc()


def record_rate_limit_exceeded(source_id: str, limit_type: str, delay: float = 0):
    """Record rate limit exceeded metrics."""
    rate_limit_exceeded.labels(source_id=source_id, limit_type=limit_type).inc()

    if delay > 0:
        rate_limit_delay.labels(source_id=source_id, limit_type=limit_type).observe(delay)


def record_content_validation_failure(source_id: str, validation_type: str):
    """Record content validation failure metrics."""
    content_validation_failures.labels(source_id=source_id, validation_type=validation_type).inc()


def record_content_quality_score(source_id: str, content_type: str, score: float):
    """Record content quality score metrics."""
    content_quality_score.labels(source_id=source_id, content_type=content_type).observe(score)


def record_memory_usage(component: str, usage_bytes: int):
    """Record memory usage metrics."""
    memory_usage_bytes.labels(component=component).set(usage_bytes)


def record_cpu_usage(component: str, usage_percent: float):
    """Record CPU usage metrics."""
    cpu_usage_percent.labels(component=component).set(usage_percent)
