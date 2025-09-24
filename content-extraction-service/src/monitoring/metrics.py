"""
Prometheus metrics for content extraction service.
"""

import time
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from loguru import logger


class MetricsCollector:
    """Prometheus metrics collector for content extraction service."""

    def __init__(self):
        """Initialize metrics collector."""
        self._setup_metrics()

    def _setup_metrics(self):
        """Setup Prometheus metrics."""
        # Request metrics
        self.requests_total = Counter(
            'content_extraction_requests_total',
            'Total number of content extraction requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.request_duration = Histogram(
            'content_extraction_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        )
        
        # Content extraction metrics
        self.extractions_total = Counter(
            'content_extractions_total',
            'Total number of content extractions',
            ['content_type', 'extraction_method', 'status']
        )
        
        self.extraction_duration = Histogram(
            'content_extraction_duration_seconds',
            'Content extraction duration in seconds',
            ['content_type', 'extraction_method'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
        )
        
        self.content_quality_score = Histogram(
            'content_quality_score',
            'Content quality score distribution',
            ['content_type'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Cache metrics
        self.cache_hits_total = Counter(
            'content_cache_hits_total',
            'Total number of cache hits',
            ['cache_type']
        )
        
        self.cache_misses_total = Counter(
            'content_cache_misses_total',
            'Total number of cache misses',
            ['cache_type']
        )
        
        self.cache_size_bytes = Gauge(
            'content_cache_size_bytes',
            'Current cache size in bytes',
            ['cache_type']
        )
        
        # Image processing metrics
        self.images_processed_total = Counter(
            'images_processed_total',
            'Total number of images processed',
            ['status', 'format']
        )
        
        self.image_processing_duration = Histogram(
            'image_processing_duration_seconds',
            'Image processing duration in seconds',
            ['format'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        # Error metrics
        self.errors_total = Counter(
            'content_extraction_errors_total',
            'Total number of errors',
            ['error_type', 'component']
        )
        
        # System metrics
        self.active_extractions = Gauge(
            'content_extraction_active_extractions',
            'Number of currently active extractions'
        )
        
        self.queue_size = Gauge(
            'content_extraction_queue_size',
            'Current queue size'
        )
        
        self.memory_usage_bytes = Gauge(
            'content_extraction_memory_usage_bytes',
            'Current memory usage in bytes'
        )
        
        # Service info
        self.service_info = Info(
            'content_extraction_service_info',
            'Service information'
        )
        
        # Set service info
        self.service_info.info({
            'version': '1.0.0',
            'environment': 'production',
            'service': 'content-extraction'
        })

    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self.requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

    def record_extraction(
        self,
        content_type: str,
        extraction_method: str,
        status: str,
        duration: float,
        quality_score: Optional[float] = None
    ):
        """Record content extraction metrics."""
        self.extractions_total.labels(
            content_type=content_type,
            extraction_method=extraction_method,
            status=status
        ).inc()
        
        self.extraction_duration.labels(
            content_type=content_type,
            extraction_method=extraction_method
        ).observe(duration)
        
        if quality_score is not None:
            self.content_quality_score.labels(
                content_type=content_type
            ).observe(quality_score)

    def record_cache_hit(self, cache_type: str = 'content'):
        """Record cache hit."""
        self.cache_hits_total.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str = 'content'):
        """Record cache miss."""
        self.cache_misses_total.labels(cache_type=cache_type).inc()

    def update_cache_size(self, size_bytes: int, cache_type: str = 'content'):
        """Update cache size."""
        self.cache_size_bytes.labels(cache_type=cache_type).set(size_bytes)

    def record_image_processing(
        self,
        status: str,
        image_format: str,
        duration: float
    ):
        """Record image processing metrics."""
        self.images_processed_total.labels(
            status=status,
            format=image_format
        ).inc()
        
        self.image_processing_duration.labels(
            format=image_format
        ).observe(duration)

    def record_error(self, error_type: str, component: str):
        """Record error metrics."""
        self.errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()

    def update_active_extractions(self, count: int):
        """Update active extractions count."""
        self.active_extractions.set(count)

    def update_queue_size(self, size: int):
        """Update queue size."""
        self.queue_size.set(size)

    def update_memory_usage(self, usage_bytes: int):
        """Update memory usage."""
        self.memory_usage_bytes.set(usage_bytes)

    def get_cache_hit_rate(self, cache_type: str = 'content') -> float:
        """Calculate cache hit rate."""
        hits = self.cache_hits_total.labels(cache_type=cache_type)._value.get()
        misses = self.cache_misses_total.labels(cache_type=cache_type)._value.get()
        
        total = hits + misses
        if total == 0:
            return 0.0
        
        return hits / total

    def get_error_rate(self, component: str = None) -> float:
        """Calculate error rate."""
        if component:
            errors = self.errors_total.labels(
                error_type='all',
                component=component
            )._value.get()
        else:
            # Sum all errors
            errors = sum(
                metric._value.get()
                for metric in self.errors_total._metrics.values()
            )
        
        total_requests = sum(
            metric._value.get()
            for metric in self.requests_total._metrics.values()
        )
        
        if total_requests == 0:
            return 0.0
        
        return errors / total_requests

    def get_throughput_per_minute(self) -> float:
        """Calculate throughput per minute."""
        # This would need to be calculated based on time windows
        # For now, return a placeholder
        return 0.0

    def get_average_processing_time(self, content_type: str = None) -> float:
        """Get average processing time."""
        if content_type:
            return self.extraction_duration.labels(
                content_type=content_type,
                extraction_method='all'
            )._sum.get() / max(
                self.extraction_duration.labels(
                    content_type=content_type,
                    extraction_method='all'
                )._count.get(), 1
            )
        else:
            # Calculate overall average
            total_sum = sum(
                metric._sum.get()
                for metric in self.extraction_duration._metrics.values()
            )
            total_count = sum(
                metric._count.get()
                for metric in self.extraction_duration._metrics.values()
            )
            
            return total_sum / max(total_count, 1)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            'requests_total': sum(
                metric._value.get()
                for metric in self.requests_total._metrics.values()
            ),
            'extractions_total': sum(
                metric._value.get()
                for metric in self.extractions_total._metrics.values()
            ),
            'cache_hit_rate': self.get_cache_hit_rate(),
            'error_rate': self.get_error_rate(),
            'active_extractions': self.active_extractions._value.get(),
            'queue_size': self.queue_size._value.get(),
            'memory_usage_bytes': self.memory_usage_bytes._value.get(),
            'average_processing_time': self.get_average_processing_time()
        }


# Global metrics instance
metrics = MetricsCollector()
