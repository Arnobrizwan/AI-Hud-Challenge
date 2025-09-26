"""Metrics collection and monitoring for Content Enrichment Service."""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
import structlog
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from config import settings

logger = structlog.get_logger(__name__)


class MetricsCollector:
    """Metrics collector using Prometheus and Redis."""

    def __init__(self):
        """Initialize the metrics collector."""
        self.registry = CollectorRegistry()
        self.redis_client = None

        # Prometheus metrics
        self._init_prometheus_metrics()

        # Internal metrics storage
        self.metrics_cache = {}
        self.cache_ttl = 300  # 5 minutes

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # Request metrics
        self.request_total = Counter(
            "content_enrichment_requests_total",
            "Total number of enrichment requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.request_duration = Histogram(
            "content_enrichment_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry,
        )

        # Enrichment metrics
        self.enrichment_total = Counter(
            "content_enrichment_processed_total",
            "Total number of content pieces processed",
            ["status", "processing_mode"],
            registry=self.registry,
        )

        self.enrichment_duration = Histogram(
            "content_enrichment_processing_duration_seconds",
            "Content enrichment processing duration",
            ["processing_mode"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

        # Entity extraction metrics
        self.entities_extracted = Counter(
            "content_enrichment_entities_extracted_total",
            "Total number of entities extracted",
            ["entity_type"],
            registry=self.registry,
        )

        self.entity_extraction_duration = Histogram(
            "content_enrichment_entity_extraction_duration_seconds",
            "Entity extraction duration",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry,
        )

        # Topic classification metrics
        self.topics_classified = Counter(
            "content_enrichment_topics_classified_total",
            "Total number of topics classified",
            ["topic_category"],
            registry=self.registry,
        )

        self.topic_classification_duration = Histogram(
            "content_enrichment_topic_classification_duration_seconds",
            "Topic classification duration",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry,
        )

        # Sentiment analysis metrics
        self.sentiment_analyzed = Counter(
            "content_enrichment_sentiment_analyzed_total",
            "Total number of sentiment analyses",
            ["sentiment_label"],
            registry=self.registry,
        )

        self.sentiment_analysis_duration = Histogram(
            "content_enrichment_sentiment_analysis_duration_seconds",
            "Sentiment analysis duration",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry,
        )

        # Content signals metrics
        self.signals_extracted = Counter(
            "content_enrichment_signals_extracted_total",
            "Total number of content signals extracted",
            ["signal_type"],
            registry=self.registry,
        )

        self.signal_extraction_duration = Histogram(
            "content_enrichment_signal_extraction_duration_seconds",
            "Signal extraction duration",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry,
        )

        # Trustworthiness metrics
        self.trust_scores_computed = Counter(
            "content_enrichment_trust_scores_computed_total",
            "Total number of trustworthiness scores computed",
            ["trust_level"],
            registry=self.registry,
        )

        self.trust_score_duration = Histogram(
            "content_enrichment_trust_score_duration_seconds",
            "Trustworthiness score computation duration",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry,
        )

        # System metrics
        self.active_requests = Gauge(
            "content_enrichment_active_requests",
            "Number of active enrichment requests",
            registry=self.registry,
        )

        self.queue_size = Gauge(
            "content_enrichment_queue_size",
            "Number of items in processing queue",
            registry=self.registry,
        )

        self.cache_hits = Counter(
            "content_enrichment_cache_hits_total",
            "Total number of cache hits",
            ["cache_type"],
            registry=self.registry,
        )

        self.cache_misses = Counter(
            "content_enrichment_cache_misses_total",
            "Total number of cache misses",
            ["cache_type"],
            registry=self.registry,
        )

        # Error metrics
        self.errors_total = Counter(
            "content_enrichment_errors_total",
            "Total number of errors",
            ["error_type", "component"],
            registry=self.registry,
        )

        # Model metrics
        self.model_inference_duration = Histogram(
            "content_enrichment_model_inference_duration_seconds",
            "Model inference duration",
            ["model_name", "model_version"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry,
        )

        self.model_accuracy = Gauge(
            "content_enrichment_model_accuracy",
            "Model accuracy score",
            ["model_name", "model_version"],
            registry=self.registry,
        )

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the metrics collector."""
        try:
            if settings.enable_metrics:
                # Initialize Redis connection for distributed metrics
                self.redis_client = redis.from_url(settings.redis_url, max_connections=10, decode_responses=True)

                # Test connection
                await self.redis_client.ping()
                logger.info("Metrics collector initialized with Redis")
            else:
                logger.info("Metrics collection disabled")

        except Exception as e:
            logger.warning("Failed to initialize Redis for metrics", error=str(e))
            self.redis_client = None

    def middleware_class(self):
        """Get the middleware class for FastAPI."""
        import time

        from fastapi import Request
        from fastapi.responses import Response

        class MetricsMiddleware:
            def __init__(self, app, metrics_collector):
                self.app = app
                self.metrics = metrics_collector

            async def __call__(self, request: Request, call_next) -> Dict[str, Any]:
                start_time = time.time()

                # Increment active requests
                self.metrics.active_requests.inc()

                try:
                    response = await call_next(request)

                    # Record request metrics
                    duration = time.time() - start_time
                    self.metrics.request_total.labels(
                        method=request.method,
                        endpoint=request.url.path,
                        status=response.status_code,
                    ).inc()

                    self.metrics.request_duration.labels(method=request.method, endpoint=request.url.path).observe(
                        duration
                    )

                    return response

                except Exception as e:
                    # Record error metrics
                    self.metrics.errors_total.labels(error_type=type(e).__name__, component="middleware").inc()
                    raise

                finally:
                    # Decrement active requests
                    self.metrics.active_requests.dec()

        return MetricsMiddleware

    async def record_enrichment(
        self,
        processing_time: int,
        success: bool,
        entities_count: int = 0,
        topics_count: int = 0,
        processing_mode: str = "realtime",
    ) -> Dict[str, Any]:
        """Record enrichment metrics."""
        try:
            # Record enrichment metrics
            status = "success" if success else "error"
            self.enrichment_total.labels(status=status, processing_mode=processing_mode).inc()

            if success:
                self.enrichment_duration.labels(processing_mode=processing_mode).observe(
                    processing_time / 1000.0
                )  # Convert to seconds

                # Record entity metrics
                if entities_count > 0:
                    self.entities_extracted.labels(entity_type="total").inc(entities_count)

                # Record topic metrics
                if topics_count > 0:
                    self.topics_classified.labels(topic_category="total").inc(topics_count)

            # Store in Redis for distributed metrics
            if self.redis_client:
                await self._store_metrics_in_redis(
                    {
                        "enrichment": {
                            "processing_time": processing_time,
                            "success": success,
                            "entities_count": entities_count,
                            "topics_count": topics_count,
                            "processing_mode": processing_mode,
                            "timestamp": time.time(),
                        }
                    }
                )

        except Exception as e:
            logger.error("Failed to record enrichment metrics", error=str(e))

    async def record_batch_enrichment(
        self, batch_size: int, processing_time: int, success_count: int
    ) -> Dict[str, Any]:
        """Record batch enrichment metrics."""
        try:
            # Record batch metrics
            self.enrichment_total.labels(status="success", processing_mode="batch").inc(success_count)

            if success_count < batch_size:
                self.enrichment_total.labels(status="error", processing_mode="batch").inc(batch_size - success_count)

            self.enrichment_duration.labels(processing_mode="batch").observe(processing_time / 1000.0)

            # Store in Redis
            if self.redis_client:
                await self._store_metrics_in_redis(
                    {
                        "batch_enrichment": {
                            "batch_size": batch_size,
                            "processing_time": processing_time,
                            "success_count": success_count,
                            "timestamp": time.time(),
                        }
                    }
                )

        except Exception as e:
            logger.error("Failed to record batch enrichment metrics", error=str(e))

    async def record_entity_extraction(self, entity_type: str, duration: float, success: bool) -> Dict[str, Any]:
        """Record entity extraction metrics."""
        try:
            if success:
                self.entities_extracted.labels(entity_type=entity_type).inc()

                self.entity_extraction_duration.observe(duration)
            else:
                self.errors_total.labels(error_type="entity_extraction_failed", component="entity_extractor").inc()

        except Exception as e:
            logger.error("Failed to record entity extraction metrics", error=str(e))

    async def record_topic_classification(self, topic_category: str, duration: float, success: bool) -> Dict[str, Any]:
        """Record topic classification metrics."""
        try:
            if success:
                self.topics_classified.labels(topic_category=topic_category).inc()

                self.topic_classification_duration.observe(duration)
            else:
                self.errors_total.labels(error_type="topic_classification_failed", component="topic_classifier").inc()

        except Exception as e:
            logger.error("Failed to record topic classification metrics", error=str(e))

    async def record_sentiment_analysis(self, sentiment_label: str, duration: float, success: bool) -> Dict[str, Any]:
        """Record sentiment analysis metrics."""
        try:
            if success:
                self.sentiment_analyzed.labels(sentiment_label=sentiment_label).inc()

                self.sentiment_analysis_duration.observe(duration)
            else:
                self.errors_total.labels(error_type="sentiment_analysis_failed", component="sentiment_analyzer").inc()

        except Exception as e:
            logger.error("Failed to record sentiment analysis metrics", error=str(e))

    async def record_signal_extraction(self, signal_type: str, duration: float, success: bool):
        """Record signal extraction metrics."""
        try:
            if success:
                self.signals_extracted.labels(signal_type=signal_type).inc()

                self.signal_extraction_duration.observe(duration)
            else:
                self.errors_total.labels(error_type="signal_extraction_failed", component="signal_extractor").inc()

        except Exception as e:
            logger.error("Failed to record signal extraction metrics", error=str(e))

    async def record_trust_score(self, trust_level: str, duration: float, success: bool):
        """Record trustworthiness score metrics."""
        try:
            if success:
                self.trust_scores_computed.labels(trust_level=trust_level).inc()

                self.trust_score_duration.observe(duration)
            else:
                self.errors_total.labels(error_type="trust_score_failed", component="signal_extractor").inc()

        except Exception as e:
            logger.error("Failed to record trust score metrics", error=str(e))

    async def record_model_inference(self, model_name: str, model_version: str, duration: float, success: bool):
        """Record model inference metrics."""
        try:
            if success:
                self.model_inference_duration.labels(model_name=model_name, model_version=model_version).observe(
                    duration
                )
            else:
                self.errors_total.labels(error_type="model_inference_failed", component=model_name).inc()

        except Exception as e:
            logger.error("Failed to record model inference metrics", error=str(e))

    async def record_cache_hit(self, cache_type: str) -> Dict[str, Any]:
        """Record cache hit."""
        try:
            self.cache_hits.labels(cache_type=cache_type).inc()
        except Exception as e:
            logger.error("Failed to record cache hit", error=str(e))

    async def record_cache_miss(self, cache_type: str) -> Dict[str, Any]:
        """Record cache miss."""
        try:
            self.cache_misses.labels(cache_type=cache_type).inc()
        except Exception as e:
            logger.error("Failed to record cache miss", error=str(e))

    async def record_error(self, error_type: str, component: str) -> Dict[str, Any]:
        """Record error metrics."""
        try:
            self.errors_total.labels(error_type=error_type, component=component).inc()
        except Exception as e:
            logger.error("Failed to record error metrics", error=str(e))

    async def _store_metrics_in_redis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Store metrics in Redis for distributed collection."""
        try:
            if not self.redis_client:
                return

            timestamp = int(time.time())
            key = f"metrics:{timestamp}"

            await self.redis_client.hset(key, mapping=metrics)
            await self.redis_client.expire(key, 3600)  # 1 hour TTL

        except Exception as e:
            logger.error("Failed to store metrics in Redis", error=str(e))

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        try:
            # Get Prometheus metrics
            prometheus_metrics = generate_latest(self.registry).decode("utf-8")

            # Get custom metrics
            custom_metrics = await self._get_custom_metrics()

            return {
                "prometheus": prometheus_metrics,
                "custom": custom_metrics,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Failed to get metrics", error=str(e))
            return {"error": str(e)}

    async def _get_custom_metrics(self) -> Dict[str, Any]:
        """Get custom metrics from Redis."""
        try:
            if not self.redis_client:
                return {}

            # Get recent metrics from Redis
            pattern = "metrics:*"
            keys = await self.redis_client.keys(pattern)

            if not keys:
                return {}

            # Get metrics from last hour
            cutoff_time = int(time.time()) - 3600
            recent_metrics = []

            for key in keys:
                timestamp = int(key.split(":")[1])
                if timestamp > cutoff_time:
                    metrics = await self.redis_client.hgetall(key)
                    recent_metrics.append(metrics)

            return {"recent_metrics": recent_metrics, "metrics_count": len(recent_metrics)}

        except Exception as e:
            logger.error("Failed to get custom metrics", error=str(e))
            return {}

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup metrics collector resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Metrics collector cleanup completed")
        except Exception as e:
            logger.error("Metrics collector cleanup failed", error=str(e))
