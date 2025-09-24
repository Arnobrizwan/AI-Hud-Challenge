"""
Comprehensive Monitoring and Metrics System
Advanced metrics collection and monitoring for the summarization service
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil
from summarization.models import ProcessingStats, QualityMetrics, SummaryResult

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ServiceMetrics:
    """Service-level metrics"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_processing_time: float = 0.0
    average_quality_score: float = 0.0
    total_processing_time: float = 0.0
    total_quality_score: float = 0.0
    error_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """Performance metrics"""

    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None


@dataclass
class QualityMetrics:
    """Quality metrics aggregation"""

    average_rouge1: float = 0.0
    average_rouge2: float = 0.0
    average_rougeL: float = 0.0
    average_bertscore: float = 0.0
    average_consistency: float = 0.0
    average_readability: float = 0.0
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    total_quality_checks: int = 0


class MetricsCollector:
    """Advanced metrics collection and monitoring system"""

    def __init__(self):
        """Initialize the metrics collector"""
        self.service_metrics = ServiceMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.quality_metrics = QualityMetrics()

        # Time-series data for trending
        self.processing_times = deque(maxlen=1000)
        self.quality_scores = deque(maxlen=1000)
        self.error_rates = deque(maxlen=1000)

        # Method-specific metrics
        self.method_metrics = defaultdict(
            lambda: {"count": 0, "total_time": 0.0, "avg_quality": 0.0, "errors": 0}
        )

        # Language-specific metrics
        self.language_metrics = defaultdict(
            lambda: {"count": 0, "avg_processing_time": 0.0, "avg_quality": 0.0}
        )

        # A/B test metrics
        self.ab_test_metrics = defaultdict(
            lambda: {
                "participants": 0,
                "conversions": 0,
                "avg_quality": 0.0,
                "avg_processing_time": 0.0,
            }
        )

        self._initialized = False
        self._monitoring_thread = None
        self._stop_monitoring = False

    async def initialize(self):
        """Initialize metrics collection"""
        try:
            logger.info("Initializing metrics collector...")

            # Start background monitoring
            self._stop_monitoring = False
            self._monitoring_thread = threading.Thread(
                target=self._monitor_system_resources, daemon=True
            )
            self._monitoring_thread.start()

            self._initialized = True
            logger.info("Metrics collector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize metrics collector: {str(e)}")
            raise

    async def cleanup(self):
        """Clean up metrics collection"""
        try:
            self._stop_monitoring = True
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=5)
        except Exception as e:
            logger.error(f"Error during metrics cleanup: {str(e)}")

    def increment_request(self):
        """Increment total request count"""
        self.service_metrics.total_requests += 1
        self.service_metrics.last_updated = datetime.now()

    def increment_success(self):
        """Increment successful request count"""
        self.service_metrics.successful_requests += 1
        self.service_metrics.last_updated = datetime.now()

    def increment_error(self):
        """Increment error count"""
        self.service_metrics.failed_requests += 1
        self.service_metrics.error_count += 1
        self.service_metrics.last_updated = datetime.now()

        # Update error rate
        self.error_rates.append(1)

    def increment_cache_hit(self):
        """Increment cache hit count"""
        self.service_metrics.cache_hits += 1
        self.service_metrics.last_updated = datetime.now()

    def increment_cache_miss(self):
        """Increment cache miss count"""
        self.service_metrics.cache_misses += 1
        self.service_metrics.last_updated = datetime.now()

    def record_summarization_metrics(self, result: SummaryResult):
        """Record metrics for a summarization result"""
        try:
            # Update service metrics
            self.service_metrics.successful_requests += 1
            self.service_metrics.total_processing_time += result.processing_stats.total_time
            self.service_metrics.total_quality_score += result.quality_metrics.overall_score

            # Update averages
            if self.service_metrics.successful_requests > 0:
                self.service_metrics.average_processing_time = (
                    self.service_metrics.total_processing_time
                    / self.service_metrics.successful_requests
                )
                self.service_metrics.average_quality_score = (
                    self.service_metrics.total_quality_score
                    / self.service_metrics.successful_requests
                )

            # Record time-series data
            self.processing_times.append(result.processing_stats.total_time)
            self.quality_scores.append(result.quality_metrics.overall_score)

            # Update method-specific metrics
            method = result.summary.method.value
            self.method_metrics[method]["count"] += 1
            self.method_metrics[method]["total_time"] += result.processing_stats.total_time
            self.method_metrics[method]["avg_quality"] = (
                self.method_metrics[method]["avg_quality"]
                * (self.method_metrics[method]["count"] - 1)
                + result.quality_metrics.overall_score
            ) / self.method_metrics[method]["count"]

            # Update language-specific metrics
            language = result.language_detected.value
            self.language_metrics[language]["count"] += 1
            self.language_metrics[language]["avg_processing_time"] = (
                self.language_metrics[language]["avg_processing_time"]
                * (self.language_metrics[language]["count"] - 1)
                + result.processing_stats.total_time
            ) / self.language_metrics[language]["count"]
            self.language_metrics[language]["avg_quality"] = (
                self.language_metrics[language]["avg_quality"]
                * (self.language_metrics[language]["count"] - 1)
                + result.quality_metrics.overall_score
            ) / self.language_metrics[language]["count"]

            # Update quality metrics
            self._update_quality_metrics(result.quality_metrics)

            self.service_metrics.last_updated = datetime.now()

        except Exception as e:
            logger.error(f"Failed to record summarization metrics: {str(e)}")

    def record_batch_metrics(self, total_requests: int, successful_requests: int):
        """Record metrics for batch processing"""
        try:
            self.service_metrics.total_requests += total_requests
            self.service_metrics.successful_requests += successful_requests
            self.service_metrics.failed_requests += total_requests - successful_requests

            # Calculate batch success rate
            if total_requests > 0:
                success_rate = successful_requests / total_requests
                self.error_rates.append(1 - success_rate)

            self.service_metrics.last_updated = datetime.now()

        except Exception as e:
            logger.error(f"Failed to record batch metrics: {str(e)}")

    def increment_headline_generation(self):
        """Increment headline generation count"""
        # This could be expanded to track headline-specific metrics
        pass

    def _update_quality_metrics(self, quality: QualityMetrics):
        """Update aggregated quality metrics"""
        try:
            self.quality_metrics.total_quality_checks += 1

            # Update averages
            n = self.quality_metrics.total_quality_checks
            self.quality_metrics.average_rouge1 = (
                self.quality_metrics.average_rouge1 * (n - 1) + quality.rouge1_f1
            ) / n
            self.quality_metrics.average_rouge2 = (
                self.quality_metrics.average_rouge2 * (n - 1) + quality.rouge2_f1
            ) / n
            self.quality_metrics.average_rougeL = (
                self.quality_metrics.average_rougeL * (n - 1) + quality.rougeL_f1
            ) / n
            self.quality_metrics.average_bertscore = (
                self.quality_metrics.average_bertscore * (n - 1) + quality.bertscore_f1
            ) / n
            self.quality_metrics.average_consistency = (
                self.quality_metrics.average_consistency * (n - 1) + quality.factual_consistency
            ) / n
            self.quality_metrics.average_readability = (
                self.quality_metrics.average_readability * (n - 1) + quality.readability
            ) / n

            # Update quality distribution
            quality_range = self._get_quality_range(quality.overall_score)
            self.quality_metrics.quality_distribution[quality_range] = (
                self.quality_metrics.quality_distribution.get(quality_range, 0) + 1
            )

        except Exception as e:
            logger.error(f"Failed to update quality metrics: {str(e)}")

    def _get_quality_range(self, score: float) -> str:
        """Get quality range for distribution"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.7:
            return "fair"
        elif score >= 0.6:
            return "poor"
        else:
            return "very_poor"

    def _monitor_system_resources(self):
        """Monitor system resources in background thread"""
        while not self._stop_monitoring:
            try:
                # CPU usage
                self.performance_metrics.cpu_usage = psutil.cpu_percent(interval=1)

                # Memory usage
                memory = psutil.virtual_memory()
                self.performance_metrics.memory_usage = memory.percent
                self.performance_metrics.memory_available = memory.available / (1024**3)  # GB

                # Disk usage
                disk = psutil.disk_usage("/")
                self.performance_metrics.disk_usage = (disk.used / disk.total) * 100

                # Network I/O
                net_io = psutil.net_io_counters()
                self.performance_metrics.network_io = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                }

                # GPU usage (if available)
                try:
                    import GPUtil

                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        self.performance_metrics.gpu_usage = gpu.load * 100
                        self.performance_metrics.gpu_memory = gpu.memoryUtil * 100
                except ImportError:
                    pass  # GPU monitoring not available

                time.sleep(10)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"System monitoring error: {str(e)}")
                time.sleep(30)  # Wait longer on error

    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        try:
            # Calculate derived metrics
            total_requests = self.service_metrics.total_requests
            success_rate = (
                self.service_metrics.successful_requests / total_requests
                if total_requests > 0
                else 0
            )

            cache_hit_rate = (
                self.service_metrics.cache_hits
                / (self.service_metrics.cache_hits + self.service_metrics.cache_misses)
                if (self.service_metrics.cache_hits + self.service_metrics.cache_misses) > 0
                else 0
            )

            # Calculate trends
            processing_time_trend = self._calculate_trend(list(self.processing_times))
            quality_score_trend = self._calculate_trend(list(self.quality_scores))
            error_rate_trend = self._calculate_trend(list(self.error_rates))

            return {
                "service_metrics": {
                    "total_requests": total_requests,
                    "successful_requests": self.service_metrics.successful_requests,
                    "failed_requests": self.service_metrics.failed_requests,
                    "success_rate": success_rate,
                    "cache_hits": self.service_metrics.cache_hits,
                    "cache_misses": self.service_metrics.cache_misses,
                    "cache_hit_rate": cache_hit_rate,
                    "average_processing_time": self.service_metrics.average_processing_time,
                    "average_quality_score": self.service_metrics.average_quality_score,
                    "error_count": self.service_metrics.error_count,
                    "last_updated": self.service_metrics.last_updated.isoformat(),
                },
                "performance_metrics": {
                    "cpu_usage": self.performance_metrics.cpu_usage,
                    "memory_usage": self.performance_metrics.memory_usage,
                    "memory_available_gb": self.performance_metrics.memory_available,
                    "disk_usage": self.performance_metrics.disk_usage,
                    "network_io": self.performance_metrics.network_io,
                    "gpu_usage": self.performance_metrics.gpu_usage,
                    "gpu_memory": self.performance_metrics.gpu_memory,
                },
                "quality_metrics": {
                    "average_rouge1": self.quality_metrics.average_rouge1,
                    "average_rouge2": self.quality_metrics.average_rouge2,
                    "average_rougeL": self.quality_metrics.average_rougeL,
                    "average_bertscore": self.quality_metrics.average_bertscore,
                    "average_consistency": self.quality_metrics.average_consistency,
                    "average_readability": self.quality_metrics.average_readability,
                    "quality_distribution": self.quality_metrics.quality_distribution,
                    "total_quality_checks": self.quality_metrics.total_quality_checks,
                },
                "method_metrics": dict(self.method_metrics),
                "language_metrics": dict(self.language_metrics),
                "ab_test_metrics": dict(self.ab_test_metrics),
                "trends": {
                    "processing_time_trend": processing_time_trend,
                    "quality_score_trend": quality_score_trend,
                    "error_rate_trend": error_rate_trend,
                },
                "time_series": {
                    "processing_times": list(self.processing_times),
                    "quality_scores": list(self.quality_scores),
                    "error_rates": list(self.error_rates),
                },
            }

        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            return {}

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend for a series of values"""
        try:
            if len(values) < 2:
                return "insufficient_data"

            # Simple linear trend calculation
            n = len(values)
            x = list(range(n))

            # Calculate slope
            x_mean = sum(x) / n
            y_mean = sum(values) / n

            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if denominator == 0:
                return "no_trend"

            slope = numerator / denominator

            if slope > 0.1:
                return "increasing"
            elif slope < -0.1:
                return "decreasing"
            else:
                return "stable"

        except Exception as e:
            logger.error(f"Trend calculation failed: {str(e)}")
            return "error"

    async def get_status(self) -> Dict[str, Any]:
        """Get metrics collector status"""
        return {
            "initialized": self._initialized,
            "monitoring_active": self._monitoring_thread is not None
            and self._monitoring_thread.is_alive(),
            "total_requests": self.service_metrics.total_requests,
            "last_updated": self.service_metrics.last_updated.isoformat(),
        }
