"""
Monitoring and analytics for notification decisioning service.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import structlog
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

logger = structlog.get_logger()

# Create custom registry for service metrics
registry = CollectorRegistry()

# Decision metrics
NOTIFICATION_DECISIONS_TOTAL = Counter(
    'notification_decisions_total',
    'Total number of notification decisions made',
    ['decision', 'notification_type', 'user_id_hash'],
    registry=registry
)

NOTIFICATION_DECISION_DURATION = Histogram(
    'notification_decision_duration_seconds',
    'Time spent making notification decisions',
    ['notification_type'],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    registry=registry
)

# Delivery metrics
NOTIFICATION_DELIVERIES_TOTAL = Counter(
    'notification_deliveries_total',
    'Total number of notification deliveries',
    ['channel', 'status', 'user_id_hash'],
    registry=registry
)

NOTIFICATION_DELIVERY_DURATION = Histogram(
    'notification_delivery_duration_seconds',
    'Time spent delivering notifications',
    ['channel'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=registry
)

# Engagement metrics
NOTIFICATION_ENGAGEMENT_TOTAL = Counter(
    'notification_engagement_total',
    'Total number of notification engagements',
    ['engagement_type', 'channel', 'user_id_hash'],
    registry=registry
)

# Fatigue metrics
FATIGUE_DETECTIONS_TOTAL = Counter(
    'fatigue_detections_total',
    'Total number of fatigue detections',
    ['fatigue_type', 'user_id_hash'],
    registry=registry
)

# A/B Testing metrics
AB_TEST_ASSIGNMENTS_TOTAL = Counter(
    'ab_test_assignments_total',
    'Total number of A/B test assignments',
    ['experiment', 'variant', 'user_id_hash'],
    registry=registry
)

# System metrics
ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections',
    registry=registry
)

MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    registry=registry
)

# Performance metrics
QUEUE_SIZE = Gauge(
    'notification_queue_size',
    'Number of notifications in queue',
    registry=registry
)

PROCESSING_RATE = Gauge(
    'notifications_processed_per_second',
    'Rate of notification processing',
    registry=registry
)


class NotificationAnalytics:
    """Analytics collector for notification service."""
    
    def __init__(self):
        self.analytics_data = {}
        self.metrics_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def record_decision(
        self,
        user_id: str,
        notification_type: str,
        decision: str,
        duration_seconds: float
    ) -> None:
        """Record notification decision metrics."""
        
        try:
            # Hash user_id for privacy
            user_hash = self._hash_user_id(user_id)
            
            # Record metrics
            NOTIFICATION_DECISIONS_TOTAL.labels(
                decision=decision,
                notification_type=notification_type,
                user_id_hash=user_hash
            ).inc()
            
            NOTIFICATION_DECISION_DURATION.labels(
                notification_type=notification_type
            ).observe(duration_seconds)
            
            logger.debug(
                "Recorded decision metrics",
                user_id_hash=user_hash,
                notification_type=notification_type,
                decision=decision,
                duration=duration_seconds
            )
            
        except Exception as e:
            logger.error(f"Error recording decision metrics: {e}")
    
    async def record_delivery(
        self,
        user_id: str,
        channel: str,
        status: str,
        duration_seconds: float
    ) -> None:
        """Record notification delivery metrics."""
        
        try:
            # Hash user_id for privacy
            user_hash = self._hash_user_id(user_id)
            
            # Record metrics
            NOTIFICATION_DELIVERIES_TOTAL.labels(
                channel=channel,
                status=status,
                user_id_hash=user_hash
            ).inc()
            
            NOTIFICATION_DELIVERY_DURATION.labels(
                channel=channel
            ).observe(duration_seconds)
            
            logger.debug(
                "Recorded delivery metrics",
                user_id_hash=user_hash,
                channel=channel,
                status=status,
                duration=duration_seconds
            )
            
        except Exception as e:
            logger.error(f"Error recording delivery metrics: {e}")
    
    async def record_engagement(
        self,
        user_id: str,
        engagement_type: str,
        channel: str
    ) -> None:
        """Record notification engagement metrics."""
        
        try:
            # Hash user_id for privacy
            user_hash = self._hash_user_id(user_id)
            
            # Record metrics
            NOTIFICATION_ENGAGEMENT_TOTAL.labels(
                engagement_type=engagement_type,
                channel=channel,
                user_id_hash=user_hash
            ).inc()
            
            logger.debug(
                "Recorded engagement metrics",
                user_id_hash=user_hash,
                engagement_type=engagement_type,
                channel=channel
            )
            
        except Exception as e:
            logger.error(f"Error recording engagement metrics: {e}")
    
    async def record_fatigue_detection(
        self,
        user_id: str,
        fatigue_type: str
    ) -> None:
        """Record fatigue detection metrics."""
        
        try:
            # Hash user_id for privacy
            user_hash = self._hash_user_id(user_id)
            
            # Record metrics
            FATIGUE_DETECTIONS_TOTAL.labels(
                fatigue_type=fatigue_type,
                user_id_hash=user_hash
            ).inc()
            
            logger.debug(
                "Recorded fatigue detection metrics",
                user_id_hash=user_hash,
                fatigue_type=fatigue_type
            )
            
        except Exception as e:
            logger.error(f"Error recording fatigue detection metrics: {e}")
    
    async def record_ab_test_assignment(
        self,
        user_id: str,
        experiment: str,
        variant: str
    ) -> None:
        """Record A/B test assignment metrics."""
        
        try:
            # Hash user_id for privacy
            user_hash = self._hash_user_id(user_id)
            
            # Record metrics
            AB_TEST_ASSIGNMENTS_TOTAL.labels(
                experiment=experiment,
                variant=variant,
                user_id_hash=user_hash
            ).inc()
            
            logger.debug(
                "Recorded A/B test assignment metrics",
                user_id_hash=user_hash,
                experiment=experiment,
                variant=variant
            )
            
        except Exception as e:
            logger.error(f"Error recording A/B test assignment metrics: {e}")
    
    async def update_system_metrics(self) -> None:
        """Update system-level metrics."""
        
        try:
            # Update active connections (mock)
            ACTIVE_CONNECTIONS.set(100)
            
            # Update memory usage (mock)
            import psutil
            memory_usage = psutil.Process().memory_info().rss
            MEMORY_USAGE.set(memory_usage)
            
            # Update queue size (mock)
            QUEUE_SIZE.set(50)
            
            # Update processing rate (mock)
            PROCESSING_RATE.set(10.5)
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary."""
        
        try:
            summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': {
                    'total_decisions': self._get_counter_value(NOTIFICATION_DECISIONS_TOTAL),
                    'total_deliveries': self._get_counter_value(NOTIFICATION_DELIVERIES_TOTAL),
                    'total_engagements': self._get_counter_value(NOTIFICATION_ENGAGEMENT_TOTAL),
                    'total_fatigue_detections': self._get_counter_value(FATIGUE_DETECTIONS_TOTAL),
                    'total_ab_assignments': self._get_counter_value(AB_TEST_ASSIGNMENTS_TOTAL)
                },
                'system_metrics': {
                    'active_connections': ACTIVE_CONNECTIONS._value._value,
                    'memory_usage_bytes': MEMORY_USAGE._value._value,
                    'queue_size': QUEUE_SIZE._value._value,
                    'processing_rate': PROCESSING_RATE._value._value
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {}
    
    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy."""
        import hashlib
        return hashlib.sha256(user_id.encode()).hexdigest()[:8]
    
    def _get_counter_value(self, counter) -> int:
        """Get counter value."""
        try:
            return int(counter._value._value)
        except:
            return 0


class PerformanceMonitor:
    """Monitor service performance."""
    
    def __init__(self):
        self.performance_data = {}
        self.alert_thresholds = {
            'decision_latency_ms': 1000,
            'delivery_latency_ms': 5000,
            'error_rate_percent': 5.0,
            'memory_usage_mb': 1000
        }
    
    async def monitor_decision_performance(
        self,
        user_id: str,
        notification_type: str,
        duration_ms: float
    ) -> None:
        """Monitor decision performance."""
        
        try:
            # Check for performance issues
            if duration_ms > self.alert_thresholds['decision_latency_ms']:
                logger.warning(
                    "High decision latency detected",
                    user_id=user_id,
                    notification_type=notification_type,
                    duration_ms=duration_ms,
                    threshold=self.alert_thresholds['decision_latency_ms']
                )
            
            # Update performance data
            key = f"decision_performance:{notification_type}"
            if key not in self.performance_data:
                self.performance_data[key] = []
            
            self.performance_data[key].append({
                'timestamp': datetime.utcnow(),
                'duration_ms': duration_ms,
                'user_id': user_id
            })
            
            # Keep only last 1000 records
            if len(self.performance_data[key]) > 1000:
                self.performance_data[key] = self.performance_data[key][-1000:]
            
        except Exception as e:
            logger.error(f"Error monitoring decision performance: {e}")
    
    async def monitor_delivery_performance(
        self,
        user_id: str,
        channel: str,
        duration_ms: float,
        success: bool
    ) -> None:
        """Monitor delivery performance."""
        
        try:
            # Check for performance issues
            if duration_ms > self.alert_thresholds['delivery_latency_ms']:
                logger.warning(
                    "High delivery latency detected",
                    user_id=user_id,
                    channel=channel,
                    duration_ms=duration_ms,
                    threshold=self.alert_thresholds['delivery_latency_ms']
                )
            
            # Update performance data
            key = f"delivery_performance:{channel}"
            if key not in self.performance_data:
                self.performance_data[key] = []
            
            self.performance_data[key].append({
                'timestamp': datetime.utcnow(),
                'duration_ms': duration_ms,
                'success': success,
                'user_id': user_id
            })
            
            # Keep only last 1000 records
            if len(self.performance_data[key]) > 1000:
                self.performance_data[key] = self.performance_data[key][-1000:]
            
        except Exception as e:
            logger.error(f"Error monitoring delivery performance: {e}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        
        try:
            summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'performance_metrics': {},
                'alerts': []
            }
            
            # Calculate performance metrics
            for key, data in self.performance_data.items():
                if not data:
                    continue
                
                # Calculate average latency
                avg_latency = sum(record['duration_ms'] for record in data) / len(data)
                
                # Calculate success rate
                if 'success' in data[0]:
                    success_rate = sum(1 for record in data if record['success']) / len(data) * 100
                else:
                    success_rate = 100.0
                
                summary['performance_metrics'][key] = {
                    'avg_latency_ms': round(avg_latency, 2),
                    'success_rate_percent': round(success_rate, 2),
                    'total_operations': len(data)
                }
                
                # Check for alerts
                if avg_latency > self.alert_thresholds['decision_latency_ms']:
                    summary['alerts'].append({
                        'type': 'high_latency',
                        'metric': key,
                        'value': avg_latency,
                        'threshold': self.alert_thresholds['decision_latency_ms']
                    })
                
                if success_rate < (100 - self.alert_thresholds['error_rate_percent']):
                    summary['alerts'].append({
                        'type': 'low_success_rate',
                        'metric': key,
                        'value': success_rate,
                        'threshold': 100 - self.alert_thresholds['error_rate_percent']
                    })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}


def setup_monitoring() -> None:
    """Setup monitoring and metrics collection."""
    
    logger.info("Setting up monitoring and metrics collection")
    
    # Initialize analytics
    analytics = NotificationAnalytics()
    
    # Initialize performance monitor
    performance_monitor = PerformanceMonitor()
    
    logger.info("Monitoring setup completed successfully")


# Global instances
analytics = NotificationAnalytics()
performance_monitor = PerformanceMonitor()
