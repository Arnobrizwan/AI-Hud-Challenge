"""
Model Monitoring Service - Comprehensive ML model monitoring and performance tracking
Supports real-time monitoring, alerting, drift detection, and performance analytics
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.config.settings import Settings
from src.infrastructure.alert_manager import AlertManager
from src.infrastructure.grafana_client import GrafanaClient
from src.infrastructure.prometheus_client import PrometheusClient
from src.models.monitoring_models import (
    AlertRule,
    DataQualityAlert,
    DriftAlert,
    DriftDetectionResult,
    MetricThreshold,
    ModelHealth,
    ModelMetrics,
    MonitoringConfig,
    MonitoringDashboard,
    PerformanceAlert,
    PerformanceReport,
)
from src.registry.model_registry import ModelRegistry
from src.utils.exceptions import MonitoringError, ValidationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelMonitoringService:
    """Comprehensive ML model monitoring and performance tracking"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.prometheus_client = PrometheusClient(settings)
        self.grafana_client = GrafanaClient(settings)
        self.alert_manager = AlertManager(settings)
        self.model_registry = ModelRegistry(settings)

        # Monitoring state
        self._monitoring_configs: Dict[str, MonitoringConfig] = {}
        self._active_alerts: Dict[str, Any] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._metrics_cache: Dict[str, List[ModelMetrics]] = {}

    async def initialize(self) -> None:
        """Initialize the monitoring service"""
        try:
            logger.info("Initializing Model Monitoring Service...")

            await self.prometheus_client.initialize()
            await self.grafana_client.initialize()
            await self.alert_manager.initialize()
            await self.model_registry.initialize()

            logger.info("Model Monitoring Service initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize Model Monitoring Service: {str(e)}")
            raise

    async def setup_model_monitoring(
        self, model_name: str, monitoring_config: MonitoringConfig
    ) -> None:
        """Set up monitoring for a model"""

        try:
            logger.info(f"Setting up monitoring for model: {model_name}")

            # Validate monitoring configuration
            await self._validate_monitoring_config(monitoring_config)

            # Store monitoring configuration
            self._monitoring_configs[model_name] = monitoring_config

            # Set up Prometheus metrics
            await self._setup_prometheus_metrics(model_name, monitoring_config)

            # Set up Grafana dashboard
            dashboard = await self._create_monitoring_dashboard(model_name, monitoring_config)

            # Set up alerting rules
            await self._setup_alerting_rules(model_name, monitoring_config)

            # Start monitoring task
            monitoring_task = asyncio.create_task(
                self._monitor_model(model_name, monitoring_config)
            )
            self._monitoring_tasks[model_name] = monitoring_task

            logger.info(f"Monitoring setup completed for {model_name}")

        except Exception as e:
            logger.error(
                f"Failed to setup monitoring for {model_name}: {str(e)}")
            raise MonitoringError(f"Monitoring setup failed: {str(e)}")

    async def setup_pipeline_monitoring(
        self, pipeline_id: str, monitoring_config: MonitoringConfig
    ) -> None:
        """Set up monitoring for a pipeline"""

        try:
            logger.info(f"Setting up pipeline monitoring: {pipeline_id}")

            # Set up pipeline-specific monitoring
            await self._setup_pipeline_metrics(pipeline_id, monitoring_config)

            # Set up pipeline dashboard
            await self._create_pipeline_dashboard(pipeline_id, monitoring_config)

            logger.info(
                f"Pipeline monitoring setup completed for {pipeline_id}")

        except Exception as e:
            logger.error(f"Failed to setup pipeline monitoring: {str(e)}")
            raise MonitoringError(
                f"Pipeline monitoring setup failed: {str(e)}")

    async def setup_deployment_monitoring(
        self,
        deployment_id: str,
        endpoint_url: str,
        model_name: str,
        monitoring_config: MonitoringConfig,
    ) -> None:
        """Set up monitoring for a model deployment"""

        try:
            logger.info(f"Setting up deployment monitoring: {deployment_id}")

            # Set up deployment-specific monitoring
            await self._setup_deployment_metrics(deployment_id, endpoint_url, monitoring_config)

            # Set up health checks
            await self._setup_health_checks(deployment_id, endpoint_url)

            # Set up performance monitoring
            await self._setup_performance_monitoring(deployment_id, model_name, monitoring_config)

            logger.info(
                f"Deployment monitoring setup completed for {deployment_id}")

        except Exception as e:
            logger.error(f"Failed to setup deployment monitoring: {str(e)}")
            raise MonitoringError(
                f"Deployment monitoring setup failed: {str(e)}")

    async def start_performance_monitoring(self) -> None:
        """Start performance monitoring for all models"""

        try:
            logger.info("Starting performance monitoring...")

            # Start monitoring task for each model
            for model_name, config in self._monitoring_configs.items():
                if model_name not in self._monitoring_tasks:
                    task = asyncio.create_task(
                        self._monitor_model_performance(
                            model_name, config))
                    self._monitoring_tasks[f"{model_name}_performance"] = task

            logger.info("Performance monitoring started")

        except Exception as e:
            logger.error(f"Failed to start performance monitoring: {str(e)}")
            raise MonitoringError(
                f"Performance monitoring startup failed: {str(e)}")

    async def stop_performance_monitoring(self) -> None:
        """Stop performance monitoring"""

        try:
            logger.info("Stopping performance monitoring...")

            # Cancel all monitoring tasks
            for task in self._monitoring_tasks.values():
                task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)

            self._monitoring_tasks.clear()

            logger.info("Performance monitoring stopped")

        except Exception as e:
            logger.error(f"Failed to stop performance monitoring: {str(e)}")

    async def log_model_metrics(
            self,
            model_name: str,
            metrics: ModelMetrics) -> None:
        """Log model metrics"""

        try:
            # Store metrics in cache
            if model_name not in self._metrics_cache:
                self._metrics_cache[model_name] = []

            self._metrics_cache[model_name].append(metrics)

            # Keep only recent metrics (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self._metrics_cache[model_name] = [
                m for m in self._metrics_cache[model_name] if m.timestamp > cutoff_time]

            # Send to Prometheus
            await self.prometheus_client.record_model_metrics(model_name, metrics)

            # Check for alerts
            await self._check_metric_alerts(model_name, metrics)

        except Exception as e:
            logger.error(f"Failed to log metrics for {model_name}: {str(e)}")

    async def log_feature_serving_metrics(
            self, metrics: Dict[str, Any]) -> None:
        """Log feature serving metrics"""

        try:
            # Send to Prometheus
            await self.prometheus_client.record_feature_serving_metrics(metrics)

        except Exception as e:
            logger.error(f"Failed to log feature serving metrics: {str(e)}")

    async def detect_data_drift(
            self,
            model_name: str,
            current_data: pd.DataFrame,
            reference_data: pd.DataFrame) -> DriftDetectionResult:
        """Detect data drift between current and reference data"""

        try:
            logger.info(f"Detecting data drift for {model_name}")

            drift_scores = {}
            drift_alerts = []

            # Calculate drift for each feature
            for column in current_data.columns:
                if column in reference_data.columns:
                    # Statistical drift detection (KS test)
                    from scipy import stats

                    current_values = current_data[column].dropna()
                    reference_values = reference_data[column].dropna()

                    if len(current_values) > 0 and len(reference_values) > 0:
                        ks_statistic, p_value = stats.ks_2samp(
                            reference_values, current_values)
                        drift_scores[column] = ks_statistic

                        # Check if drift is significant
                        if ks_statistic > 0.1:  # Threshold for significant drift
                            drift_alerts.append(
                                {
                                    "feature": column,
                                    "drift_score": ks_statistic,
                                    "p_value": p_value,
                                    "severity": "high" if ks_statistic > 0.2 else "medium",
                                })

            # Calculate overall drift score
            overall_drift_score = max(
                drift_scores.values()) if drift_scores else 0

            # Create drift detection result
            result = DriftDetectionResult(
                model_name=model_name,
                overall_drift_score=overall_drift_score,
                feature_drift_scores=drift_scores,
                drift_alerts=drift_alerts,
                detected_at=datetime.utcnow(),
                is_drift_detected=overall_drift_score > 0.1,
            )

            # Send drift alerts if detected
            if result.is_drift_detected:
    await self._send_drift_alert(result)

            logger.info(
                f"Data drift detection completed for {model_name}: {overall_drift_score}")
            return result

        except Exception as e:
            logger.error(
                f"Failed to detect data drift for {model_name}: {str(e)}")
            raise MonitoringError(f"Data drift detection failed: {str(e)}")

    async def generate_performance_report(
        self, model_name: str, start_time: datetime, end_time: datetime
    ) -> PerformanceReport:
        """Generate performance report for a model"""

        try:
            logger.info(f"Generating performance report for {model_name}")

            # Get metrics for the time period
            metrics = await self._get_metrics_for_period(model_name, start_time, end_time)

            if not metrics:
                return PerformanceReport(
                    model_name=model_name,
                    start_time=start_time,
                    end_time=end_time,
                    metrics_summary={},
                    alerts_summary=[],
                    generated_at=datetime.utcnow(),
                )

            # Calculate summary statistics
            metrics_summary = self._calculate_metrics_summary(metrics)

            # Get alerts for the period
            alerts_summary = await self._get_alerts_for_period(model_name, start_time, end_time)

            # Calculate model health score
            health_score = self._calculate_health_score(
                metrics_summary, alerts_summary)

            report = PerformanceReport(
                model_name=model_name,
                start_time=start_time,
                end_time=end_time,
                metrics_summary=metrics_summary,
                alerts_summary=alerts_summary,
                health_score=health_score,
                generated_at=datetime.utcnow(),
            )

            logger.info(f"Performance report generated for {model_name}")
            return report

        except Exception as e:
            logger.error(
                f"Failed to generate performance report for {model_name}: {str(e)}")
            raise MonitoringError(
                f"Performance report generation failed: {str(e)}")

    async def get_model_health(self, model_name: str) -> ModelHealth:
        """Get current model health status"""

        try:
            # Get recent metrics
            recent_metrics = await self._get_recent_metrics(model_name, hours=1)

            # Get active alerts
            active_alerts = await self._get_active_alerts(model_name)

            # Calculate health score
            health_score = self._calculate_health_score_from_metrics(
                recent_metrics)

            # Determine health status
            if health_score >= 0.9:
                status = "healthy"
            elif health_score >= 0.7:
                status = "degraded"
            else:
                status = "unhealthy"

            return ModelHealth(
                model_name=model_name,
                status=status,
                health_score=health_score,
                active_alerts=len(active_alerts),
                last_updated=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(
                f"Failed to get model health for {model_name}: {str(e)}")
            raise MonitoringError(f"Model health retrieval failed: {str(e)}")

    async def _monitor_model(
            self,
            model_name: str,
            config: MonitoringConfig) -> None:
        """Monitor a specific model"""

        while True:
            try:
                # Collect metrics
                metrics = await self._collect_model_metrics(model_name, config)

                # Log metrics
                await self.log_model_metrics(model_name, metrics)

                # Check for data drift
                if config.drift_detection_enabled:
    await self._check_data_drift(model_name, config)

                # Wait before next check
                await asyncio.sleep(config.monitoring_interval_seconds)

            except asyncio.CancelledError:
                logger.info(f"Model monitoring cancelled for {model_name}")
                break
            except Exception as e:
                logger.error(f"Error monitoring {model_name}: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _monitor_model_performance(
            self,
            model_name: str,
            config: MonitoringConfig) -> None:
        """Monitor model performance metrics"""

        while True:
            try:
                # Collect performance metrics
                performance_metrics = await self._collect_performance_metrics(model_name, config)

                # Log performance metrics
                await self.log_model_metrics(model_name, performance_metrics)

                # Wait before next check
                await asyncio.sleep(config.performance_monitoring_interval_seconds)

            except asyncio.CancelledError:
                logger.info(
                    f"Performance monitoring cancelled for {model_name}")
                break
            except Exception as e:
                logger.error(
                    f"Error in performance monitoring for {model_name}: {str(e)}")
                await asyncio.sleep(60)

    async def _collect_model_metrics(
        self, model_name: str, config: MonitoringConfig
    ) -> ModelMetrics:
        """Collect model metrics"""

        # This would implement actual metrics collection
        # For now, return dummy metrics
        return ModelMetrics(
            model_name=model_name,
            accuracy=0.85,
            latency_ms=100,
            throughput_rps=50,
            error_rate=0.02,
            timestamp=datetime.utcnow(),
        )

    async def _collect_performance_metrics(
        self, model_name: str, config: MonitoringConfig
    ) -> ModelMetrics:
        """Collect performance-specific metrics"""

        # This would implement performance metrics collection
        return ModelMetrics(
            model_name=model_name,
            accuracy=0.85,
            latency_ms=100,
            throughput_rps=50,
            error_rate=0.02,
            timestamp=datetime.utcnow(),
        )

    async def _check_metric_alerts(
            self,
            model_name: str,
            metrics: ModelMetrics) -> None:
        """Check if metrics trigger any alerts"""

        config = self._monitoring_configs.get(model_name)
        if not config:
            return

        # Check each alert rule
        for rule in config.alert_rules:
            if await self._evaluate_alert_rule(rule, metrics):
    await self._trigger_alert(model_name, rule, metrics)

    async def _evaluate_alert_rule(
            self,
            rule: AlertRule,
            metrics: ModelMetrics) -> bool:
        """Evaluate if an alert rule should trigger"""

        metric_value = getattr(metrics, rule.metric_name, None)
        if metric_value is None:
            return False

        if rule.operator == "greater_than":
            return metric_value > rule.threshold
        elif rule.operator == "less_than":
            return metric_value < rule.threshold
        elif rule.operator == "equals":
            return metric_value == rule.threshold
        elif rule.operator == "not_equals":
            return metric_value != rule.threshold

        return False

    async def _trigger_alert(
            self,
            model_name: str,
            rule: AlertRule,
            metrics: ModelMetrics) -> None:
        """Trigger an alert"""

        alert = PerformanceAlert(
            id=str(uuid4()),
            model_name=model_name,
            alert_type=rule.alert_type,
            severity=rule.severity,
            message=f"{rule.metric_name} {rule.operator} {rule.threshold}",
            metric_value=getattr(metrics, rule.metric_name),
            threshold=rule.threshold,
            triggered_at=datetime.utcnow(),
        )

        # Store alert
        self._active_alerts[alert.id] = alert

        # Send alert
        await self.alert_manager.send_alert(alert)

        logger.warning(f"Alert triggered for {model_name}: {alert.message}")

    async def _send_drift_alert(
            self, drift_result: DriftDetectionResult) -> None:
        """Send data drift alert"""

        alert = DriftAlert(
            id=str(
                uuid4()),
            model_name=drift_result.model_name,
            drift_score=drift_result.overall_drift_score,
            affected_features=len(
                drift_result.drift_alerts),
            message=f"Data drift detected: {drift_result.overall_drift_score:.3f}",
            triggered_at=datetime.utcnow(),
        )

        # Store alert
        self._active_alerts[alert.id] = alert

        # Send alert
        await self.alert_manager.send_alert(alert)

        logger.warning(f"Drift alert sent for {drift_result.model_name}")

    def _calculate_health_score(
        self, metrics_summary: Dict[str, Any], alerts_summary: List[Any]
    ) -> float:
        """Calculate model health score"""

        # Base score
        health_score = 1.0

        # Penalize for alerts
        alert_penalty = len(alerts_summary) * 0.1
        health_score -= alert_penalty

        # Penalize for poor metrics
        if "error_rate" in metrics_summary:
            error_rate = metrics_summary["error_rate"]
            if error_rate > 0.05:  # 5% error rate threshold
                health_score -= (error_rate - 0.05) * 2

        if "latency_ms" in metrics_summary:
            latency = metrics_summary["latency_ms"]
            if latency > 1000:  # 1 second latency threshold
                health_score -= (latency - 1000) / 10000

        return max(0.0, min(1.0, health_score))

    def _calculate_health_score_from_metrics(
            self, metrics: List[ModelMetrics]) -> float:
        """Calculate health score from recent metrics"""

        if not metrics:
            return 0.5  # Unknown health

        # Calculate average metrics
        avg_error_rate = statistics.mean([m.error_rate for m in metrics])
        avg_latency = statistics.mean([m.latency_ms for m in metrics])

        # Calculate health score
        health_score = 1.0
        health_score -= avg_error_rate * 5  # Error rate penalty
        health_score -= min(avg_latency / 1000, 0.5)  # Latency penalty

        return max(0.0, min(1.0, health_score))

    def _calculate_metrics_summary(
            self, metrics: List[ModelMetrics]) -> Dict[str, Any]:
    """Calculate summary statistics for metrics"""
        if not metrics:
            return {}

        return {
            "accuracy": statistics.mean([m.accuracy for m in metrics]),
            "latency_ms": statistics.mean([m.latency_ms for m in metrics]),
            "throughput_rps": statistics.mean([m.throughput_rps for m in metrics]),
            "error_rate": statistics.mean([m.error_rate for m in metrics]),
            "sample_count": len(metrics),
        }

    async def _get_metrics_for_period(
        self, model_name: str, start_time: datetime, end_time: datetime
    ) -> List[ModelMetrics]:
        """Get metrics for a specific time period"""

        if model_name not in self._metrics_cache:
            return []

        return [m for m in self._metrics_cache[model_name]
                if start_time <= m.timestamp <= end_time]

    async def _get_recent_metrics(
            self,
            model_name: str,
            hours: int) -> List[ModelMetrics]:
        """Get recent metrics for a model"""

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return await self._get_metrics_for_period(model_name, cutoff_time, datetime.utcnow())

    async def _get_alerts_for_period(
        self, model_name: str, start_time: datetime, end_time: datetime
    ) -> List[Any]:
        """Get alerts for a specific time period"""

        return [
            alert
            for alert in self._active_alerts.values()
            if (alert.model_name == model_name and start_time <= alert.triggered_at <= end_time)
        ]

    async def _get_active_alerts(self, model_name: str) -> List[Any]:
        """Get active alerts for a model"""

        return [
            alert
            for alert in self._active_alerts.values()
            if alert.model_name == model_name and alert.status == "active"
        ]

    async def _validate_monitoring_config(
            self, config: MonitoringConfig) -> None:
        """Validate monitoring configuration"""

        if not config.monitoring_interval_seconds:
            raise ValidationError("Monitoring interval is required")

        if not config.alert_rules:
            raise ValidationError("At least one alert rule is required")

    async def _setup_prometheus_metrics(
            self,
            model_name: str,
            config: MonitoringConfig) -> None:
        """Set up Prometheus metrics for a model"""

        await self.prometheus_client.setup_model_metrics(model_name, config)

    async def _create_monitoring_dashboard(
        self, model_name: str, config: MonitoringConfig
    ) -> MonitoringDashboard:
        """Create monitoring dashboard for a model"""

        dashboard = await self.grafana_client.create_model_dashboard(model_name, config)
        return dashboard

    async def _setup_alerting_rules(
            self,
            model_name: str,
            config: MonitoringConfig) -> None:
        """Set up alerting rules for a model"""

        await self.alert_manager.setup_model_alerts(model_name, config.alert_rules)

    async def _setup_pipeline_metrics(
            self,
            pipeline_id: str,
            config: MonitoringConfig) -> None:
        """Set up pipeline-specific metrics"""

        await self.prometheus_client.setup_pipeline_metrics(pipeline_id, config)

    async def _create_pipeline_dashboard(
            self,
            pipeline_id: str,
            config: MonitoringConfig) -> None:
        """Create pipeline monitoring dashboard"""

        await self.grafana_client.create_pipeline_dashboard(pipeline_id, config)

    async def _setup_deployment_metrics(
        self, deployment_id: str, endpoint_url: str, config: MonitoringConfig
    ) -> None:
        """Set up deployment-specific metrics"""

        await self.prometheus_client.setup_deployment_metrics(deployment_id, endpoint_url, config)

    async def _setup_health_checks(
            self,
            deployment_id: str,
            endpoint_url: str) -> None:
        """Set up health checks for deployment"""

        # Implement health check setup
        pass

    async def _setup_performance_monitoring(
        self, deployment_id: str, model_name: str, config: MonitoringConfig
    ) -> None:
        """Set up performance monitoring for deployment"""

        await self.prometheus_client.setup_performance_metrics(deployment_id, model_name, config)

    async def _check_data_drift(
            self,
            model_name: str,
            config: MonitoringConfig) -> None:
        """Check for data drift"""

        # This would implement actual drift detection
        # For now, it's a placeholder
        pass
