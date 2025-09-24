"""
Observability Engine - Central orchestrator for all monitoring components
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config.settings import ObservabilityConfig, ObservabilityStatus
from monitoring.alerting import AlertingSystem
from monitoring.chaos import ChaosEngine
from monitoring.cost import CostMonitor
from monitoring.health import HealthChecker
from monitoring.incidents import IncidentManager
from monitoring.logging import LogAggregator
from monitoring.metrics import MetricsCollector
from monitoring.runbooks import RunbookEngine
from monitoring.slo import SLOMonitor
from monitoring.tracing import TraceManager

logger = logging.getLogger(__name__)


@dataclass
class SystemHealthReport:
    """Comprehensive system health report"""

    overall_health_score: float
    service_health: Dict[str, Any]
    infrastructure_health: Dict[str, Any]
    pipeline_health: Dict[str, Any]
    ml_model_health: Dict[str, Any]
    dependency_health: Dict[str, Any]
    health_check_timestamp: datetime
    recommendations: List[str] = None


class ObservabilityEngine:
    """Comprehensive observability and monitoring system"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.trace_manager = TraceManager()
        self.log_aggregator = LogAggregator()
        self.alerting_system = AlertingSystem()
        self.runbook_engine = RunbookEngine()
        self.slo_monitor = SLOMonitor()
        self.incident_manager = IncidentManager()
        self.cost_monitor = CostMonitor()
        self.chaos_engine = ChaosEngine()
        self.health_checker = HealthChecker()

        self.monitoring_tasks = []
        self.is_initialized = False

    async def initialize_observability(self, config: ObservabilityConfig) -> ObservabilityStatus:
        """Initialize comprehensive observability stack"""

        logger.info("Initializing observability components...")

        initialization_tasks = [
            self.metrics_collector.initialize(config.metrics_config),
            self.trace_manager.initialize(config.tracing_config),
            self.log_aggregator.initialize(config.logging_config),
            self.alerting_system.initialize(config.alerting_config),
            self.runbook_engine.initialize(config.runbook_config),
            self.slo_monitor.initialize(config.slo_config),
            self.incident_manager.initialize(config.incident_config),
            self.cost_monitor.initialize(config.cost_config),
            self.chaos_engine.initialize(config.chaos_config),
        ]

        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)

        # Check initialization success
        failed_components = [result for result in results if isinstance(result, Exception)]

        if failed_components:
            logger.error(f"Failed to initialize components: {failed_components}")

        # Start monitoring loops
        await self.start_monitoring_loops()

        self.is_initialized = True

        return ObservabilityStatus(
            initialized_components=len([r for r in results if not isinstance(r, Exception)]),
            failed_components=len(failed_components),
            status="operational" if not failed_components else "degraded",
            initialization_timestamp=datetime.utcnow(),
        )

    async def start_monitoring_loops(self):
        """Start background monitoring tasks"""

        # Health monitoring loop
        self.monitoring_tasks.append(asyncio.create_task(self._health_monitoring_loop()))

        # Metrics collection loop
        self.monitoring_tasks.append(asyncio.create_task(self._metrics_collection_loop()))

        # Alert processing loop
        self.monitoring_tasks.append(asyncio.create_task(self._alert_processing_loop()))

        # SLO monitoring loop
        self.monitoring_tasks.append(asyncio.create_task(self._slo_monitoring_loop()))

        # Cost monitoring loop
        self.monitoring_tasks.append(asyncio.create_task(self._cost_monitoring_loop()))

        # Chaos engineering loop
        self.monitoring_tasks.append(asyncio.create_task(self._chaos_engineering_loop()))

        logger.info("Started monitoring loops")

    async def collect_system_health(self) -> SystemHealthReport:
        """Collect comprehensive system health metrics"""

        health_checks = await asyncio.gather(
            self.check_service_health(),
            self.check_infrastructure_health(),
            self.check_data_pipeline_health(),
            self.check_ml_model_health(),
            self.check_external_dependencies_health(),
            return_exceptions=True,
        )

        service_health, infra_health, pipeline_health, ml_health, deps_health = health_checks

        # Calculate overall health score
        overall_health = self.calculate_overall_health_score(
            [health for health in health_checks if not isinstance(health, Exception)]
        )

        # Generate recommendations
        recommendations = await self.generate_health_recommendations(health_checks)

        return SystemHealthReport(
            overall_health_score=overall_health,
            service_health=service_health if not isinstance(service_health, Exception) else {},
            infrastructure_health=infra_health if not isinstance(infra_health, Exception) else {},
            pipeline_health=pipeline_health if not isinstance(pipeline_health, Exception) else {},
            ml_model_health=ml_health if not isinstance(ml_health, Exception) else {},
            dependency_health=deps_health if not isinstance(deps_health, Exception) else {},
            health_check_timestamp=datetime.utcnow(),
            recommendations=recommendations,
        )

    async def check_service_health(self) -> Dict[str, Any]:
        """Check health of all microservices"""
        return await self.health_checker.check_all_services()

    async def check_infrastructure_health(self) -> Dict[str, Any]:
        """Check infrastructure health (Kubernetes, databases, etc.)"""
        return await self.health_checker.check_infrastructure()

    async def check_data_pipeline_health(self) -> Dict[str, Any]:
        """Check data pipeline health"""
        return await self.health_checker.check_data_pipeline()

    async def check_ml_model_health(self) -> Dict[str, Any]:
        """Check ML model health and performance"""
        return await self.health_checker.check_ml_models()

    async def check_external_dependencies_health(self) -> Dict[str, Any]:
        """Check external dependencies health"""
        return await self.health_checker.check_external_dependencies()

    def calculate_overall_health_score(self, health_checks: List[Dict[str, Any]]) -> float:
        """Calculate overall system health score"""
        if not health_checks:
            return 0.0

        total_score = 0.0
        valid_checks = 0

        for health_check in health_checks:
            if isinstance(health_check, dict) and "health_score" in health_check:
                total_score += health_check["health_score"]
                valid_checks += 1

        return total_score / valid_checks if valid_checks > 0 else 0.0

    async def generate_health_recommendations(self, health_checks: List[Any]) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []

        for health_check in health_checks:
            if isinstance(health_check, dict) and "recommendations" in health_check:
                recommendations.extend(health_check["recommendations"])

        return list(set(recommendations))  # Remove duplicates

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self.is_initialized:
            return {"status": "not_initialized"}

        try:
            # Get health report
            health_report = await self.collect_system_health()

            # Get metrics summary
            metrics_summary = await self.metrics_collector.get_metrics_summary()

            # Get active incidents
            active_incidents = await self.incident_manager.get_active_incidents()

            # Get SLO status
            slo_status = await self.slo_monitor.get_overall_slo_status()

            return {
                "status": "operational",
                "health_score": health_report.overall_health_score,
                "metrics_summary": metrics_summary,
                "active_incidents": len(active_incidents),
                "slo_status": slo_status,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get system status: {str(e)}")
            return {"status": "error", "error": str(e), "timestamp": datetime.utcnow().isoformat()}

    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while True:
            try:
                health_report = await self.collect_system_health()

                # Check for critical health issues
                if health_report.overall_health_score < 0.7:
                    await self.alerting_system.create_alert(
                        {
                            "type": "system_health_critical",
                            "severity": "critical",
                            "message": f"System health score is {health_report.overall_health_score}",
                            "metadata": health_report.__dict__,
                        }
                    )

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Health monitoring loop error: {str(e)}")
                await asyncio.sleep(60)

    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while True:
            try:
                await self.metrics_collector.collect_real_time_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds

            except Exception as e:
                logger.error(f"Metrics collection loop error: {str(e)}")
                await asyncio.sleep(30)

    async def _alert_processing_loop(self):
        """Background alert processing loop"""
        while True:
            try:
                await self.alerting_system.process_pending_alerts()
                await asyncio.sleep(10)  # Process every 10 seconds

            except Exception as e:
                logger.error(f"Alert processing loop error: {str(e)}")
                await asyncio.sleep(10)

    async def _slo_monitoring_loop(self):
        """Background SLO monitoring loop"""
        while True:
            try:
                await self.slo_monitor.monitor_all_slos()
                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"SLO monitoring loop error: {str(e)}")
                await asyncio.sleep(300)

    async def _cost_monitoring_loop(self):
        """Background cost monitoring loop"""
        while True:
            try:
                await self.cost_monitor.collect_cost_metrics()
                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Cost monitoring loop error: {str(e)}")
                await asyncio.sleep(3600)

    async def _chaos_engineering_loop(self):
        """Background chaos engineering loop"""
        while True:
            try:
                await self.chaos_engine.run_chaos_experiments()
                await asyncio.sleep(1800)  # Run every 30 minutes

            except Exception as e:
                logger.error(f"Chaos engineering loop error: {str(e)}")
                await asyncio.sleep(1800)

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up observability engine...")

        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)

        # Cleanup components
        if hasattr(self.metrics_collector, "cleanup"):
            await self.metrics_collector.cleanup()

        if hasattr(self.trace_manager, "cleanup"):
            await self.trace_manager.cleanup()

        if hasattr(self.log_aggregator, "cleanup"):
            await self.log_aggregator.cleanup()

        logger.info("Observability engine cleanup completed")
