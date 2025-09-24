"""
Chaos engineering and reliability testing
"""

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ChaosExperimentType(Enum):
    """Chaos experiment types"""

    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_STRESS = "disk_stress"
    SERVICE_FAILURE = "service_failure"
    DATABASE_FAILURE = "database_failure"
    CACHE_FAILURE = "cache_failure"
    RANDOM_KILL = "random_kill"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class ExperimentStatus(Enum):
    """Experiment status"""

    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentSeverity(Enum):
    """Experiment severity"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChaosExperiment:
    """Chaos experiment definition"""

    id: str
    name: str
    description: str
    experiment_type: ChaosExperimentType
    severity: ExperimentSeverity
    target_services: List[str]
    parameters: Dict[str, Any]
    duration_minutes: int
    schedule: Optional[str] = None  # Cron expression
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"


@dataclass
class ExperimentExecution:
    """Chaos experiment execution"""

    id: str
    experiment_id: str
    status: ExperimentStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    executed_by: str = "system"
    results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_during: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReliabilityReport:
    """Reliability testing report"""

    id: str
    experiment_id: str
    execution_id: str
    overall_reliability_score: float
    service_impact: Dict[str, Any]
    recovery_time: float
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.utcnow)


class ChaosEngine:
    """Main chaos engineering engine"""

    def __init__(self):
        self.experiments = {}
        self.executions = {}
        self.active_experiments = set()
        self.is_initialized = False

    async def initialize(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Initialize chaos engineering"""

        # Load experiment definitions
        await self.load_experiment_definitions(config.get("experiments", []) if config else [])

        self.is_initialized = True
        logger.info("Chaos engine initialized")

    async def load_experiment_definitions(
            self, experiments: List[Dict[str, Any]]):
         -> Dict[str, Any]:"""Load chaos experiment definitions"""

        for exp_data in experiments:
            experiment = ChaosExperiment(
                id=exp_data["id"],
                name=exp_data["name"],
                description=exp_data["description"],
                experiment_type=ChaosExperimentType(exp_data["type"]),
                severity=ExperimentSeverity(exp_data["severity"]),
                target_services=exp_data["target_services"],
                parameters=exp_data["parameters"],
                duration_minutes=exp_data["duration_minutes"],
                schedule=exp_data.get("schedule"),
                enabled=exp_data.get("enabled", True),
            )
            self.experiments[experiment.id] = experiment

    async def run_chaos_experiments(self) -> Dict[str, Any]:
        """Run scheduled chaos experiments"""

        for experiment in self.experiments.values():
            if not experiment.enabled:
                continue

            # Check if experiment should run based on schedule
            if await self._should_run_experiment(experiment):
                try:
    await self.execute_experiment(experiment)
                except Exception as e:
                    logger.error(
                        f"Failed to execute experiment {experiment.id}: {str(e)}")

    async def _should_run_experiment(
            self, experiment: ChaosExperiment) -> bool:
        """Check if experiment should run based on schedule"""

        if not experiment.schedule:
            return False

        # Simple schedule check - in practice, you'd use a proper cron parser
        # For now, run experiments randomly with low probability
        return random.random() < 0.1  # 10% chance per check

    async def execute_experiment(
            self, experiment: ChaosExperiment) -> ExperimentExecution:
        """Execute a chaos experiment"""

        execution = ExperimentExecution(
            id=f"{experiment.id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            experiment_id=experiment.id,
            status=ExperimentStatus.RUNNING,
            started_at=datetime.utcnow(),
            executed_by="system",
        )

        self.executions[execution.id] = execution
        self.active_experiments.add(experiment.id)

        try:
            logger.info(f"Starting chaos experiment: {experiment.name}")

            # Collect metrics before experiment
            execution.metrics_before = await self._collect_system_metrics()

            # Execute experiment based on type
            if experiment.experiment_type == ChaosExperimentType.NETWORK_LATENCY:
    await self._execute_network_latency_experiment(experiment, execution)
            elif experiment.experiment_type == ChaosExperimentType.CPU_STRESS:
    await self._execute_cpu_stress_experiment(experiment, execution)
            elif experiment.experiment_type == ChaosExperimentType.MEMORY_STRESS:
    await self._execute_memory_stress_experiment(experiment, execution)
            elif experiment.experiment_type == ChaosExperimentType.SERVICE_FAILURE:
    await self._execute_service_failure_experiment(experiment, execution)
            elif experiment.experiment_type == ChaosExperimentType.DATABASE_FAILURE:
    await self._execute_database_failure_experiment(experiment, execution)
            elif experiment.experiment_type == ChaosExperimentType.RANDOM_KILL:
    await self._execute_random_kill_experiment(experiment, execution)
            else:
                raise ValueError(
                    f"Unknown experiment type: {experiment.experiment_type}")

            # Collect metrics during experiment
            execution.metrics_during = await self._collect_system_metrics()

            # Wait for experiment duration
            await asyncio.sleep(experiment.duration_minutes * 60)

            # Collect metrics after experiment
            execution.metrics_after = await self._collect_system_metrics()

            # Complete experiment
            execution.status = ExperimentStatus.COMPLETED
            execution.completed_at = datetime.utcnow()

            # Generate reliability report
            report = await self._generate_reliability_report(experiment, execution)
            execution.results = report.__dict__

            logger.info(f"Completed chaos experiment: {experiment.name}")

        except Exception as e:
            execution.status = ExperimentStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            logger.error(
                f"Chaos experiment failed: {experiment.name} - {str(e)}")

        finally:
            self.active_experiments.discard(experiment.id)

        return execution

    async def _execute_network_latency_experiment(
        self, experiment: ChaosExperiment, execution: ExperimentExecution
    ):
         -> Dict[str, Any]:"""Execute network latency experiment"""

        latency_ms = experiment.parameters.get("latency_ms", 1000)
        target_services = experiment.target_services

        logger.info(
            f"Simulating network latency of {latency_ms}ms for services: {target_services}")

        # This would integrate with network simulation tools
        # For now, just log the action
        execution.results["network_latency_ms"] = latency_ms
        execution.results["affected_services"] = target_services

    async def _execute_cpu_stress_experiment(
        self, experiment: ChaosExperiment, execution: ExperimentExecution
    ):
         -> Dict[str, Any]:"""Execute CPU stress experiment"""

        cpu_percentage = experiment.parameters.get("cpu_percentage", 80)
        duration_minutes = experiment.parameters.get("duration_minutes", 5)

        logger.info(
            f"Simulating CPU stress of {cpu_percentage}% for {duration_minutes} minutes")

        # This would integrate with CPU stress tools
        # For now, just log the action
        execution.results["cpu_percentage"] = cpu_percentage
        execution.results["duration_minutes"] = duration_minutes

    async def _execute_memory_stress_experiment(
        self, experiment: ChaosExperiment, execution: ExperimentExecution
    ):
         -> Dict[str, Any]:"""Execute memory stress experiment"""

        memory_percentage = experiment.parameters.get("memory_percentage", 80)
        duration_minutes = experiment.parameters.get("duration_minutes", 5)

        logger.info(
            f"Simulating memory stress of {memory_percentage}% for {duration_minutes} minutes"
        )

        # This would integrate with memory stress tools
        # For now, just log the action
        execution.results["memory_percentage"] = memory_percentage
        execution.results["duration_minutes"] = duration_minutes

    async def _execute_service_failure_experiment(
        self, experiment: ChaosExperiment, execution: ExperimentExecution
    ):
         -> Dict[str, Any]:"""Execute service failure experiment"""

        target_services = experiment.target_services
        failure_duration_minutes = experiment.parameters.get(
            "failure_duration_minutes", 2)

        logger.info(
            f"Simulating failure of services: {target_services} for {failure_duration_minutes} minutes"
        )

        # This would integrate with service orchestration tools
        # For now, just log the action
        execution.results["failed_services"] = target_services
        execution.results["failure_duration_minutes"] = failure_duration_minutes

    async def _execute_database_failure_experiment(
        self, experiment: ChaosExperiment, execution: ExperimentExecution
    ):
         -> Dict[str, Any]:"""Execute database failure experiment"""

        database_type = experiment.parameters.get("database_type", "primary")
        failure_duration_minutes = experiment.parameters.get(
            "failure_duration_minutes", 1)

        logger.info(
            f"Simulating {database_type} database failure for {failure_duration_minutes} minutes"
        )

        # This would integrate with database management tools
        # For now, just log the action
        execution.results["database_type"] = database_type
        execution.results["failure_duration_minutes"] = failure_duration_minutes

    async def _execute_random_kill_experiment(
        self, experiment: ChaosExperiment, execution: ExperimentExecution
    ):
         -> Dict[str, Any]:"""Execute random kill experiment"""

        target_services = experiment.target_services
        kill_percentage = experiment.parameters.get("kill_percentage", 50)

        # Randomly select services to kill
        num_to_kill = max(1, int(len(target_services) * kill_percentage / 100))
        services_to_kill = random.sample(
            target_services, min(
                num_to_kill, len(target_services)))

        logger.info(f"Randomly killing services: {services_to_kill}")

        # This would integrate with container orchestration tools
        # For now, just log the action
        execution.results["killed_services"] = services_to_kill
        execution.results["kill_percentage"] = kill_percentage

    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics for experiment analysis"""
        try:
            # This would collect actual system metrics
            # For now, return mock data

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu_usage": random.uniform(20, 80),
                "memory_usage": random.uniform(30, 70),
                "disk_usage": random.uniform(40, 90),
                "network_latency": random.uniform(1, 100),
                "active_connections": random.randint(100, 1000),
                "request_rate": random.uniform(50, 500),
            }

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            return {}

    async def _generate_reliability_report(
        self, experiment: ChaosExperiment, execution: ExperimentExecution
    ) -> ReliabilityReport:
        """Generate reliability report for experiment"""

        # Calculate reliability score based on metrics
        reliability_score = self._calculate_reliability_score(
            execution.metrics_before, execution.metrics_during, execution.metrics_after)

        # Calculate service impact
        service_impact = self._calculate_service_impact(execution)

        # Calculate recovery time
        recovery_time = self._calculate_recovery_time(execution)

        # Generate recommendations
        recommendations = self._generate_reliability_recommendations(
            experiment, execution, reliability_score
        )

        report = ReliabilityReport(
            id=f"report_{execution.id}",
            experiment_id=experiment.id,
            execution_id=execution.id,
            overall_reliability_score=reliability_score,
            service_impact=service_impact,
            recovery_time=recovery_time,
            recommendations=recommendations,
        )

        return report

    def _calculate_reliability_score(
        self,
        metrics_before: Dict[str, Any],
        metrics_during: Dict[str, Any],
        metrics_after: Dict[str, Any],
    ) -> float:
        """Calculate overall reliability score"""

        # Simple reliability calculation
        # In practice, this would be more sophisticated

        if not metrics_before or not metrics_during or not metrics_after:
            return 0.0

        # Calculate degradation during experiment
        cpu_degradation = abs(
            metrics_during.get(
                "cpu_usage",
                0) -
            metrics_before.get(
                "cpu_usage",
                0))
        memory_degradation = abs(
            metrics_during.get(
                "memory_usage",
                0) -
            metrics_before.get(
                "memory_usage",
                0))

        # Calculate recovery
        cpu_recovery = abs(
            metrics_after.get(
                "cpu_usage",
                0) -
            metrics_before.get(
                "cpu_usage",
                0))
        memory_recovery = abs(
            metrics_after.get(
                "memory_usage",
                0) -
            metrics_before.get(
                "memory_usage",
                0))

        # Simple scoring (0-1, higher is better)
        degradation_score = max(
            0, 1 - (cpu_degradation + memory_degradation) / 200)
        recovery_score = max(0, 1 - (cpu_recovery + memory_recovery) / 100)

        return (degradation_score + recovery_score) / 2

    def _calculate_service_impact(
            self, execution: ExperimentExecution) -> Dict[str, Any]:
    """Calculate service impact during experiment"""
        # This would analyze actual service impact
        # For now, return mock data

        return {
            "services_affected": len(
                execution.results.get(
                    "affected_services", [])), "availability_drop": random.uniform(
                0, 0.3), "response_time_increase": random.uniform(
                    0, 2.0), "error_rate_increase": random.uniform(
                        0, 0.1), }

    def _calculate_recovery_time(
            self, execution: ExperimentExecution) -> float:
        """Calculate recovery time in minutes"""

        if not execution.completed_at or not execution.started_at:
            return 0.0

        total_duration = (execution.completed_at -
                          execution.started_at).total_seconds() / 60

        # Assume recovery takes 20% of total experiment time
        return total_duration * 0.2

    def _generate_reliability_recommendations(
            self,
            experiment: ChaosExperiment,
            execution: ExperimentExecution,
            reliability_score: float) -> List[str]:
        """Generate reliability improvement recommendations"""

        recommendations = []

        if reliability_score < 0.5:
            recommendations.append(
                "System showed poor resilience during chaos experiment")
            recommendations.append(
                "Implement better error handling and circuit breakers")
            recommendations.append("Add redundancy and failover mechanisms")
        elif reliability_score < 0.8:
            recommendations.append("System showed moderate resilience")
            recommendations.append(
                "Consider improving monitoring and alerting")
            recommendations.append(
                "Review resource allocation and scaling policies")
        else:
            recommendations.append(
                "System showed good resilience during chaos experiment")
            recommendations.append(
                "Continue regular chaos testing to maintain reliability")

        # Add experiment-specific recommendations
        if experiment.experiment_type == ChaosExperimentType.NETWORK_LATENCY:
            recommendations.append("Review network timeout configurations")
            recommendations.append("Implement network retry policies")
        elif experiment.experiment_type == ChaosExperimentType.SERVICE_FAILURE:
            recommendations.append(
                "Implement service discovery and health checks")
            recommendations.append("Add automatic failover mechanisms")

        return recommendations

    async def create_experiment(
            self, experiment_data: Dict[str, Any]) -> ChaosExperiment:
        """Create new chaos experiment"""

        experiment = ChaosExperiment(
            id=experiment_data["id"],
            name=experiment_data["name"],
            description=experiment_data["description"],
            experiment_type=ChaosExperimentType(experiment_data["type"]),
            severity=ExperimentSeverity(experiment_data["severity"]),
            target_services=experiment_data["target_services"],
            parameters=experiment_data["parameters"],
            duration_minutes=experiment_data["duration_minutes"],
            schedule=experiment_data.get("schedule"),
            enabled=experiment_data.get("enabled", True),
            created_by=experiment_data.get("created_by", "system"),
        )

        self.experiments[experiment.id] = experiment
        logger.info(f"Created chaos experiment: {experiment.name}")

        return experiment

    async def get_experiment_results(
            self, experiment_id: str) -> List[ExperimentExecution]:
        """Get results for a specific experiment"""

        return [
            execution
            for execution in self.executions.values()
            if execution.experiment_id == experiment_id
        ]

    async def get_reliability_summary(self) -> Dict[str, Any]:
        """Get overall reliability summary"""
        completed_executions = [
            e for e in self.executions.values() if e.status == ExperimentStatus.COMPLETED]

        if not completed_executions:
            return {
                "total_experiments": 0,
                "average_reliability_score": 0.0,
                "reliability_trend": "unknown",
            }

        # Calculate average reliability score
        reliability_scores = []
        for execution in completed_executions:
            if "overall_reliability_score" in execution.results:
                reliability_scores.append(
                    execution.results["overall_reliability_score"])

        avg_reliability = (
            sum(reliability_scores) /
            len(reliability_scores) if reliability_scores else 0.0)

        # Calculate trend (simplified)
        recent_executions = sorted(
            completed_executions, key=lambda x: x.started_at)[-5:]
        recent_scores = [
            e.results.get("overall_reliability_score", 0)
            for e in recent_executions
            if "overall_reliability_score" in e.results
        ]

        if len(recent_scores) >= 2:
            trend = "improving" if recent_scores[-1] > recent_scores[0] else "declining"
        else:
            trend = "stable"

        return {
            "total_experiments": len(completed_executions),
            "average_reliability_score": avg_reliability,
            "reliability_trend": trend,
            "last_experiment": (
                completed_executions[-1].started_at.isoformat() if completed_executions else None
            ),
        }

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup chaos engine"""
        self.is_initialized = False
        logger.info("Chaos engine cleaned up")
