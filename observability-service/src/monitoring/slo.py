"""
SLI/SLO monitoring and error budget tracking
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SLOStatus(Enum):
    """SLO status"""

    HEALTHY = "healthy"
    WARNING = "warning"
    BREACHED = "breached"
    UNKNOWN = "unknown"


class SLIType(Enum):
    """SLI types"""

    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CUSTOM = "custom"


@dataclass
class SLIDefinition:
    """SLI definition"""

    id: str
    name: str
    description: str
    sli_type: SLIType
    query: str
    evaluation_window: int  # seconds
    target_percentage: float  # 0-100
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SLODefinition:
    """SLO definition"""

    id: str
    name: str
    description: str
    sli_definitions: List[SLIDefinition]
    target_percentage: float  # 0-100
    evaluation_window: int  # seconds
    error_budget_policy: Dict[str, Any]
    alerting_thresholds: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.utcnow)
    enabled: bool = True


@dataclass
class SLIResult:
    """SLI calculation result"""

    sli_id: str
    value: float
    target: float
    status: SLOStatus
    evaluation_window: int
    calculated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLOStatus:
    """SLO status result"""

    slo_id: str
    overall_status: SLOStatus
    sli_results: List[SLIResult]
    error_budget_remaining: float
    error_budget_consumed: float
    burn_rate: float
    time_to_breach: Optional[timedelta]
    calculated_at: datetime
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ErrorBudget:
    """Error budget tracking"""

    slo_id: str
    total_budget: float
    consumed_budget: float
    remaining_budget: float
    burn_rate: float
    time_to_exhaustion: Optional[timedelta]
    last_updated: datetime


class SLICalculator:
    """Calculate SLI values from metrics"""

    def __init__(self):
        self.metrics_storage = {}

    async def calculate_sli(
        self, sli_definition: SLIDefinition, time_window: timedelta
    ) -> SLIResult:
        """Calculate SLI value for given time window"""

        try:
            if sli_definition.sli_type == SLIType.AVAILABILITY:
                value = await self._calculate_availability_sli(sli_definition, time_window)
            elif sli_definition.sli_type == SLIType.LATENCY:
                value = await self._calculate_latency_sli(sli_definition, time_window)
            elif sli_definition.sli_type == SLIType.ERROR_RATE:
                value = await self._calculate_error_rate_sli(sli_definition, time_window)
            elif sli_definition.sli_type == SLIType.THROUGHPUT:
                value = await self._calculate_throughput_sli(sli_definition, time_window)
        else:
                value = await self._calculate_custom_sli(sli_definition, time_window)

            # Determine status
            status = self._determine_sli_status(
                value, sli_definition.target_percentage)

            return SLIResult(
                sli_id=sli_definition.id,
                value=value,
                target=sli_definition.target_percentage,
                status=status,
                evaluation_window=time_window.total_seconds(),
                calculated_at=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(
                f"Failed to calculate SLI {sli_definition.id}: {str(e)}")
            return SLIResult(
                sli_id=sli_definition.id,
                value=0.0,
                target=sli_definition.target_percentage,
                status=SLOStatus.UNKNOWN,
                evaluation_window=time_window.total_seconds(),
                calculated_at=datetime.utcnow(),
                metadata={"error": str(e)},
            )

    async def _calculate_availability_sli(
        self, sli_definition: SLIDefinition, time_window: timedelta
    ) -> float:
        """Calculate availability SLI"""

        # This would query actual metrics data
        # For now, return mock data

        # Simulate availability calculation
        total_requests = 1000
        successful_requests = 950  # 95% availability

        return (successful_requests / total_requests) * 100

    async def _calculate_latency_sli(
        self, sli_definition: SLIDefinition, time_window: timedelta
    ) -> float:
        """Calculate latency SLI (percentage of requests under threshold)"""

        # This would query actual latency metrics
        # For now, return mock data

        # Simulate latency calculation
        total_requests = 1000
        requests_under_threshold = 900  # 90% under threshold

        return (requests_under_threshold / total_requests) * 100

    async def _calculate_error_rate_sli(
        self, sli_definition: SLIDefinition, time_window: timedelta
    ) -> float:
        """Calculate error rate SLI (percentage of successful requests)"""

        # This would query actual error metrics
        # For now, return mock data

        # Simulate error rate calculation
        total_requests = 1000
        successful_requests = 980  # 98% success rate

        return (successful_requests / total_requests) * 100

    async def _calculate_throughput_sli(
        self, sli_definition: SLIDefinition, time_window: timedelta
    ) -> float:
        """Calculate throughput SLI"""

        # This would query actual throughput metrics
        # For now, return mock data

        # Simulate throughput calculation
        target_throughput = 1000  # requests per second
        actual_throughput = 950  # actual requests per second

        return min((actual_throughput / target_throughput) * 100, 100)

    async def _calculate_custom_sli(
        self, sli_definition: SLIDefinition, time_window: timedelta
    ) -> float:
        """Calculate custom SLI"""

        # This would execute custom query
        # For now, return mock data

        return 95.0  # Mock value

    def _determine_sli_status(self, value: float, target: float) -> SLOStatus:
        """Determine SLI status based on value and target"""

        if value >= target:
            return SLOStatus.HEALTHY
        elif value >= target * 0.9:  # 90% of target
            return SLOStatus.WARNING
        else:
            return SLOStatus.BREACHED


class ErrorBudgetManager:
    """Manage error budgets for SLOs"""

    def __init__(self):
        self.error_budgets = {}

    async def calculate_error_budget(
        self, slo_definition: SLODefinition, sli_results: List[SLIResult]
    ) -> ErrorBudget:
        """Calculate error budget for SLO"""

        # Calculate overall SLO performance
        overall_performance = self._calculate_overall_performance(sli_results)

        # Calculate error budget
        total_budget = 100.0 - slo_definition.target_percentage
        consumed_budget = max(
            0,
            slo_definition.target_percentage -
            overall_performance)
        remaining_budget = max(0, total_budget - consumed_budget)

        # Calculate burn rate
        burn_rate = self._calculate_burn_rate(
            consumed_budget, slo_definition.evaluation_window)

        # Calculate time to exhaustion
        time_to_exhaustion = self._calculate_time_to_exhaustion(
            remaining_budget, burn_rate)

        error_budget = ErrorBudget(
            slo_id=slo_definition.id,
            total_budget=total_budget,
            consumed_budget=consumed_budget,
            remaining_budget=remaining_budget,
            burn_rate=burn_rate,
            time_to_exhaustion=time_to_exhaustion,
            last_updated=datetime.utcnow(),
        )

        self.error_budgets[slo_definition.id] = error_budget
        return error_budget

    def _calculate_overall_performance(
            self, sli_results: List[SLIResult]) -> float:
        """Calculate overall SLO performance from SLI results"""

        if not sli_results:
            return 0.0

        # Simple average - in practice, you might weight different SLIs
        return sum(result.value for result in sli_results) / len(sli_results)

    def _calculate_burn_rate(
            self,
            consumed_budget: float,
            evaluation_window: int) -> float:
        """Calculate error budget burn rate"""

        # Burn rate as percentage per hour
        hours_in_window = evaluation_window / 3600
        return consumed_budget / hours_in_window if hours_in_window > 0 else 0

    def _calculate_time_to_exhaustion(
        self, remaining_budget: float, burn_rate: float
    ) -> Optional[timedelta]:
        """Calculate time until error budget is exhausted"""

        if burn_rate <= 0:
            return None

        hours_to_exhaustion = remaining_budget / burn_rate
        return timedelta(hours=hours_to_exhaustion)


class BurnRateAlerting:
    """Alert on error budget burn rate"""

    def __init__(self):
        self.alerting_rules = {}

    async def check_burn_rate_alerts(
        self, error_budget: ErrorBudget, slo_definition: SLODefinition
    ) -> List[Dict[str, Any]]:
        """Check for burn rate alerts"""

        alerts = []

        # Check burn rate thresholds
        burn_rate_thresholds = slo_definition.alerting_thresholds.get(
            "burn_rate", {})

        for threshold_name, threshold_value in burn_rate_thresholds.items():
            if error_budget.burn_rate >= threshold_value:
                alerts.append(
                    {
                        "type": "burn_rate_alert",
                        "severity": self._get_threshold_severity(threshold_name),
                        "message": f"Error budget burn rate {error_budget.burn_rate:.2f}% exceeds {threshold_name} threshold {threshold_value}%",
                        "slo_id": error_budget.slo_id,
                        "burn_rate": error_budget.burn_rate,
                        "threshold": threshold_value,
                        "time_to_exhaustion": error_budget.time_to_exhaustion,
                    })

        return alerts

    def _get_threshold_severity(self, threshold_name: str) -> str:
        """Get severity for threshold name"""

        if "critical" in threshold_name.lower():
            return "critical"
        elif "warning" in threshold_name.lower():
            return "warning"
        else:
            return "info"


class SLODashboard:
    """SLO dashboard and reporting"""

    def __init__(self):
        self.dashboard_data = {}

    async def generate_dashboard_data(
        self,
        slo_definitions: List[SLODefinition],
        sli_results: List[SLIResult],
        error_budgets: List[ErrorBudget],
    ) -> Dict[str, Any]:
    """Generate dashboard data for SLOs"""
        dashboard = {
            "overview":
    await self._generate_overview(slo_definitions, sli_results, error_budgets),
            "slo_details":
    await self._generate_slo_details(
                slo_definitions, sli_results, error_budgets
            ),
            "trends":
    await self._generate_trends(slo_definitions),
            "recommendations":
    await self._generate_recommendations(
                slo_definitions, sli_results, error_budgets
            ),
            "generated_at": datetime.utcnow().isoformat(),
        }

        return dashboard

    async def _generate_overview(
        self,
        slo_definitions: List[SLODefinition],
        sli_results: List[SLIResult],
        error_budgets: List[ErrorBudget],
    ) -> Dict[str, Any]:
    """Generate overview data"""
        total_slos = len(slo_definitions)
        healthy_slos = len(
            [slo for slo in slo_definitions if self._is_slo_healthy(slo, sli_results)]
        )
        warning_slos = len(
            [slo for slo in slo_definitions if self._is_slo_warning(slo, sli_results)]
        )
        breached_slos = total_slos - healthy_slos - warning_slos

        return {
            "total_slos": total_slos,
            "healthy_slos": healthy_slos,
            "warning_slos": warning_slos,
            "breached_slos": breached_slos,
            "health_percentage": (
                healthy_slos /
                total_slos *
                100) if total_slos > 0 else 0,
        }

    async def _generate_slo_details(
        self,
        slo_definitions: List[SLODefinition],
        sli_results: List[SLIResult],
        error_budgets: List[ErrorBudget],
    ) -> List[Dict[str, Any]]:
        """Generate detailed SLO data"""

        details = []

        for slo in slo_definitions:
            slo_sli_results = [
                r for r in sli_results if r.sli_id in [
                    sli.id for sli in slo.sli_definitions]]
            slo_error_budget = next(
                (eb for eb in error_budgets if eb.slo_id == slo.id), None)

            details.append(
                {
                    "slo_id": slo.id,
                    "name": slo.name,
                    "target_percentage": slo.target_percentage,
                    "current_performance": self._calculate_overall_performance(slo_sli_results),
                    "status": self._determine_slo_status(slo, slo_sli_results),
                    "error_budget_remaining": (
                        slo_error_budget.remaining_budget if slo_error_budget else 0
                    ),
                    "burn_rate": slo_error_budget.burn_rate if slo_error_budget else 0,
                    "time_to_exhaustion": (
                        slo_error_budget.time_to_exhaustion.isoformat()
                        if slo_error_budget and slo_error_budget.time_to_exhaustion
                        else None
                    ),
                }
            )

        return details

    async def _generate_trends(
            self, slo_definitions: List[SLODefinition]) -> Dict[str, Any]:
    """Generate trend data"""
        # This would query historical data
        # For now, return mock trends

        return {
            "performance_trend": "stable",
            "error_budget_trend": "decreasing",
            "burn_rate_trend": "increasing",
        }

    async def _generate_recommendations(
        self,
        slo_definitions: List[SLODefinition],
        sli_results: List[SLIResult],
        error_budgets: List[ErrorBudget],
    ) -> List[str]:
        """Generate recommendations for SLO improvement"""

        recommendations = []

        for slo in slo_definitions:
            slo_sli_results = [
                r for r in sli_results if r.sli_id in [
                    sli.id for sli in slo.sli_definitions]]
            slo_error_budget = next(
                (eb for eb in error_budgets if eb.slo_id == slo.id), None)

            if slo_error_budget and slo_error_budget.burn_rate > 10:
                recommendations.append(
                    f"High burn rate for {slo.name}: {slo_error_budget.burn_rate:.2f}%/hour"
                )

            if (
                slo_error_budget
                and slo_error_budget.time_to_exhaustion
                and slo_error_budget.time_to_exhaustion < timedelta(hours=24)
            ):
                recommendations.append(
                    f"Error budget for {slo.name} will be exhausted in {slo_error_budget.time_to_exhaustion}"
                )

            failed_slis = [
                r for r in slo_sli_results if r.status == SLOStatus.BREACHED]
            if failed_slis:
                recommendations.append(
                    f"SLO {slo.name} has {len(failed_slis)} failed SLIs")

        return recommendations

    def _is_slo_healthy(
            self,
            slo: SLODefinition,
            sli_results: List[SLIResult]) -> bool:
        """Check if SLO is healthy"""
        slo_sli_results = [
            r for r in sli_results if r.sli_id in [
                sli.id for sli in slo.sli_definitions]]
        return all(
            result.status == SLOStatus.HEALTHY for result in slo_sli_results)

    def _is_slo_warning(
            self,
            slo: SLODefinition,
            sli_results: List[SLIResult]) -> bool:
        """Check if SLO is in warning state"""
        slo_sli_results = [
            r for r in sli_results if r.sli_id in [
                sli.id for sli in slo.sli_definitions]]
        return any(
            result.status == SLOStatus.WARNING for result in slo_sli_results)

    def _calculate_overall_performance(
            self, sli_results: List[SLIResult]) -> float:
        """Calculate overall performance from SLI results"""
        if not sli_results:
            return 0.0
        return sum(result.value for result in sli_results) / len(sli_results)

    def _determine_slo_status(
            self,
            slo: SLODefinition,
            sli_results: List[SLIResult]) -> str:
        """Determine SLO status"""
        if not sli_results:
            return "unknown"

        if all(result.status == SLOStatus.HEALTHY for result in sli_results):
            return "healthy"
        elif any(result.status == SLOStatus.BREACHED for result in sli_results):
            return "breached"
        else:
            return "warning"


class SLOMonitor:
    """Main SLO monitoring system"""

    def __init__(self):
        self.sli_calculator = SLICalculator()
        self.error_budget_manager = ErrorBudgetManager()
        self.slo_dashboard = SLODashboard()
        self.burn_rate_alerting = BurnRateAlerting()
        self.slo_definitions = {}
        self.is_initialized = False

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize SLO monitoring"""

        # Load SLO definitions
        await self.load_slo_definitions(config.get("slo_definitions", []))

        # Set up SLI collection
        await self.setup_sli_collection()

        # Initialize error budgets
        await self.initialize_error_budgets()

        # Start monitoring loops
        await self.start_slo_monitoring()

        self.is_initialized = True
        logger.info("SLO monitor initialized")

    async def load_slo_definitions(self, definitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Load SLO definitions from configuration"""

        for slo_config in definitions:
            sli_definitions = []
            for sli_config in slo_config.get("sli_definitions", []):
                sli = SLIDefinition(
                    id=sli_config["id"],
                    name=sli_config["name"],
                    description=sli_config["description"],
                    sli_type=SLIType(sli_config["type"]),
                    query=sli_config["query"],
                    evaluation_window=sli_config["evaluation_window"],
                    target_percentage=sli_config["target_percentage"],
                )
                sli_definitions.append(sli)

            slo = SLODefinition(
                id=slo_config["id"],
                name=slo_config["name"],
                description=slo_config["description"],
                sli_definitions=sli_definitions,
                target_percentage=slo_config["target_percentage"],
                evaluation_window=slo_config["evaluation_window"],
                error_budget_policy=slo_config.get("error_budget_policy", {}),
                alerting_thresholds=slo_config.get("alerting_thresholds", {}),
            )

            self.slo_definitions[slo.id] = slo

    async def setup_sli_collection(self) -> Dict[str, Any]:
        """Set up SLI data collection"""
        # This would set up metrics collection for SLIs
        logger.info("SLI collection setup completed")

    async def initialize_error_budgets(self) -> Dict[str, Any]:
        """Initialize error budgets for all SLOs"""
        for slo in self.slo_definitions.values():
            error_budget = ErrorBudget(
                slo_id=slo.id,
                total_budget=100.0 - slo.target_percentage,
                consumed_budget=0.0,
                remaining_budget=100.0 - slo.target_percentage,
                burn_rate=0.0,
                time_to_exhaustion=None,
                last_updated=datetime.utcnow(),
            )
            self.error_budget_manager.error_budgets[slo.id] = error_budget

    async def start_slo_monitoring(self) -> Dict[str, Any]:
        """Start SLO monitoring loops"""
        asyncio.create_task(self._slo_monitoring_loop())

    async def _slo_monitoring_loop(self) -> Dict[str, Any]:
        """Background SLO monitoring loop"""
        while True:
            try:
    await self.monitor_all_slos()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"SLO monitoring loop error: {str(e)}")
                await asyncio.sleep(300)

    async def monitor_all_slos(self) -> Dict[str, Any]:
        """Monitor all SLOs"""

        for slo in self.slo_definitions.values():
            try:
    await self.monitor_slo(slo)
            except Exception as e:
                logger.error(f"Failed to monitor SLO {slo.id}: {str(e)}")

    async def monitor_slo(self, slo: SLODefinition) -> Dict[str, Any]:
        """Monitor individual SLO"""

        # Calculate SLI results
        sli_results = []
        for sli in slo.sli_definitions:
            time_window = timedelta(seconds=sli.evaluation_window)
            result = await self.sli_calculator.calculate_sli(sli, time_window)
            sli_results.append(result)

        # Calculate error budget
        error_budget = await self.error_budget_manager.calculate_error_budget(slo, sli_results)

        # Check for burn rate alerts
        alerts = await self.burn_rate_alerting.check_burn_rate_alerts(error_budget, slo)

        # Process alerts
        for alert in alerts:
            logger.warning(f"SLO Alert: {alert['message']}")

    async def calculate_slo_status(
            self,
            slo_id: str,
            time_window: timedelta) -> SLOStatus:
        """Calculate current SLO status"""

        slo = self.slo_definitions.get(slo_id)
        if not slo:
            raise ValueError(f"SLO {slo_id} not found")

        # Calculate SLI results
        sli_results = []
        for sli in slo.sli_definitions:
            result = await self.sli_calculator.calculate_sli(sli, time_window)
            sli_results.append(result)

        # Calculate error budget
        error_budget = await self.error_budget_manager.calculate_error_budget(slo, sli_results)

        # Determine overall status
        overall_status = self._determine_overall_status(
            sli_results, error_budget)

        # Generate recommendations
        recommendations = await self._generate_slo_recommendations(slo, sli_results, error_budget)

        return SLOStatus(
            slo_id=slo_id,
            overall_status=overall_status,
            sli_results=sli_results,
            error_budget_remaining=error_budget.remaining_budget,
            error_budget_consumed=error_budget.consumed_budget,
            burn_rate=error_budget.burn_rate,
            time_to_breach=error_budget.time_to_exhaustion,
            calculated_at=datetime.utcnow(),
            recommendations=recommendations,
        )

    def _determine_overall_status(
        self, sli_results: List[SLIResult], error_budget: ErrorBudget
    ) -> SLOStatus:
        """Determine overall SLO status"""

        if not sli_results:
            return SLOStatus.UNKNOWN

        # Check if any SLI is breached
        if any(result.status == SLOStatus.BREACHED for result in sli_results):
            return SLOStatus.BREACHED

        # Check error budget
        if error_budget.remaining_budget <= 0:
            return SLOStatus.BREACHED

        # Check if any SLI is in warning
        if any(result.status == SLOStatus.WARNING for result in sli_results):
            return SLOStatus.WARNING

        return SLOStatus.HEALTHY

    async def _generate_slo_recommendations(
            self,
            slo: SLODefinition,
            sli_results: List[SLIResult],
            error_budget: ErrorBudget) -> List[str]:
        """Generate recommendations for SLO improvement"""

        recommendations = []

        # Check burn rate
        if error_budget.burn_rate > 10:
            recommendations.append(
                f"High burn rate: {error_budget.burn_rate:.2f}%/hour. Consider investigating root causes."
            )

        # Check time to exhaustion
        if error_budget.time_to_exhaustion and error_budget.time_to_exhaustion < timedelta(
                hours=24):
            recommendations.append(
                f"Error budget will be exhausted in {error_budget.time_to_exhaustion}. Take immediate action."
            )

        # Check individual SLIs
        for result in sli_results:
            if result.status == SLOStatus.BREACHED:
                recommendations.append(
                    f"SLI {result.sli_id} is breached: {result.value:.2f}% vs target {result.target:.2f}%"
                )
            elif result.status == SLOStatus.WARNING:
                recommendations.append(
                    f"SLI {result.sli_id} is in warning: {result.value:.2f}% vs target {result.target:.2f}%"
                )

        return recommendations

    async def get_overall_slo_status(self) -> Dict[str, Any]:
        """Get overall SLO status across all SLOs"""
        slo_statuses = []
        for slo_id in self.slo_definitions.keys():
            try:
                status = await self.calculate_slo_status(slo_id, timedelta(hours=1))
                slo_statuses.append(status)
            except Exception as e:
                logger.error(
                    f"Failed to get status for SLO {slo_id}: {str(e)}")

        # Calculate overall health
        healthy_count = len(
            [s for s in slo_statuses if s.overall_status == SLOStatus.HEALTHY])
        total_count = len(slo_statuses)

        return {
            "total_slos": total_count,
            "healthy_slos": healthy_count,
            "health_percentage": (
                healthy_count /
                total_count *
                100) if total_count > 0 else 0,
            "slo_statuses": slo_statuses,
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup SLO monitor"""
        self.is_initialized = False
        logger.info("SLO monitor cleaned up")
