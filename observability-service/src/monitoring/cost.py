"""
Cost monitoring and optimization alerts
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Cost categories"""

    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    ML_MODELS = "ml_models"
    THIRD_PARTY = "third_party"
    MONITORING = "monitoring"
    OTHER = "other"


class CostAlertType(Enum):
    """Cost alert types"""

    BUDGET_EXCEEDED = "budget_exceeded"
    UNUSUAL_SPIKE = "unusual_spike"
    WASTE_DETECTED = "waste_detected"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"
    THRESHOLD_REACHED = "threshold_reached"


@dataclass
class CostData:
    """Cost data point"""

    timestamp: datetime
    category: CostCategory
    service: str
    amount: float
    currency: str = "USD"
    region: Optional[str] = None
    resource_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostAlert:
    """Cost alert definition"""

    id: str
    alert_type: CostAlertType
    category: CostCategory
    service: str
    current_cost: float
    threshold: float
    message: str
    severity: str
    created_at: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    recommendations: List[str] = field(default_factory=list)


@dataclass
class CostBudget:
    """Cost budget definition"""

    id: str
    name: str
    category: CostCategory
    service: str
    monthly_limit: float
    daily_limit: Optional[float] = None
    currency: str = "USD"
    alert_thresholds: List[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    enabled: bool = True


@dataclass
class CostOptimization:
    """Cost optimization recommendation"""

    id: str
    category: CostCategory
    service: str
    current_cost: float
    potential_savings: float
    recommendation: str
    implementation_effort: str  # low, medium, high
    priority: str  # low, medium, high, critical
    created_at: datetime = field(default_factory=datetime.utcnow)


class CostCollector:
    """Collect cost data from various sources"""

    def __init__(self):
        self.cost_data = []
        self.is_initialized = False

    async def initialize(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Initialize cost collector"""
        # Set up data sources
        self.data_sources = config.get("data_sources", {}) if config else {}

        self.is_initialized = True
        logger.info("Cost collector initialized")

    async def collect_cost_metrics(self) -> List[CostData]:
        """Collect cost metrics from all sources"""

        cost_data = []

        # Collect from different sources
        cost_data.extend(await self._collect_aws_costs())
        cost_data.extend(await self._collect_gcp_costs())
        cost_data.extend(await self._collect_azure_costs())
        cost_data.extend(await self._collect_kubernetes_costs())
        cost_data.extend(await self._collect_third_party_costs())

        # Store collected data
        self.cost_data.extend(cost_data)

        # Keep only last 30 days of data
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        self.cost_data = [
            d for d in self.cost_data if d.timestamp >= cutoff_date]

        return cost_data

    async def _collect_aws_costs(self) -> List[CostData]:
        """Collect AWS costs"""

        try:
            # This would integrate with AWS Cost Explorer API
            # For now, return mock data

            mock_costs = [
                CostData(
                    timestamp=datetime.utcnow(),
                    category=CostCategory.COMPUTE,
                    service="ec2",
                    amount=150.50,
                    region="us-east-1",
                    metadata={"instance_type": "t3.medium", "hours": 24},
                ),
                CostData(
                    timestamp=datetime.utcnow(),
                    category=CostCategory.STORAGE,
                    service="s3",
                    amount=25.30,
                    region="us-east-1",
                    metadata={"storage_class": "standard", "gb": 1000},
                ),
                CostData(
                    timestamp=datetime.utcnow(),
                    category=CostCategory.DATABASE,
                    service="rds",
                    amount=80.75,
                    region="us-east-1",
                    metadata={"instance_type": "db.t3.micro", "hours": 24},
                ),
            ]

            return mock_costs

        except Exception as e:
            logger.error(f"Failed to collect AWS costs: {str(e)}")
            return []

    async def _collect_gcp_costs(self) -> List[CostData]:
        """Collect GCP costs"""

        try:
            # This would integrate with GCP Billing API
            # For now, return mock data

            mock_costs = [
                CostData(
                    timestamp=datetime.utcnow(),
                    category=CostCategory.COMPUTE,
                    service="compute-engine",
                    amount=120.25,
                    region="us-central1",
                    metadata={"machine_type": "n1-standard-2", "hours": 24},
                ),
                CostData(
                    timestamp=datetime.utcnow(),
                    category=CostCategory.STORAGE,
                    service="cloud-storage",
                    amount=15.80,
                    region="us-central1",
                    metadata={"storage_class": "standard", "gb": 500},
                ),
            ]

            return mock_costs

        except Exception as e:
            logger.error(f"Failed to collect GCP costs: {str(e)}")
            return []

    async def _collect_azure_costs(self) -> List[CostData]:
        """Collect Azure costs"""

        try:
            # This would integrate with Azure Cost Management API
            # For now, return mock data

            mock_costs = [
                CostData(
                    timestamp=datetime.utcnow(),
                    category=CostCategory.COMPUTE,
                    service="virtual-machines",
                    amount=95.40,
                    region="eastus",
                    metadata={"vm_size": "Standard_B2s", "hours": 24},
                )
            ]

            return mock_costs

        except Exception as e:
            logger.error(f"Failed to collect Azure costs: {str(e)}")
            return []

    async def _collect_kubernetes_costs(self) -> List[CostData]:
        """Collect Kubernetes costs"""

        try:
            # This would integrate with Kubernetes cost monitoring tools
            # For now, return mock data

            mock_costs = [
                CostData(
                    timestamp=datetime.utcnow(),
                    category=CostCategory.COMPUTE,
                    service="kubernetes",
                    amount=200.00,
                    metadata={
                        "pods": 15,
                        "cpu_hours": 360,
                        "memory_gb_hours": 720},
                )]

            return mock_costs

        except Exception as e:
            logger.error(f"Failed to collect Kubernetes costs: {str(e)}")
            return []

    async def _collect_third_party_costs(self) -> List[CostData]:
        """Collect third-party service costs"""

        try:
            # This would integrate with various third-party APIs
            # For now, return mock data

            mock_costs = [
                CostData(
                    timestamp=datetime.utcnow(),
                    category=CostCategory.THIRD_PARTY,
                    service="openai",
                    amount=45.60,
                    metadata={"api_calls": 10000, "tokens": 500000},
                ),
                CostData(
                    timestamp=datetime.utcnow(),
                    category=CostCategory.THIRD_PARTY,
                    service="elasticsearch",
                    amount=30.25,
                    metadata={"cluster_size": "small", "storage_gb": 100},
                ),
                CostData(
                    timestamp=datetime.utcnow(),
                    category=CostCategory.MONITORING,
                    service="datadog",
                    amount=75.00,
                    metadata={"hosts": 10, "custom_metrics": 50},
                ),
            ]

            return mock_costs

        except Exception as e:
            logger.error(f"Failed to collect third-party costs: {str(e)}")
            return []


class CostAnalyzer:
    """Analyze cost data and identify optimization opportunities"""

    def __init__(self):
        self.cost_data = []
        self.budgets = {}
        self.optimizations = []

    async def analyze_costs(self, cost_data: List[CostData]) -> Dict[str, Any]:
        """Analyze cost data and generate insights"""
        self.cost_data = cost_data

        # Calculate total costs
        total_cost = sum(d.amount for d in cost_data)

        # Group by category
        category_costs = self._group_by_category(cost_data)

        # Group by service
        service_costs = self._group_by_service(cost_data)

        # Calculate trends
        trends = self._calculate_trends(cost_data)

        # Identify anomalies
        anomalies = self._identify_anomalies(cost_data)

        # Generate optimizations
        optimizations = await self._generate_optimizations(cost_data)

        return {
            "total_cost": total_cost,
            "category_breakdown": category_costs,
            "service_breakdown": service_costs,
            "trends": trends,
            "anomalies": anomalies,
            "optimizations": optimizations,
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    def _group_by_category(
            self, cost_data: List[CostData]) -> Dict[str, float]:
        """Group costs by category"""

        category_costs = {}
        for data in cost_data:
            category = data.category.value
            if category not in category_costs:
                category_costs[category] = 0.0
            category_costs[category] += data.amount

        return category_costs

    def _group_by_service(self, cost_data: List[CostData]) -> Dict[str, float]:
        """Group costs by service"""

        service_costs = {}
        for data in cost_data:
            service = data.service
            if service not in service_costs:
                service_costs[service] = 0.0
            service_costs[service] += data.amount

        return service_costs

    def _calculate_trends(self, cost_data: List[CostData]) -> Dict[str, Any]:
        """Calculate cost trends"""
        # Group by day
        daily_costs = {}
        for data in cost_data:
            day = data.timestamp.date()
            if day not in daily_costs:
                daily_costs[day] = 0.0
            daily_costs[day] += data.amount

        # Calculate trend
        if len(daily_costs) >= 2:
            days = sorted(daily_costs.keys())
            first_cost = daily_costs[days[0]]
            last_cost = daily_costs[days[-1]]
            days_diff = (days[-1] - days[0]).days

            if days_diff > 0:
                daily_change = (last_cost - first_cost) / days_diff
                trend_direction = "increasing" if daily_change > 0 else "decreasing"
            else:
                daily_change = 0
                trend_direction = "stable"
        else:
            daily_change = 0
            trend_direction = "stable"

        return {
            "daily_change": daily_change,
            "trend_direction": trend_direction,
            "daily_costs": daily_costs,
        }

    def _identify_anomalies(
            self, cost_data: List[CostData]) -> List[Dict[str, Any]]:
        """Identify cost anomalies"""

        anomalies = []

        # Group by service and day
        service_daily_costs = {}
        for data in cost_data:
            service = data.service
            day = data.timestamp.date()
            key = (service, day)
            if key not in service_daily_costs:
                service_daily_costs[key] = 0.0
            service_daily_costs[key] += data.amount

        # Calculate average and standard deviation for each service
        for service in set(data.service for data in cost_data):
            service_costs = [
                cost for (
                    s,
                    day),
                cost in service_daily_costs.items() if s == service]

            if len(service_costs) >= 3:
                avg_cost = sum(service_costs) / len(service_costs)
                variance = sum(
                    (cost - avg_cost) ** 2 for cost in service_costs) / len(service_costs)
                std_dev = variance**0.5

                # Identify outliers (costs > 2 standard deviations from mean)
                for (s, day), cost in service_daily_costs.items():
                    if s == service and cost > avg_cost + 2 * std_dev:
                        anomalies.append(
                            {
                                "service": service,
                                "date": day.isoformat(),
                                "cost": cost,
                                "expected_cost": avg_cost,
                                "deviation": cost -
                                avg_cost,
                                "severity": "high" if cost > avg_cost +
                                3 *
                                std_dev else "medium",
                            })

        return anomalies

    async def _generate_optimizations(
            self, cost_data: List[CostData]) -> List[CostOptimization]:
        """Generate cost optimization recommendations"""

        optimizations = []

        # Analyze compute costs
        compute_costs = [
            d for d in cost_data if d.category == CostCategory.COMPUTE]
        if compute_costs:
            compute_optimizations = await self._analyze_compute_costs(compute_costs)
            optimizations.extend(compute_optimizations)

        # Analyze storage costs
        storage_costs = [
            d for d in cost_data if d.category == CostCategory.STORAGE]
        if storage_costs:
            storage_optimizations = await self._analyze_storage_costs(storage_costs)
            optimizations.extend(storage_optimizations)

        # Analyze database costs
        db_costs = [d for d in cost_data if d.category ==
                    CostCategory.DATABASE]
        if db_costs:
            db_optimizations = await self._analyze_database_costs(db_costs)
            optimizations.extend(db_optimizations)

        return optimizations

    async def _analyze_compute_costs(
            self, compute_costs: List[CostData]) -> List[CostOptimization]:
        """Analyze compute costs for optimization opportunities"""

        optimizations = []

        # Group by service
        service_costs = {}
        for cost in compute_costs:
            if cost.service not in service_costs:
                service_costs[cost.service] = []
            service_costs[cost.service].append(cost)

        for service, costs in service_costs.items():
            total_cost = sum(c.amount for c in costs)

            # Check for underutilized resources
            if total_cost > 100:  # Only analyze services with significant costs
                optimization = CostOptimization(
                    id=f"compute_{service}_{datetime.utcnow().strftime('%Y%m%d')}",
                    category=CostCategory.COMPUTE,
                    service=service,
                    current_cost=total_cost,
                    potential_savings=total_cost * 0.2,  # Assume 20% savings possible
                    recommendation=f"Review {service} resource utilization and consider rightsizing",
                    implementation_effort="medium",
                    priority="medium",
                )
                optimizations.append(optimization)

        return optimizations

    async def _analyze_storage_costs(
            self, storage_costs: List[CostData]) -> List[CostOptimization]:
        """Analyze storage costs for optimization opportunities"""

        optimizations = []

        for cost in storage_costs:
            if cost.amount > 50:  # Only analyze significant storage costs
                optimization = CostOptimization(
                    id=f"storage_{cost.service}_{datetime.utcnow().strftime('%Y%m%d')}",
                    category=CostCategory.STORAGE,
                    service=cost.service,
                    current_cost=cost.amount,
                    potential_savings=cost.amount * 0.3,  # Assume 30% savings possible
                    recommendation=f"Review {cost.service} storage classes and lifecycle policies",
                    implementation_effort="low",
                    priority="low",
                )
                optimizations.append(optimization)

        return optimizations

    async def _analyze_database_costs(
            self, db_costs: List[CostData]) -> List[CostOptimization]:
        """Analyze database costs for optimization opportunities"""

        optimizations = []

        for cost in db_costs:
            if cost.amount > 100:  # Only analyze significant database costs
                optimization = CostOptimization(
                    id=f"database_{cost.service}_{datetime.utcnow().strftime('%Y%m%d')}",
                    category=CostCategory.DATABASE,
                    service=cost.service,
                    current_cost=cost.amount,
                    potential_savings=cost.amount * 0.15,  # Assume 15% savings possible
                    recommendation=f"Review {cost.service} instance sizing and query optimization",
                    implementation_effort="high",
                    priority="high",
                )
                optimizations.append(optimization)

        return optimizations


class CostAlertManager:
    """Manage cost alerts and notifications"""

    def __init__(self):
        self.alerts = []
        self.budgets = {}
        self.alert_rules = {}

    async def check_cost_alerts(
            self, cost_data: List[CostData]) -> List[CostAlert]:
        """Check for cost alerts based on current data"""

        alerts = []

        # Check budget alerts
        budget_alerts = await self._check_budget_alerts(cost_data)
        alerts.extend(budget_alerts)

        # Check spike alerts
        spike_alerts = await self._check_spike_alerts(cost_data)
        alerts.extend(spike_alerts)

        # Check waste alerts
        waste_alerts = await self._check_waste_alerts(cost_data)
        alerts.extend(waste_alerts)

        # Store alerts
        self.alerts.extend(alerts)

        return alerts

    async def _check_budget_alerts(
            self, cost_data: List[CostData]) -> List[CostAlert]:
        """Check for budget threshold alerts"""

        alerts = []

        # Group costs by service and category
        service_category_costs = {}
        for data in cost_data:
            key = (data.service, data.category)
            if key not in service_category_costs:
                service_category_costs[key] = 0.0
            service_category_costs[key] += data.amount

        # Check against budgets
        for (service, category), cost in service_category_costs.items():
            budget_key = f"{service}_{category.value}"
            budget = self.budgets.get(budget_key)

            if budget and cost > budget.monthly_limit:
                alert = CostAlert(
                    id=f"budget_{budget_key}_{datetime.utcnow().strftime('%Y%m%d%H%M')}",
                    alert_type=CostAlertType.BUDGET_EXCEEDED,
                    category=category,
                    service=service,
                    current_cost=cost,
                    threshold=budget.monthly_limit,
                    message=f"Budget exceeded for {service} {category.value}: ${cost:.2f} > ${budget.monthly_limit:.2f}",
                    severity="high",
                    created_at=datetime.utcnow(),
                    recommendations=[
                        f"Review {service} usage patterns",
                        "Consider implementing cost controls",
                        "Evaluate resource optimization opportunities",
                    ],
                )
                alerts.append(alert)

        return alerts

    async def _check_spike_alerts(
            self, cost_data: List[CostData]) -> List[CostAlert]:
        """Check for unusual cost spikes"""

        alerts = []

        # Group by service and day
        service_daily_costs = {}
        for data in cost_data:
            service = data.service
            day = data.timestamp.date()
            key = (service, day)
            if key not in service_daily_costs:
                service_daily_costs[key] = 0.0
            service_daily_costs[key] += data.amount

        # Calculate average daily cost for each service
        for service in set(data.service for data in cost_data):
            service_costs = [
                cost for (
                    s,
                    day),
                cost in service_daily_costs.items() if s == service]

            if len(service_costs) >= 3:
                avg_cost = sum(service_costs) / len(service_costs)

                # Check for spikes (> 200% of average)
                for (s, day), cost in service_daily_costs.items():
                    if s == service and cost > avg_cost * 2:
                        alert = CostAlert(
                            id=f"spike_{service}_{day}_{datetime.utcnow().strftime('%H%M')}",
                            alert_type=CostAlertType.UNUSUAL_SPIKE,
                            category=CostCategory.OTHER,
                            service=service,
                            current_cost=cost,
                            threshold=avg_cost * 2,
                            message=f"Unusual cost spike for {service} on {day}: ${cost:.2f} (avg: ${avg_cost:.2f})",
                            severity="medium",
                            created_at=datetime.utcnow(),
                            recommendations=[
                                "Investigate recent changes to service configuration",
                                "Check for resource leaks or inefficient operations",
                                "Review monitoring and alerting setup",
                            ],
                        )
                        alerts.append(alert)

        return alerts

    async def _check_waste_alerts(
            self, cost_data: List[CostData]) -> List[CostAlert]:
        """Check for resource waste"""

        alerts = []

        # Check for idle resources
        for data in cost_data:
            if data.category == CostCategory.COMPUTE and data.amount > 50:
                # This would check actual resource utilization
                # For now, create mock waste alert
                alert = CostAlert(
                    id=f"waste_{data.service}_{datetime.utcnow().strftime('%Y%m%d%H%M')}",
                    alert_type=CostAlertType.WASTE_DETECTED,
                    category=data.category,
                    service=data.service,
                    current_cost=data.amount,
                    threshold=data.amount * 0.5,
                    message=f"Potential waste detected in {data.service}: ${data.amount:.2f} with low utilization",
                    severity="low",
                    created_at=datetime.utcnow(),
                    recommendations=[
                        f"Review {data.service} resource utilization",
                        "Consider downsizing or stopping unused resources",
                        "Implement automated resource management",
                    ],
                )
                alerts.append(alert)

        return alerts

    async def create_budget(self, budget_data: Dict[str, Any]) -> CostBudget:
        """Create cost budget"""

        budget = CostBudget(
            id=budget_data["id"],
            name=budget_data["name"],
            category=CostCategory(budget_data["category"]),
            service=budget_data["service"],
            monthly_limit=budget_data["monthly_limit"],
            daily_limit=budget_data.get("daily_limit"),
            currency=budget_data.get("currency", "USD"),
            alert_thresholds=budget_data.get("alert_thresholds", []),
        )

        self.budgets[budget.id] = budget
        return budget


class CostMonitor:
    """Main cost monitoring system"""

    def __init__(self):
        self.cost_collector = CostCollector()
        self.cost_analyzer = CostAnalyzer()
        self.alert_manager = CostAlertManager()
        self.is_initialized = False

    async def initialize(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Initialize cost monitoring"""
        await self.cost_collector.initialize(config)
        self.is_initialized = True
        logger.info("Cost monitor initialized")

    async def collect_cost_metrics(self) -> List[CostData]:
        """Collect cost metrics from all sources"""
        return await self.cost_collector.collect_cost_metrics()

    async def analyze_costs(self) -> Dict[str, Any]:
        """Analyze costs and generate insights"""
        # Collect fresh cost data
        cost_data = await self.collect_cost_metrics()

        # Analyze costs
        analysis = await self.cost_analyzer.analyze_costs(cost_data)

        # Check for alerts
        alerts = await self.alert_manager.check_cost_alerts(cost_data)
        analysis["alerts"] = alerts

        return analysis

    async def get_cost_summary(
            self, time_window: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """Get cost summary for time window"""
        # Collect cost data
        cost_data = await self.collect_cost_metrics()

        # Filter by time window
        cutoff_time = datetime.utcnow() - time_window
        recent_costs = [d for d in cost_data if d.timestamp >= cutoff_time]

        # Calculate summary
        total_cost = sum(d.amount for d in recent_costs)
        category_breakdown = self.cost_analyzer._group_by_category(
            recent_costs)
        service_breakdown = self.cost_analyzer._group_by_service(recent_costs)

        # Calculate daily average
        days = time_window.days or 1
        daily_average = total_cost / days

        return {
            "total_cost": total_cost,
            "daily_average": daily_average,
            "category_breakdown": category_breakdown,
            "service_breakdown": service_breakdown,
            "time_window_days": days,
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def create_budget(self, budget_data: Dict[str, Any]) -> CostBudget:
        """Create cost budget"""
        return await self.alert_manager.create_budget(budget_data)

    async def get_active_alerts(self) -> List[CostAlert]:
        """Get active cost alerts"""
        return [alert for alert in self.alert_manager.alerts if not alert.resolved]

    async def resolve_alert(
            self,
            alert_id: str,
            user: str,
            notes: Optional[str] = None):
         -> Dict[str, Any]:"""Resolve cost alert"""

        for alert in self.alert_manager.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                logger.info(f"Cost alert {alert_id} resolved by {user}")
                break

    async def cleanup(self) -> Dict[str, Any]:
    """Cleanup cost monitor"""
        self.is_initialized = False
        logger.info("Cost monitor cleaned up")
