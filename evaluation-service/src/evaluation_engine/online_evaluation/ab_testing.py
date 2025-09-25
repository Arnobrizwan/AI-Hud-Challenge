"""
Advanced A/B Testing Framework with Statistical Rigor
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

from ..database import get_db_session
from ..models import (
    BayesianAnalysis,
    Experiment,
    ExperimentAnalysis,
    ExperimentConfig,
    ExperimentData,
    ExperimentStatus,
    FrequentistAnalysis,
    SequentialAnalysis,
    StatisticalTestType,
    VariantAnalysis,
)
from ..statistical_testing import PowerAnalyzer, StatisticalTester

logger = logging.getLogger(__name__)


class ABTestingFramework:
    """Advanced A/B testing with statistical rigor"""

    def __init__(self):
        self.statistical_tester = StatisticalTester()
        self.power_analyzer = PowerAnalyzer()
        self.sequential_tester = None  # Will be initialized
        self.bayesian_analyzer = None  # Will be initialized

        # Experiment tracking
        self.active_experiments: Dict[str, Experiment] = {}
        self.experiment_data: Dict[str, ExperimentData] = {}

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the A/B testing framework"""
        try:
            logger.info("Initializing A/B testing framework...")

            await self.statistical_tester.initialize()
            await self.power_analyzer.initialize()

            # Initialize sequential and Bayesian testers
            from .bayesian_testing import BayesianTestingFramework
            from .sequential_testing import SequentialTestingFramework

            self.sequential_tester = SequentialTestingFramework()
            self.bayesian_analyzer = BayesianTestingFramework()

            await self.sequential_tester.initialize()
            await self.bayesian_analyzer.initialize()

            logger.info("A/B testing framework initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize A/B testing framework: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup A/B testing framework resources"""
        try:
            logger.info("Cleaning up A/B testing framework...")

            await self.statistical_tester.cleanup()
            await self.power_analyzer.cleanup()

            if self.sequential_tester:
                await self.sequential_tester.cleanup()
            if self.bayesian_analyzer:
                await self.bayesian_analyzer.cleanup()

            logger.info("A/B testing framework cleanup completed")

        except Exception as e:
            logger.error(f"Error during A/B testing framework cleanup: {str(e)}")

    async def create_experiment(self, experiment_config: ExperimentConfig) -> Experiment:
        """Create A/B test experiment"""

        logger.info(f"Creating experiment: {experiment_config.name}")

        # Validate experiment design
        design_validation = await self.validate_experiment_design(experiment_config)
        if not design_validation["is_valid"]:
            raise ExperimentDesignError(design_validation["error_message"])

        # Calculate required sample size
        sample_size = await self.power_analyzer.calculate_sample_size(
            effect_size=experiment_config.minimum_detectable_effect,
            alpha=experiment_config.alpha,
            power=experiment_config.power,
            baseline_conversion_rate=experiment_config.baseline_rate,
        )

        # Ensure minimum sample size
        sample_size = max(sample_size, experiment_config.get("min_sample_size", 1000))

        experiment = Experiment(
            id=str(uuid4()),
            name=experiment_config.name,
            hypothesis=experiment_config.hypothesis,
            variants=experiment_config.variants,
            traffic_allocation=experiment_config.traffic_allocation,
            primary_metric=experiment_config.primary_metric,
            secondary_metrics=experiment_config.secondary_metrics,
            guardrail_metrics=experiment_config.guardrail_metrics,
            sample_size_per_variant=sample_size,
            start_date=experiment_config.start_date,
            estimated_end_date=self._calculate_estimated_end_date(sample_size, experiment_config),
            status=ExperimentStatus.DRAFT,
            created_at=datetime.utcnow(),
        )

        # Store experiment
        self.active_experiments[experiment.id] = experiment

        logger.info(f"Experiment {experiment.id} created successfully")
        return experiment

    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""

        if experiment_id not in self.active_experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False

        experiment = self.active_experiments[experiment_id]

        if experiment.status != ExperimentStatus.DRAFT:
            logger.error(f"Experiment {experiment_id} is not in draft status")
            return False

        # Update status
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_date = datetime.utcnow()

        # Initialize experiment data collection
        self.experiment_data[experiment_id] = ExperimentData(
            experiment_id=experiment_id,
            variant_data={variant: {} for variant in experiment.variants},
        )

        logger.info(f"Experiment {experiment_id} started successfully")
        return True

    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment"""

        if experiment_id not in self.active_experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False

        experiment = self.active_experiments[experiment_id]

        if experiment.status != ExperimentStatus.RUNNING:
            logger.error(f"Experiment {experiment_id} is not running")
            return False

        # Update status
        experiment.status = ExperimentStatus.COMPLETED
        experiment.actual_end_date = datetime.utcnow()

        logger.info(f"Experiment {experiment_id} stopped successfully")
        return True

    async def record_event(
        self,
        experiment_id: str,
        user_id: str,
        variant: str,
        event_type: str,
        value: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record an event for an experiment"""

        if experiment_id not in self.experiment_data:
            logger.warning(f"Experiment {experiment_id} data not found")
            return False

        experiment_data = self.experiment_data[experiment_id]

        if variant not in experiment_data.variant_data:
            logger.warning(f"Variant {variant} not found in experiment {experiment_id}")
            return False

        # Initialize event tracking for variant if not exists
        if event_type not in experiment_data.variant_data[variant]:
            experiment_data.variant_data[variant][event_type] = []

        # Record event
        event_record = {
            "user_id": user_id,
            "value": value,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {},
        }

        experiment_data.variant_data[variant][event_type].append(event_record)

        return True

    async def analyze_experiment(self, experiment_id: str, analysis_type: str = "frequentist") -> ExperimentAnalysis:
        """Analyze experiment results with statistical testing"""

        logger.info(f"Analyzing experiment {experiment_id} with {analysis_type} analysis")

        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.active_experiments[experiment_id]

        if experiment_id not in self.experiment_data:
            raise ValueError(f"Experiment {experiment_id} data not found")

        experiment_data = self.experiment_data[experiment_id]

        # Perform analysis based on type
        if analysis_type == "frequentist":
            analysis = await self._frequentist_analysis(experiment, experiment_data)
        elif analysis_type == "bayesian":
            analysis = await self._bayesian_analysis(experiment, experiment_data)
        elif analysis_type == "sequential":
            analysis = await self._sequential_analysis(experiment, experiment_data)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        # Check for statistical significance
        significance_results = await self._check_statistical_significance(analysis)

        # Analyze guardrail metrics
        guardrail_analysis = await self._analyze_guardrail_metrics(experiment, experiment_data)

        # Generate business recommendations
        recommendations = await self._generate_experiment_recommendations(
            analysis, significance_results, guardrail_analysis
        )

        return ExperimentAnalysis(
            experiment_id=experiment_id,
            analysis_type=analysis_type,
            statistical_results=analysis,
            significance_results=significance_results,
            guardrail_analysis=guardrail_analysis,
            recommendations=recommendations,
            analysis_timestamp=datetime.utcnow(),
        )

    async def _frequentist_analysis(
        self, experiment: Experiment, experiment_data: ExperimentData
    ) -> FrequentistAnalysis:
        """Perform frequentist statistical analysis"""

        control_data = experiment_data.get_variant_data("control")
        if not control_data:
            raise ValueError("Control variant data not found")

        results = {}

        for variant_name in experiment.variants:
            if variant_name == "control":
                continue

            variant_data = experiment_data.get_variant_data(variant_name)
            if not variant_data:
                continue

            # Primary metric analysis
            control_values = self._extract_metric_values(control_data, experiment.primary_metric)
            variant_values = self._extract_metric_values(variant_data, experiment.primary_metric)

            if len(control_values) == 0 or len(variant_values) == 0:
                continue

            primary_test_result = await self.statistical_tester.two_sample_test(
                control_values, variant_values, test_type=StatisticalTestType.WELCH_T_TEST
            )

            # Secondary metrics analysis
            secondary_results = {}
            for metric in experiment.secondary_metrics:
                control_metric_values = self._extract_metric_values(control_data, metric)
                variant_metric_values = self._extract_metric_values(variant_data, metric)

                if len(control_metric_values) > 0 and len(variant_metric_values) > 0:
                    secondary_test_result = await self.statistical_tester.two_sample_test(
                        control_metric_values,
                        variant_metric_values,
                        test_type=StatisticalTestType.WELCH_T_TEST,
                    )
                    secondary_results[metric] = secondary_test_result

            # Calculate effect size and confidence intervals
            effect_size = await self.statistical_tester.calculate_effect_size(control_values, variant_values)

            confidence_interval = await self.statistical_tester.calculate_confidence_interval(
                control_values, variant_values
            )

            results[variant_name] = VariantAnalysis(
                variant_name=variant_name,
                primary_metric_result=primary_test_result,
                secondary_metric_results=secondary_results,
                effect_size=effect_size,
                confidence_interval=confidence_interval,
                sample_size=len(variant_values),
                conversion_rate=np.mean(variant_values),
            )

        # Determine overall significance
        overall_significance = any(
            result.primary_metric_result.get("is_significant", False) for result in results.values()
        )

        # Apply multiple testing correction
        multiple_testing_correction = await self._apply_multiple_testing_correction(results)

        return FrequentistAnalysis(
            experiment_id=experiment.id,
            variant_results=results,
            overall_significance=overall_significance,
            multiple_testing_correction=multiple_testing_correction,
        )

    async def _bayesian_analysis(self, experiment: Experiment, experiment_data: ExperimentData) -> BayesianAnalysis:
        """Perform Bayesian statistical analysis"""

        if not self.bayesian_analyzer:
            raise ValueError("Bayesian analyzer not initialized")

        return await self.bayesian_analyzer.analyze_experiment(experiment, experiment_data)

    async def _sequential_analysis(self, experiment: Experiment, experiment_data: ExperimentData) -> SequentialAnalysis:
        """Perform sequential statistical analysis"""

        if not self.sequential_tester:
            raise ValueError("Sequential tester not initialized")

        return await self.sequential_tester.analyze_experiment(experiment, experiment_data)

    async def _extract_metric_values(self, variant_data: Dict[str, Any], metric_name: str) -> List[float]:
        """Extract metric values from variant data"""

        if metric_name not in variant_data:
            return []

        events = variant_data[metric_name]
        if not isinstance(events, list):
            return []

        return [event["value"] for event in events if isinstance(event, dict) and "value" in event]

    async def _check_statistical_significance(self, analysis) -> Dict[str, Any]:
        """Check statistical significance of analysis results"""
        if isinstance(analysis, FrequentistAnalysis):
            return {
                "overall_significant": analysis.overall_significance,
                "variant_significance": {
                    variant: result.primary_metric_result.get("is_significant", False)
                    for variant, result in analysis.variant_results.items()
                },
            }
        elif isinstance(analysis, BayesianAnalysis):
            return {
                "overall_significant": any(prob > 0.95 for prob in analysis.posterior_probabilities.values()),
                "posterior_probabilities": analysis.posterior_probabilities,
            }
        elif isinstance(analysis, SequentialAnalysis):
            return {
                "overall_significant": analysis.early_stopping,
                "stopping_boundaries": analysis.stopping_boundaries,
            }
        else:
            return {}

    async def _analyze_guardrail_metrics(
        self, experiment: Experiment, experiment_data: ExperimentData
    ) -> Dict[str, Any]:
        """Analyze guardrail metrics for safety"""
        guardrail_analysis = {}

        for metric in experiment.guardrail_metrics:
            control_values = self._extract_metric_values(experiment_data.get_variant_data("control"), metric)

            metric_analysis = {}

            for variant_name in experiment.variants:
                if variant_name == "control":
                    continue

                variant_values = self._extract_metric_values(experiment_data.get_variant_data(variant_name), metric)

                if len(control_values) > 0 and len(variant_values) > 0:
                    # Check if variant is significantly worse
                    test_result = await self.statistical_tester.two_sample_test(
                        control_values, variant_values, test_type=StatisticalTestType.WELCH_T_TEST
                    )

                    metric_analysis[variant_name] = {
                        "is_significantly_worse": (
                            test_result.get("is_significant", False)
                            and np.mean(variant_values) < np.mean(control_values)
                        ),
                        "control_mean": np.mean(control_values),
                        "variant_mean": np.mean(variant_values),
                        "p_value": test_result.get("p_value", 1.0),
                    }

            guardrail_analysis[metric] = metric_analysis

        return guardrail_analysis

    async def _generate_experiment_recommendations(
        self, analysis, significance_results: Dict[str, Any], guardrail_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate business recommendations based on analysis"""

        recommendations = []

        # Check for guardrail violations
        for metric, metric_analysis in guardrail_analysis.items():
            for variant, variant_analysis in metric_analysis.items():
                if variant_analysis.get("is_significantly_worse", False):
                    recommendations.append(
                        {
                            "type": "guardrail_violation",
                            "priority": "high",
                            "title": f"Guardrail violation: {metric}",
                            "description": f"Variant {variant} significantly underperforms control on {metric}",
                            "action": "Consider stopping experiment or adjusting variant",
                        }
                    )

        # Check for statistical significance
        if significance_results.get("overall_significant", False):
            if isinstance(analysis, FrequentistAnalysis):
                # Find winning variant
                winning_variants = [
                    variant
                    for variant, result in analysis.variant_results.items()
                    if result.primary_metric_result.get("is_significant", False)
                ]

                if winning_variants:
                    recommendations.append(
                        {
                            "type": "significant_winner",
                            "priority": "medium",
                            "title": "Significant winner found",
                            "description": f"Variants {winning_variants} show significant improvement",
                            "action": "Consider deploying winning variant(s)",
                        }
                    )

        # Check for insufficient power
        if isinstance(analysis, FrequentistAnalysis):
            for variant, result in analysis.variant_results.items():
                if result.sample_size < experiment.sample_size_per_variant * 0.8:
                    recommendations.append(
                        {
                            "type": "insufficient_sample_size",
                            "priority": "low",
                            "title": f"Insufficient sample size: {variant}",
                            "description": f"Variant {variant} has only {result.sample_size} samples",
                            "action": "Continue experiment to reach target sample size",
                        }
                    )

        return recommendations

    async def _apply_multiple_testing_correction(self, results: Dict[str, VariantAnalysis]) -> Dict[str, Any]:
    """Apply multiple testing correction (Bonferroni)"""
        n_tests = len(results)
        if n_tests <= 1:
            return {"method": "none", "corrected_alpha": 0.05}

        # Bonferroni correction
        corrected_alpha = 0.05 / n_tests

        corrected_results = {}
        for variant, result in results.items():
            original_p = result.primary_metric_result.get("p_value", 1.0)
            corrected_significant = original_p < corrected_alpha

            corrected_results[variant] = {
                "original_p_value": original_p,
                "corrected_alpha": corrected_alpha,
                "corrected_significant": corrected_significant,
            }

        return {
            "method": "bonferroni",
            "n_tests": n_tests,
            "corrected_alpha": corrected_alpha,
            "variant_results": corrected_results,
        }

    async def validate_experiment_design(self, config: ExperimentConfig) -> Dict[str, Any]:
    """Validate experiment design"""
        errors = []

        # Check variants
        if len(config.variants) < 2:
            errors.append("At least 2 variants required")

        if "control" not in config.variants:
            errors.append("Control variant is required")

        # Check traffic allocation
        total_allocation = sum(config.traffic_allocation.values())
        if abs(total_allocation - 1.0) > 0.01:
            errors.append(f"Traffic allocation must sum to 1.0, got {total_allocation}")

        # Check sample size
        if config.minimum_detectable_effect <= 0:
            errors.append("Minimum detectable effect must be positive")

        if config.alpha <= 0 or config.alpha >= 1:
            errors.append("Alpha must be between 0 and 1")

        if config.power <= 0 or config.power >= 1:
            errors.append("Power must be between 0 and 1")

        # Check dates
        if config.start_date < datetime.utcnow():
            errors.append("Start date cannot be in the past")

        if config.end_date and config.end_date <= config.start_date:
            errors.append("End date must be after start date")

        return {
            "is_valid": len(errors) == 0,
            "error_message": "; ".join(errors) if errors else None,
        }

    def _calculate_estimated_end_date(self, sample_size: int, config: ExperimentConfig) -> datetime:
        """Calculate estimated end date based on sample size and traffic"""

        # Estimate based on daily traffic (this would come from actual traffic
        # data)
        estimated_daily_traffic = 10000  # Mock value
        daily_sample_size = sample_size * sum(config.traffic_allocation.values())
        estimated_days = max(1, sample_size / daily_sample_size)

        return config.start_date + timedelta(days=estimated_days)

    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        return self.active_experiments.get(experiment_id)

    async def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Experiment]:
        """List experiments with optional status filter"""

        experiments = list(self.active_experiments.values())

        if status:
            experiments = [exp for exp in experiments if exp.status == status]

        return sorted(experiments, key=lambda x: x.created_at, reverse=True)


class ExperimentDesignError(Exception):
    """Exception for experiment design validation errors"""

    pass
