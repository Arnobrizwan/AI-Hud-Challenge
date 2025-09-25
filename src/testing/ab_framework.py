"""A/B testing framework for ranking algorithms."""

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog

from ..optimization.cache import CacheManager
from ..schemas import ABTestExperiment, ABTestVariant

logger = structlog.get_logger(__name__)


@dataclass
class ExperimentResult:
    """Result of an A/B test experiment."""

    experiment_id: str
    variant_id: str
    user_id: str
    timestamp: datetime
    metrics: Dict[str, float]


class ABTestingFramework:
    """A/B testing framework for ranking algorithms."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.experiments: Dict[str, ABTestExperiment] = {}
        self.results: List[ExperimentResult] = []
        self.assignment_cache: Dict[str, str] = {}  # user_id -> variant_id

        # Initialize with default experiments lazily
        self._initialized = False

    async def _initialize_experiments(self) -> Dict[str, Any]:
        """Initialize default A/B test experiments."""
        try:
            # Create default ranking algorithm experiment
            ranking_experiment = ABTestExperiment(
                experiment_id="ranking_algorithm_v4",
                name="Ranking Algorithm Comparison",
                variants=[
                    ABTestVariant(
                        variant_id="ml_ranker",
                        name="ML-based Ranking",
                        weight=0.4,
                        config={
                            "algorithm": "lightgbm",
                            "model_version": "v3",
                            "features": "comprehensive",
                        },
                    ),
                    ABTestVariant(
                        variant_id="hybrid",
                        name="Hybrid ML + Heuristic",
                        weight=0.3,
                        config={
                            "algorithm": "hybrid",
                            "ml_weight": 0.7,
                            "heuristic_weight": 0.3},
                    ),
                    ABTestVariant(
                        variant_id="heuristic",
                        name="Heuristic-only",
                        weight=0.3,
                        config={
                            "algorithm": "heuristic",
                            "weights": {
                                "freshness": 0.3,
                                "relevance": 0.25,
                                "authority": 0.2,
                                "personalization": 0.15,
                                "diversity": 0.1,
                            },
                        },
                    ),
                ],
                start_date=datetime.utcnow(),
                is_active=True,
            )

            self.experiments[ranking_experiment.experiment_id] = ranking_experiment

            # Create personalization experiment
            personalization_experiment = ABTestExperiment(
                experiment_id="personalization_weights",
                name="Personalization Weight Tuning",
                variants=[
                    ABTestVariant(
                        variant_id="high_personalization",
                        name="High Personalization",
                        weight=0.5,
                        config={
                            "personalization_weight": 0.3,
                            "topic_weight": 0.3,
                            "source_weight": 0.2,
                            "cf_weight": 0.2,
                            "cb_weight": 0.2,
                            "time_weight": 0.1,
                        },
                    ),
                    ABTestVariant(
                        variant_id="low_personalization",
                        name="Low Personalization",
                        weight=0.5,
                        config={
                            "personalization_weight": 0.1,
                            "topic_weight": 0.2,
                            "source_weight": 0.15,
                            "cf_weight": 0.15,
                            "cb_weight": 0.15,
                            "time_weight": 0.05,
                        },
                    ),
                ],
                start_date=datetime.utcnow(),
                is_active=True,
            )

            self.experiments[personalization_experiment.experiment_id] = personalization_experiment

            logger.info(
                "A/B test experiments initialized",
                count=len(
                    self.experiments))

        except Exception as e:
            logger.error(
                "Failed to initialize A/B test experiments",
                error=str(e))

    async def get_variant(self, user_id: str, experiment: str) -> str:
        """Get variant assignment for user in experiment."""
        try:
            # Ensure experiments are initialized
            if not self._initialized:
                await self._initialize_experiments()
                self._initialized = True

            # Check if user is already assigned
            cache_key = f"ab_assignment:{experiment}:{user_id}"
            cached_variant = await self.cache_manager.get(cache_key)
            if cached_variant:
                return cached_variant

            # Get experiment
            exp = self.experiments.get(experiment)
            if not exp or not exp.is_active:
                logger.warning(
                    "Experiment not found or inactive",
                    experiment=experiment)
                return "control"  # Default variant

            # Check if experiment is still active
            if exp.end_date and datetime.utcnow() > exp.end_date:
                exp.is_active = False
                return "control"

            # Assign variant using consistent hashing
            variant_id = await self._assign_variant(user_id, exp)

            # Cache assignment
            # 24 hours
            await self.cache_manager.set(cache_key, variant_id, ttl=86400)

            # Log assignment
            await self._log_assignment(user_id, experiment, variant_id)

            return variant_id

        except Exception as e:
            logger.error(
                "Variant assignment failed",
                error=str(e),
                user_id=user_id,
                experiment=experiment)
            return "control"

    async def _assign_variant(
            self,
            user_id: str,
            experiment: ABTestExperiment) -> str:
        """Assign variant to user using consistent hashing."""
        # Use consistent hashing to ensure same user gets same variant
        hash_input = f"{user_id}:{experiment.experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Normalize to 0-1 range
        normalized_hash = (hash_value % 10000) / 10000.0

        # Assign based on cumulative weights
        cumulative_weight = 0.0
        for variant in experiment.variants:
            cumulative_weight += variant.weight
            if normalized_hash <= cumulative_weight:
                return variant.variant_id

        # Fallback to last variant
        return experiment.variants[-1].variant_id

    async def _log_assignment(
            self,
            user_id: str,
            experiment: str,
            variant: str) -> Dict[str, Any]:
        """Log variant assignment for analysis."""
        try:
            assignment = {
                "user_id": user_id,
                "experiment": experiment,
                "variant": variant,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # In production, this would log to a data warehouse
            logger.info("Variant assigned", **assignment)

        except Exception as e:
            logger.warning("Failed to log assignment", error=str(e))

    async def record_result(self,
                            user_id: str,
                            experiment: str,
                            variant: str,
                            metrics: Dict[str,
                                          float]) -> Dict[str, Any]:
        """Record experiment result for analysis."""
        try:
            result = ExperimentResult(
                experiment_id=experiment,
                variant_id=variant,
                user_id=user_id,
                timestamp=datetime.utcnow(),
                metrics=metrics,
            )

            self.results.append(result)

            # In production, this would store in a database
            logger.info(
                "Experiment result recorded",
                experiment=experiment,
                variant=variant,
                user_id=user_id,
            )

        except Exception as e:
            logger.error("Failed to record experiment result", error=str(e))

    async def get_experiment_stats(self, experiment: str) -> Dict[str, Any]:
        """Get statistics for an experiment."""
        try:
            # Ensure experiments are initialized
            if not self._initialized:
                await self._initialize_experiments()
                self._initialized = True

            exp = self.experiments.get(experiment)
            if not exp:
                return {"error": "Experiment not found"}

            # Filter results for this experiment
            exp_results = [
                r for r in self.results if r.experiment_id == experiment]

            if not exp_results:
                return {
                    "experiment_id": experiment,
                    "total_users": 0,
                    "variants": {}}

            # Group by variant
            variant_stats = {}
            for variant in exp.variants:
                variant_results = [
                    r for r in exp_results if r.variant_id == variant.variant_id]

                if variant_results:
                    # Calculate metrics
                    user_count = len(set(r.user_id for r in variant_results))

                    # Calculate average metrics
                    avg_metrics = {}
                    for metric in variant_results[0].metrics.keys():
                        values = [r.metrics.get(metric, 0)
                                  for r in variant_results]
                        avg_metrics[metric] = sum(
                            values) / len(values) if values else 0

                    variant_stats[variant.variant_id] = {
                        "name": variant.name,
                        "user_count": user_count,
                        "avg_metrics": avg_metrics,
                        "weight": variant.weight,
                    }
                else:
                    variant_stats[variant.variant_id] = {
                        "name": variant.name,
                        "user_count": 0,
                        "avg_metrics": {},
                        "weight": variant.weight,
                    }

            return {
                "experiment_id": experiment,
                "total_users": len(set(r.user_id for r in exp_results)),
                "variants": variant_stats,
                "is_active": exp.is_active,
                "start_date": exp.start_date.isoformat(),
                "end_date": exp.end_date.isoformat() if exp.end_date else None,
            }

        except Exception as e:
            logger.error("Failed to get experiment stats", error=str(e))
            return {"error": str(e)}

    async def create_experiment(self, experiment: ABTestExperiment) -> bool:
        """Create a new A/B test experiment."""
        try:
            # Validate experiment
            if not self._validate_experiment(experiment):
                return False

            # Store experiment
            self.experiments[experiment.experiment_id] = experiment

            # Cache experiment config
            await self.cache_manager.set(
                f"experiment:{experiment.experiment_id}", experiment.dict(), ttl=86400
            )

            logger.info(
                "Experiment created",
                experiment_id=experiment.experiment_id)
            return True

        except Exception as e:
            logger.error("Failed to create experiment", error=str(e))
            return False

    async def update_experiment(
            self, experiment_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing experiment."""
        try:
            exp = self.experiments.get(experiment_id)
            if not exp:
                return False

            # Update experiment fields
            for key, value in updates.items():
                if hasattr(exp, key):
                    setattr(exp, key, value)

            # Update cache
            await self.cache_manager.set(f"experiment:{experiment_id}", exp.dict(), ttl=86400)

            logger.info("Experiment updated", experiment_id=experiment_id)
            return True

        except Exception as e:
            logger.error("Failed to update experiment", error=str(e))
            return False

    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        try:
            exp = self.experiments.get(experiment_id)
            if not exp:
                return False

            exp.is_active = False
            exp.end_date = datetime.utcnow()

            # Update cache
            await self.cache_manager.set(f"experiment:{experiment_id}", exp.dict(), ttl=86400)

            logger.info("Experiment stopped", experiment_id=experiment_id)
            return True

        except Exception as e:
            logger.error("Failed to stop experiment", error=str(e))
            return False

    def _validate_experiment(self, experiment: ABTestExperiment) -> bool:
        """Validate experiment configuration."""
        try:
            # Check required fields
            if not experiment.experiment_id or not experiment.name:
                return False

            # Check variants
            if not experiment.variants or len(experiment.variants) < 2:
                return False

            # Check weights sum to 1.0
            total_weight = sum(v.weight for v in experiment.variants)
            if abs(total_weight - 1.0) > 0.01:
                return False

            # Check variant IDs are unique
            variant_ids = [v.variant_id for v in experiment.variants]
            if len(variant_ids) != len(set(variant_ids)):
                return False

            return True

        except Exception as e:
            logger.error("Experiment validation failed", error=str(e))
            return False

    async def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments with their status."""
        experiments = []
        for exp in self.experiments.values():
            stats = await self.get_experiment_stats(exp.experiment_id)
            experiments.append(
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "is_active": exp.is_active,
                    "start_date": exp.start_date.isoformat(),
                    "end_date": exp.end_date.isoformat() if exp.end_date else None,
                    "variant_count": len(
                        exp.variants),
                    "stats": stats,
                })

        return experiments

    async def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Perform statistical analysis of experiment results."""
        try:
            exp = self.experiments.get(experiment_id)
            if not exp:
                return {"error": "Experiment not found"}

            # Get experiment results
            exp_results = [
                r for r in self.results if r.experiment_id == experiment_id]
            if not exp_results:
                return {"error": "No results found"}

            # Group by variant
            variant_data = {}
            for variant in exp.variants:
                variant_results = [
                    r for r in exp_results if r.variant_id == variant.variant_id]
                variant_data[variant.variant_id] = variant_results

            # Perform statistical tests
            analysis = {
                "experiment_id": experiment_id,
                "total_users": len(set(r.user_id for r in exp_results)),
                "analysis_date": datetime.utcnow().isoformat(),
                "variants": {},
            }

            for variant_id, results in variant_data.items():
                if not results:
                    continue

                # Calculate metrics
                user_count = len(set(r.user_id for r in results))

                # Calculate average metrics
                avg_metrics = {}
                for metric in results[0].metrics.keys():
                    values = [r.metrics.get(metric, 0) for r in results]
                    avg_metrics[metric] = {
                        "mean": sum(values) / len(values) if values else 0,
                        "count": len(values),
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0,
                    }

                analysis["variants"][variant_id] = {
                    "user_count": user_count,
                    "metrics": avg_metrics,
                }

            return analysis

        except Exception as e:
            logger.error("Experiment analysis failed", error=str(e))
            return {"error": str(e)}
