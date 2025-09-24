"""
A/B Testing Framework for Summarization
Advanced A/B testing for different summarization approaches and parameters
"""

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import settings

from .models import ABTestResult, ABTestVariant, SummarizationRequest, SummaryResult

logger = logging.getLogger(__name__)


@dataclass
class TestMetrics:
    """Metrics for A/B testing"""

    quality_score: float
    processing_time: float
    user_satisfaction: Optional[float] = None
    click_through_rate: Optional[float] = None
    engagement_score: Optional[float] = None


class ABTestManager:
    """Advanced A/B testing framework for summarization"""

    def __init__(self):
        """Initialize the A/B test manager"""
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.test_results: Dict[str, List[TestMetrics]] = {}
        self.user_assignments: Dict[str, str] = {}  # user_id -> variant_id
        self._initialized = False

    async def initialize(self) -> Dict[str, Any]:
    """Initialize A/B testing framework"""
        try:
            logger.info("Initializing A/B testing framework...")

            # Load existing tests from storage (in production, this would be
            # from a database)
            await self._load_existing_tests()

            self._initialized = True
            logger.info("A/B testing framework initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize A/B testing framework: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
    """Clean up resources"""
        try:
            # Save test data (in production, this would be to a database)
            await self._save_test_data()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def create_test(self, test_name: str, variants: List[A / BTestVariant], duration_days: int = 7) -> str:
        """Create a new A/B test"""
        try:
            test_id = str(uuid.uuid4())

            # Validate variants
            if len(variants) < 2:
                raise ValueError("At least 2 variants required for A/B testing")

            # Check traffic allocation
            total_traffic = sum(v.traffic_percentage for v in variants)
            if abs(total_traffic - 1.0) > 0.01:
                raise ValueError("Traffic percentages must sum to 1.0")

            # Create test configuration
            test_config = {
                "test_id": test_id,
                "test_name": test_name,
                "variants": [v.dict() for v in variants],
                "created_at": datetime.now(),
                "end_date": datetime.now() + timedelta(days=duration_days),
                "status": "active",
                "total_participants": 0,
                "results": {},
            }

            self.active_tests[test_id] = test_config
            self.test_results[test_id] = []

            logger.info(f"Created A/B test: {test_name} (ID: {test_id})")
            return test_id

        except Exception as e:
            logger.error(f"Failed to create A/B test: {str(e)}")
            raise

    async def assign_variant(self, test_id: str, user_id: str) -> Optional[str]:
        """Assign a variant to a user for a specific test"""
        try:
            if test_id not in self.active_tests:
                logger.warning(f"Test {test_id} not found")
                return None

            test = self.active_tests[test_id]

            # Check if test is still active
            if test["status"] != "active" or datetime.now() > test["end_date"]:
                logger.warning(f"Test {test_id} is not active")
                return None

            # Check if user already assigned
            user_key = f"{test_id}:{user_id}"
            if user_key in self.user_assignments:
                return self.user_assignments[user_key]

            # Assign variant based on traffic allocation
            variant_id = await self._select_variant(test["variants"])

            if variant_id:
                self.user_assignments[user_key] = variant_id
                test["total_participants"] += 1
                logger.info(f"Assigned variant {variant_id} to user {user_id} for test {test_id}")

            return variant_id

        except Exception as e:
            logger.error(f"Failed to assign variant: {str(e)}")
            return None

    async def _select_variant(self, variants: List[Dict[str, Any]]) -> Optional[str]:
        """Select a variant based on traffic allocation"""
        try:
            # Generate random number
            random_value = np.random.random()

            # Cumulative traffic allocation
            cumulative = 0.0
            for variant in variants:
                cumulative += variant["traffic_percentage"]
                if random_value <= cumulative:
                    return variant["variant_id"]

            # Fallback to first variant
            return variants[0]["variant_id"] if variants else None

        except Exception as e:
            logger.error(f"Variant selection failed: {str(e)}")
            return None

    async def record_metrics(self, test_id: str, variant_id: str, metrics: TestMetrics) -> bool:
        """Record metrics for a test variant"""
        try:
            if test_id not in self.test_results:
                logger.warning(f"Test {test_id} not found")
                return False

            # Add variant information to metrics
            metrics_dict = {
                "variant_id": variant_id,
                "timestamp": datetime.now(),
                "quality_score": metrics.quality_score,
                "processing_time": metrics.processing_time,
                "user_satisfaction": metrics.user_satisfaction,
                "click_through_rate": metrics.click_through_rate,
                "engagement_score": metrics.engagement_score,
            }

            self.test_results[test_id].append(metrics_dict)

            logger.info(f"Recorded metrics for test {test_id}, variant {variant_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to record metrics: {str(e)}")
            return False

    async def get_test_results(self, test_id: str) -> Optional[A / BTestResult]:
        """Get A/B test results and analysis"""
        try:
            if test_id not in self.active_tests:
                logger.warning(f"Test {test_id} not found")
                return None

            test = self.active_tests[test_id]
            results = self.test_results.get(test_id, [])

            if not results:
                logger.warning(f"No results found for test {test_id}")
                return None

            # Analyze results by variant
            variant_metrics = {}
            for result in results:
                variant_id = result["variant_id"]
                if variant_id not in variant_metrics:
                    variant_metrics[variant_id] = []
                variant_metrics[variant_id].append(result)

            # Calculate statistics for each variant
            variant_stats = {}
            for variant_id, metrics_list in variant_metrics.items():
                stats = self._calculate_variant_stats(metrics_list)
                variant_stats[variant_id] = stats

            # Determine winning variant
            winning_variant = self._determine_winning_variant(variant_stats)

            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(variant_stats)

            # Calculate test duration
            test_duration = (datetime.now() - test["created_at"]).days

            return A / BTestResult(
                test_id=test_id,
                variant_id=winning_variant,
                metrics=variant_stats,
                confidence_level=confidence_level,
                sample_size=len(results),
                duration_days=test_duration,
            )

        except Exception as e:
            logger.error(f"Failed to get test results: {str(e)}")
            return None

    def _calculate_variant_stats(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate statistics for a variant"""
        try:
            if not metrics_list:
                return {}

            # Extract metrics
            quality_scores = [m["quality_score"] for m in metrics_list if m["quality_score"] is not None]
            processing_times = [m["processing_time"] for m in metrics_list if m["processing_time"] is not None]
            user_satisfaction = [m["user_satisfaction"] for m in metrics_list if m["user_satisfaction"] is not None]
            click_through_rates = [m["click_through_rate"] for m in metrics_list if m["click_through_rate"] is not None]
            engagement_scores = [m["engagement_score"] for m in metrics_list if m["engagement_score"] is not None]

            stats = {}

            if quality_scores:
                stats["avg_quality_score"] = np.mean(quality_scores)
                stats["std_quality_score"] = np.std(quality_scores)
                stats["min_quality_score"] = np.min(quality_scores)
                stats["max_quality_score"] = np.max(quality_scores)

            if processing_times:
                stats["avg_processing_time"] = np.mean(processing_times)
                stats["std_processing_time"] = np.std(processing_times)
                stats["min_processing_time"] = np.min(processing_times)
                stats["max_processing_time"] = np.max(processing_times)

            if user_satisfaction:
                stats["avg_user_satisfaction"] = np.mean(user_satisfaction)
                stats["std_user_satisfaction"] = np.std(user_satisfaction)

            if click_through_rates:
                stats["avg_click_through_rate"] = np.mean(click_through_rates)
                stats["std_click_through_rate"] = np.std(click_through_rates)

            if engagement_scores:
                stats["avg_engagement_score"] = np.mean(engagement_scores)
                stats["std_engagement_score"] = np.std(engagement_scores)

            stats["sample_size"] = len(metrics_list)

            return stats

        except Exception as e:
            logger.error(f"Failed to calculate variant stats: {str(e)}")
            return {}

    def _determine_winning_variant(self, variant_stats: Dict[str, Dict[str, float]]) -> str:
        """Determine the winning variant based on metrics"""
        try:
            if not variant_stats:
                return ""

            # Score each variant based on key metrics
            variant_scores = {}

            for variant_id, stats in variant_stats.items():
                score = 0.0
                weight = 0.0

                # Quality score (40% weight)
                if "avg_quality_score" in stats:
                    score += stats["avg_quality_score"] * 0.4
                    weight += 0.4

                # Processing time (20% weight, lower is better)
                if "avg_processing_time" in stats:
                    # Normalize processing time (assume max 10 seconds)
                    normalized_time = max(0, 1 - (stats["avg_processing_time"] / 10))
                    score += normalized_time * 0.2
                    weight += 0.2

                # User satisfaction (20% weight)
                if "avg_user_satisfaction" in stats:
                    score += stats["avg_user_satisfaction"] * 0.2
                    weight += 0.2

                # Engagement score (20% weight)
                if "avg_engagement_score" in stats:
                    score += stats["avg_engagement_score"] * 0.2
                    weight += 0.2

                # Normalize by weight
                if weight > 0:
                    variant_scores[variant_id] = score / weight
                else:
                    variant_scores[variant_id] = 0.0

            # Return variant with highest score
            return max(variant_scores, key=variant_scores.get) if variant_scores else ""

        except Exception as e:
            logger.error(f"Failed to determine winning variant: {str(e)}")
            return ""

    def _calculate_confidence_level(self, variant_stats: Dict[str, Dict[str, float]]) -> float:
        """Calculate statistical confidence level"""
        try:
            if len(variant_stats) < 2:
                return 0.0

            # Simple confidence calculation based on sample size and variance
            total_samples = sum(stats.get("sample_size", 0) for stats in variant_stats.values())

            if total_samples < 30:
                return 0.5  # Low confidence for small samples
            elif total_samples < 100:
                return 0.7  # Medium confidence
            else:
                return 0.9  # High confidence for large samples

        except Exception as e:
            logger.error(f"Failed to calculate confidence level: {str(e)}")
            return 0.0

    async def end_test(self, test_id: str) -> bool:
        """End an A/B test"""
        try:
            if test_id not in self.active_tests:
                logger.warning(f"Test {test_id} not found")
                return False

            self.active_tests[test_id]["status"] = "ended"
            self.active_tests[test_id]["ended_at"] = datetime.now()

            logger.info(f"Ended A/B test: {test_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to end test: {str(e)}")
            return False

    async def get_active_tests(self) -> List[Dict[str, Any]]:
        """Get list of active A/B tests"""
        try:
            active_tests = []
            for test_id, test in self.active_tests.items():
                if test["status"] == "active" and datetime.now() <= test["end_date"]:
                    active_tests.append(
                        {
                            "test_id": test_id,
                            "test_name": test["test_name"],
                            "created_at": test["created_at"],
                            "end_date": test["end_date"],
                            "total_participants": test["total_participants"],
                        }
                    )

            return active_tests

        except Exception as e:
            logger.error(f"Failed to get active tests: {str(e)}")
            return []

    async def _load_existing_tests(self) -> Dict[str, Any]:
    """Load existing tests from storage"""
        # In production, this would load from a database
        # For now, we'll start with empty tests
        pass

    async def _save_test_data(self) -> Dict[str, Any]:
    """Save test data to storage"""
        # In production, this would save to a database
        # For now, we'll just log the data
        logger.info(
            f"Saving {len(self.active_tests)} tests and {sum(len(results) for results in self.test_results.values())} results"
        )

    async def get_status(self) -> Dict[str, Any]:
    """Get A/B testing framework status"""
        return {
            "initialized": self._initialized,
            "active_tests": len([t for t in self.active_tests.values() if t["status"] == "active"]),
            "total_tests": len(self.active_tests),
            "total_results": sum(len(results) for results in self.test_results.values()),
        }
