"""A/B testing framework for personalization algorithms."""

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from scipy import stats

from ..database.postgres_client import PostgreSQLClient
from ..database.redis_client import RedisClient
from ..models.schemas import ABExperiment, UserProfile

logger = structlog.get_logger()


class ABTestingFramework:
    """A/B testing framework for personalization algorithms."""

    def __init__(self, redis_client: RedisClient,
                 postgres_client: PostgreSQLClient):
        self.redis = redis_client
        self.postgres = postgres_client
        self.cache_ttl = 3600  # 1 hour

    async def create_experiment(self, experiment: ABExperiment) -> str:
        """Create a new A/B test experiment."""
        try:
            # Save experiment to database
            query = """
            INSERT INTO ab_experiments (
                experiment_id, name, description, variants, traffic_allocation,
                start_date, end_date, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """

            await self.postgres.execute(
                query,
                experiment.experiment_id,
                experiment.name,
                experiment.description,
                experiment.variants,
                experiment.traffic_allocation,
                experiment.start_date,
                experiment.end_date,
                experiment.status,
            )

            logger.info(
                f"Created A/B test experiment: {experiment.experiment_id}")
            return experiment.experiment_id

        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise

    async def get_algorithm_variant(
            self, user_id: str, experiment: str) -> str:
        """Get algorithm variant for a user in an experiment."""
        # Check if user is already assigned
        cached_variant = await self.redis.get(f"ab_test:{experiment}:{user_id}")
        if cached_variant:
            return cached_variant

        # Get experiment details
        experiment_data = await self._get_experiment(experiment)
        if not experiment_data:
            return "default"

        # Check if experiment is active
        if not self._is_experiment_active(experiment_data):
            return "default"

        # Assign user to variant
        variant = await self._assign_user_to_variant(user_id, experiment_data)

        # Cache assignment
        await self.redis.setex(f"ab_test:{experiment}:{user_id}", self.cache_ttl, variant)

        return variant

    async def _get_experiment(
            self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment details from database."""
        query = """
        SELECT * FROM ab_experiments WHERE experiment_id = $1
        """

        return await self.postgres.fetch_one(query, experiment_id)

    def _is_experiment_active(self, experiment_data: Dict[str, Any]) -> bool:
        """Check if experiment is active."""
        if experiment_data["status"] != "active":
            return False

        now = datetime.utcnow()

        if experiment_data["start_date"] and now < experiment_data["start_date"]:
            return False

        if experiment_data["end_date"] and now > experiment_data["end_date"]:
            return False

        return True

    async def _assign_user_to_variant(
            self, user_id: str, experiment_data: Dict[str, Any]) -> str:
        """Assign user to a variant based on traffic allocation."""
        variants = experiment_data["variants"]
        traffic_allocation = experiment_data["traffic_allocation"]

        # Use consistent hashing for user assignment
        user_hash = self._hash_user_id(user_id)

        # Assign based on traffic allocation
        cumulative_prob = 0.0
        for variant, probability in traffic_allocation.items():
            cumulative_prob += probability
            if user_hash <= cumulative_prob:
                # Record assignment
                await self._record_user_assignment(
                    user_id, experiment_data["experiment_id"], variant
                )
                return variant

        # Fallback to first variant
        return list(variants.keys())[0]

    def _hash_user_id(self, user_id: str) -> float:
        """Generate consistent hash for user ID."""
        # Use SHA-256 and normalize to [0, 1]
        hash_obj = hashlib.sha256(user_id.encode())
        hash_int = int(hash_obj.hexdigest()[:8], 16)
        return hash_int / (2**32)

    async def _record_user_assignment(
            self,
            user_id: str,
            experiment_id: str,
            variant: str):
         -> Dict[str, Any]:"""Record user assignment to variant."""
        query = """
        INSERT INTO user_experiments (user_id, experiment_id, variant, assigned_at)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, experiment_id) DO UPDATE SET
        variant = $3, assigned_at = $4
        """

        await self.postgres.execute(query, user_id, experiment_id, variant, datetime.utcnow())

    async def record_experiment_event(self,
                                      user_id: str,
                                      experiment_id: str,
                                      event_type: str,
                                      event_data: Dict[str,
                                                       Any] = None):
         -> Dict[str, Any]:"""Record an event for an experiment."""
        try:
            # Get user's variant
            variant = await self.get_algorithm_variant(user_id, experiment_id)

            # Record event (this would be stored in a separate events table)
            event = {
                "user_id": user_id,
                "experiment_id": experiment_id,
                "variant": variant,
                "event_type": event_type,
                "event_data": event_data or {},
                "timestamp": datetime.utcnow(),
            }

            # Store in Redis for real-time analytics
            await self.redis.lpush(f"experiment_events:{experiment_id}", event)

            logger.info(
                f"Recorded experiment event: {event_type} for user {user_id}")

        except Exception as e:
            logger.error(f"Failed to record experiment event: {e}")

    async def get_experiment_results(
            self, experiment_id: str) -> Dict[str, Any]:
    """Get experiment results and statistics."""
        try:
            # Get experiment details
            experiment = await self._get_experiment(experiment_id)
            if not experiment:
                return {"error": "Experiment not found"}

            # Get user assignments
            assignments = await self._get_user_assignments(experiment_id)

            # Get events for this experiment
            events = await self._get_experiment_events(experiment_id)

            # Calculate metrics for each variant
            variant_metrics = {}
            for variant in experiment["variants"].keys():
                variant_users = [
                    a for a in assignments if a["variant"] == variant]
                variant_events = [e for e in events if e["variant"] == variant]

                metrics = await self._calculate_variant_metrics(variant_users, variant_events)
                variant_metrics[variant] = metrics

            # Calculate statistical significance
            significance = await self._calculate_statistical_significance(variant_metrics)

            return {
                "experiment_id": experiment_id,
                "experiment_name": experiment["name"],
                "status": experiment["status"],
                "variants": variant_metrics,
                "statistical_significance": significance,
                "total_users": len(assignments),
                "total_events": len(events),
            }

        except Exception as e:
            logger.error(f"Failed to get experiment results: {e}")
            return {"error": str(e)}

    async def _get_user_assignments(
            self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get user assignments for an experiment."""
        query = """
        SELECT user_id, variant, assigned_at
        FROM user_experiments
        WHERE experiment_id = $1
        """

        return await self.postgres.fetch_all(query, experiment_id)

    async def _get_experiment_events(
            self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get events for an experiment."""
        # Get events from Redis
        events = await self.redis.lrange(f"experiment_events:{experiment_id}", 0, -1)

        # Parse events
        parsed_events = []
        for event in events:
            try:
                parsed_events.append(event)
            except Exception as e:
                logger.error(f"Failed to parse event: {e}")

        return parsed_events

    async def _calculate_variant_metrics(
        self, users: List[Dict[str, Any]], events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
    """Calculate metrics for a variant."""
        if not users:
            return {}

        # Basic metrics
        total_users = len(users)
        total_events = len(events)

        # Event types
        event_types = {}
        for event in events:
            event_type = event.get("event_type", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1

        # Conversion rate (users with at least one event)
        users_with_events = len(set(event["user_id"] for event in events))
        conversion_rate = users_with_events / total_users if total_users > 0 else 0

        # Average events per user
        avg_events_per_user = total_events / total_users if total_users > 0 else 0

        return {
            "total_users": total_users,
            "total_events": total_events,
            "conversion_rate": conversion_rate,
            "avg_events_per_user": avg_events_per_user,
            "event_types": event_types,
        }

    async def _calculate_statistical_significance(
        self, variant_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Calculate statistical significance between variants."""
        if len(variant_metrics) < 2:
            return {"error": "Need at least 2 variants for significance testing"}

        variants = list(variant_metrics.keys())

        # Compare first two variants (in production, compare all pairs)
        variant1 = variants[0]
        variant2 = variants[1]

        metrics1 = variant_metrics[variant1]
        metrics2 = variant_metrics[variant2]

        # Chi-square test for conversion rate
        n1 = metrics1["total_users"]
        n2 = metrics2["total_users"]
        x1 = metrics1["total_events"]
        x2 = metrics2["total_events"]

        if n1 > 0 and n2 > 0:
            # Create contingency table
            observed = np.array([[x1, n1 - x1], [x2, n2 - x2]])

            # Chi-square test
            chi2, p_value = stats.chi2_contingency(observed)[:2]

            significance = {
                "chi_square": chi2,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "confidence_level": 0.95,
            }
        else:
            significance = {
                "error": "Insufficient data for significance testing"}

        return significance

    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        try:
            query = """
            UPDATE ab_experiments
            SET status = 'stopped', end_date = $1
            WHERE experiment_id = $2
            """

            await self.postgres.execute(query, datetime.utcnow(), experiment_id)

            # Clear cache
            await self.redis.delete_pattern(f"ab_test:{experiment_id}:*")

            logger.info(f"Stopped experiment: {experiment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to stop experiment: {e}")
            return False

    async def get_active_experiments(self) -> List[Dict[str, Any]]:
        """Get all active experiments."""
        query = """
        SELECT * FROM ab_experiments
        WHERE status = 'active'
        AND (start_date IS NULL OR start_date <= $1)
        AND (end_date IS NULL OR end_date >= $1)
        """

        return await self.postgres.fetch_all(query, datetime.utcnow())

    async def get_experiment_analytics(self) -> Dict[str, Any]:
        """Get analytics about all experiments."""
        try:
            # Get all experiments
            all_experiments = await self.postgres.fetch_all(
                "SELECT * FROM ab_experiments ORDER BY created_at DESC"
            )

            # Get active experiments
            active_experiments = await self.get_active_experiments()

            # Get total user assignments
            total_assignments = await self.postgres.fetch_one(
                "SELECT COUNT(*) as count FROM user_experiments"
            )

            return {
                "total_experiments": len(all_experiments),
                "active_experiments": len(active_experiments),
                "total_user_assignments": total_assignments["count"],
                "experiments": all_experiments,
            }

        except Exception as e:
            logger.error(f"Failed to get experiment analytics: {e}")
            return {"error": str(e)}
