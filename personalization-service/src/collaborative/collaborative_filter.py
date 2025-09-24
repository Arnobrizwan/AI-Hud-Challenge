"""Collaborative filtering implementation."""

from typing import Dict, List, Optional

import structlog

from ..database.postgres_client import PostgreSQLClient
from ..database.redis_client import RedisClient
from ..models.schemas import Recommendation, SimilarUser, UserInteraction
from .matrix_factorization import ImplicitMatrixFactorization

logger = structlog.get_logger()


class CollaborativeFilter:
    """Collaborative filtering with matrix factorization."""

    def __init__(self, redis_client: RedisClient, postgres_client: PostgreSQLClient):
        self.redis = redis_client
        self.postgres = postgres_client
        self.mf_model = ImplicitMatrixFactorization()
        self.cache_ttl = 3600  # 1 hour

    async def initialize(self) -> None:
        """Initialize the collaborative filtering model."""
        logger.info("Initializing collaborative filtering model")

        # Load recent interactions for training
        interactions = await self._load_recent_interactions()

        if interactions:
    await self.mf_model.train_model(interactions)
            logger.info("Collaborative filtering model trained successfully")
        else:
            logger.warning("No interactions found for training")

    async def _load_recent_interactions(self, limit: int = 100000) -> List[UserInteraction]:
        """Load recent user interactions for training."""
        query = """
        SELECT user_id, item_id, interaction_type, rating, timestamp, context,
               session_id, device_type, location
        FROM user_interactions
        ORDER BY timestamp DESC
        LIMIT $1
        """

        rows = await self.postgres.fetch_all(query, limit)

        interactions = []
        for row in rows:
            interaction = UserInteraction(
                user_id=row["user_id"],
                item_id=row["item_id"],
                interaction_type=row["interaction_type"],
                rating=row["rating"],
                timestamp=row["timestamp"],
                context=row["context"] or {},
                session_id=row["session_id"],
                device_type=row["device_type"],
                location=row["location"],
            )
            interactions.append(interaction)

        return interactions

    async def predict_batch(self, user_id: str, item_ids: List[str]) -> List[float]:
        """Batch prediction for multiple items."""
        # Try cache first
        cache_key = f"cf_predictions:{user_id}:{hash(tuple(sorted(item_ids)))}"
        cached_scores = await self.redis.get(cache_key)

        if cached_scores:
            return cached_scores

        # Get predictions from model
        scores = await self.mf_model.predict_batch(user_id, item_ids)

        # Cache results
        await self.redis.setex(cache_key, self.cache_ttl, scores)

        return scores

    async def get_similar_users(self, user_id: str, n_users: int = 10) -> List[SimilarUser]:
        """Find similar users for collaborative filtering."""
        cache_key = f"similar_users:{user_id}:{n_users}"
        cached_users = await self.redis.get(cache_key)

        if cached_users:
            return [SimilarUser(**user) for user in cached_users]

        # Get similar users from model
        similar_users = await self.mf_model.get_similar_users(user_id, n_users)

        # Cache results
        await self.redis.setex(cache_key, self.cache_ttl, [user.dict() for user in similar_users])

        return similar_users

    async def get_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Recommendation]:
        """Get collaborative filtering recommendations."""
        cache_key = f"cf_recommendations:{user_id}:{n_recommendations}"
        cached_recs = await self.redis.get(cache_key)

        if cached_recs:
            return [Recommendation(**rec) for rec in cached_recs]

        # Get recommendations from model
        recommendations = await self.mf_model.get_recommendations(user_id, n_recommendations)

        # Convert to Recommendation objects
        rec_objects = []
        for item_id, score in recommendations:
            rec_objects.append(
                Recommendation(
                    item_id=item_id,
                    score=score,
                    method="collaborative",
                    features={"cf_score": score},
                )
            )

        # Cache results
        await self.redis.setex(cache_key, self.cache_ttl, [rec.dict() for rec in rec_objects])

        return rec_objects

    async def update_user_factors(self, user_id: str, item_id: str, rating: float) -> None:
        """Update user factors with new interaction."""
        await self.mf_model.update_user_factors(user_id, item_id, rating)

        # Invalidate cache for this user
        await self._invalidate_user_cache(user_id)

    async def _invalidate_user_cache(self, user_id: str) -> None:
        """Invalidate cache entries for a user."""
        patterns = [
            f"cf_predictions:{user_id}:*",
            f"similar_users:{user_id}:*",
            f"cf_recommendations:{user_id}:*",
        ]

        for pattern in patterns:
    await self.redis.delete_pattern(pattern)

    async def retrain_model(self) -> None:
        """Retrain the collaborative filtering model."""
        logger.info("Retraining collaborative filtering model")

        # Load fresh interactions
        interactions = await self._load_recent_interactions()

        if interactions:
    await self.mf_model.train_model(interactions)
            logger.info("Collaborative filtering model retrained successfully")

            # Clear all caches
            await self.redis.delete_pattern("cf_*")
        else:
            logger.warning("No interactions found for retraining")

    async def get_model_metrics(self) -> Dict[str, any]:
        """Get model performance metrics."""
        model_info = self.mf_model.get_model_info()

        # Add additional metrics
        metrics = {
            **model_info,
            "cache_hit_rate":
    await self._get_cache_hit_rate(),
            "last_training_time":
    await self._get_last_training_time(),
        }

        return metrics

    async def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate for collaborative filtering."""
        # This would be implemented with proper cache metrics
        return 0.0

    async def _get_last_training_time(self) -> Optional[str]:
        """Get last training time."""
        # This would be stored in the database
        return None

    async def get_user_factors(self, user_id: str) -> Optional[List[float]]:
        """Get user factor vector."""
        factors = await self.mf_model.get_user_factors(user_id)
        if factors is not None:
            return factors.tolist()
        return None

    async def get_item_factors(self, item_id: str) -> Optional[List[float]]:
        """Get item factor vector."""
        factors = await self.mf_model.get_item_factors(item_id)
        if factors is not None:
            return factors.tolist()
        return None
