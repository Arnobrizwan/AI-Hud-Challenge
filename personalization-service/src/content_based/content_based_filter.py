"""Content-based filtering implementation."""

import asyncio
from typing import Dict, List

import structlog

from ..database.postgres_client import PostgreSQLClient
from ..database.redis_client import RedisClient
from ..models.schemas import ContentItem, Recommendation, UserProfile
from .embedding_filter import EmbeddingContentFilter
from .tfidf_filter import TFIDFContentFilter

logger = structlog.get_logger()


class ContentBasedFilter:
    """Content-based filtering with TF-IDF and embeddings."""

    def __init__(self, redis_client: RedisClient, postgres_client: PostgreSQLClient):
        self.redis = redis_client
        self.postgres = postgres_client
        self.tfidf_filter = TFIDFContentFilter()
        self.embedding_filter = EmbeddingContentFilter()
        self.cache_ttl = 3600  # 1 hour

    async def initialize(self) -> None:
        """Initialize the content-based filtering models."""
        logger.info("Initializing content-based filtering models")

        # Load content items for training
        content_items = await self._load_content_items()

        if content_items:
            # Fit both models
            await asyncio.gather(self.tfidf_filter.fit(content_items), self.embedding_filter.fit(content_items))
            logger.info("Content-based filtering models trained successfully")
        else:
            logger.warning("No content items found for training")

    async def _load_content_items(self, limit: int = 50000) -> List[ContentItem]:
        """Load content items for training."""
        query = """
        SELECT item_id, title, content, topics, source, author, published_at, content_features
        FROM content_items
        WHERE content IS NOT NULL AND title IS NOT NULL
        ORDER BY published_at DESC
        LIMIT $1
        """

        rows = await self.postgres.fetch_all(query, limit)

        content_items = []
        for row in rows:
            item = ContentItem(
                id=row["item_id"],
                title=row["title"],
                content=row["content"],
                topics=row["topics"] or [],
                source=row["source"],
                author=row["author"],
                published_at=row["published_at"],
                content_features=row["content_features"] or {},
            )
            content_items.append(item)

        return content_items

    async def compute_similarities(self, user_profile: UserProfile, candidates: List[ContentItem]) -> List[float]:
        """Compute content-based similarity scores for candidates."""
        # Try cache first
        cache_key = f"cb_similarities:{user_profile.user_id}:{hash(tuple(c.id for c in candidates))}"
        cached_scores = await self.redis.get(cache_key)

        if cached_scores:
            return cached_scores

        # Compute similarities using both methods
        tfidf_scores = await self.tfidf_filter.compute_similarities(user_profile, candidates)
        embedding_scores = await self.embedding_filter.compute_similarities(user_profile, candidates)

        # Combine scores (weighted average)
        combined_scores = []
        for tfidf_score, embedding_score in zip(tfidf_scores, embedding_scores):
            # Weight based on user profile preferences
            tfidf_weight = 0.4
            embedding_weight = 0.6

            combined_score = tfidf_score * tfidf_weight + embedding_score * embedding_weight
            combined_scores.append(combined_score)

        # Cache results
        await self.redis.setex(cache_key, self.cache_ttl, combined_scores)

        return combined_scores

    async def get_recommendations(self, user_profile: UserProfile, n_recommendations: int = 10) -> List[Recommendation]:
        """Get content-based recommendations."""
        cache_key = f"cb_recommendations:{user_profile.user_id}:{n_recommendations}"
        cached_recs = await self.redis.get(cache_key)

        if cached_recs:
            return [Recommendation(**rec) for rec in cached_recs]

        # Get content items for recommendation
        content_items = await self._get_recommendation_candidates(user_profile, n_recommendations * 2)

        if not content_items:
            return []

        # Compute similarities
        similarities = await self.compute_similarities(user_profile, content_items)

        # Create recommendations
        recommendations = []
        for item, score in zip(content_items, similarities):
            recommendations.append(
                Recommendation(
                    item_id=item.id,
                    score=score,
                    method="content_based",
                    features=await self._extract_recommendation_features(item, user_profile),
                    topics=item.topics,
                    source=item.source,
                )
            )

        # Sort by score and return top N
        recommendations.sort(key=lambda x: x.score, reverse=True)
        top_recommendations = recommendations[:n_recommendations]

        # Cache results
        await self.redis.setex(cache_key, self.cache_ttl, [rec.dict() for rec in top_recommendations])

        return top_recommendations

    async def _get_recommendation_candidates(self, user_profile: UserProfile, limit: int) -> List[ContentItem]:
        """Get candidate content items for recommendation."""
        # Get content based on user preferences
        query = """
        SELECT item_id, title, content, topics, source, author, published_at, content_features
        FROM content_items
        WHERE content IS NOT NULL AND title IS NOT NULL
        ORDER BY published_at DESC
        LIMIT $1
        """

        rows = await self.postgres.fetch_all(query, limit)

        content_items = []
        for row in rows:
            item = ContentItem(
                id=row["item_id"],
                title=row["title"],
                content=row["content"],
                topics=row["topics"] or [],
                source=row["source"],
                author=row["author"],
                published_at=row["published_at"],
                content_features=row["content_features"] or {},
            )
            content_items.append(item)

        return content_items

    async def _extract_recommendation_features(
        self, content_item: ContentItem, user_profile: UserProfile
    ) -> Dict[str, any]:
        """Extract features for a recommendation."""
        features = {}

        # TF-IDF features
        tfidf_features = await self.tfidf_filter.get_content_features(content_item)
        features.update(tfidf_features)

        # Embedding features
        embedding_features = await self.embedding_filter.get_content_features(content_item)
        features.update(embedding_features)

        # Topic similarity
        topic_similarity = await self.embedding_filter.compute_topic_similarity(user_profile, content_item)
        features["topic_similarity"] = topic_similarity

        # Source preference
        source_preference = user_profile.source_preferences.get(content_item.source, 0.0)
        features["source_preference"] = source_preference

        return features

    async def get_similar_content(self, content_id: str, n_similar: int = 10) -> List[Recommendation]:
        """Get similar content based on content similarity."""
        cache_key = f"similar_content:{content_id}:{n_similar}"
        cached_recs = await self.redis.get(cache_key)

        if cached_recs:
            return [Recommendation(**rec) for rec in cached_recs]

        # Get similar content from both models
        tfidf_similar = await self.tfidf_filter.get_similar_content(content_id, n_similar)
        embedding_similar = await self.embedding_filter.get_similar_content(content_id, n_similar)

        # Combine results
        combined_similar = {}
        for item_id, score in tfidf_similar:
            combined_similar[item_id] = combined_similar.get(item_id, 0) + score * 0.4

        for item_id, score in embedding_similar:
            combined_similar[item_id] = combined_similar.get(item_id, 0) + score * 0.6

        # Create recommendations
        recommendations = []
        for item_id, score in sorted(combined_similar.items(), key=lambda x: x[1], reverse=True):
            recommendations.append(
                Recommendation(
                    item_id=item_id,
                    score=score,
                    method="content_similarity",
                    features={"similarity_score": score},
                )
            )

        # Cache results
        await self.redis.setex(cache_key, self.cache_ttl, [rec.dict() for rec in recommendations])

        return recommendations

    async def retrain_models(self) -> None:
        """Retrain the content-based filtering models."""
        logger.info("Retraining content-based filtering models")

        # Load fresh content items
        content_items = await self._load_content_items()

        if content_items:
            # Retrain both models
            await asyncio.gather(self.tfidf_filter.fit(content_items), self.embedding_filter.fit(content_items))
            logger.info("Content-based filtering models retrained successfully")

            # Clear caches
            await self.redis.delete_pattern("cb_*")
            await self.redis.delete_pattern("similar_content:*")
        else:
            logger.warning("No content items found for retraining")

    async def get_model_metrics(self) -> Dict[str, any]:
        """Get model performance metrics."""
        tfidf_info = self.tfidf_filter.get_model_info()
        embedding_info = self.embedding_filter.get_model_info()

        return {
            "tfidf_model": tfidf_info,
            "embedding_model": embedding_info,
            "cache_hit_rate": await self._get_cache_hit_rate(),
        }

    async def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate for content-based filtering."""
        # This would be implemented with proper cache metrics
        return 0.0
