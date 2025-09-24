"""Cold start handling strategies for new users and content."""

import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from ..database.postgres_client import PostgreSQLClient
from ..database.redis_client import RedisClient
from ..models.schemas import ContentItem, Recommendation, UserContext, UserProfile

logger = structlog.get_logger()


class ColdStartHandler:
    """Handle cold start scenarios for new users and content."""

    def __init__(self, redis_client: RedisClient,
                 postgres_client: PostgreSQLClient):
        self.redis = redis_client
        self.postgres = postgres_client
        self.cache_ttl = 3600  # 1 hour

    async def handle_user_cold_start(
        self, user_id: str, context: Optional[UserContext] = None
    ) -> List[Recommendation]:
        """Handle cold start for new users."""
        logger.info(f"Handling cold start for user: {user_id}")

        # Try demographic-based initialization
        demographic_recs = await self._get_demographic_recommendations(user_id, context)

        # Try popularity-based fallback
        popularity_recs = await self._get_popularity_recommendations()

        # Try onboarding preference collection
        onboarding_recs = await self._get_onboarding_recommendations()

        # Combine strategies
        combined_recs = await self._combine_cold_start_strategies(
            demographic_recs, popularity_recs, onboarding_recs
        )

        return combined_recs[:10]  # Return top 10

    async def handle_content_cold_start(
            self, content_item: ContentItem) -> Dict[str, Any]:
    """Handle cold start for new content items."""
        logger.info(f"Handling cold start for content: {content_item.id}")

        # Extract content features
        features = await self._extract_content_features(content_item)

        # Find similar content
        similar_content = await self._find_similar_content(content_item)

        # Estimate popularity
        popularity_score = await self._estimate_content_popularity(content_item)

        return {
            "features": features,
            "similar_content": similar_content,
            "popularity_score": popularity_score,
            "cold_start_handled": True,
        }

    async def _get_demographic_recommendations(
        self, user_id: str, context: Optional[UserContext]
    ) -> List[Recommendation]:
        """Get recommendations based on demographic data."""
        # Get demographic template
        template = await self._get_demographic_template(user_id, context)

        # Get content based on demographic preferences
        query = """
        SELECT item_id, title, content, topics, source, author, published_at, content_features
        FROM content_items
        WHERE topics && $1::text[] OR source = ANY($2::text[])
        ORDER BY published_at DESC
        LIMIT 20
        """

        preferred_topics = list(template["topic_preferences"].keys())
        preferred_sources = list(template["source_preferences"].keys())

        rows = await self.postgres.fetch_all(query, preferred_topics, preferred_sources)

        recommendations = []
        for row in rows:
            # Compute demographic-based score
            score = await self._compute_demographic_score(row, template)

            recommendations.append(
                Recommendation(
                    item_id=row["item_id"],
                    score=score,
                    method="demographic",
                    features={"demographic_score": score},
                    topics=row["topics"] or [],
                    source=row["source"],
                )
            )

        return sorted(recommendations, key=lambda x: x.score, reverse=True)

    async def _get_popularity_recommendations(self) -> List[Recommendation]:
        """Get recommendations based on content popularity."""
        # Get popular content from recent interactions
        query = """
        SELECT ci.item_id, ci.title, ci.content, ci.topics, ci.source, ci.author,
               ci.published_at, ci.content_features,
               COUNT(ui.id) as interaction_count,
               AVG(CASE WHEN ui.rating IS NOT NULL THEN ui.rating ELSE 0 END) as avg_rating
        FROM content_items ci
        LEFT JOIN user_interactions ui ON ci.item_id = ui.item_id
        WHERE ui.timestamp > $1 OR ui.timestamp IS NULL
        GROUP BY ci.item_id, ci.title, ci.content, ci.topics, ci.source, ci.author,
                 ci.published_at, ci.content_features
        ORDER BY interaction_count DESC, avg_rating DESC
        LIMIT 20
        """

        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        rows = await self.postgres.fetch_all(query, recent_cutoff)

        recommendations = []
        for row in rows:
            # Compute popularity score
            interaction_count = row["interaction_count"] or 0
            avg_rating = row["avg_rating"] or 0.0

            # Normalize scores
            # Normalize by 100 interactions
            popularity_score = min(1.0, interaction_count / 100.0)
            rating_score = avg_rating / 5.0  # Normalize by 5-star rating

            combined_score = popularity_score * 0.7 + rating_score * 0.3

            recommendations.append(
                Recommendation(
                    item_id=row["item_id"],
                    score=combined_score,
                    method="popularity",
                    features={
                        "popularity_score": popularity_score,
                        "rating_score": rating_score,
                        "interaction_count": interaction_count,
                    },
                    topics=row["topics"] or [],
                    source=row["source"],
                )
            )

        return sorted(recommendations, key=lambda x: x.score, reverse=True)

    async def _get_onboarding_recommendations(self) -> List[Recommendation]:
        """Get recommendations for onboarding preference collection."""
        # Get diverse content for preference collection
        query = """
        SELECT item_id, title, content, topics, source, author, published_at, content_features
        FROM content_items
        WHERE content IS NOT NULL AND title IS NOT NULL
        ORDER BY published_at DESC
        LIMIT 30
        """

        rows = await self.postgres.fetch_all(query)

        recommendations = []
        for i, row in enumerate(rows):
            # Assign onboarding score based on diversity
            # Decreasing score for diversity
            onboarding_score = 1.0 - (i / len(rows))

            recommendations.append(
                Recommendation(
                    item_id=row["item_id"],
                    score=onboarding_score,
                    method="onboarding",
                    features={"onboarding_score": onboarding_score},
                    topics=row["topics"] or [],
                    source=row["source"],
                )
            )

        return recommendations

    async def _combine_cold_start_strategies(
        self,
        demographic_recs: List[Recommendation],
        popularity_recs: List[Recommendation],
        onboarding_recs: List[Recommendation],
    ) -> List[Recommendation]:
        """Combine different cold start strategies."""
        # Create a combined recommendation list
        combined = {}

        # Add demographic recommendations with weight 0.4
        for rec in demographic_recs[:5]:
            rec.score *= 0.4
            combined[rec.item_id] = rec

        # Add popularity recommendations with weight 0.4
        for rec in popularity_recs[:5]:
            rec.score *= 0.4
            if rec.item_id in combined:
                combined[rec.item_id].score += rec.score
            else:
                combined[rec.item_id] = rec

        # Add onboarding recommendations with weight 0.2
        for rec in onboarding_recs[:5]:
            rec.score *= 0.2
            if rec.item_id in combined:
                combined[rec.item_id].score += rec.score
            else:
                combined[rec.item_id] = rec

        # Sort by combined score
        return sorted(combined.values(), key=lambda x: x.score, reverse=True)

    async def _get_demographic_template(
        self, user_id: str, context: Optional[UserContext]
    ) -> Dict[str, Any]:
    """Get demographic-based profile template."""
        # This would typically use demographic data or similar user patterns
        # For now, return a default template with some variation

        base_template = {
            "topic_preferences": {
                "technology": 0.3,
                "business": 0.2,
                "news": 0.2,
                "sports": 0.1,
                "entertainment": 0.1,
                "science": 0.1,
            },
            "source_preferences": {
                "cnn": 0.2,
                "bbc": 0.2,
                "reuters": 0.2,
                "ap": 0.2,
                "guardian": 0.2,
            },
        }

        # Add context-based adjustments
        if context:
            if context.device_type == "mobile":
                base_template["topic_preferences"]["technology"] += 0.1
            if context.time_of_day == "morning":
                base_template["topic_preferences"]["news"] += 0.1

        return base_template

    async def _compute_demographic_score(
        self, content_row: Dict[str, Any], template: Dict[str, Any]
    ) -> float:
        """Compute demographic-based score for content."""
        score = 0.0

        # Topic score
        content_topics = content_row.get("topics", [])
        topic_preferences = template.get("topic_preferences", {})

        for topic in content_topics:
            topic_score = topic_preferences.get(topic, 0.0)
            score += topic_score

        if content_topics:
            score /= len(content_topics)

        # Source score
        content_source = content_row.get("source")
        source_preferences = template.get("source_preferences", {})

        if content_source:
            source_score = source_preferences.get(content_source, 0.0)
            score = (score + source_score) / 2.0

        return score

    async def _extract_content_features(
            self, content_item: ContentItem) -> Dict[str, Any]:
    """Extract features for new content item."""
        features = {}

        # Basic features
        features["title_length"] = len(content_item.title)
        features["content_length"] = len(content_item.content or "")
        features["topic_count"] = len(content_item.topics)
        features["has_author"] = 1.0 if content_item.author else 0.0
        features["has_source"] = 1.0 if content_item.source else 0.0

        # Topic features
        for topic in content_item.topics:
            features[f"topic_{topic}"] = 1.0

        # Source features
        if content_item.source:
            features[f"source_{content_item.source}"] = 1.0

        return features

    async def _find_similar_content(
            self, content_item: ContentItem) -> List[Dict[str, Any]]:
        """Find similar content for new content item."""
        # Get content with similar topics
        query = """
        SELECT item_id, title, topics, source,
               array_length(topics, 1) as topic_count
        FROM content_items
        WHERE item_id != $1 AND topics && $2::text[]
        ORDER BY array_length(topics, 1) DESC
        LIMIT 5
        """

        rows = await self.postgres.fetch_all(query, content_item.id, content_item.topics)

        similar_content = []
        for row in rows:
            # Compute similarity score
            common_topics = len(
                set(content_item.topics).intersection(set(row["topics"] or [])))
            total_topics = len(set(content_item.topics).union(
                set(row["topics"] or [])))

            similarity_score = common_topics / total_topics if total_topics > 0 else 0.0

            similar_content.append(
                {
                    "item_id": row["item_id"],
                    "title": row["title"],
                    "similarity_score": similarity_score,
                    "common_topics": common_topics,
                }
            )

        return similar_content

    async def _estimate_content_popularity(
            self, content_item: ContentItem) -> float:
        """Estimate popularity for new content item."""
        # Use content features to estimate popularity
        popularity_score = 0.0

        # Title length factor
        title_length = len(content_item.title)
        if 20 <= title_length <= 100:
            popularity_score += 0.2

        # Content length factor
        content_length = len(content_item.content or "")
        if 200 <= content_length <= 2000:
            popularity_score += 0.2

        # Topic popularity
        topic_popularity = await self._get_topic_popularity(content_item.topics)
        popularity_score += topic_popularity * 0.3

        # Source popularity
        source_popularity = await self._get_source_popularity(content_item.source)
        popularity_score += source_popularity * 0.3

        return min(1.0, popularity_score)

    async def _get_topic_popularity(self, topics: List[str]) -> float:
        """Get popularity score for topics."""
        if not topics:
            return 0.0

        # Get topic popularity from interactions
        query = """
        SELECT COUNT(*) as interaction_count
        FROM user_interactions ui
        JOIN content_items ci ON ui.item_id = ci.item_id
        WHERE ci.topics && $1::text[]
        AND ui.timestamp > $2
        """

        recent_cutoff = datetime.utcnow() - timedelta(days=30)
        result = await self.postgres.fetch_one(query, topics, recent_cutoff)

        interaction_count = result["interaction_count"] or 0
        # Normalize by 1000 interactions
        return min(1.0, interaction_count / 1000.0)

    async def _get_source_popularity(self, source: Optional[str]) -> float:
        """Get popularity score for source."""
        if not source:
            return 0.0

        # Get source popularity from interactions
        query = """
        SELECT COUNT(*) as interaction_count
        FROM user_interactions ui
        JOIN content_items ci ON ui.item_id = ci.item_id
        WHERE ci.source = $1
        AND ui.timestamp > $2
        """

        recent_cutoff = datetime.utcnow() - timedelta(days=30)
        result = await self.postgres.fetch_one(query, source, recent_cutoff)

        interaction_count = result["interaction_count"] or 0
        # Normalize by 500 interactions
        return min(1.0, interaction_count / 500.0)

    async def get_cold_start_analytics(self) -> Dict[str, Any]:
        """Get analytics about cold start handling."""
        # Get new users in last 30 days
        new_users_query = """
        SELECT COUNT(*) as count
        FROM user_profiles
        WHERE created_at > $1
        """

        recent_cutoff = datetime.utcnow() - timedelta(days=30)
        new_users_result = await self.postgres.fetch_one(new_users_query, recent_cutoff)

        # Get new content in last 30 days
        new_content_query = """
        SELECT COUNT(*) as count
        FROM content_items
        WHERE created_at > $1
        """

        new_content_result = await self.postgres.fetch_one(new_content_query, recent_cutoff)

        return {
            "new_users_30_days": new_users_result["count"],
            "new_content_30_days": new_content_result["count"],
            "cold_start_strategies": [
                "demographic",
                "popularity",
                "onboarding"],
        }
