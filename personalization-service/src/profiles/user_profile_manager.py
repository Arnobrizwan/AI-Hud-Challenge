"""User profile management with real-time updates."""

import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from ..config.settings import settings
from ..database.postgres_client import PostgreSQLClient
from ..database.redis_client import RedisClient
from ..models.schemas import InteractionType, UserInteraction, UserProfile

logger = structlog.get_logger()


class UserProfileManager:
    """Dynamic user profiling with real-time updates."""

    def __init__(self, redis_client: RedisClient, postgres_client: PostgreSQLClient):
        self.redis = redis_client
        self.postgres = postgres_client
        self.cache_ttl = settings.profile_cache_ttl

    async def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Retrieve or initialize user profile."""
        # Try cache first
        cached_profile = await self.redis.get(f"profile:{user_id}")
        if cached_profile:
            return UserProfile.parse_raw(cached_profile)

        # Try database
        profile_data = await self.postgres.fetch_one(
            "SELECT * FROM user_profiles WHERE user_id = $1", user_id
        )

        if profile_data:
            profile = UserProfile(**profile_data)
        else:
            # Create new profile for cold start
            profile = await self.create_cold_start_profile(user_id)
            await self.save_profile(profile)

        # Cache profile
        await self.redis.setex(f"profile:{user_id}", self.cache_ttl, profile.json())

        return profile

    async def create_cold_start_profile(self, user_id: str) -> UserProfile:
        """Create a new user profile for cold start."""
        logger.info(f"Creating cold start profile for user: {user_id}")

        # Get demographic-based template if available
        template = await self._get_demographic_template(user_id)

        profile = UserProfile(
            user_id=user_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            total_interactions=0,
            last_interaction_at=None,
            topic_preferences=template.get("topic_preferences", {}),
            source_preferences=template.get("source_preferences", {}),
            reading_patterns=template.get("reading_patterns", {}),
            collaborative_weight=0.5,
            content_weight=0.5,
            diversity_preference=0.3,
            serendipity_preference=0.2,
            demographic_data=template.get("demographic_data", {}),
            privacy_settings=template.get("privacy_settings", {}),
        )

        return profile

    async def _get_demographic_template(self, user_id: str) -> Dict[str, Any]:
        """Get demographic-based profile template."""
        # This would typically use demographic data or similar user patterns
        # For now, return a default template
        return {
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
            "reading_patterns": {
                "preferred_length": "medium",
                "reading_time": "morning",
                "device_preference": "mobile",
            },
            "demographic_data": {},
            "privacy_settings": {
                "allow_personalization": True,
                "allow_data_collection": True,
                "allow_analytics": True,
            },
        }

    async def update_profile_from_interaction(
        self, user_id: str, interaction: UserInteraction
    ) -> None:
        """Update user profile based on interaction."""
        profile = await self.get_or_create_profile(user_id)

        # Update topic preferences
        if interaction.context.get("topics"):
            profile.topic_preferences = await self.update_topic_preferences(
                profile.topic_preferences,
                interaction.context["topics"],
                interaction.interaction_type,
            )

        # Update source preferences
        if interaction.context.get("source"):
            profile.source_preferences = await self.update_source_preferences(
                profile.source_preferences,
                interaction.context["source"],
                interaction.interaction_type,
            )

        # Update reading patterns
        profile.reading_patterns = await self.update_reading_patterns(
            profile.reading_patterns, interaction
        )

        # Update collaborative filtering factors
        if interaction.interaction_type in [
            InteractionType.CLICK,
            InteractionType.SHARE,
            InteractionType.SAVE,
        ]:
            # This would be handled by the collaborative filter
            pass

        # Increment interaction count and update timestamps
        profile.total_interactions += 1
        profile.last_interaction_at = interaction.timestamp
        profile.updated_at = datetime.utcnow()

        # Save updated profile
        await self.save_profile(profile)

        # Update cache
        await self.redis.setex(f"profile:{user_id}", self.cache_ttl, profile.json())

    async def update_topic_preferences(
        self,
        current_preferences: Dict[str, float],
        topics: List[str],
        interaction_type: InteractionType,
    ) -> Dict[str, float]:
        """Update topic preferences based on interaction."""
        # Get interaction weight
        weight = self._get_interaction_weight(interaction_type)

        # Update preferences
        for topic in topics:
            current_preferences[topic] = current_preferences.get(topic, 0.0) + weight

        # Normalize preferences
        total_weight = sum(current_preferences.values())
        if total_weight > 0:
            current_preferences = {
                topic: weight / total_weight for topic, weight in current_preferences.items()
            }

        return current_preferences

    async def update_source_preferences(
        self, current_preferences: Dict[str, float], source: str, interaction_type: InteractionType
    ) -> Dict[str, float]:
        """Update source preferences based on interaction."""
        # Get interaction weight
        weight = self._get_interaction_weight(interaction_type)

        # Update preferences
        current_preferences[source] = current_preferences.get(source, 0.0) + weight

        # Normalize preferences
        total_weight = sum(current_preferences.values())
        if total_weight > 0:
            current_preferences = {
                source: weight / total_weight for source, weight in current_preferences.items()
            }

        return current_preferences

    async def update_reading_patterns(
        self, current_patterns: Dict[str, Any], interaction: UserInteraction
    ) -> Dict[str, Any]:
        """Update reading patterns based on interaction."""
        patterns = current_patterns.copy()

        # Update device preference
        if interaction.device_type:
            device_counts = patterns.get("device_counts", {})
            device_counts[interaction.device_type] = (
                device_counts.get(interaction.device_type, 0) + 1
            )
            patterns["device_counts"] = device_counts

            # Update preferred device
            preferred_device = max(device_counts.items(), key=lambda x: x[1])[0]
            patterns["preferred_device"] = preferred_device

        # Update time patterns
        if interaction.timestamp:
            hour = interaction.timestamp.hour
            time_period = self._get_time_period(hour)

            time_counts = patterns.get("time_counts", {})
            time_counts[time_period] = time_counts.get(time_period, 0) + 1
            patterns["time_counts"] = time_counts

            # Update preferred time
            preferred_time = max(time_counts.items(), key=lambda x: x[1])[0]
            patterns["preferred_time"] = preferred_time

        # Update content length preference
        if interaction.context.get("content_length"):
            length = interaction.context["content_length"]
            length_category = self._get_length_category(length)

            length_counts = patterns.get("length_counts", {})
            length_counts[length_category] = length_counts.get(length_category, 0) + 1
            patterns["length_counts"] = length_counts

            # Update preferred length
            preferred_length = max(length_counts.items(), key=lambda x: x[1])[0]
            patterns["preferred_length"] = preferred_length

        return patterns

    def _get_interaction_weight(self, interaction_type: InteractionType) -> float:
        """Get weight for interaction based on type."""
        weights = {
            InteractionType.CLICK: 1.0,
            InteractionType.VIEW: 0.5,
            InteractionType.SHARE: 2.0,
            InteractionType.SAVE: 2.5,
            InteractionType.LIKE: 1.5,
            InteractionType.DISLIKE: -1.0,
            InteractionType.READ: 3.0,
            InteractionType.SKIP: -0.5,
        }
        return weights.get(interaction_type, 1.0)

    def _get_time_period(self, hour: int) -> str:
        """Get time period from hour."""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"

    def _get_length_category(self, length: int) -> str:
        """Get content length category."""
        if length < 200:
            return "short"
        elif length < 1000:
            return "medium"
        else:
            return "long"

    async def save_profile(self, profile: UserProfile) -> None:
        """Save user profile to database."""
        query = """
        INSERT INTO user_profiles (
            user_id, created_at, updated_at, total_interactions, last_interaction_at,
            topic_preferences, source_preferences, reading_patterns, collaborative_weight,
            content_weight, diversity_preference, serendipity_preference,
            demographic_data, privacy_settings
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
        ) ON CONFLICT (user_id) DO UPDATE SET
            updated_at = $3,
            total_interactions = $4,
            last_interaction_at = $5,
            topic_preferences = $6,
            source_preferences = $7,
            reading_patterns = $8,
            collaborative_weight = $9,
            content_weight = $10,
            diversity_preference = $11,
            serendipity_preference = $12,
            demographic_data = $13,
            privacy_settings = $14
        """

        await self.postgres.execute(
            query,
            profile.user_id,
            profile.created_at,
            profile.updated_at,
            profile.total_interactions,
            profile.last_interaction_at,
            profile.topic_preferences,
            profile.source_preferences,
            profile.reading_patterns,
            profile.collaborative_weight,
            profile.content_weight,
            profile.diversity_preference,
            profile.serendipity_preference,
            profile.demographic_data,
            profile.privacy_settings,
        )

    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics and activity."""
        profile = await self.get_or_create_profile(user_id)

        # Get recent interactions
        recent_interactions = await self.postgres.fetch_all(
            """
            SELECT interaction_type, COUNT(*) as count
            FROM user_interactions 
            WHERE user_id = $1 AND timestamp > $2
            GROUP BY interaction_type
            """,
            user_id,
            datetime.utcnow() - timedelta(days=30),
        )

        interaction_stats = {row["interaction_type"]: row["count"] for row in recent_interactions}

        return {
            "user_id": user_id,
            "total_interactions": profile.total_interactions,
            "last_interaction": profile.last_interaction_at,
            "recent_interactions": interaction_stats,
            "topic_preferences": profile.topic_preferences,
            "source_preferences": profile.source_preferences,
            "reading_patterns": profile.reading_patterns,
        }

    async def delete_user_profile(self, user_id: str) -> None:
        """Delete user profile and related data."""
        # Delete from database
        await self.postgres.execute("DELETE FROM user_profiles WHERE user_id = $1", user_id)

        # Delete from cache
        await self.redis.delete(f"profile:{user_id}")

        logger.info(f"Deleted profile for user: {user_id}")

    async def get_profile_analytics(self) -> Dict[str, Any]:
        """Get analytics about user profiles."""
        # Get total users
        total_users = await self.postgres.fetch_one("SELECT COUNT(*) as count FROM user_profiles")

        # Get active users (last 30 days)
        active_users = await self.postgres.fetch_one(
            """
            SELECT COUNT(*) as count 
            FROM user_profiles 
            WHERE last_interaction_at > $1
            """,
            datetime.utcnow() - timedelta(days=30),
        )

        # Get average interactions per user
        avg_interactions = await self.postgres.fetch_one(
            "SELECT AVG(total_interactions) as avg FROM user_profiles"
        )

        return {
            "total_users": total_users["count"],
            "active_users": active_users["count"],
            "average_interactions": float(avg_interactions["avg"] or 0),
        }
