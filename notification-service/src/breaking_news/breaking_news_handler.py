"""
Breaking news handler for immediate notification processing.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from redis.asyncio import Redis

from ..exceptions import NotificationError
from ..models.schemas import NewsItem, NotificationCandidate, NotificationType, Priority

logger = structlog.get_logger()


class BreakingNewsHandler:
    """Special handling for breaking news notifications."""

    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client
        self.breaking_news_criteria = {
            "urgency_threshold": 0.8, "recency_hours": 1, "keyword_boost": [
                "breaking", "urgent", "crisis", "emergency", "alert"], "source_priority": [
                "reuters", "ap", "bbc", "cnn", "nytimes"], }

        # Breaking news cache to prevent duplicates
        self.breaking_news_cache = {}
        self.cache_ttl = 3600  # 1 hour

    async def initialize(self) -> None:
        """Initialize breaking news handler."""
        logger.info("Initializing breaking news handler")
        # No specific initialization needed
        logger.info("Breaking news handler initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup breaking news handler."""
        logger.info("Cleaning up breaking news handler")
        self.breaking_news_cache.clear()

    async def handle_breaking_news(
            self, news_item: NewsItem) -> List[NotificationCandidate]:
        """Process breaking news for immediate notification."""

        try:
            logger.info(
                "Processing breaking news",
                news_item_id=news_item.id,
                title=news_item.title,
                is_breaking=news_item.is_breaking,
            )

            # Verify breaking news criteria
            if not await self._verify_breaking_news(news_item):
                logger.debug(
                    "News item does not meet breaking news criteria",
                    news_item_id=news_item.id)
                return []

            # Check for duplicate breaking news
            if await self._is_duplicate_breaking_news(news_item):
                logger.debug(
                    "Duplicate breaking news detected",
                    news_item_id=news_item.id)
                return []

            # Get users who should receive breaking news
            eligible_users = await self._get_breaking_news_eligible_users(news_item)

            if not eligible_users:
                logger.warning("No eligible users for breaking news")
                return []

            # Create high-priority notification candidates
            candidates = []
            for user_id in eligible_users:
                candidate = NotificationCandidate(
                    user_id=user_id,
                    content=news_item,
                    notification_type=NotificationType.BREAKING_NEWS,
                    urgency_score=0.9,  # High urgency for breaking news
                    priority=Priority.URGENT,
                    bypass_fatigue=True,  # Bypass fatigue for breaking news
                    immediate_delivery=True,
                    metadata={
                        "breaking_news": True,
                        "processed_at": datetime.utcnow().isoformat(),
                        "news_item_id": news_item.id,
                    },
                )
                candidates.append(candidate)

            # Cache breaking news to prevent duplicates
            await self._cache_breaking_news(news_item)

            logger.info(
                "Breaking news processed successfully",
                news_item_id=news_item.id,
                candidates_created=len(candidates),
            )

            return candidates

        except Exception as e:
            logger.error(
                "Error processing breaking news",
                news_item_id=news_item.id,
                error=str(e),
                exc_info=True,
            )
            raise NotificationError(
                f"Failed to process breaking news: {str(e)}")

    async def _verify_breaking_news(self, news_item: NewsItem) -> bool:
        """Verify if news item meets breaking news criteria."""

        try:
            # Check if already marked as breaking
            if news_item.is_breaking:
                return True

            # Check urgency score
            if news_item.urgency_score >= self.breaking_news_criteria["urgency_threshold"]:
                return True

            # Check recency
            age_hours = (datetime.utcnow() -
                         news_item.published_at).total_seconds() / 3600
            if age_hours > self.breaking_news_criteria["recency_hours"]:
                return False

            # Check for breaking news keywords
            title_lower = news_item.title.lower()
            content_lower = news_item.content.lower()

            keyword_matches = sum(
                1
                for keyword in self.breaking_news_criteria["keyword_boost"]
                if keyword in title_lower or keyword in content_lower
            )

            if keyword_matches >= 1:
                return True

            # Check source priority
            if news_item.source.lower(
            ) in self.breaking_news_criteria["source_priority"]:
                return True

            return False

        except Exception as e:
            logger.error(f"Error verifying breaking news criteria: {e}")
            return False

    async def _is_duplicate_breaking_news(self, news_item: NewsItem) -> bool:
        """Check if breaking news is duplicate."""

        try:
            # Create content hash for duplicate detection
            content_hash = self._create_content_hash(news_item)

            # Check cache
            if content_hash in self.breaking_news_cache:
                cached_time = self.breaking_news_cache[content_hash]
                if (datetime.utcnow() - cached_time).seconds < self.cache_ttl:
                    return True

            # Check Redis for recent similar content
            redis_key = f"breaking_news:{content_hash}"
            exists = await self.redis_client.exists(redis_key)

            if exists:
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking duplicate breaking news: {e}")
            return False

    def _create_content_hash(self, news_item: NewsItem) -> str:
        """Create content hash for duplicate detection."""

        import hashlib

        # Create hash from title and first 100 characters of content
        content_for_hash = f"{news_item.title}:{news_item.content[:100]}"
        return hashlib.md5(content_for_hash.encode()).hexdigest()

    async def _cache_breaking_news(self, news_item: NewsItem) -> None:
        """Cache breaking news to prevent duplicates."""

        try:
            content_hash = self._create_content_hash(news_item)

            # Cache in memory
            self.breaking_news_cache[content_hash] = datetime.utcnow()

            # Cache in Redis
            redis_key = f"breaking_news:{content_hash}"
            await self.redis_client.setex(redis_key, self.cache_ttl, news_item.id)

        except Exception as e:
            logger.error(f"Error caching breaking news: {e}")

    async def _get_breaking_news_eligible_users(
            self, news_item: NewsItem) -> List[str]:
        """Get users who should receive breaking news."""

        try:
            # In real implementation, this would query database for users
            # who have breaking news notifications enabled

            # Mock implementation - return sample users
            eligible_users = []

            # Get users with breaking news enabled
            breaking_news_users = await self._get_users_with_breaking_news_enabled()

            # Filter by geographic relevance if applicable
            if news_item.locations:
                geo_relevant_users = await self._get_geo_relevant_users(news_item.locations)
                eligible_users = list(
                    set(breaking_news_users) & set(geo_relevant_users))
            else:
                eligible_users = breaking_news_users

            # Limit to prevent overwhelming the system
            max_breaking_news_users = 10000
            if len(eligible_users) > max_breaking_news_users:
                # Prioritize users based on engagement history
                eligible_users = await self._prioritize_users(
                    eligible_users, max_breaking_news_users
                )

            logger.debug(
                "Found eligible users for breaking news",
                total_eligible=len(eligible_users),
                news_item_id=news_item.id,
            )

            return eligible_users

        except Exception as e:
            logger.error(f"Error getting eligible users: {e}")
            return []

    async def _get_users_with_breaking_news_enabled(self) -> List[str]:
        """Get users who have breaking news notifications enabled."""

        try:
            # Mock implementation - would query database
            # SELECT user_id FROM user_preferences
            # WHERE 'breaking_news' = ANY(enabled_types)

            # Return mock user IDs
            return [f"user_{i}" for i in range(1000)]

        except Exception as e:
            logger.error(
                f"Error getting users with breaking news enabled: {e}")
            return []

    async def _get_geo_relevant_users(self, locations: List[str]) -> List[str]:
        """Get users relevant to geographic locations."""

        try:
            # Mock implementation - would query database based on user location preferences
            # For now, return all users (no geographic filtering)
            return await self._get_users_with_breaking_news_enabled()

        except Exception as e:
            logger.error(f"Error getting geo-relevant users: {e}")
            return []

    async def _prioritize_users(
            self,
            users: List[str],
            max_count: int) -> List[str]:
        """Prioritize users based on engagement history."""

        try:
            # Mock implementation - would prioritize based on:
            # - Recent engagement with breaking news
            # - User activity level
            # - Notification response rate

            # For now, return first max_count users
            return users[:max_count]

        except Exception as e:
            logger.error(f"Error prioritizing users: {e}")
            return users[:max_count]

    async def get_breaking_news_analytics(self) -> Dict[str, Any]:
    """Get breaking news analytics."""
        try:
            analytics = {
                "breaking_news_criteria": self.breaking_news_criteria,
                "cached_breaking_news_count": len(self.breaking_news_cache),
                "cache_ttl_seconds": self.cache_ttl,
                "recent_breaking_news":
    await self._get_recent_breaking_news(),
                "user_coverage":
    await self._get_breaking_news_user_coverage(),
            }

            return analytics

        except Exception as e:
            logger.error(f"Error getting breaking news analytics: {e}")
            return {}

    async def _get_recent_breaking_news(self) -> List[Dict[str, Any]]:
        """Get recent breaking news items."""

        try:
            # Mock implementation - would query database for recent breaking
            # news
            return [
                {
                    "id": "news_1",
                    "title": "Breaking: Major Event Occurs",
                    "published_at": datetime.utcnow().isoformat(),
                    "urgency_score": 0.9,
                }
            ]

        except Exception as e:
            logger.error(f"Error getting recent breaking news: {e}")
            return []

    async def _get_breaking_news_user_coverage(self) -> Dict[str, int]:
        """Get breaking news user coverage statistics."""

        try:
            # Mock implementation - would calculate from actual data
            return {
                "total_eligible_users": 1000,
                "users_notified_last_hour": 50,
                "users_notified_last_day": 200,
                "average_notification_time_seconds": 2.5,
            }

        except Exception as e:
            logger.error(f"Error getting breaking news user coverage: {e}")
            return {}

    async def update_breaking_news_criteria(
            self, criteria: Dict[str, Any]) -> None:
        """Update breaking news criteria."""

        try:
            self.breaking_news_criteria.update(criteria)

            logger.info("Updated breaking news criteria", criteria=criteria)

        except Exception as e:
            logger.error(f"Error updating breaking news criteria: {e}")
            raise NotificationError(
                f"Failed to update breaking news criteria: {str(e)}")

    async def clear_breaking_news_cache(self) -> None:
        """Clear breaking news cache."""

        try:
            self.breaking_news_cache.clear()

            # Clear Redis cache
            pattern = "breaking_news:*"
            keys = await self.redis_client.keys(pattern)
            if keys:
    await self.redis_client.delete(*keys)

            logger.info("Cleared breaking news cache")

        except Exception as e:
            logger.error(f"Error clearing breaking news cache: {e}")
            raise NotificationError(
                f"Failed to clear breaking news cache: {str(e)}")
