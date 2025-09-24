"""
Cache Policies - Intelligent caching strategies and policies
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types"""

    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    CUSTOM = "custom"


class ContentType(Enum):
    """Content type categories"""

    ARTICLE = "article"
    SEARCH_RESULT = "search_result"
    USER_SESSION = "user_session"
    METADATA = "metadata"
    MEDIA = "media"


class CachePolicies:
    """Manage intelligent caching policies and strategies"""

    def __init__(self):
        self._initialized = False
        self._policies: Dict[str, Dict[str, Any]] = {}
        self._strategy_configs: Dict[CacheStrategy, Dict[str, Any]] = {}
        self._content_type_configs: Dict[ContentType, Dict[str, Any]] = {}

    async def initialize(self) -> Dict[str, Any]:
        """Initialize cache policies"""
        if self._initialized:
            return

        logger.info("Initializing Cache Policies...")

        try:
            # Initialize strategy configurations
            await self._initialize_strategy_configs()

            # Initialize content type configurations
            await self._initialize_content_type_configs()

            # Load custom policies
            await self._load_custom_policies()

            self._initialized = True
            logger.info("Cache Policies initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Cache Policies: {e}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup cache policies"""
        self._initialized = False
        logger.info("Cache Policies cleanup complete")

    async def _initialize_strategy_configs(self) -> Dict[str, Any]:
        """Initialize cache strategy configurations"""
        self._strategy_configs = {
            CacheStrategy.AGGRESSIVE: {
                "memory_ttl": 1800,  # 30 minutes
                "redis_ttl": 7200,  # 2 hours
                "cdn_ttl": 14400,  # 4 hours
                "use_memory_cache": True,
                "use_redis_cache": True,
                "use_cdn_cache": True,
                "preload_popular": True,
                "warm_cache": True,
            },
            CacheStrategy.BALANCED: {
                "memory_ttl": 600,  # 10 minutes
                "redis_ttl": 3600,  # 1 hour
                "cdn_ttl": 7200,  # 2 hours
                "use_memory_cache": True,
                "use_redis_cache": True,
                "use_cdn_cache": True,
                "preload_popular": False,
                "warm_cache": True,
            },
            CacheStrategy.CONSERVATIVE: {
                "memory_ttl": 300,  # 5 minutes
                "redis_ttl": 1800,  # 30 minutes
                "cdn_ttl": 3600,  # 1 hour
                "use_memory_cache": True,
                "use_redis_cache": True,
                "use_cdn_cache": False,
                "preload_popular": False,
                "warm_cache": False,
            },
        }

    async def _initialize_content_type_configs(self) -> Dict[str, Any]:
        """Initialize content type specific configurations"""
        self._content_type_configs = {
            ContentType.ARTICLE: {
                "base_ttl_multiplier": 1.0,
                "memory_priority": "high",
                "redis_priority": "high",
                "cdn_priority": "medium",
                "invalidation_strategy": "immediate",
                "tags": ["content", "article"],
            },
            ContentType.SEARCH_RESULT: {
                "base_ttl_multiplier": 0.5,
                "memory_priority": "high",
                "redis_priority": "medium",
                "cdn_priority": "low",
                "invalidation_strategy": "delayed",
                "tags": ["search", "query"],
            },
            ContentType.USER_SESSION: {
                "base_ttl_multiplier": 2.0,
                "memory_priority": "high",
                "redis_priority": "high",
                "cdn_priority": "none",
                "invalidation_strategy": "immediate",
                "tags": ["user", "session"],
            },
            ContentType.METADATA: {
                "base_ttl_multiplier": 3.0,
                "memory_priority": "medium",
                "redis_priority": "high",
                "cdn_priority": "low",
                "invalidation_strategy": "delayed",
                "tags": ["metadata", "system"],
            },
            ContentType.MEDIA: {
                "base_ttl_multiplier": 4.0,
                "memory_priority": "low",
                "redis_priority": "low",
                "cdn_priority": "high",
                "invalidation_strategy": "delayed",
                "tags": ["media", "static"],
            },
        }

    async def _load_custom_policies(self) -> Dict[str, Any]:
        """Load custom cache policies from configuration"""
        # This would typically load from a database or configuration file
        # For now, we'll use default policies
        self._policies = {
            "popular_content": {
                "strategy": CacheStrategy.AGGRESSIVE,
                "content_types": [ContentType.ARTICLE],
                "conditions": {"min_views": 100, "min_engagement": 0.7},
            },
            "breaking_news": {
                "strategy": CacheStrategy.CONSERVATIVE,
                "content_types": [ContentType.ARTICLE],
                "conditions": {"age_hours": 1, "categories": ["breaking", "urgent"]},
            },
            "user_personalized": {
                "strategy": CacheStrategy.BALANCED,
                "content_types": [ContentType.SEARCH_RESULT, ContentType.USER_SESSION],
                "conditions": {"user_specific": True},
            },
        }

    async def get_cache_config(
        self, content_type: ContentType, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
    """Get cache configuration for content type and context"""
        if not self._initialized:
            return self._get_default_config()

        try:
            # Get base configuration for content type
            base_config = self._content_type_configs.get(content_type, {})

            # Determine strategy based on context
            strategy = await self._determine_strategy(content_type, context or {})
            strategy_config = self._strategy_configs.get(strategy, {})

            # Merge configurations
            config = self._merge_configs(base_config, strategy_config)

            # Apply context-specific adjustments
            config = await self._apply_context_adjustments(config, context or {})

            return config

        except Exception as e:
            logger.error(f"Failed to get cache config: {e}")
            return self._get_default_config()

    async def _determine_strategy(
        self, content_type: ContentType, context: Dict[str, Any]
    ) -> CacheStrategy:
        """Determine optimal cache strategy based on context"""
        try:
            # Check for specific policy matches
            for policy_name, policy in self._policies.items():
                if await self._matches_policy(policy, content_type, context):
                    return policy["strategy"]

            # Default strategy based on content type
            if content_type == ContentType.ARTICLE:
                # Check article characteristics
                if context.get("is_breaking_news", False):
                    return CacheStrategy.CONSERVATIVE
                elif context.get("is_popular", False):
                    return CacheStrategy.AGGRESSIVE
                else:
                    return CacheStrategy.BALANCED

            elif content_type == ContentType.SEARCH_RESULT:
                return CacheStrategy.BALANCED

            elif content_type == ContentType.USER_SESSION:
                return CacheStrategy.AGGRESSIVE

            elif content_type == ContentType.METADATA:
                return CacheStrategy.AGGRESSIVE

            elif content_type == ContentType.MEDIA:
                return CacheStrategy.AGGRESSIVE

            return CacheStrategy.BALANCED

        except Exception as e:
            logger.error(f"Failed to determine strategy: {e}")
            return CacheStrategy.BALANCED

    async def _matches_policy(
        self, policy: Dict[str, Any], content_type: ContentType, context: Dict[str, Any]
    ) -> bool:
        """Check if context matches a policy"""
        try:
            # Check content type
            if content_type not in policy.get("content_types", []):
                return False

            # Check conditions
            conditions = policy.get("conditions", {})
            for condition_key, condition_value in conditions.items():
                if not self._evaluate_condition(
                        condition_key, condition_value, context):
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to evaluate policy match: {e}")
            return False

    def _evaluate_condition(
        self, condition_key: str, condition_value: Any, context: Dict[str, Any]
    ) -> bool:
        """Evaluate a single condition"""
        try:
            if condition_key == "min_views":
                return context.get("views", 0) >= condition_value

            elif condition_key == "min_engagement":
                return context.get("engagement_score", 0) >= condition_value

            elif condition_key == "age_hours":
                if "published_at" in context:
                    age_hours = (
                        datetime.utcnow() - context["published_at"]).total_seconds() / 3600
                    return age_hours <= condition_value
                return False

            elif condition_key == "categories":
                article_categories = context.get("categories", [])
                return any(
                    cat in article_categories for cat in condition_value)

            elif condition_key == "user_specific":
                return context.get("user_id") is not None

            return True

        except Exception as e:
            logger.error(f"Failed to evaluate condition {condition_key}: {e}")
            return False

    def _merge_configs(
        self, base_config: Dict[str, Any], strategy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Merge base and strategy configurations"""
        config = base_config.copy()

        # Override with strategy config
        for key, value in strategy_config.items():
            if key in ["memory_ttl", "redis_ttl", "cdn_ttl"]:
                # Apply TTL multiplier
                multiplier = base_config.get("base_ttl_multiplier", 1.0)
                config[key] = int(value * multiplier)
            else:
                config[key] = value

        return config

    async def _apply_context_adjustments(
        self, config: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Apply context-specific adjustments to config"""
        try:
            # Adjust TTL based on content age
            if "published_at" in context:
                age_hours = (
                    datetime.utcnow() - context["published_at"]).total_seconds() / 3600

                # Older content gets longer cache
                if age_hours > 24:
                    config["memory_ttl"] = int(config["memory_ttl"] * 2)
                    config["redis_ttl"] = int(config["redis_ttl"] * 2)
                    config["cdn_ttl"] = int(config["cdn_ttl"] * 2)
                elif age_hours < 1:
                    # Very recent content gets shorter cache
                    config["memory_ttl"] = int(config["memory_ttl"] * 0.5)
                    config["redis_ttl"] = int(config["redis_ttl"] * 0.5)

            # Adjust based on popularity
            if context.get("is_popular", False):
                config["memory_ttl"] = int(config["memory_ttl"] * 1.5)
                config["redis_ttl"] = int(config["redis_ttl"] * 1.5)
                config["preload_popular"] = True

            # Adjust based on user activity
            if context.get("user_id"):
                config["memory_ttl"] = int(config["memory_ttl"] * 1.2)
                config["warm_cache"] = True

            return config

        except Exception as e:
            logger.error(f"Failed to apply context adjustments: {e}")
            return config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default cache configuration"""
        return {
            "memory_ttl": 600,
            "redis_ttl": 3600,
            "cdn_ttl": 7200,
            "use_memory_cache": True,
            "use_redis_cache": True,
            "use_cdn_cache": False,
            "preload_popular": False,
            "warm_cache": False,
        }

    async def get_invalidation_strategy(
            self, content_type: ContentType) -> str:
        """Get invalidation strategy for content type"""
        if not self._initialized:
            return "immediate"

        config = self._content_type_configs.get(content_type, {})
        return config.get("invalidation_strategy", "immediate")

    async def get_cache_tags(
        self, content_type: ContentType, context: Dict[str, Any] = None
    ) -> List[str]:
        """Get cache tags for content type and context"""
        if not self._initialized:
            return []

        try:
            tags = []

            # Base tags from content type
            config = self._content_type_configs.get(content_type, {})
            tags.extend(config.get("tags", []))

            # Context-specific tags
            if context:
                if "user_id" in context:
                    tags.append(f"user:{context['user_id']}")

                if "source" in context:
                    tags.append(f"source:{context['source']}")

                if "categories" in context:
                    for category in context["categories"]:
                        tags.append(f"category:{category}")

                if "article_id" in context:
                    tags.append(f"article:{context['article_id']}")

            return tags

        except Exception as e:
            logger.error(f"Failed to get cache tags: {e}")
            return []

    async def should_preload(
        self, content_type: ContentType, context: Dict[str, Any] = None
    ) -> bool:
        """Determine if content should be preloaded"""
        if not self._initialized:
            return False

        try:
            config = await self.get_cache_config(content_type, context)
            return config.get("preload_popular", False)

        except Exception as e:
            logger.error(f"Failed to determine preload decision: {e}")
            return False

    async def should_warm_cache(
        self, content_type: ContentType, context: Dict[str, Any] = None
    ) -> bool:
        """Determine if cache should be warmed"""
        if not self._initialized:
            return False

        try:
            config = await self.get_cache_config(content_type, context)
            return config.get("warm_cache", False)

        except Exception as e:
            logger.error(f"Failed to determine cache warming decision: {e}")
            return False
