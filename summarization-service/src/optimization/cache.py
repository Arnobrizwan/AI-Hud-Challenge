"""
Advanced Caching System for Summarization
Redis-based caching with intelligent invalidation and optimization
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis
from summarization.models import SummarizationRequest, SummaryResult

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = None
    metadata: Dict[str, Any] = None


class SummaryCache:
    """Advanced caching system for summarization results"""

    def __init__(self):
        """Initialize the cache system"""
        self.redis_client = None
        self.local_cache = {}  # In-memory cache for hot data
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0}
        self._initialized = False

    async def initialize(self) -> Dict[str, Any]:
    """Initialize cache system"""
        try:
            logger.info("Initializing summary cache...")

            # Initialize Redis client
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )

            # Test connection
            await self.redis_client.ping()

            self._initialized = True
            logger.info("Summary cache initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize cache: {str(e)}")
            # Fallback to local cache only
            self._initialized = True
            logger.warning("Using local cache only (Redis unavailable)")

    async def cleanup(self) -> Dict[str, Any]:
    """Clean up cache resources"""
        try:
            if self.redis_client:
    await self.redis_client.close()
        except Exception as e:
            logger.error(f"Error during cache cleanup: {str(e)}")

    def generate_key(self, request: SummarizationRequest) -> str:
        """Generate cache key for a summarization request"""
        try:
            # Create a deterministic key based on request content
            key_data = {
                "text": request.content.text[:1000],  # First 1000 chars
                "target_lengths": sorted(request.target_lengths),
                "methods": [m.value for m in request.methods],
                "headline_styles": request.headline_styles,
                "language": request.content.language.value,
                "content_type": request.content.content_type.value,
            }

            # Create hash
            key_string = json.dumps(key_data, sort_keys=True)
            key_hash = hashlib.md5(key_string.encode()).hexdigest()

            return f"summary:{key_hash}"

        except Exception as e:
            logger.error(f"Failed to generate cache key: {str(e)}")
            return f"summary:{hash(str(request))}"

    async def get(self, key: str) -> Optional[SummaryResult]:
        """Get cached result"""
        try:
            if not self._initialized:
                return None

            # Try local cache first
            if key in self.local_cache:
                entry = self.local_cache[key]
                if self._is_entry_valid(entry):
                    self._update_access_stats(entry)
                    self.cache_stats["hits"] += 1
                    return self._deserialize_result(entry.value)
                else:
                    # Remove expired entry
                    del self.local_cache[key]

            # Try Redis cache
            if self.redis_client:
                try:
                    cached_data = await self.redis_client.get(key)
                    if cached_data:
                        entry_data = json.loads(cached_data)
                        entry = CacheEntry(**entry_data)

                        if self._is_entry_valid(entry):
                            # Store in local cache for faster access
                            self.local_cache[key] = entry
                            self._update_access_stats(entry)
                            self.cache_stats["hits"] += 1
                            return self._deserialize_result(entry.value)
                        else:
                            # Remove expired entry
                            await self.redis_client.delete(key)
                except Exception as e:
                    logger.error(f"Redis get failed: {str(e)}")

            self.cache_stats["misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Cache get failed: {str(e)}")
            self.cache_stats["misses"] += 1
            return None

    async def set(
            self,
            key: str,
            value: SummaryResult,
            ttl: int = None) -> bool:
        """Set cached result"""
        try:
            if not self._initialized:
                return False

            # Use default TTL if not specified
            if ttl is None:
                ttl = settings.CACHE_TTL

            # Create cache entry
            now = datetime.now()
            entry = CacheEntry(
                key=key,
                value=self._serialize_result(value),
                created_at=now,
                expires_at=now + timedelta(seconds=ttl),
                last_accessed=now,
                metadata={
                    "content_length": len(value.summary.text),
                    "processing_time": value.processing_stats.total_time,
                    "quality_score": value.quality_metrics.overall_score,
                },
            )

            # Store in local cache
            self.local_cache[key] = entry

            # Store in Redis cache
            if self.redis_client:
                try:
                    entry_data = asdict(entry)
                    # Convert datetime objects to strings for JSON
                    # serialization
                    entry_data["created_at"] = entry.created_at.isoformat()
                    entry_data["expires_at"] = entry.expires_at.isoformat()
                    entry_data["last_accessed"] = entry.last_accessed.isoformat()

                    await self.redis_client.setex(key, ttl, json.dumps(entry_data))
                except Exception as e:
                    logger.error(f"Redis set failed: {str(e)}")

            self.cache_stats["sets"] += 1
            return True

        except Exception as e:
            logger.error(f"Cache set failed: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete cached result"""
        try:
            # Remove from local cache
            if key in self.local_cache:
                del self.local_cache[key]

            # Remove from Redis cache
            if self.redis_client:
                try:
    await self.redis_client.delete(key)
                except Exception as e:
                    logger.error(f"Redis delete failed: {str(e)}")

            self.cache_stats["deletes"] += 1
            return True

        except Exception as e:
            logger.error(f"Cache delete failed: {str(e)}")
            return False

    async def clear(self) -> bool:
        """Clear all cached results"""
        try:
            # Clear local cache
            self.local_cache.clear()

            # Clear Redis cache
            if self.redis_client:
                try:
                    # Get all summary keys
                    keys = await self.redis_client.keys("summary:*")
                    if keys:
    await self.redis_client.delete(*keys)
                except Exception as e:
                    logger.error(f"Redis clear failed: {str(e)}")

            # Reset stats
            self.cache_stats = {
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "deletes": 0,
                "evictions": 0}

            return True

        except Exception as e:
            logger.error(f"Cache clear failed: {str(e)}")
            return False

    async def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        try:
            cleaned_count = 0
            now = datetime.now()

            # Clean local cache
            expired_keys = []
            for key, entry in self.local_cache.items():
                if not self._is_entry_valid(entry):
                    expired_keys.append(key)

            for key in expired_keys:
                del self.local_cache[key]
                cleaned_count += 1

            # Clean Redis cache (let Redis handle TTL)
            # We could implement a more aggressive cleanup here if needed

            self.cache_stats["evictions"] += cleaned_count
            return cleaned_count

        except Exception as e:
            logger.error(f"Cache cleanup failed: {str(e)}")
            return 0

    def _is_entry_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid"""
        return datetime.now() < entry.expires_at

    def _update_access_stats(self, entry: CacheEntry):
        """Update access statistics for cache entry"""
        entry.access_count += 1
        entry.last_accessed = datetime.now()

    def _serialize_result(self, result: SummaryResult) -> Dict[str, Any]:
    """Serialize SummaryResult for caching"""
        try:
            # Convert to dictionary
            result_dict = result.dict()

            # Convert datetime objects to strings
            if result.processing_stats:
                result_dict["processing_stats"]["created_at"] = datetime.now(
                ).isoformat()

            return result_dict

        except Exception as e:
            logger.error(f"Result serialization failed: {str(e)}")
            return {}

    def _deserialize_result(self, data: Dict[str, Any]) -> SummaryResult:
        """Deserialize cached data back to SummaryResult"""
        try:
            # Convert string dates back to datetime objects
            if "processing_stats" in data and data["processing_stats"]:
                if "created_at" in data["processing_stats"]:
                    data["processing_stats"]["created_at"] = datetime.fromisoformat(
                        data["processing_stats"]["created_at"])

            return SummaryResult(**data)

        except Exception as e:
            logger.error(f"Result deserialization failed: {str(e)}")
            return None

    async def get_stats(self) -> Dict[str, Any]:
    """Get cache statistics"""
        try:
            total_requests = self.cache_stats["hits"] + \
                self.cache_stats["misses"]
            hit_rate = self.cache_stats["hits"] / \
                total_requests if total_requests > 0 else 0

            return {
                "hits": self.cache_stats["hits"],
                "misses": self.cache_stats["misses"],
                "sets": self.cache_stats["sets"],
                "deletes": self.cache_stats["deletes"],
                "evictions": self.cache_stats["evictions"],
                "hit_rate": hit_rate,
                "local_cache_size": len(self.local_cache),
                "redis_available": self.redis_client is not None,
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {}

    async def get_status(self) -> Dict[str, Any]:
    """Get cache system status"""
        return {
            "initialized": self._initialized,
            "redis_available": self.redis_client is not None,
            "local_cache_size": len(self.local_cache),
            "stats":
    await self.get_stats(),
        }
