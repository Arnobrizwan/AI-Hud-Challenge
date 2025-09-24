"""
Redis Manager - High-performance caching and session management
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis
from redis.asyncio import Redis

from config import Settings
from models import Article, RetrievedArticle

logger = logging.getLogger(__name__)


class RedisManager:
    """Manage Redis for caching and session management"""

    def __init__(self):
        self.redis_client: Optional[Redis] = None
        self.settings = Settings()
        self._initialized = False

    async def initialize(self) -> Dict[str, Any]:
    """Initialize Redis client"""
        if self._initialized:
            return

        logger.info("Initializing Redis Manager...")

        try:
            # Create Redis client
            redis_config = self.settings.get_redis_config()

            self.redis_client = redis.Redis(
                host=redis_config.host,
                port=redis_config.port,
                password=redis_config.password,
                db=redis_config.db,
                max_connections=redis_config.max_connections,
                decode_responses=False,  # We'll handle encoding/decoding ourselves
            )

            # Test connection
            await self.redis_client.ping()

            self._initialized = True
            logger.info("Redis Manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Redis Manager: {e}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
    """Cleanup Redis client"""
        if self.redis_client:
    await self.redis_client.close()
            self.redis_client = None

        self._initialized = False
        logger.info("Redis Manager cleanup complete")

    async def cache_article(self, article: Article, ttl: int = 3600) -> bool:
        """Cache article in Redis"""
        if not self._initialized or not self.redis_client:
            raise RuntimeError("Redis Manager not initialized")

        try:
            cache_key = f"article:{article.id}"
            article_data = self._serialize_article(article)

            await self.redis_client.setex(cache_key, ttl, article_data)

            # Also cache by URL for quick lookups
            if article.url:
                url_key = f"article_url:{article.url}"
                await self.redis_client.setex(url_key, ttl, article.id)

            logger.debug(f"Article {article.id} cached in Redis")
            return True

        except Exception as e:
            logger.error(f"Failed to cache article {article.id}: {e}")
            return False

    async def get_cached_article(
            self, article_id: str) -> Optional[RetrievedArticle]:
        """Get cached article from Redis"""
        if not self._initialized or not self.redis_client:
            raise RuntimeError("Redis Manager not initialized")

        try:
            cache_key = f"article:{article_id}"
            article_data = await self.redis_client.get(cache_key)

            if article_data:
                article = self._deserialize_article(article_data)
                if article:
                    # Mark as cache hit
                    article.cache_hit = True
                    article.retrieval_timestamp = datetime.utcnow()
                    logger.debug(
                        f"Article {article_id} retrieved from Redis cache")
                    return article

            return None

        except Exception as e:
            logger.error(f"Failed to get cached article {article_id}: {e}")
            return None

    async def get_cached_article_by_url(
            self, url: str) -> Optional[RetrievedArticle]:
        """Get cached article by URL"""
        if not self._initialized or not self.redis_client:
            raise RuntimeError("Redis Manager not initialized")

        try:
            url_key = f"article_url:{url}"
            article_id = await self.redis_client.get(url_key)

            if article_id:
                article_id = article_id.decode("utf-8")
                return await self.get_cached_article(article_id)

            return None

        except Exception as e:
            logger.error(f"Failed to get cached article by URL {url}: {e}")
            return None

    async def cache_search_results(
        self, query: str, results: List[Dict[str, Any]], ttl: int = 1800
    ) -> bool:
        """Cache search results"""
        if not self._initialized or not self.redis_client:
            raise RuntimeError("Redis Manager not initialized")

        try:
            cache_key = f"search:{hash(query)}"
            results_data = json.dumps(results, default=str)

            await self.redis_client.setex(cache_key, ttl, results_data)

            logger.debug(f"Search results cached for query: {query[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to cache search results: {e}")
            return False

    async def get_cached_search_results(
            self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results"""
        if not self._initialized or not self.redis_client:
            raise RuntimeError("Redis Manager not initialized")

        try:
            cache_key = f"search:{hash(query)}"
            results_data = await self.redis_client.get(cache_key)

            if results_data:
                results = json.loads(results_data.decode("utf-8"))
                logger.debug(
                    f"Search results retrieved from cache for query: {query[:50]}...")
                return results

            return None

        except Exception as e:
            logger.error(f"Failed to get cached search results: {e}")
            return None

    async def cache_user_session(
        self, user_id: str, session_data: Dict[str, Any], ttl: int = 3600
    ) -> bool:
        """Cache user session data"""
        if not self._initialized or not self.redis_client:
            raise RuntimeError("Redis Manager not initialized")

        try:
            cache_key = f"session:{user_id}"
            session_json = json.dumps(session_data, default=str)

            await self.redis_client.setex(cache_key, ttl, session_json)

            logger.debug(f"User session {user_id} cached in Redis")
            return True

        except Exception as e:
            logger.error(f"Failed to cache user session {user_id}: {e}")
            return False

    async def get_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user session data"""
        if not self._initialized or not self.redis_client:
            raise RuntimeError("Redis Manager not initialized")

        try:
            cache_key = f"session:{user_id}"
            session_data = await self.redis_client.get(cache_key)

            if session_data:
                session = json.loads(session_data.decode("utf-8"))
                logger.debug(f"User session {user_id} retrieved from Redis")
                return session

            return None

        except Exception as e:
            logger.error(f"Failed to get user session {user_id}: {e}")
            return None

    async def invalidate_article_cache(self, article_id: str) -> bool:
        """Invalidate article cache"""
        if not self._initialized or not self.redis_client:
            raise RuntimeError("Redis Manager not initialized")

        try:
            # Delete article cache
            article_key = f"article:{article_id}"
            await self.redis_client.delete(article_key)

            # Delete from any search result caches that might contain this article
            # This is a simplified approach - in production, you might want to track
            # which search results contain which articles
            await self._invalidate_search_caches_containing(article_id)

            logger.debug(f"Article cache invalidated for {article_id}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to invalidate article cache {article_id}: {e}")
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        if not self._initialized or not self.redis_client:
            raise RuntimeError("Redis Manager not initialized")

        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted_count = await self.redis_client.delete(*keys)
                logger.debug(
                    f"Invalidated {deleted_count} cache entries matching pattern: {pattern}"
                )
                return deleted_count
            return 0

        except Exception as e:
            logger.error(f"Failed to invalidate pattern {pattern}: {e}")
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
    """Get Redis cache statistics"""
        if not self._initialized or not self.redis_client:
            raise RuntimeError("Redis Manager not initialized")

        try:
            info = await self.redis_client.info()

            # Count different types of cache entries
            article_keys = await self.redis_client.keys("article:*")
            search_keys = await self.redis_client.keys("search:*")
            session_keys = await self.redis_client.keys("session:*")

            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "hit_rate": self._calculate_hit_rate(info),
                "cache_entries": {
                    "articles": len(article_keys),
                    "search_results": len(search_keys),
                    "sessions": len(session_keys),
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    async def flush_cache(self) -> bool:
        """Flush all cache data"""
        if not self._initialized or not self.redis_client:
            raise RuntimeError("Redis Manager not initialized")

        try:
    await self.redis_client.flushdb()
            logger.info("Redis cache flushed")
            return True

        except Exception as e:
            logger.error(f"Failed to flush cache: {e}")
            return False

    def _serialize_article(self, article: Article) -> bytes:
        """Serialize article for caching"""
        try:
            # Convert to dict and serialize
            article_dict = {
                "id": article.id,
                "title": article.title,
                "content": article.content,
                "summary": article.summary,
                "author": article.author,
                "source": article.source,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "categories": article.categories,
                "tags": article.tags,
                "language": article.language,
                "url": article.url,
                "embeddings": article.embeddings,
                "media_files": article.media_files,
                "metadata": article.metadata,
                "cached_at": datetime.utcnow().isoformat(),
            }

            return pickle.dumps(article_dict)

        except Exception as e:
            logger.error(f"Failed to serialize article: {e}")
            return b""

    def _deserialize_article(self, data: bytes) -> Optional[RetrievedArticle]:
        """Deserialize article from cache"""
        try:
            article_dict = pickle.loads(data)

            # Convert back to RetrievedArticle
            return RetrievedArticle(
                id=article_dict["id"],
                title=article_dict["title"],
                content=article_dict["content"],
                summary=article_dict["summary"],
                author=article_dict["author"],
                source=article_dict["source"],
                published_at=(
                    datetime.fromisoformat(
                        article_dict["published_at"]) if article_dict["published_at"] else None),
                categories=article_dict["categories"],
                tags=article_dict["tags"],
                language=article_dict["language"],
                url=article_dict["url"],
                embeddings=article_dict["embeddings"],
                media_files=article_dict["media_files"],
                metadata=article_dict["metadata"],
                retrieval_timestamp=datetime.fromisoformat(
                    article_dict["cached_at"]),
                cache_hit=True,
                retrieval_sources=[],
            )

        except Exception as e:
            logger.error(f"Failed to deserialize article: {e}")
            return None

    def _calculate_hit_rate(self, info: Dict[str, Any]) -> float:
        """Calculate cache hit rate"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses

        if total == 0:
            return 0.0

        return (hits / total) * 100

    async def _invalidate_search_caches_containing(self, article_id: str) -> Dict[str, Any]:
    """Invalidate search caches that might contain the article"""
        try:
            # This is a simplified approach - in production, you might want to
            # maintain a mapping of which search results contain which articles
            search_keys = await self.redis_client.keys("search:*")

            for key in search_keys:
                try:
                    # Check if the search result contains this article
                    search_data = await self.redis_client.get(key)
                    if search_data and article_id.encode() in search_data:
    await self.redis_client.delete(key)
                        logger.debug(
                            f"Invalidated search cache {key} containing article {article_id}"
                        )
                except Exception:
                    # If we can't check the content, just delete the key
                    await self.redis_client.delete(key)

        except Exception as e:
            logger.warning(
                f"Failed to invalidate search caches containing {article_id}: {e}")
