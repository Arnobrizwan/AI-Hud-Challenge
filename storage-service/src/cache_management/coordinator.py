"""
Cache Coordinator - Multi-layer cache management and coordination
"""

import asyncio
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from collections import OrderedDict

from models import (
    Article, RetrievedArticle, CacheConfig, CachedData, CacheResult,
    CacheInvalidationRequest, InvalidationResult, CacheLevel
)
from .memory_cache import MemoryCache
from .cdn_manager import CDNManager
from .cache_policies import CachePolicies
from storage_backends.redis import RedisManager

logger = logging.getLogger(__name__)

class CacheCoordinator:
    """Coordinate caching across multiple layers"""
    
    def __init__(self):
        self.redis_manager: Optional[RedisManager] = None
        self.memory_cache: Optional[MemoryCache] = None
        self.cdn_manager: Optional[CDNManager] = None
        self.cache_policies: Optional[CachePolicies] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize cache coordination components"""
        if self._initialized:
            return
            
        logger.info("Initializing Cache Coordinator...")
        
        try:
            # Initialize Redis manager
            self.redis_manager = RedisManager()
            await self.redis_manager.initialize()
            
            # Initialize memory cache
            self.memory_cache = MemoryCache()
            await self.memory_cache.initialize()
            
            # Initialize CDN manager
            self.cdn_manager = CDNManager()
            await self.cdn_manager.initialize()
            
            # Initialize cache policies
            self.cache_policies = CachePolicies()
            await self.cache_policies.initialize()
            
            self._initialized = True
            logger.info("Cache Coordinator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cache Coordinator: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup cache coordination components"""
        logger.info("Cleaning up Cache Coordinator...")
        
        cleanup_tasks = []
        
        if self.redis_manager:
            cleanup_tasks.append(self.redis_manager.cleanup())
        if self.memory_cache:
            cleanup_tasks.append(self.memory_cache.cleanup())
        if self.cdn_manager:
            cleanup_tasks.append(self.cdn_manager.cleanup())
        if self.cache_policies:
            cleanup_tasks.append(self.cache_policies.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self._initialized = False
        logger.info("Cache Coordinator cleanup complete")
    
    async def get_cached_data(self, cache_key: str, 
                            cache_config: CacheConfig) -> CachedData:
        """Multi-layer cache retrieval"""
        if not self._initialized:
            raise RuntimeError("Cache Coordinator not initialized")
        
        logger.debug(f"Retrieving cached data for key: {cache_key}")
        
        try:
            # L1: Memory cache (fastest)
            if cache_config.use_memory_cache and self.memory_cache:
                memory_data = await self.memory_cache.get(cache_key)
                if memory_data:
                    logger.debug(f"Cache hit in memory for key: {cache_key}")
                    return CachedData(
                        data=memory_data,
                        cache_level=CacheLevel.MEMORY,
                        cache_hit=True
                    )
            
            # L2: Redis cache (fast, shared)
            if cache_config.use_redis_cache and self.redis_manager:
                redis_data = await self.redis_manager.get_cached_article(cache_key)
                if redis_data:
                    # Populate memory cache
                    if cache_config.use_memory_cache and self.memory_cache:
                        await self.memory_cache.set(
                            cache_key, redis_data, ttl=cache_config.memory_ttl
                        )
                    
                    logger.debug(f"Cache hit in Redis for key: {cache_key}")
                    return CachedData(
                        data=redis_data,
                        cache_level=CacheLevel.REDIS,
                        cache_hit=True
                    )
            
            # L3: CDN cache (for public content)
            if cache_config.use_cdn_cache and cache_config.is_public and self.cdn_manager:
                cdn_data = await self.cdn_manager.get_cached_content(cache_key)
                if cdn_data:
                    logger.debug(f"Cache hit in CDN for key: {cache_key}")
                    return CachedData(
                        data=cdn_data,
                        cache_level=CacheLevel.CDN,
                        cache_hit=True
                    )
            
            logger.debug(f"Cache miss for key: {cache_key}")
            return CachedData(cache_hit=False)
            
        except Exception as e:
            logger.error(f"Failed to get cached data for key {cache_key}: {e}")
            return CachedData(cache_hit=False)
    
    async def cache_data(self, cache_key: str, data: Any,
                        cache_config: CacheConfig) -> CacheResult:
        """Multi-layer cache storage"""
        if not self._initialized:
            raise RuntimeError("Cache Coordinator not initialized")
        
        logger.debug(f"Caching data for key: {cache_key}")
        
        try:
            cache_operations = []
            cached_levels = []
            
            # Cache in memory
            if cache_config.use_memory_cache and self.memory_cache:
                cache_operations.append(
                    self.memory_cache.set(cache_key, data, ttl=cache_config.memory_ttl)
                )
                cached_levels.append(CacheLevel.MEMORY)
            
            # Cache in Redis
            if cache_config.use_redis_cache and self.redis_manager:
                # Convert data to Article if needed
                if isinstance(data, dict) and 'id' in data:
                    article = RetrievedArticle(**data)
                    cache_operations.append(
                        self.redis_manager.cache_article(article, ttl=cache_config.redis_ttl)
                    )
                else:
                    # For non-article data, use a generic caching method
                    cache_operations.append(
                        self._cache_generic_data(cache_key, data, cache_config.redis_ttl)
                    )
                cached_levels.append(CacheLevel.REDIS)
            
            # Cache in CDN
            if cache_config.use_cdn_cache and cache_config.is_public and self.cdn_manager:
                cache_operations.append(
                    self.cdn_manager.cache_content(cache_key, data, cache_config.cdn_ttl)
                )
                cached_levels.append(CacheLevel.CDN)
            
            # Execute cache operations
            cache_results = await asyncio.gather(*cache_operations, return_exceptions=True)
            
            # Check for failures
            successful_levels = []
            for i, result in enumerate(cache_results):
                if not isinstance(result, Exception):
                    successful_levels.append(cached_levels[i])
                else:
                    logger.warning(f"Cache operation failed for level {cached_levels[i]}: {result}")
            
            logger.debug(f"Cached data for key {cache_key} in {len(successful_levels)} levels")
            
            return CacheResult(
                cache_key=cache_key,
                cached_levels=successful_levels,
                cache_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to cache data for key {cache_key}: {e}")
            raise
    
    async def cache_article(self, article: Article, ttl: int = 3600) -> CacheResult:
        """Cache article with intelligent TTL"""
        if not self._initialized:
            raise RuntimeError("Cache Coordinator not initialized")
        
        try:
            # Determine optimal cache configuration
            cache_config = await self._get_optimal_cache_config(article)
            
            # Generate cache key
            cache_key = self._generate_article_cache_key(article.id)
            
            # Cache the article
            return await self.cache_data(cache_key, article, cache_config)
            
        except Exception as e:
            logger.error(f"Failed to cache article {article.id}: {e}")
            raise
    
    async def get_cached_article(self, article_id: str) -> Optional[RetrievedArticle]:
        """Get cached article"""
        if not self._initialized:
            raise RuntimeError("Cache Coordinator not initialized")
        
        try:
            cache_key = self._generate_article_cache_key(article_id)
            
            # Use default cache config for retrieval
            cache_config = CacheConfig(
                use_memory_cache=True,
                use_redis_cache=True,
                use_cdn_cache=False,
                memory_ttl=300,
                redis_ttl=3600,
                is_public=False
            )
            
            cached_data = await self.get_cached_data(cache_key, cache_config)
            
            if cached_data.cache_hit and isinstance(cached_data.data, RetrievedArticle):
                return cached_data.data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached article {article_id}: {e}")
            return None
    
    async def update_cache(self, article: RetrievedArticle) -> CacheResult:
        """Update cache with new article data"""
        if not self._initialized:
            raise RuntimeError("Cache Coordinator not initialized")
        
        try:
            cache_key = self._generate_article_cache_key(article.id)
            
            # Determine cache configuration
            cache_config = await self._get_optimal_cache_config(article)
            
            # Update cache
            return await self.cache_data(cache_key, article, cache_config)
            
        except Exception as e:
            logger.error(f"Failed to update cache for article {article.id}: {e}")
            raise
    
    async def invalidate_cache(self, invalidation_request: CacheInvalidationRequest) -> InvalidationResult:
        """Intelligent cache invalidation"""
        if not self._initialized:
            raise RuntimeError("Cache Coordinator not initialized")
        
        logger.info(f"Invalidating cache with request: {invalidation_request}")
        
        try:
            invalidation_tasks = []
            total_invalidated = 0
            
            # Invalidate by key patterns
            if invalidation_request.key_patterns:
                for pattern in invalidation_request.key_patterns:
                    # Memory cache invalidation
                    if self.memory_cache:
                        invalidation_tasks.append(
                            self.memory_cache.delete_pattern(pattern)
                        )
                    
                    # Redis cache invalidation
                    if self.redis_manager:
                        invalidation_tasks.append(
                            self.redis_manager.invalidate_pattern(pattern)
                        )
                    
                    # CDN cache invalidation
                    if self.cdn_manager:
                        invalidation_tasks.append(
                            self.cdn_manager.purge_pattern(pattern)
                        )
            
            # Invalidate by tags
            if invalidation_request.cache_tags:
                for tag in invalidation_request.cache_tags:
                    # Memory cache invalidation by tag
                    if self.memory_cache:
                        invalidation_tasks.append(
                            self.memory_cache.delete_by_tag(tag)
                        )
                    
                    # Redis cache invalidation by tag
                    if self.redis_manager:
                        invalidation_tasks.append(
                            self.redis_manager.invalidate_pattern(f"tag:{tag}:*")
                        )
                    
                    # CDN cache invalidation by tag
                    if self.cdn_manager:
                        invalidation_tasks.append(
                            self.cdn_manager.purge_by_tag(tag)
                        )
            
            # Execute invalidation
            if invalidation_tasks:
                invalidation_results = await asyncio.gather(
                    *invalidation_tasks, return_exceptions=True
                )
                
                # Count successful invalidations
                for result in invalidation_results:
                    if not isinstance(result, Exception) and isinstance(result, int):
                        total_invalidated += result
            
            logger.info(f"Invalidated {total_invalidated} cache entries")
            
            return InvalidationResult(
                invalidated_keys=total_invalidated,
                invalidation_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
            raise
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        if not self._initialized:
            raise RuntimeError("Cache Coordinator not initialized")
        
        try:
            stats = {
                'timestamp': datetime.utcnow().isoformat(),
                'layers': {}
            }
            
            # Memory cache stats
            if self.memory_cache:
                memory_stats = await self.memory_cache.get_statistics()
                stats['layers']['memory'] = memory_stats
            
            # Redis cache stats
            if self.redis_manager:
                redis_stats = await self.redis_manager.get_cache_stats()
                stats['layers']['redis'] = redis_stats
            
            # CDN cache stats
            if self.cdn_manager:
                cdn_stats = await self.cdn_manager.get_statistics()
                stats['layers']['cdn'] = cdn_stats
            
            # Overall hit rate calculation
            total_hits = 0
            total_requests = 0
            
            for layer_stats in stats['layers'].values():
                if 'hits' in layer_stats:
                    total_hits += layer_stats['hits']
                if 'requests' in layer_stats:
                    total_requests += layer_stats['requests']
            
            if total_requests > 0:
                stats['overall_hit_rate'] = (total_hits / total_requests) * 100
            else:
                stats['overall_hit_rate'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {}
    
    def _generate_article_cache_key(self, article_id: str) -> str:
        """Generate cache key for article"""
        return f"article:{article_id}"
    
    def _generate_search_cache_key(self, query: str, filters: Dict[str, Any] = None) -> str:
        """Generate cache key for search results"""
        key_data = f"search:{query}"
        if filters:
            key_data += f":{hash(str(sorted(filters.items())))}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_optimal_cache_config(self, article: Union[Article, RetrievedArticle]) -> CacheConfig:
        """Get optimal cache configuration for article"""
        if not self.cache_policies:
            return CacheConfig()  # Default configuration
        
        try:
            # Determine if article is public (simplified logic)
            is_public = article.source in ['reuters', 'ap', 'bbc', 'cnn']
            
            # Determine TTL based on article age and source
            base_ttl = 3600  # 1 hour
            if article.published_at:
                age_hours = (datetime.utcnow() - article.published_at).total_seconds() / 3600
                if age_hours > 24:  # Older articles get longer cache
                    base_ttl = 86400  # 24 hours
                elif age_hours > 1:  # Recent articles get shorter cache
                    base_ttl = 1800  # 30 minutes
            
            return CacheConfig(
                use_memory_cache=True,
                use_redis_cache=True,
                use_cdn_cache=is_public,
                memory_ttl=min(base_ttl, 300),  # Memory cache max 5 minutes
                redis_ttl=base_ttl,
                cdn_ttl=base_ttl * 2 if is_public else 0,
                is_public=is_public
            )
            
        except Exception as e:
            logger.warning(f"Failed to get optimal cache config: {e}")
            return CacheConfig()
    
    async def _cache_generic_data(self, cache_key: str, data: Any, ttl: int) -> bool:
        """Cache generic data in Redis"""
        try:
            if self.redis_manager and self.redis_manager.redis_client:
                import json
                serialized_data = json.dumps(data, default=str)
                await self.redis_manager.redis_client.setex(
                    cache_key, ttl, serialized_data
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cache generic data: {e}")
            return False
