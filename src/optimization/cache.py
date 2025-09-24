"""Redis caching and optimization layer."""

import asyncio
import json
import pickle
import time
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)


class CacheManager:
    """High-performance Redis cache manager with optimization strategies."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 max_connections: int = 100):
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.redis_pool = None
        self.redis = None
        
        # Cache statistics
        self.hit_count = 0
        self.miss_count = 0
        self.total_requests = 0
        
        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.compression_threshold = 1024  # 1KB
        
        # Initialize Redis connection
        asyncio.create_task(self._initialize_redis())
    
    async def _initialize_redis(self):
        """Initialize Redis connection pool."""
        try:
            self.redis_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                decode_responses=False  # We'll handle encoding ourselves
            )
            self.redis = redis.Redis(connection_pool=self.redis_pool)
            
            # Test connection
            await self.redis.ping()
            logger.info("Redis connection established", url=self.redis_url)
            
        except Exception as e:
            logger.error("Failed to initialize Redis", error=str(e))
            self.redis = None
    
    async def get(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """Get value from cache."""
        if not self.redis:
            return None
        
        try:
            self.total_requests += 1
            
            # Get raw value
            raw_value = await self.redis.get(key)
            if raw_value is None:
                self.miss_count += 1
                return None
            
            self.hit_count += 1
            
            if not deserialize:
                return raw_value
            
            # Deserialize value
            return await self._deserialize(raw_value)
            
        except Exception as e:
            logger.error("Cache get failed", error=str(e), key=key)
            self.miss_count += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                 serialize: bool = True) -> bool:
        """Set value in cache."""
        if not self.redis:
            return False
        
        try:
            # Serialize value if needed
            if serialize:
                serialized_value = await self._serialize(value)
            else:
                serialized_value = value
            
            # Set with TTL
            ttl = ttl or self.default_ttl
            await self.redis.setex(key, ttl, serialized_value)
            
            return True
            
        except Exception as e:
            logger.error("Cache set failed", error=str(e), key=key)
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.redis:
            return False
        
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error("Cache delete failed", error=str(e), key=key)
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.redis:
            return False
        
        try:
            result = await self.redis.exists(key)
            return result > 0
        except Exception as e:
            logger.error("Cache exists check failed", error=str(e), key=key)
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key."""
        if not self.redis:
            return False
        
        try:
            result = await self.redis.expire(key, ttl)
            return result
        except Exception as e:
            logger.error("Cache expire failed", error=str(e), key=key)
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not self.redis or not keys:
            return {}
        
        try:
            # Get all values
            values = await self.redis.mget(keys)
            
            # Build result dictionary
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = await self._deserialize(value)
            
            return result
            
        except Exception as e:
            logger.error("Cache get_many failed", error=str(e))
            return {}
    
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache."""
        if not self.redis or not mapping:
            return False
        
        try:
            # Serialize all values
            serialized_mapping = {}
            for key, value in mapping.items():
                serialized_mapping[key] = await self._serialize(value)
            
            # Set all values
            ttl = ttl or self.default_ttl
            pipe = self.redis.pipeline()
            for key, value in serialized_mapping.items():
                pipe.setex(key, ttl, value)
            await pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error("Cache set_many failed", error=str(e))
            return False
    
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> Optional[int]:
        """Increment counter in cache."""
        if not self.redis:
            return None
        
        try:
            # Use pipeline for atomic operation
            pipe = self.redis.pipeline()
            pipe.incrby(key, amount)
            if ttl:
                pipe.expire(key, ttl)
            results = await pipe.execute()
            
            return results[0]
            
        except Exception as e:
            logger.error("Cache increment failed", error=str(e), key=key)
            return None
    
    async def get_or_set(self, key: str, factory_func, ttl: Optional[int] = None) -> Any:
        """Get value from cache or set using factory function."""
        # Try to get from cache
        value = await self.get(key)
        if value is not None:
            return value
        
        # Generate value using factory function
        try:
            if asyncio.iscoroutinefunction(factory_func):
                value = await factory_func()
            else:
                value = factory_func()
            
            # Set in cache
            await self.set(key, value, ttl)
            
            return value
            
        except Exception as e:
            logger.error("Cache get_or_set failed", error=str(e), key=key)
            return None
    
    async def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Try JSON first for simple types
            if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                json_str = json.dumps(value, default=str)
                if len(json_str) > self.compression_threshold:
                    # Use pickle for large objects
                    return pickle.dumps(value)
                return json_str.encode('utf-8')
            else:
                # Use pickle for complex objects
                return pickle.dumps(value)
                
        except (TypeError, ValueError):
            # Fallback to pickle
            return pickle.dumps(value)
    
    async def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try JSON first
            try:
                json_str = data.decode('utf-8')
                return json.loads(json_str)
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            
            # Fallback to pickle
            return pickle.loads(data)
            
        except Exception as e:
            logger.error("Deserialization failed", error=str(e))
            return None
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hit_count / self.total_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "total_requests": self.total_requests,
            "hit_rate": self.get_hit_rate(),
            "redis_connected": self.redis is not None
        }
    
    async def clear_stats(self):
        """Clear cache statistics."""
        self.hit_count = 0
        self.miss_count = 0
        self.total_requests = 0
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")


class FeatureCache:
    """Specialized cache for ML features with precomputation."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.feature_prefix = "features"
        self.precomputed_prefix = "precomputed"
        
        # Feature TTLs (in seconds)
        self.feature_ttls = {
            "content": 3600,      # 1 hour
            "freshness": 300,     # 5 minutes
            "authority": 1800,    # 30 minutes
            "personalization": 600,  # 10 minutes
            "contextual": 300,    # 5 minutes
            "interaction": 60,    # 1 minute
        }
    
    async def get_feature(self, feature_type: str, key: str) -> Optional[Any]:
        """Get feature from cache."""
        cache_key = f"{self.feature_prefix}:{feature_type}:{key}"
        return await self.cache_manager.get(cache_key)
    
    async def set_feature(self, feature_type: str, key: str, value: Any) -> bool:
        """Set feature in cache."""
        cache_key = f"{self.feature_prefix}:{feature_type}:{key}"
        ttl = self.feature_ttls.get(feature_type, 3600)
        return await self.cache_manager.set(cache_key, value, ttl)
    
    async def get_precomputed_ranking(self, request_hash: str) -> Optional[Any]:
        """Get precomputed ranking results."""
        cache_key = f"{self.precomputed_prefix}:ranking:{request_hash}"
        return await self.cache_manager.get(cache_key)
    
    async def set_precomputed_ranking(self, request_hash: str, results: Any, ttl: int = 300) -> bool:
        """Set precomputed ranking results."""
        cache_key = f"{self.precomputed_prefix}:ranking:{request_hash}"
        return await self.cache_manager.set(cache_key, results, ttl)
    
    async def invalidate_user_features(self, user_id: str):
        """Invalidate all features for a user."""
        pattern = f"{self.feature_prefix}:*:{user_id}*"
        await self._invalidate_pattern(pattern)
    
    async def invalidate_article_features(self, article_id: str):
        """Invalidate all features for an article."""
        pattern = f"{self.feature_prefix}:*:{article_id}*"
        await self._invalidate_pattern(pattern)
    
    async def _invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern."""
        if not self.cache_manager.redis:
            return
        
        try:
            keys = await self.cache_manager.redis.keys(pattern)
            if keys:
                await self.cache_manager.redis.delete(*keys)
        except Exception as e:
            logger.error("Pattern invalidation failed", error=str(e), pattern=pattern)


class RankingCache:
    """Specialized cache for ranking results and models."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.ranking_prefix = "ranking"
        self.model_prefix = "models"
    
    async def get_ranking_result(self, request_hash: str) -> Optional[Any]:
        """Get cached ranking result."""
        cache_key = f"{self.ranking_prefix}:result:{request_hash}"
        return await self.cache_manager.get(cache_key)
    
    async def set_ranking_result(self, request_hash: str, result: Any, ttl: int = 300) -> bool:
        """Cache ranking result."""
        cache_key = f"{self.ranking_prefix}:result:{request_hash}"
        return await self.cache_manager.set(cache_key, result, ttl)
    
    async def get_model_prediction(self, model_name: str, features_hash: str) -> Optional[Any]:
        """Get cached model prediction."""
        cache_key = f"{self.model_prefix}:{model_name}:{features_hash}"
        return await self.cache_manager.get(cache_key)
    
    async def set_model_prediction(self, model_name: str, features_hash: str, 
                                 prediction: Any, ttl: int = 1800) -> bool:
        """Cache model prediction."""
        cache_key = f"{self.model_prefix}:{model_name}:{features_hash}"
        return await self.cache_manager.set(cache_key, prediction, ttl)
    
    async def warm_up_cache(self, common_queries: List[Dict[str, Any]]):
        """Warm up cache with common queries."""
        for query in common_queries:
            # This would precompute and cache common ranking results
            pass


class CacheOptimizer:
    """Cache optimization strategies and monitoring."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.optimization_strategies = {
            "lazy_loading": True,
            "precomputation": True,
            "compression": True,
            "ttl_optimization": True
        }
    
    async def optimize_ttl(self, key_pattern: str, access_frequency: Dict[str, int]):
        """Optimize TTL based on access frequency."""
        # Implement TTL optimization based on access patterns
        pass
    
    async def precompute_features(self, article_ids: List[str]):
        """Precompute features for articles."""
        # Implement feature precomputation
        pass
    
    async def compress_large_values(self, threshold: int = 1024):
        """Compress large values in cache."""
        # Implement compression for large values
        pass
    
    async def get_cache_efficiency_metrics(self) -> Dict[str, Any]:
        """Get cache efficiency metrics."""
        stats = self.cache_manager.get_stats()
        
        return {
            "hit_rate": stats["hit_rate"],
            "total_requests": stats["total_requests"],
            "memory_usage": await self._get_memory_usage(),
            "key_count": await self._get_key_count(),
            "efficiency_score": self._calculate_efficiency_score(stats)
        }
    
    async def _get_memory_usage(self) -> int:
        """Get Redis memory usage."""
        if not self.cache_manager.redis:
            return 0
        
        try:
            info = await self.cache_manager.redis.info('memory')
            return info.get('used_memory', 0)
        except Exception:
            return 0
    
    async def _get_key_count(self) -> int:
        """Get total key count in Redis."""
        if not self.cache_manager.redis:
            return 0
        
        try:
            info = await self.cache_manager.redis.info('keyspace')
            total_keys = 0
            for db_info in info.values():
                if isinstance(db_info, dict) and 'keys' in db_info:
                    total_keys += db_info['keys']
            return total_keys
        except Exception:
            return 0
    
    def _calculate_efficiency_score(self, stats: Dict[str, Any]) -> float:
        """Calculate cache efficiency score."""
        hit_rate = stats["hit_rate"]
        total_requests = stats["total_requests"]
        
        # Efficiency score based on hit rate and request volume
        if total_requests < 100:
            return 0.5  # Not enough data
        
        # Higher hit rate and more requests = better efficiency
        efficiency = hit_rate * min(total_requests / 1000, 1.0)
        return min(efficiency, 1.0)
