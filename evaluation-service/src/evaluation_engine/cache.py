"""
Cache configuration and management
"""

import logging
import redis
from typing import Optional, Any
import json
import pickle

from .config import get_redis_url

logger = logging.getLogger(__name__)

# Redis client
redis_client = None

def init_cache():
    """Initialize Redis cache"""
    global redis_client
    
    try:
        redis_url = get_redis_url()
        redis_client = redis.from_url(redis_url, decode_responses=False)
        
        # Test connection
        redis_client.ping()
        
        logger.info("Cache initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize cache: {str(e)}")
        raise

def get_cache_client():
    """Get Redis cache client"""
    if redis_client is None:
        raise RuntimeError("Cache not initialized")
    return redis_client

async def cache_get(key: str) -> Optional[Any]:
    """Get value from cache"""
    try:
        if redis_client is None:
            return None
        
        value = redis_client.get(key)
        if value is None:
            return None
        
        # Try to deserialize as JSON first, then pickle
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return pickle.loads(value)
            
    except Exception as e:
        logger.error(f"Error getting from cache: {str(e)}")
        return None

async def cache_set(key: str, value: Any, ttl: int = 3600) -> bool:
    """Set value in cache"""
    try:
        if redis_client is None:
            return False
        
        # Try to serialize as JSON first, then pickle
        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError):
            serialized = pickle.dumps(value)
        
        redis_client.setex(key, ttl, serialized)
        return True
        
    except Exception as e:
        logger.error(f"Error setting cache: {str(e)}")
        return False

async def cache_delete(key: str) -> bool:
    """Delete value from cache"""
    try:
        if redis_client is None:
            return False
        
        redis_client.delete(key)
        return True
        
    except Exception as e:
        logger.error(f"Error deleting from cache: {str(e)}")
        return False

async def cache_clear() -> bool:
    """Clear all cache"""
    try:
        if redis_client is None:
            return False
        
        redis_client.flushdb()
        return True
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return False
