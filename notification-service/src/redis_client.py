"""
Redis client configuration for notification decisioning service.
"""

import asyncio
from typing import Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from .config import get_settings

_redis_client: Optional[Redis] = None


async def get_redis_client() -> Redis:
    """Get Redis client instance."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_keepalive=True,
            socket_keepalive_options={},
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Test connection
        try:
            await _redis_client.ping()
            print("Redis connection established successfully")
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            raise
    
    return _redis_client


async def close_redis_client() -> None:
    """Close Redis client connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
