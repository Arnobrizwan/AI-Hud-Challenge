"""Cache management utilities."""

import json
import pickle
from typing import Any, Optional, Union

import redis.asyncio as redis

from ..config.settings import settings


class CacheManager:
    """Redis cache manager."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize cache manager.

        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
        self.default_ttl = settings.cache_ttl

    async def initialize(self):
        """Initialize cache manager."""
        # Test connection
        await self.redis.ping()

    async def get(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key
            deserialize: Whether to deserialize the value

        Returns:
            Cached value or None
        """
        try:
            value = await self.redis.get(key)
            if value is None:
                return None

            if deserialize:
                try:
                    return pickle.loads(value)
                except (pickle.PickleError, TypeError):
                    # Try JSON deserialization
                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        return value.decode("utf-8")
            else:
                return value
        except Exception:
            return None

    async def set(
        self, key: str, value: Any, ttl: Optional[int] = None, serialize: bool = True
    ) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialize: Whether to serialize the value

        Returns:
            True if successful
        """
        try:
            if serialize:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                else:
                    value = pickle.dumps(value)

            ttl = ttl or self.default_ttl
            await self.redis.setex(key, ttl, value)
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if successful
        """
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        try:
            result = await self.redis.exists(key)
            return result > 0
        except Exception:
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key.

        Args:
            key: Cache key
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        try:
            result = await self.redis.expire(key, ttl)
            return result
        except Exception:
            return False

    async def get_ttl(self, key: str) -> int:
        """Get time to live for key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds (-1 if no expiration, -2 if key doesn't exist)
        """
        try:
            return await self.redis.ttl(key)
        except Exception:
            return -2

    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Increment counter in cache.

        Args:
            key: Cache key
            amount: Amount to increment by
            ttl: Time to live in seconds

        Returns:
            New value
        """
        try:
            value = await self.redis.incrby(key, amount)
            if ttl and await self.redis.ttl(key) == -1:
                await self.redis.expire(key, ttl)
            return value
        except Exception:
            return 0

    async def decrement(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Decrement counter in cache.

        Args:
            key: Cache key
            amount: Amount to decrement by
            ttl: Time to live in seconds

        Returns:
            New value
        """
        try:
            value = await self.redis.decrby(key, amount)
            if ttl and await self.redis.ttl(key) == -1:
                await self.redis.expire(key, ttl)
            return value
        except Exception:
            return 0

    async def get_hash(self, key: str, field: str) -> Optional[str]:
        """Get hash field value.

        Args:
            key: Hash key
            field: Hash field

        Returns:
            Field value or None
        """
        try:
            return await self.redis.hget(key, field)
        except Exception:
            return None

    async def set_hash(self, key: str, field: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set hash field value.

        Args:
            key: Hash key
            field: Hash field
            value: Field value
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        try:
            await self.redis.hset(key, field, value)
            if ttl and await self.redis.ttl(key) == -1:
                await self.redis.expire(key, ttl)
            return True
        except Exception:
            return False

    async def get_all_hash(self, key: str) -> dict:
        """Get all hash fields.

        Args:
            key: Hash key

        Returns:
            Dictionary of field-value pairs
        """
        try:
            return await self.redis.hgetall(key)
        except Exception:
            return {}

    async def set_hash_mapping(self, key: str, mapping: dict, ttl: Optional[int] = None) -> bool:
        """Set multiple hash fields.

        Args:
            key: Hash key
            mapping: Dictionary of field-value pairs
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        try:
            await self.redis.hset(key, mapping=mapping)
            if ttl and await self.redis.ttl(key) == -1:
                await self.redis.expire(key, ttl)
            return True
        except Exception:
            return False

    async def delete_hash_field(self, key: str, field: str) -> bool:
        """Delete hash field.

        Args:
            key: Hash key
            field: Hash field

        Returns:
            True if successful
        """
        try:
            result = await self.redis.hdel(key, field)
            return result > 0
        except Exception:
            return False

    async def list_push(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Push value to list.

        Args:
            key: List key
            value: Value to push
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif not isinstance(value, (str, bytes)):
                value = str(value)

            await self.redis.lpush(key, value)
            if ttl and await self.redis.ttl(key) == -1:
                await self.redis.expire(key, ttl)
            return True
        except Exception:
            return False

    async def list_pop(self, key: str) -> Optional[str]:
        """Pop value from list.

        Args:
            key: List key

        Returns:
            Popped value or None
        """
        try:
            return await self.redis.rpop(key)
        except Exception:
            return None

    async def list_get_range(self, key: str, start: int = 0, end: int = -1) -> list:
        """Get list range.

        Args:
            key: List key
            start: Start index
            end: End index

        Returns:
            List of values
        """
        try:
            return await self.redis.lrange(key, start, end)
        except Exception:
            return []

    async def list_trim(self, key: str, start: int, end: int) -> bool:
        """Trim list to range.

        Args:
            key: List key
            start: Start index
            end: End index

        Returns:
            True if successful
        """
        try:
            await self.redis.ltrim(key, start, end)
            return True
        except Exception:
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern.

        Args:
            pattern: Key pattern

        Returns:
            Number of keys deleted
        """
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception:
            return 0

    async def close(self):
        """Close cache manager."""
        await self.redis.close()
