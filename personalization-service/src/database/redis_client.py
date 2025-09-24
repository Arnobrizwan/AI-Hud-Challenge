"""Redis client for caching and real-time features."""

import asyncio
import json
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
import structlog

from ..config.settings import settings

logger = structlog.get_logger()


class RedisClient:
    """Redis client for caching and real-time features."""

    def __init__(self):
        self.redis_url = settings.redis_url
        self.client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )

            # Test connection
            await self.client.ping()
            logger.info("Connected to Redis successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from Redis")

    async def get(self, key: str) -> Any:
        """Get value from Redis."""
        if not self.client:
            return None

        try:
            value = await self.client.get(key)
            if value is None:
                return None

            # Try to parse as JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        if not self.client:
            return False

        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)

            if ttl:
                result = await self.client.setex(key, ttl, serialized_value)
            else:
                result = await self.client.set(key, serialized_value)

            return result

        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False

    async def setex(self, key: str, ttl: int, value: Any) -> bool:
        """Set value with expiration in Redis."""
        return await self.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self.client:
            return False

        try:
            result = await self.client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        if not self.client:
            return 0

        try:
            keys = await self.client.keys(pattern)
            if keys:
                return await self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error deleting pattern {pattern}: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self.client:
            return False

        try:
            result = await self.client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Error checking existence of key {key}: {e}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key."""
        if not self.client:
            return False

        try:
            result = await self.client.expire(key, ttl)
            return bool(result)
        except Exception as e:
            logger.error(f"Error setting expiration for key {key}: {e}")
            return False

    async def ttl(self, key: str) -> int:
        """Get TTL for key."""
        if not self.client:
            return -1

        try:
            return await self.client.ttl(key)
        except Exception as e:
            logger.error(f"Error getting TTL for key {key}: {e}")
            return -1

    async def incr(self, key: str) -> int:
        """Increment key value."""
        if not self.client:
            return 0

        try:
            return await self.client.incr(key)
        except Exception as e:
            logger.error(f"Error incrementing key {key}: {e}")
            return 0

    async def decr(self, key: str) -> int:
        """Decrement key value."""
        if not self.client:
            return 0

        try:
            return await self.client.decr(key)
        except Exception as e:
            logger.error(f"Error decrementing key {key}: {e}")
            return 0

    async def hget(self, name: str, key: str) -> Any:
        """Get hash field value."""
        if not self.client:
            return None

        try:
            value = await self.client.hget(name, key)
            if value is None:
                return None

            # Try to parse as JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.error(f"Error getting hash field {name}.{key}: {e}")
            return None

    async def hset(self, name: str, key: str, value: Any) -> bool:
        """Set hash field value."""
        if not self.client:
            return False

        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)

            result = await self.client.hset(name, key, serialized_value)
            return bool(result)
        except Exception as e:
            logger.error(f"Error setting hash field {name}.{key}: {e}")
            return False

    async def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all hash fields."""
        if not self.client:
            return {}

        try:
            result = await self.client.hgetall(name)

            # Parse JSON values
            parsed_result = {}
            for k, v in result.items():
                try:
                    parsed_result[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    parsed_result[k] = v

            return parsed_result
        except Exception as e:
            logger.error(f"Error getting all hash fields {name}: {e}")
            return {}

    async def hdel(self, name: str, key: str) -> bool:
        """Delete hash field."""
        if not self.client:
            return False

        try:
            result = await self.client.hdel(name, key)
            return bool(result)
        except Exception as e:
            logger.error(f"Error deleting hash field {name}.{key}: {e}")
            return False

    async def lpush(self, key: str, *values: Any) -> int:
        """Push values to list."""
        if not self.client:
            return 0

        try:
            # Serialize values
            serialized_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    serialized_values.append(json.dumps(value))
                else:
                    serialized_values.append(str(value))

            return await self.client.lpush(key, *serialized_values)
        except Exception as e:
            logger.error(f"Error pushing to list {key}: {e}")
            return 0

    async def rpop(self, key: str) -> Any:
        """Pop value from list."""
        if not self.client:
            return None

        try:
            value = await self.client.rpop(key)
            if value is None:
                return None

            # Try to parse as JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.error(f"Error popping from list {key}: {e}")
            return None

    async def llen(self, key: str) -> int:
        """Get list length."""
        if not self.client:
            return 0

        try:
            return await self.client.llen(key)
        except Exception as e:
            logger.error(f"Error getting list length {key}: {e}")
            return 0

    async def get_health(self) -> Dict[str, Any]:
        """Get Redis health status."""
        if not self.client:
            return {"status": "disconnected", "error": "No client"}

        try:
            info = await self.client.info()
            return {
                "status": "connected",
                "version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
