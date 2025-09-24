"""
Redis-based rate limiting service with sliding window algorithm.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import aioredis

from src.config.settings import settings
from src.models.common import RateLimitInfo
from src.utils.logging import get_logger
from src.utils.metrics import metrics_collector

logger = get_logger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
            self,
            limit_type: str,
            limit: int,
            window_seconds: int,
            reset_time: datetime):
        self.limit_type = limit_type
        self.limit = limit
        self.window_seconds = window_seconds
        self.reset_time = reset_time
        super().__init__(
            f"Rate limit exceeded for {limit_type}: {limit} requests per {window_seconds}s"
        )


class SlidingWindowRateLimiter:
    """Redis-based sliding window rate limiter."""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis_pool: Optional[aioredis.Redis] = None
        self._lock = asyncio.Lock()

    async def _get_redis(self) -> aioredis.Redis:
        """Get Redis connection with connection pooling."""
        if self.redis_pool is None:
    async with self._lock:
                if self.redis_pool is None:
                    self.redis_pool = aioredis.from_url(
                        self.redis_url,
                        max_connections=settings.REDIS_MAX_CONNECTIONS,
                        retry_on_timeout=True,
                        decode_responses=True,
                    )
                    logger.info("Redis connection pool initialized")
        return self.redis_pool

    async def check_rate_limit(
        self, key: str, limit: int, window_seconds: int = 60
    ) -> Tuple[bool, RateLimitInfo]:
        """
        Check if request is within rate limit using sliding window algorithm.

        Args:
            key: Unique identifier for the rate limit (e.g., user_id, ip_address)
            limit: Maximum number of requests allowed
            window_seconds: Time window in seconds

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        redis = await self._get_redis()
        current_time = time.time()
        window_start = current_time - window_seconds

        # Redis key for the sliding window
        redis_key = f"rate_limit:{key}"

        try:
            # Use Redis pipeline for atomic operations
            async with redis.pipeline() as pipe:
                # Remove expired entries
                await pipe.zremrangebyscore(redis_key, 0, window_start)

                # Count current requests in window
                await pipe.zcard(redis_key)

                # Execute pipeline
                results = await pipe.execute()
                current_count = results[1]

                # Check if limit exceeded
                if current_count >= limit:
                    # Calculate reset time
                    reset_time = datetime.fromtimestamp(
                        current_time + window_seconds)

                    rate_limit_info = RateLimitInfo(
                        limit=limit,
                        remaining=0,
                        reset_time=reset_time,
                        window_seconds=window_seconds,
                    )

                    # Record metrics
                    metrics_collector.record_rate_limit_block("sliding_window")

                    logger.warning(
                        "Rate limit exceeded",
                        key=key,
                        limit=limit,
                        current_count=current_count,
                        window_seconds=window_seconds,
                    )

                    return False, rate_limit_info

                # Add current request to window
                await redis.zadd(redis_key, {str(current_time): current_time})

                # Set expiration for cleanup
                await redis.expire(redis_key, window_seconds + 1)

                # Calculate remaining requests and reset time
                remaining = max(0, limit - current_count - 1)
                reset_time = datetime.fromtimestamp(
                    current_time + window_seconds)

                rate_limit_info = RateLimitInfo(
                    limit=limit,
                    remaining=remaining,
                    reset_time=reset_time,
                    window_seconds=window_seconds,
                )

                # Record metrics
                metrics_collector.record_rate_limit_hit("sliding_window", key)

                return True, rate_limit_info

        except Exception as e:
            logger.error("Rate limit check failed", key=key, error=str(e))
            # On Redis failure, allow request (fail open)
            reset_time = datetime.fromtimestamp(current_time + window_seconds)
            rate_limit_info = RateLimitInfo(
                limit=limit,
                remaining=limit - 1,
                reset_time=reset_time,
                window_seconds=window_seconds,
            )
            return True, rate_limit_info

    async def reset_rate_limit(self, key: str) -> bool:
        """Reset rate limit for a key."""
        try:
            redis = await self._get_redis()
            redis_key = f"rate_limit:{key}"
            deleted = await redis.delete(redis_key)

            logger.info(f"Rate limit reset for key: {key}")
            return deleted > 0

        except Exception as e:
            logger.error(f"Failed to reset rate limit for {key}", error=str(e))
            return False

    async def get_rate_limit_info(
        self, key: str, limit: int, window_seconds: int = 60
    ) -> RateLimitInfo:
        """Get current rate limit information without consuming a request."""
        redis = await self._get_redis()
        current_time = time.time()
        window_start = current_time - window_seconds

        redis_key = f"rate_limit:{key}"

        try:
            # Remove expired entries and count current
            async with redis.pipeline() as pipe:
    await pipe.zremrangebyscore(redis_key, 0, window_start)
                await pipe.zcard(redis_key)
                results = await pipe.execute()
                current_count = results[1]

            remaining = max(0, limit - current_count)
            reset_time = datetime.fromtimestamp(current_time + window_seconds)

            return RateLimitInfo(
                limit=limit,
                remaining=remaining,
                reset_time=reset_time,
                window_seconds=window_seconds,
            )

        except Exception as e:
            logger.error(
                "Failed to get rate limit info",
                key=key,
                error=str(e))
            # Return safe defaults on error
            reset_time = datetime.fromtimestamp(current_time + window_seconds)
            return RateLimitInfo(
                limit=limit,
                remaining=limit,
                reset_time=reset_time,
                window_seconds=window_seconds)

    async def close(self) -> Dict[str, Any]:
    """Close Redis connection."""
        if self.redis_pool:
    await self.redis_pool.close()
            logger.info("Redis connection closed")


class DistributedRateLimiter:
    """Distributed rate limiter with multiple limit types."""

    def __init__(self):
        self.limiter = SlidingWindowRateLimiter()

        # Default rate limits from settings
        self.default_limits = {
            "per_user": {
                "limit": settings.RATE_LIMIT_PER_USER,
                "window": settings.RATE_LIMIT_WINDOW_SECONDS,
            },
            "per_ip": {
                "limit": settings.RATE_LIMIT_PER_IP,
                "window": settings.RATE_LIMIT_WINDOW_SECONDS,
            },
        }

    async def check_user_rate_limit(
        self, user_id: str, limit: Optional[int] = None, window_seconds: Optional[int] = None
    ) -> Tuple[bool, RateLimitInfo]:
        """Check rate limit for a specific user."""
        limit = limit or self.default_limits["per_user"]["limit"]
        window_seconds = window_seconds or self.default_limits["per_user"]["window"]

        key = f"user:{user_id}"
        return await self.limiter.check_rate_limit(key, limit, window_seconds)

    async def check_ip_rate_limit(self,
                                  ip_address: str,
                                  limit: Optional[int] = None,
                                  window_seconds: Optional[int] = None) -> Tuple[bool,
                                                                                 RateLimitInfo]:
        """Check rate limit for an IP address."""
        limit = limit or self.default_limits["per_ip"]["limit"]
        window_seconds = window_seconds or self.default_limits["per_ip"]["window"]

        key = f"ip:{ip_address}"
        return await self.limiter.check_rate_limit(key, limit, window_seconds)

    async def check_endpoint_rate_limit(
        self, endpoint: str, identifier: str, limit: int, window_seconds: int = 60
    ) -> Tuple[bool, RateLimitInfo]:
        """Check rate limit for a specific endpoint."""
        key = f"endpoint:{endpoint}:{identifier}"
        return await self.limiter.check_rate_limit(key, limit, window_seconds)

    async def check_global_rate_limit(
        self, limit: int = 100000, window_seconds: int = 60
    ) -> Tuple[bool, RateLimitInfo]:
        """Check global rate limit for the entire service."""
        key = "global"
        return await self.limiter.check_rate_limit(key, limit, window_seconds)

    async def check_multiple_limits(
            self, checks: list) -> Dict[str, Tuple[bool, RateLimitInfo]]:
        """Check multiple rate limits concurrently."""
        tasks = []
        check_names = []

        for check in checks:
            if check["type"] == "user":
                task = self.check_user_rate_limit(
                    check["identifier"],
                    check.get("limit"),
                    check.get("window_seconds"))
            elif check["type"] == "ip":
                task = self.check_ip_rate_limit(
                    check["identifier"],
                    check.get("limit"),
                    check.get("window_seconds"))
            elif check["type"] == "endpoint":
                task = self.check_endpoint_rate_limit(
                    check["endpoint"],
                    check["identifier"],
                    check["limit"],
                    check.get("window_seconds", 60),
                )
            elif check["type"] == "global":
                task = self.check_global_rate_limit(
                    check.get("limit", 100000), check.get("window_seconds", 60)
                )
            else:
                continue

            tasks.append(task)
            check_names.append(check.get("name", check["type"]))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            name: result if not isinstance(
                result, Exception) else (
                False, None) for name, result in zip(
                check_names, results)}

    async def reset_user_rate_limit(self, user_id: str) -> bool:
        """Reset rate limit for a user."""
        key = f"user:{user_id}"
        return await self.limiter.reset_rate_limit(key)

    async def reset_ip_rate_limit(self, ip_address: str) -> bool:
        """Reset rate limit for an IP address."""
        key = f"ip:{ip_address}"
        return await self.limiter.reset_rate_limit(key)

    async def get_user_rate_limit_info(
            self,
            user_id: str,
            limit: Optional[int] = None,
            window_seconds: Optional[int] = None) -> RateLimitInfo:
        """Get rate limit information for a user."""
        limit = limit or self.default_limits["per_user"]["limit"]
        window_seconds = window_seconds or self.default_limits["per_user"]["window"]

        key = f"user:{user_id}"
        return await self.limiter.get_rate_limit_info(key, limit, window_seconds)

    async def get_ip_rate_limit_info(
            self,
            ip_address: str,
            limit: Optional[int] = None,
            window_seconds: Optional[int] = None) -> RateLimitInfo:
        """Get rate limit information for an IP address."""
        limit = limit or self.default_limits["per_ip"]["limit"]
        window_seconds = window_seconds or self.default_limits["per_ip"]["window"]

        key = f"ip:{ip_address}"
        return await self.limiter.get_rate_limit_info(key, limit, window_seconds)

    async def health_check(self) -> bool:
        """Check if rate limiter is healthy."""
        try:
            redis = await self.limiter._get_redis()
            await redis.ping()
            return True
        except Exception:
            return False

    async def close(self) -> Dict[str, Any]:
    """Close rate limiter connections."""
        await self.limiter.close()


# Global rate limiter instance
rate_limiter = DistributedRateLimiter()
