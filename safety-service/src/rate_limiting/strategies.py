"""
Rate Limiting Strategies
Various rate limiting strategies and implementations
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from safety_engine.config import get_rate_limit_config

logger = logging.getLogger(__name__)


class BaseRateLimiter:
    """Base class for rate limiting strategies"""

    def __init__(self):
        self.config = get_rate_limit_config()
        self.is_initialized = False
        self.redis_client = None

    async def initialize(self, redis_client) -> Dict[str, Any]:
        """Initialize the rate limiter"""
        self.redis_client = redis_client
        self.is_initialized = True

    async def check_limit(
        self, user_id: str, endpoint: str, request_size: int = 1
    ) -> Optional[Any]:
        """Check if request is within limits"""
        pass

    def get_key(self, user_id: str, endpoint: str) -> str:
        """Generate Redis key for rate limiting"""
        key_data = f"{self.__class__.__name__}:{user_id}:{endpoint}"
        return hashlib.md5(key_data.encode()).hexdigest()


class SlidingWindowRateLimiter(BaseRateLimiter):
    """Sliding window rate limiter"""

    def __init__(self):
        super().__init__()
        self.window_size = self.config.window_size
        self.default_limit = self.config.default_limit

    async def check_limit(
        self, user_id: str, endpoint: str, request_size: int = 1
    ) -> Optional[Any]:
        """Check sliding window rate limit"""
        try:
            if not self.is_initialized:
                return None

            key = self.get_key(user_id, endpoint)
            current_time = time.time()
            window_start = current_time - self.window_size

            # Get current window data
            window_data = await self.redis_client.get(key)
            if window_data:
                requests = json.loads(window_data)
            else:
                requests = []

            # Remove old requests outside the window
            requests = [
                req_time for req_time in requests if req_time > window_start]

            # Check if adding this request would exceed the limit
            if len(requests) + request_size > self.default_limit:
                return type(
                    "RateLimitResult",
                    (),
                    {
                        "is_limited": True,
                        "limit_type": "sliding_window",
                        "remaining_capacity": max(0, self.default_limit - len(requests)),
                        "retry_after": int(self.window_size - (current_time - window_start)),
                    },
                )()

            # Add current request
            requests.extend([current_time] * request_size)

            # Store updated window data
            await self.redis_client.setex(key, int(self.window_size), json.dumps(requests))

            return type(
                "RateLimitResult",
                (),
                {
                    "is_limited": False,
                    "limit_type": "sliding_window",
                    "remaining_capacity": self.default_limit - len(requests),
                    "retry_after": 0,
                },
            )()

        except Exception as e:
            logger.error(f"Sliding window rate limit check failed: {str(e)}")
            return None

    async def set_dynamic_limit(
            self,
            user_id: str,
            endpoint: str,
            new_limit: int):
         -> Dict[str, Any]:"""Set dynamic limit for user and endpoint"""
        try:
            # Store dynamic limit
            limit_key = f"dynamic_limit:{user_id}:{endpoint}"
            # 1 hour TTL
            await self.redis_client.setex(limit_key, 3600, str(new_limit))

        except Exception as e:
            logger.error(f"Dynamic limit setting failed: {str(e)}")

    async def reset_limit(self, user_id: str, endpoint: str) -> Dict[str, Any]:
        """Reset rate limit for user and endpoint"""
        try:
            key = self.get_key(user_id, endpoint)
            await self.redis_client.delete(key)

        except Exception as e:
            logger.error(f"Rate limit reset failed: {str(e)}")

    async def reset_user_limits(self, user_id: str) -> Dict[str, Any]:
        """Reset all rate limits for user"""
        try:
            pattern = f"{self.__class__.__name__}:{user_id}:*"
            keys = await self.redis_client.keys(pattern)
            if keys:
    await self.redis_client.delete(*keys)

        except Exception as e:
            logger.error(f"User rate limit reset failed: {str(e)}")

    async def get_status(self, user_id: str, endpoint: str) -> Dict[str, Any]:
        """Get current rate limit status"""
        try:
            key = self.get_key(user_id, endpoint)
            window_data = await self.redis_client.get(key)

            if window_data:
                requests = json.loads(window_data)
                current_time = time.time()
                window_start = current_time - self.window_size
                active_requests = [
                    req_time for req_time in requests if req_time > window_start]

                return {
                    "current_requests": len(active_requests),
                    "limit": self.default_limit,
                    "remaining": max(
                        0,
                        self.default_limit -
                        len(active_requests)),
                    "window_size": self.window_size,
                }
            else:
                return {
                    "current_requests": 0,
                    "limit": self.default_limit,
                    "remaining": self.default_limit,
                    "window_size": self.window_size,
                }

        except Exception as e:
            logger.error(f"Status retrieval failed: {str(e)}")
            return {"error": str(e)}

    async def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        try:
            # Get all keys for this limiter
            pattern = f"{self.__class__.__name__}:*"
            keys = await self.redis_client.keys(pattern)

            return {
                "active_limits": len(keys),
                "window_size": self.window_size,
                "default_limit": self.default_limit,
            }

        except Exception as e:
            logger.error(f"Statistics retrieval failed: {str(e)}")
            return {"error": str(e)}


class TokenBucketRateLimiter(BaseRateLimiter):
    """Token bucket rate limiter"""

    def __init__(self):
        super().__init__()
        self.capacity = self.config.default_limit
        self.refill_rate = self.config.default_limit / \
            self.config.window_size  # tokens per second
        self.last_refill = {}

    async def check_limit(
        self, user_id: str, endpoint: str, request_size: int = 1
    ) -> Optional[Any]:
        """Check token bucket rate limit"""
        try:
            if not self.is_initialized:
                return None

            key = self.get_key(user_id, endpoint)
            current_time = time.time()

            # Get current bucket state
            bucket_data = await self.redis_client.get(key)
            if bucket_data:
                bucket = json.loads(bucket_data)
                tokens = bucket["tokens"]
                last_refill = bucket["last_refill"]
            else:
                tokens = self.capacity
                last_refill = current_time

            # Calculate tokens to add based on time elapsed
            time_elapsed = current_time - last_refill
            tokens_to_add = time_elapsed * self.refill_rate
            tokens = min(self.capacity, tokens + tokens_to_add)

            # Check if we have enough tokens
            if tokens < request_size:
                # Calculate retry after time
                retry_after = (request_size - tokens) / self.refill_rate

                return type(
                    "RateLimitResult",
                    (),
                    {
                        "is_limited": True,
                        "limit_type": "token_bucket",
                        "remaining_capacity": int(tokens),
                        "retry_after": int(retry_after),
                    },
                )()

            # Consume tokens
            tokens -= request_size

            # Update bucket state
            bucket_state = {"tokens": tokens, "last_refill": current_time}
            await self.redis_client.setex(
                key, int(self.config.window_size), json.dumps(bucket_state)
            )

            return type(
                "RateLimitResult",
                (),
                {
                    "is_limited": False,
                    "limit_type": "token_bucket",
                    "remaining_capacity": int(tokens),
                    "retry_after": 0,
                },
            )()

        except Exception as e:
            logger.error(f"Token bucket rate limit check failed: {str(e)}")
            return None

    async def set_dynamic_limit(
            self,
            user_id: str,
            endpoint: str,
            new_limit: int):
         -> Dict[str, Any]:"""Set dynamic limit for user and endpoint"""
        try:
            # Store dynamic limit
            limit_key = f"dynamic_limit:{user_id}:{endpoint}"
            # 1 hour TTL
            await self.redis_client.setex(limit_key, 3600, str(new_limit))

        except Exception as e:
            logger.error(f"Dynamic limit setting failed: {str(e)}")

    async def reset_limit(self, user_id: str, endpoint: str) -> Dict[str, Any]:
        """Reset rate limit for user and endpoint"""
        try:
            key = self.get_key(user_id, endpoint)
            await self.redis_client.delete(key)

        except Exception as e:
            logger.error(f"Rate limit reset failed: {str(e)}")

    async def reset_user_limits(self, user_id: str) -> Dict[str, Any]:
        """Reset all rate limits for user"""
        try:
            pattern = f"{self.__class__.__name__}:{user_id}:*"
            keys = await self.redis_client.keys(pattern)
            if keys:
    await self.redis_client.delete(*keys)

        except Exception as e:
            logger.error(f"User rate limit reset failed: {str(e)}")

    async def get_status(self, user_id: str, endpoint: str) -> Dict[str, Any]:
        """Get current rate limit status"""
        try:
            key = self.get_key(user_id, endpoint)
            bucket_data = await self.redis_client.get(key)

            if bucket_data:
                bucket = json.loads(bucket_data)
                tokens = bucket["tokens"]
                last_refill = bucket["last_refill"]

                # Refill tokens based on time elapsed
                current_time = time.time()
                time_elapsed = current_time - last_refill
                tokens_to_add = time_elapsed * self.refill_rate
                tokens = min(self.capacity, tokens + tokens_to_add)

                return {
                    "tokens": int(tokens),
                    "capacity": self.capacity,
                    "refill_rate": self.refill_rate,
                }
            else:
                return {
                    "tokens": self.capacity,
                    "capacity": self.capacity,
                    "refill_rate": self.refill_rate,
                }

        except Exception as e:
            logger.error(f"Status retrieval failed: {str(e)}")
            return {"error": str(e)}

    async def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        try:
            # Get all keys for this limiter
            pattern = f"{self.__class__.__name__}:*"
            keys = await self.redis_client.keys(pattern)

            return {
                "active_buckets": len(keys),
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
            }

        except Exception as e:
            logger.error(f"Statistics retrieval failed: {str(e)}")
            return {"error": str(e)}


class AdaptiveRateLimiter(BaseRateLimiter):
    """Adaptive rate limiter that adjusts based on system load and user behavior"""

    def __init__(self):
        super().__init__()
        self.base_limit = self.config.default_limit
        self.load_threshold = 0.8
        self.user_behavior_history = {}

    async def check_limit(
        self,
        user_id: str,
        endpoint: str,
        current_load: float,
        reputation_score: float,
        abuse_score: float,
    ) -> Optional[Any]:
        """Check adaptive rate limit"""
        try:
            if not self.is_initialized:
                return None

            # Calculate adaptive limit based on system load and user behavior
            adaptive_limit = self.calculate_adaptive_limit(
                current_load, reputation_score, abuse_score
            )

            # Use sliding window with adaptive limit
            key = self.get_key(user_id, endpoint)
            current_time = time.time()
            window_size = self.config.window_size
            window_start = current_time - window_size

            # Get current window data
            window_data = await self.redis_client.get(key)
            if window_data:
                requests = json.loads(window_data)
            else:
                requests = []

            # Remove old requests outside the window
            requests = [
                req_time for req_time in requests if req_time > window_start]

            # Check if adding this request would exceed the adaptive limit
            if len(requests) + 1 > adaptive_limit:
                return type(
                    "RateLimitResult",
                    (),
                    {
                        "is_limited": True,
                        "limit_type": "adaptive",
                        "remaining_capacity": max(0, adaptive_limit - len(requests)),
                        "retry_after": int(window_size - (current_time - window_start)),
                    },
                )()

            # Add current request
            requests.append(current_time)

            # Store updated window data
            await self.redis_client.setex(key, int(window_size), json.dumps(requests))

            return type(
                "RateLimitResult",
                (),
                {
                    "is_limited": False,
                    "limit_type": "adaptive",
                    "remaining_capacity": adaptive_limit - len(requests),
                    "retry_after": 0,
                },
            )()

        except Exception as e:
            logger.error(f"Adaptive rate limit check failed: {str(e)}")
            return None

    def calculate_adaptive_limit(
        self, current_load: float, reputation_score: float, abuse_score: float
    ) -> int:
        """Calculate adaptive limit based on system load and user behavior"""
        try:
            # Base limit
            limit = self.base_limit

            # Adjust based on system load
            if current_load > self.load_threshold:
                load_factor = 1.0 - (current_load - self.load_threshold) * 0.5
                limit = int(limit * load_factor)

            # Adjust based on user reputation
            reputation_factor = 0.5 + (reputation_score * 0.5)  # 0.5 to 1.0
            limit = int(limit * reputation_factor)

            # Adjust based on abuse score
            abuse_factor = 1.0 - (abuse_score * 0.5)  # 1.0 to 0.5
            limit = int(limit * abuse_factor)

            # Ensure minimum limit
            return max(1, limit)

        except Exception as e:
            logger.error(f"Adaptive limit calculation failed: {str(e)}")
            return self.base_limit

    async def set_dynamic_limit(
            self,
            user_id: str,
            endpoint: str,
            new_limit: int):
         -> Dict[str, Any]:"""Set dynamic limit for user and endpoint"""
        try:
            # Store dynamic limit
            limit_key = f"dynamic_limit:{user_id}:{endpoint}"
            # 1 hour TTL
            await self.redis_client.setex(limit_key, 3600, str(new_limit))

        except Exception as e:
            logger.error(f"Dynamic limit setting failed: {str(e)}")

    async def reset_limit(self, user_id: str, endpoint: str) -> Dict[str, Any]:
        """Reset rate limit for user and endpoint"""
        try:
            key = self.get_key(user_id, endpoint)
            await self.redis_client.delete(key)

        except Exception as e:
            logger.error(f"Rate limit reset failed: {str(e)}")

    async def reset_user_limits(self, user_id: str) -> Dict[str, Any]:
        """Reset all rate limits for user"""
        try:
            pattern = f"{self.__class__.__name__}:{user_id}:*"
            keys = await self.redis_client.keys(pattern)
            if keys:
    await self.redis_client.delete(*keys)

        except Exception as e:
            logger.error(f"User rate limit reset failed: {str(e)}")

    async def get_status(self, user_id: str, endpoint: str) -> Dict[str, Any]:
        """Get current rate limit status"""
        try:
            key = self.get_key(user_id, endpoint)
            window_data = await self.redis_client.get(key)

            if window_data:
                requests = json.loads(window_data)
                current_time = time.time()
                window_size = self.config.window_size
                window_start = current_time - window_size
                active_requests = [
                    req_time for req_time in requests if req_time > window_start]

                return {
                    "current_requests": len(active_requests),
                    # Would need to calculate current adaptive limit
                    "adaptive_limit": self.base_limit,
                    "remaining": max(0, self.base_limit - len(active_requests)),
                    "window_size": window_size,
                }
            else:
                return {
                    "current_requests": 0,
                    "adaptive_limit": self.base_limit,
                    "remaining": self.base_limit,
                    "window_size": self.config.window_size,
                }

        except Exception as e:
            logger.error(f"Status retrieval failed: {str(e)}")
            return {"error": str(e)}

    async def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        try:
            # Get all keys for this limiter
            pattern = f"{self.__class__.__name__}:*"
            keys = await self.redis_client.keys(pattern)

            return {
                "active_limits": len(keys),
                "base_limit": self.base_limit,
                "load_threshold": self.load_threshold,
            }

        except Exception as e:
            logger.error(f"Statistics retrieval failed: {str(e)}")
            return {"error": str(e)}


class GeolocationBasedLimiter(BaseRateLimiter):
    """Rate limiter based on geographic location"""

    def __init__(self):
        super().__init__()
        self.location_limits = {
            "US": 1000,
            "EU": 800,
            "ASIA": 600,
            "OTHER": 400}
        self.suspicious_locations = set()

    async def check_limit(
        self, ip_address: str, endpoint: str, request_size: int = 1
    ) -> Optional[Any]:
        """Check geolocation-based rate limit"""
        try:
            if not self.is_initialized:
                return None

            # Determine location from IP (simplified)
            location = self.get_location_from_ip(ip_address)
            limit = self.location_limits.get(
                location, self.location_limits["OTHER"])

            # Check if location is suspicious
            if location in self.suspicious_locations:
                limit = int(limit * 0.1)  # 10% of normal limit

            # Use sliding window with location-based limit
            key = f"geo_limit:{location}:{endpoint}"
            current_time = time.time()
            window_size = self.config.window_size
            window_start = current_time - window_size

            # Get current window data
            window_data = await self.redis_client.get(key)
            if window_data:
                requests = json.loads(window_data)
            else:
                requests = []

            # Remove old requests outside the window
            requests = [
                req_time for req_time in requests if req_time > window_start]

            # Check if adding this request would exceed the limit
            if len(requests) + request_size > limit:
                return type(
                    "RateLimitResult",
                    (),
                    {
                        "is_limited": True,
                        "limit_type": "geolocation",
                        "remaining_capacity": max(0, limit - len(requests)),
                        "retry_after": int(window_size - (current_time - window_start)),
                    },
                )()

            # Add current request
            requests.extend([current_time] * request_size)

            # Store updated window data
            await self.redis_client.setex(key, int(window_size), json.dumps(requests))

            return type(
                "RateLimitResult",
                (),
                {
                    "is_limited": False,
                    "limit_type": "geolocation",
                    "remaining_capacity": limit - len(requests),
                    "retry_after": 0,
                },
            )()

        except Exception as e:
            logger.error(f"Geolocation rate limit check failed: {str(e)}")
            return None

    def get_location_from_ip(self, ip_address: str) -> str:
        """Get location from IP address (simplified)"""
        try:
            # In a real implementation, this would use a GeoIP service
            # For now, we'll simulate based on IP patterns
            if ip_address.startswith(
                    "192.168.") or ip_address.startswith("10."):
                return "US"  # Local network
            elif ip_address.startswith("172."):
                return "EU"  # Simulated
            elif ip_address.startswith("203."):
                return "ASIA"  # Simulated
        else:
                return "OTHER"

        except Exception as e:
            logger.error(f"Location detection failed: {str(e)}")
            return "OTHER"

    async def mark_location_suspicious(self, location: str) -> Dict[str, Any]:
        """Mark a location as suspicious"""
        try:
            self.suspicious_locations.add(location)
            logger.info(f"Marked location {location} as suspicious")

        except Exception as e:
            logger.error(f"Failed to mark location as suspicious: {str(e)}")

    async def unmark_location_suspicious(self, location: str) -> Dict[str, Any]:
        """Remove suspicious mark from location"""
        try:
            self.suspicious_locations.discard(location)
            logger.info(f"Removed suspicious mark from location {location}")

        except Exception as e:
            logger.error(f"Failed to unmark location: {str(e)}")

    async def get_status(self, ip_address: str,
                         endpoint: str) -> Dict[str, Any]:
    """Get current rate limit status"""
        try:
            location = self.get_location_from_ip(ip_address)
            limit = self.location_limits.get(
                location, self.location_limits["OTHER"])

            if location in self.suspicious_locations:
                limit = int(limit * 0.1)

            key = f"geo_limit:{location}:{endpoint}"
            window_data = await self.redis_client.get(key)

            if window_data:
                requests = json.loads(window_data)
                current_time = time.time()
                window_size = self.config.window_size
                window_start = current_time - window_size
                active_requests = [
                    req_time for req_time in requests if req_time > window_start]

                return {
                    "location": location,
                    "current_requests": len(active_requests),
                    "limit": limit,
                    "remaining": max(0, limit - len(active_requests)),
                    "is_suspicious": location in self.suspicious_locations,
                }
            else:
                return {
                    "location": location,
                    "current_requests": 0,
                    "limit": limit,
                    "remaining": limit,
                    "is_suspicious": location in self.suspicious_locations,
                }

        except Exception as e:
            logger.error(f"Status retrieval failed: {str(e)}")
            return {"error": str(e)}

    async def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        try:
            # Get all keys for this limiter
            pattern = "geo_limit:*"
            keys = await self.redis_client.keys(pattern)

            return {
                "active_limits": len(keys),
                "location_limits": self.location_limits,
                "suspicious_locations": list(self.suspicious_locations),
            }

        except Exception as e:
            logger.error(f"Statistics retrieval failed: {str(e)}")
            return {"error": str(e)}
