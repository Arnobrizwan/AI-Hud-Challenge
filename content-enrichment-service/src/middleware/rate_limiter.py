"""Rate limiting middleware for Content Enrichment Service."""

import asyncio
import time
from typing import Dict, Optional
import redis.asyncio as redis
import structlog
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

from ..config import settings

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Rate limiter using Redis for distributed rate limiting."""
    
    def __init__(self):
        """Initialize the rate limiter."""
        self.redis_client = None
        self.local_cache = {}  # Fallback local cache
        self.cache_ttl = 60  # 1 minute TTL for local cache
        
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Rate limiter initialized with Redis")
            
        except Exception as e:
            logger.warning("Failed to initialize Redis for rate limiting", error=str(e))
            self.redis_client = None
    
    async def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limit."""
        try:
            if self.redis_client:
                return await self._check_redis_rate_limit(identifier)
            else:
                return await self._check_local_rate_limit(identifier)
                
        except Exception as e:
            logger.error("Rate limit check failed", identifier=identifier, error=str(e))
            return True  # Allow request if rate limiting fails
    
    async def _check_redis_rate_limit(self, identifier: str) -> bool:
        """Check rate limit using Redis."""
        try:
            current_time = int(time.time())
            window_start = current_time - settings.rate_limit_window
            
            # Use sliding window approach
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(f"rate_limit:{identifier}", 0, window_start)
            
            # Count current requests
            pipe.zcard(f"rate_limit:{identifier}")
            
            # Add current request
            pipe.zadd(f"rate_limit:{identifier}", {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(f"rate_limit:{identifier}", settings.rate_limit_window)
            
            results = await pipe.execute()
            current_count = results[1]
            
            if current_count >= settings.rate_limit_requests:
                logger.warning("Rate limit exceeded", 
                             identifier=identifier, 
                             current_count=current_count,
                             limit=settings.rate_limit_requests)
                return False
            
            return True
            
        except Exception as e:
            logger.error("Redis rate limit check failed", error=str(e))
            return True
    
    async def _check_local_rate_limit(self, identifier: str) -> bool:
        """Check rate limit using local cache."""
        try:
            current_time = time.time()
            window_start = current_time - settings.rate_limit_window
            
            # Clean up old entries
            if identifier in self.local_cache:
                self.local_cache[identifier] = [
                    timestamp for timestamp in self.local_cache[identifier]
                    if timestamp > window_start
                ]
            else:
                self.local_cache[identifier] = []
            
            # Check if under limit
            if len(self.local_cache[identifier]) >= settings.rate_limit_requests:
                logger.warning("Local rate limit exceeded", 
                             identifier=identifier,
                             current_count=len(self.local_cache[identifier]),
                             limit=settings.rate_limit_requests)
                return False
            
            # Add current request
            self.local_cache[identifier].append(current_time)
            
            return True
            
        except Exception as e:
            logger.error("Local rate limit check failed", error=str(e))
            return True
    
    async def get_rate_limit_info(self, identifier: str) -> Dict[str, int]:
        """Get rate limit information for an identifier."""
        try:
            if self.redis_client:
                current_time = int(time.time())
                window_start = current_time - settings.rate_limit_window
                
                # Get current count
                current_count = await self.redis_client.zcount(
                    f"rate_limit:{identifier}", 
                    window_start, 
                    current_time
                )
                
                return {
                    "current_requests": current_count,
                    "limit": settings.rate_limit_requests,
                    "window_seconds": settings.rate_limit_window,
                    "remaining": max(0, settings.rate_limit_requests - current_count)
                }
            else:
                current_time = time.time()
                window_start = current_time - settings.rate_limit_window
                
                if identifier in self.local_cache:
                    current_count = len([
                        timestamp for timestamp in self.local_cache[identifier]
                        if timestamp > window_start
                    ])
                else:
                    current_count = 0
                
                return {
                    "current_requests": current_count,
                    "limit": settings.rate_limit_requests,
                    "window_seconds": settings.rate_limit_window,
                    "remaining": max(0, settings.rate_limit_requests - current_count)
                }
                
        except Exception as e:
            logger.error("Rate limit info retrieval failed", error=str(e))
            return {
                "current_requests": 0,
                "limit": settings.rate_limit_requests,
                "window_seconds": settings.rate_limit_window,
                "remaining": settings.rate_limit_requests
            }
    
    async def reset_rate_limit(self, identifier: str) -> bool:
        """Reset rate limit for an identifier."""
        try:
            if self.redis_client:
                await self.redis_client.delete(f"rate_limit:{identifier}")
            else:
                if identifier in self.local_cache:
                    del self.local_cache[identifier]
            
            logger.info("Rate limit reset", identifier=identifier)
            return True
            
        except Exception as e:
            logger.error("Rate limit reset failed", error=str(e))
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Rate limiter cleanup completed")
        except Exception as e:
            logger.error("Rate limiter cleanup failed", error=str(e))


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting."""
        try:
            # Get client identifier
            client_ip = request.client.host
            user_agent = request.headers.get("user-agent", "")
            identifier = f"{client_ip}:{hash(user_agent)}"
            
            # Check rate limit
            if not await self.rate_limiter.check_rate_limit(identifier):
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": "Too many requests. Please try again later.",
                        "retry_after": settings.rate_limit_window
                    },
                    headers={"Retry-After": str(settings.rate_limit_window)}
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            rate_info = await self.rate_limiter.get_rate_limit_info(identifier)
            response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + settings.rate_limit_window)
            
            return response
            
        except Exception as e:
            logger.error("Rate limit middleware failed", error=str(e))
            # Allow request to proceed if middleware fails
            return await call_next(request)
