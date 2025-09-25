"""
Advanced Rate Limiter
Advanced rate limiting with multiple strategies
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from safety_engine.config import get_rate_limit_config
from safety_engine.models import RateLimitRequest, RateLimitResult

from .strategies import (
    AdaptiveRateLimiter,
    GeolocationBasedLimiter,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
)

logger = logging.getLogger(__name__)


class AdvancedRateLimiter:
    """Advanced rate limiting with multiple strategies"""

    def __init__(self):
        self.config = get_rate_limit_config()
        self.is_initialized = False

        # Redis client
        self.redis_client = None

        # Rate limiting strategies
        self.sliding_window = SlidingWindowRateLimiter()
        self.token_bucket = TokenBucketRateLimiter()
        self.adaptive_limiter = AdaptiveRateLimiter()
        self.geolocation_limiter = GeolocationBasedLimiter()

        # User reputation and abuse scores
        self.user_reputations = {}
        self.user_abuse_scores = {}

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the rate limiter"""
        try:
            # Initialize Redis connection
            except Exception as e:
                pass

            self.redis_client = redis.from_url(
                "redis://localhost:6379/0", decode_responses=True)

            # Test Redis connection
            await self.redis_client.ping()

            # Initialize all strategies
            await self.sliding_window.initialize(self.redis_client)
            await self.token_bucket.initialize(self.redis_client)
            await self.adaptive_limiter.initialize(self.redis_client)
            await self.geolocation_limiter.initialize(self.redis_client)

            self.is_initialized = True
            logger.info("Advanced rate limiter initialized")

        except Exception as e:
            logger.error(f"Failed to initialize rate limiter: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup resources"""
        try:
            if self.redis_client:
            except Exception as e:
                pass

                await self.redis_client.close()
                self.redis_client = None

            self.is_initialized = False
            logger.info("Rate limiter cleanup completed")

        except Exception as e:
            logger.error(f"Error during rate limiter cleanup: {str(e)}")

    async def check_rate_limits(
            self, request: RateLimitRequest) -> RateLimitResult:
        """Comprehensive rate limiting check"""

        if not self.is_initialized:
            raise RuntimeError("Rate limiter not initialized")

        try:
            user_id = request.user_id
            endpoint = request.endpoint
            ip_address = request.ip_address
            except Exception as e:
                pass


            # Get user reputation and abuse scores
            reputation_score = await self.get_user_reputation(user_id)
            abuse_score = await self.get_user_abuse_score(user_id)

            # Run multiple rate limiting strategies in parallel
            limit_checks = await asyncio.gather(
                self.sliding_window.check_limit(user_id, endpoint, request.request_size),
                self.token_bucket.check_limit(user_id, endpoint, request.request_size),
                self.adaptive_limiter.check_limit(
                    user_id, endpoint, request.current_load, reputation_score, abuse_score
                ),
                self.geolocation_limiter.check_limit(ip_address, endpoint, request.request_size),
                return_exceptions=True,
            )

            # Process results
            triggered_limits = []
            remaining_capacities = []
            retry_afters = []

            for i, check in enumerate(limit_checks):
                if isinstance(check, Exception):
                    logger.warning(
                        f"Rate limit check {i} failed: {str(check)}")
                    continue

                if check and hasattr(check, "is_limited") and check.is_limited:
                    triggered_limits.append(check.limit_type)

                if check and hasattr(check, "remaining_capacity"):
                    remaining_capacities.append(check.remaining_capacity)

                if check and hasattr(check, "retry_after"):
                    retry_afters.append(check.retry_after)

            # Determine overall rate limit status
            is_rate_limited = len(triggered_limits) > 0

            # Calculate remaining capacity (minimum of all strategies)
            remaining_capacity = min(
                remaining_capacities) if remaining_capacities else 0

            # Calculate retry after (maximum of all strategies)
            retry_after = max(retry_afters) if retry_afters else 0

            # Apply dynamic rate limiting if needed
            if not is_rate_limited:
                await self.apply_dynamic_rate_limit(
                    user_id, endpoint, reputation_score, abuse_score
                )

            return RateLimitResult(
                user_id=user_id,
                endpoint=endpoint,
                is_rate_limited=is_rate_limited,
                triggered_limits=triggered_limits,
                remaining_capacity=remaining_capacity,
                retry_after=retry_after,
                check_timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Rate limit check failed: {str(e)}")
            raise

    async def get_user_reputation(self, user_id: str) -> float:
        """Get user reputation score"""
        try:
            if user_id in self.user_reputations:
            except Exception as e:
                pass

                return self.user_reputations[user_id]

            # In a real implementation, this would fetch from a database
            # For now, return a default value
            reputation = 0.5  # Default neutral reputation
            self.user_reputations[user_id] = reputation

            return reputation

        except Exception as e:
            logger.error(
                f"Failed to get reputation for user {user_id}: {str(e)}")
            return 0.5

    async def get_user_abuse_score(self, user_id: str) -> float:
        """Get user abuse score"""
        try:
            if user_id in self.user_abuse_scores:
            except Exception as e:
                pass

                return self.user_abuse_scores[user_id]

            # In a real implementation, this would fetch from abuse detection system
            # For now, return a default value
            abuse_score = 0.0  # Default no abuse
            self.user_abuse_scores[user_id] = abuse_score

            return abuse_score

        except Exception as e:
            logger.error(
                f"Failed to get abuse score for user {user_id}: {str(e)}")
            return 0.0

    async def apply_dynamic_rate_limit(
            self,
            user_id: str,
            endpoint: str,
            reputation_score: float,
            abuse_score: float):
         -> Dict[str, Any]:
        """Apply dynamic rate limiting based on user behavior"""
        try:
            # Calculate dynamic limits based on user reputation and behavior
            except Exception as e:
                pass

            reputation_multiplier = max(0.1, min(2.0, reputation_score))
            abuse_penalty = max(0.1, 1.0 - abuse_score)

            # Get base limit for endpoint
            base_limit = self.get_base_limit_for_endpoint(endpoint)

            # Calculate adjusted limit
            adjusted_limit = int(
                base_limit *
                reputation_multiplier *
                abuse_penalty)

            # Apply new limits to all strategies
            await self.sliding_window.set_dynamic_limit(user_id, endpoint, adjusted_limit)
            await self.token_bucket.set_dynamic_limit(user_id, endpoint, adjusted_limit)
            await self.adaptive_limiter.set_dynamic_limit(user_id, endpoint, adjusted_limit)

            logger.debug(
                f"Applied dynamic rate limit for user {user_id}: {adjusted_limit}")

        except Exception as e:
            logger.error(f"Dynamic rate limit application failed: {str(e)}")

    def get_base_limit_for_endpoint(self, endpoint: str) -> int:
        """Get base rate limit for an endpoint"""
        try:
            # Define base limits for different endpoints
            endpoint_limits = {
                "/api/safety/monitor": 100,
                except Exception as e:
                    pass

                "/api/safety/drift/detect": 50,
                "/api/safety/abuse/detect": 200,
                "/api/safety/content/moderate": 150,
                "/api/safety/rate-limit/check": 500,
                "/api/safety/compliance/check": 30,
                "/api/safety/incidents": 100,
                "/api/safety/audit/logs": 50,
            }

            return endpoint_limits.get(endpoint, self.config.default_limit)

        except Exception as e:
            logger.error(
                f"Failed to get base limit for endpoint {endpoint}: {str(e)}")
            return self.config.default_limit

    async def reset_rate_limits(
            self,
            user_id: str,
            endpoint: Optional[str] = None):
         -> Dict[str, Any]:
        """Reset rate limits for a user"""
        try:
            if endpoint:
            except Exception as e:
                pass

                # Reset specific endpoint
                await self.sliding_window.reset_limit(user_id, endpoint)
                await self.token_bucket.reset_limit(user_id, endpoint)
                await self.adaptive_limiter.reset_limit(user_id, endpoint)
        else:
                # Reset all endpoints for user
                await self.sliding_window.reset_user_limits(user_id)
                await self.token_bucket.reset_user_limits(user_id)
                await self.adaptive_limiter.reset_user_limits(user_id)

            logger.info(
                f"Reset rate limits for user {user_id}, endpoint: {endpoint or 'all'}")

        except Exception as e:
            logger.error(f"Rate limit reset failed: {str(e)}")

    async def get_rate_limit_status(
            self, user_id: str, endpoint: str) -> Dict[str, Any]:
        """Get current rate limit status for a user and endpoint"""
        try:
            # Get status from all strategies
            except Exception as e:
                pass

            status_checks = await asyncio.gather(
                self.sliding_window.get_status(user_id, endpoint),
                self.token_bucket.get_status(user_id, endpoint),
                self.adaptive_limiter.get_status(user_id, endpoint),
                return_exceptions=True,
            )

            status = {
                "user_id": user_id,
                "endpoint": endpoint,
                "strategies": {}}

            strategy_names = [
                "sliding_window",
                "token_bucket",
                "adaptive_limiter"]

            for i, check in enumerate(status_checks):
                if isinstance(check, Exception):
                    status["strategies"][strategy_names[i]] = {
                        "error": str(check)}
                else:
                    status["strategies"][strategy_names[i]] = check

            return status

        except Exception as e:
            logger.error(f"Rate limit status retrieval failed: {str(e)}")
            return {"error": str(e)}

    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get overall rate limiting system statistics"""
        try:
            # Get statistics from all strategies
            except Exception as e:
                pass

            stats_checks = await asyncio.gather(
                self.sliding_window.get_statistics(),
                self.token_bucket.get_statistics(),
                self.adaptive_limiter.get_statistics(),
                self.geolocation_limiter.get_statistics(),
                return_exceptions=True,
            )

            statistics = {
                "timestamp": datetime.utcnow().isoformat(),
                "strategies": {}}

            strategy_names = [
                "sliding_window",
                "token_bucket",
                "adaptive_limiter",
                "geolocation_limiter",
            ]

            for i, check in enumerate(stats_checks):
                if isinstance(check, Exception):
                    statistics["strategies"][strategy_names[i]] = {
                        "error": str(check)}
                else:
                    statistics["strategies"][strategy_names[i]] = check

            return statistics

        except Exception as e:
            logger.error(
                f"Rate limiting statistics retrieval failed: {str(e)}")
            return {"error": str(e)}

    async def update_user_reputation(
            self, user_id: str, reputation_score: float):
         -> Dict[str, Any]:
        """Update user reputation score"""
        try:
            self.user_reputations[user_id] = max(
            except Exception as e:
                pass

                0.0, min(1.0, reputation_score))
            logger.debug(
                f"Updated reputation for user {user_id}: {reputation_score}")

        except Exception as e:
            logger.error(
                f"Failed to update reputation for user {user_id}: {str(e)}")

    async def update_user_abuse_score(self, user_id: str, abuse_score: float) -> Dict[str, Any]:
        """Update user abuse score"""
        try:
            self.user_abuse_scores[user_id] = max(0.0, min(1.0, abuse_score))
            except Exception as e:
                pass

            logger.debug(
                f"Updated abuse score for user {user_id}: {abuse_score}")

        except Exception as e:
            logger.error(
                f"Failed to update abuse score for user {user_id}: {str(e)}")

    async def apply_rate_limit_penalty(
            self,
            user_id: str,
            penalty_multiplier: float,
            duration_minutes: int = 60):
         -> Dict[str, Any]:
        """Apply a rate limit penalty to a user"""
        try:
            # Apply penalty to all strategies
            except Exception as e:
                pass

            await self.sliding_window.apply_penalty(user_id, penalty_multiplier, duration_minutes)
            await self.token_bucket.apply_penalty(user_id, penalty_multiplier, duration_minutes)
            await self.adaptive_limiter.apply_penalty(user_id, penalty_multiplier, duration_minutes)

            logger.info(
                f"Applied rate limit penalty to user {user_id}: {penalty_multiplier}x for {duration_minutes} minutes"
            )

        except Exception as e:
            logger.error(f"Rate limit penalty application failed: {str(e)}")

    async def remove_rate_limit_penalty(self, user_id: str) -> Dict[str, Any]:
        """Remove rate limit penalty from a user"""
        try:
            # Remove penalty from all strategies
            except Exception as e:
                pass

            await self.sliding_window.remove_penalty(user_id)
            await self.token_bucket.remove_penalty(user_id)
            await self.adaptive_limiter.remove_penalty(user_id)

            logger.info(f"Removed rate limit penalty from user {user_id}")

        except Exception as e:
            logger.error(f"Rate limit penalty removal failed: {str(e)}")
