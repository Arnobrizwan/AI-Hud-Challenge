"""
Rate limiting middleware
"""

import logging
import time
from collections import defaultdict, deque
from typing import Dict, Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from config.settings import get_settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware:
    """Rate limiting middleware using sliding window algorithm"""

    def __init__(self, app):
        self.app = app
        self.settings = get_settings()
        self.requests = defaultdict(lambda: deque())
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()

    async def __call__(self, scope, receive, send) -> Dict[str, Any]:
        if scope["type"] == "http":
            request = Request(scope, receive)

            # Skip rate limiting for health checks
            if self._is_health_endpoint(request.url.path):
    await self.app(scope, receive, send)
                return

            # Check rate limit
            client_id = self._get_client_id(request)
            if not self._is_allowed(client_id, request.url.path):
                response = self._create_rate_limit_response()
                await response(scope, receive, send)
                return

            # Cleanup old entries periodically
            await self._cleanup_old_entries()

        await self.app(scope, receive, send)

    def _is_health_endpoint(self, path: str) -> bool:
        """Check if endpoint is a health check"""

        health_paths = ["/health", "/metrics", "/"]

        return any(path.startswith(health_path)
                   for health_path in health_paths)

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""

        # Try to get user ID from request state (if authenticated)
        if hasattr(request.state, "user_id") and request.state.user_id:
            return f"user:{request.state.user_id}"

        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get the first IP in the chain
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        return f"ip:{client_ip}"

    def _is_allowed(self, client_id: str, path: str) -> bool:
        """Check if request is allowed based on rate limits"""

        current_time = time.time()
        window_size = self.settings.rate_limit_window

        # Get rate limit for the endpoint
        rate_limit = self._get_rate_limit_for_path(path)

        # Clean old requests outside the window
        client_requests = self.requests[client_id]
        while client_requests and client_requests[0] <= current_time - window_size:
            client_requests.popleft()

        # Check if under rate limit
        if len(client_requests) >= rate_limit:
            logger.warning(f"Rate limit exceeded for {client_id} on {path}")
            return False

        # Add current request
        client_requests.append(current_time)
        return True

    def _get_rate_limit_for_path(self, path: str) -> int:
        """Get rate limit for specific path"""

        # Different rate limits for different endpoints
        if path.startswith("/api/v1/alerts"):
            return 50  # 50 requests per window for alerts
        elif path.startswith("/api/v1/incidents"):
            return 30  # 30 requests per window for incidents
        elif path.startswith("/api/v1/runbooks"):
            return 20  # 20 requests per window for runbooks
        elif path.startswith("/api/v1/chaos"):
            return 10  # 10 requests per window for chaos experiments
        else:
            return self.settings.rate_limit_requests

    def _create_rate_limit_response(self):
        """Create rate limit exceeded response"""

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": "Rate limit exceeded",
                "retry_after": self.settings.rate_limit_window,
            },
            headers={"Retry-After": str(self.settings.rate_limit_window)},
        )

    async def _cleanup_old_entries(self) -> Dict[str, Any]:
        """Cleanup old rate limiting entries"""

        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        self.last_cleanup = current_time
        window_size = self.settings.rate_limit_window
        cutoff_time = current_time - window_size

        # Remove old entries
        clients_to_remove = []
        for client_id, requests in self.requests.items():
            while requests and requests[0] <= cutoff_time:
                requests.popleft()

            # Remove empty entries
            if not requests:
                clients_to_remove.append(client_id)

        for client_id in clients_to_remove:
            del self.requests[client_id]

        logger.info(
            f"Cleaned up rate limiting data for {len(clients_to_remove)} clients")


class AdaptiveRateLimitMiddleware(RateLimitMiddleware):
    """Adaptive rate limiting that adjusts based on system load"""

    def __init__(self, app):
        super().__init__(app)
        self.load_history = deque(maxlen=100)
        self.current_load = 0.0

    def _get_rate_limit_for_path(self, path: str) -> int:
        """Get adaptive rate limit based on system load"""

        base_rate_limit = super()._get_rate_limit_for_path(path)

        # Adjust based on current system load
        if self.current_load > 0.8:  # High load
            return max(1, int(base_rate_limit * 0.5))
        elif self.current_load > 0.6:  # Medium load
            return max(1, int(base_rate_limit * 0.7))
        else:  # Low load
            return base_rate_limit

    def _update_system_load(self, load: float):
        """Update system load metric"""

        self.current_load = load
        self.load_history.append(load)

    def _get_average_load(self) -> float:
        """Get average system load over recent history"""

        if not self.load_history:
            return 0.0

        return sum(self.load_history) / len(self.load_history)
