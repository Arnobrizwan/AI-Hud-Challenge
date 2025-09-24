"""
Rate limiting middleware for MLOps Orchestration Service
"""

import time
from typing import Dict, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from src.config.settings import Settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware:
    """Rate limiting middleware using sliding window algorithm"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.requests_per_window = settings.rate_limit_requests
        self.window_seconds = settings.rate_limit_window
        self.requests: Dict[str, list] = {}

    async def __call__(self, request: Request, call_next) -> Dict[str, Any]:
        """Process rate limiting for each request"""

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/live", "/health/ready"]:
            return await call_next(request)

        # Get client identifier
        client_id = self.get_client_id(request)

        # Check rate limit
        if not self.is_allowed(client_id):
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {self.requests_per_window} per {self.window_seconds} seconds",
                    "retry_after": self.get_retry_after(client_id),
                },
                headers={
                    "Retry-After": str(self.get_retry_after(client_id)),
                    "X-RateLimit-Limit": str(self.requests_per_window),
                    "X-RateLimit-Remaining": str(self.get_remaining_requests(client_id)),
                    "X-RateLimit-Reset": str(int(time.time()) + self.window_seconds),
                },
            )

        # Record request
        self.record_request(client_id)

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(
            self.get_remaining_requests(client_id))
        response.headers["X-RateLimit-Reset"] = str(
            int(time.time()) + self.window_seconds)

        return response

    def get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""

        # Try to get user ID from request state (if authenticated)
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.get('user_id', 'anonymous')}"

        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"

        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client"""

        current_time = time.time()
        window_start = current_time - self.window_seconds

        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id] if req_time > window_start]
        else:
            self.requests[client_id] = []

        # Check if under limit
        return len(self.requests[client_id]) < self.requests_per_window

    def record_request(self, client_id: str) -> None:
        """Record a request for client"""

        current_time = time.time()

        if client_id not in self.requests:
            self.requests[client_id] = []

        self.requests[client_id].append(current_time)

    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""

        if client_id not in self.requests:
            return self.requests_per_window

        current_time = time.time()
        window_start = current_time - self.window_seconds

        # Count requests in current window
        recent_requests = [
            req_time for req_time in self.requests[client_id] if req_time > window_start]

        return max(0, self.requests_per_window - len(recent_requests))

    def get_retry_after(self, client_id: str) -> int:
        """Get retry after seconds for client"""

        if client_id not in self.requests or not self.requests[client_id]:
            return self.window_seconds

        # Get oldest request in current window
        current_time = time.time()
        window_start = current_time - self.window_seconds

        recent_requests = [
            req_time for req_time in self.requests[client_id] if req_time > window_start]

        if not recent_requests:
            return self.window_seconds

        oldest_request = min(recent_requests)
        retry_after = int(oldest_request + self.window_seconds - current_time)

        return max(1, retry_after)


class AdvancedRateLimitMiddleware:
    """Advanced rate limiting with different limits per endpoint"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.endpoint_limits = {
            "/api/v1/pipelines/": {"requests": 50, "window": 60},
            "/api/v1/training/": {"requests": 20, "window": 60},
            "/api/v1/deployment/": {"requests": 10, "window": 60},
            "/api/v1/monitoring/": {"requests": 100, "window": 60},
            "/api/v1/features/": {"requests": 200, "window": 60},
            "/api/v1/retraining/": {"requests": 10, "window": 60},
        }
        self.requests: Dict[str, Dict[str, list]] = {}

    async def __call__(self, request: Request, call_next) -> Dict[str, Any]:
        """Process advanced rate limiting for each request"""

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/live", "/health/ready"]:
            return await call_next(request)

        # Get endpoint-specific limits
        endpoint_limits = self.get_endpoint_limits(request.url.path)
        if not endpoint_limits:
            return await call_next(request)

        # Get client identifier
        client_id = self.get_client_id(request)

        # Check rate limit
        if not self.is_allowed(client_id, request.url.path, endpoint_limits):
            logger.warning(
                f"Rate limit exceeded for client: {client_id} on endpoint: {request.url.path}"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests for endpoint. Limit: {endpoint_limits['requests']} per {endpoint_limits['window']} seconds",
                    "retry_after": self.get_retry_after(
                        client_id,
                        request.url.path,
                        endpoint_limits),
                },
            )

        # Record request
        self.record_request(client_id, request.url.path)

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(
            endpoint_limits["requests"])
        response.headers["X-RateLimit-Remaining"] = str(
            self.get_remaining_requests(
                client_id, request.url.path, endpoint_limits))
        response.headers["X-RateLimit-Reset"] = str(
            int(time.time()) + endpoint_limits["window"])

        return response

    def get_endpoint_limits(self, path: str) -> Optional[Dict[str, int]]:
        """Get rate limits for endpoint"""

        for endpoint, limits in self.endpoint_limits.items():
            if path.startswith(endpoint):
                return limits

        return None

    def get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""

        # Try to get user ID from request state (if authenticated)
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.get('user_id', 'anonymous')}"

        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"

        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    def is_allowed(self, client_id: str, endpoint: str,
                   limits: Dict[str, int]) -> bool:
        """Check if request is allowed for client and endpoint"""

        current_time = time.time()
        window_start = current_time - limits["window"]

        # Clean old requests
        if client_id in self.requests and endpoint in self.requests[client_id]:
            self.requests[client_id][endpoint] = [
                req_time
                for req_time in self.requests[client_id][endpoint]
                if req_time > window_start
            ]
        else:
            if client_id not in self.requests:
                self.requests[client_id] = {}
            self.requests[client_id][endpoint] = []

        # Check if under limit
        return len(self.requests[client_id][endpoint]) < limits["requests"]

    def record_request(self, client_id: str, endpoint: str) -> None:
        """Record a request for client and endpoint"""

        current_time = time.time()

        if client_id not in self.requests:
            self.requests[client_id] = {}
        if endpoint not in self.requests[client_id]:
            self.requests[client_id][endpoint] = []

        self.requests[client_id][endpoint].append(current_time)

    def get_remaining_requests(
            self, client_id: str, endpoint: str, limits: Dict[str, int]) -> int:
        """Get remaining requests for client and endpoint"""

        if client_id not in self.requests or endpoint not in self.requests[client_id]:
            return limits["requests"]

        current_time = time.time()
        window_start = current_time - limits["window"]

        # Count requests in current window
        recent_requests = [
            req_time for req_time in self.requests[client_id][endpoint] if req_time > window_start]

        return max(0, limits["requests"] - len(recent_requests))

    def get_retry_after(self, client_id: str, endpoint: str,
                        limits: Dict[str, int]) -> int:
        """Get retry after seconds for client and endpoint"""

        if client_id not in self.requests or endpoint not in self.requests[client_id]:
            return limits["window"]

        # Get oldest request in current window
        current_time = time.time()
        window_start = current_time - limits["window"]

        recent_requests = [
            req_time for req_time in self.requests[client_id][endpoint] if req_time > window_start]

        if not recent_requests:
            return limits["window"]

        oldest_request = min(recent_requests)
        retry_after = int(oldest_request + limits["window"] - current_time)

        return max(1, retry_after)
