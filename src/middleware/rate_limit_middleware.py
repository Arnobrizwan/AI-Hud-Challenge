"""
Rate limiting middleware using Redis-based sliding window algorithm.
"""

import time
from typing import Any, Dict, List, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# from src.config.settings import settings  # Unused import
from src.models.common import RateLimitInfo, RateLimitResponse, ResponseStatus
from src.services.rate_limiter import RateLimitExceeded, rate_limiter
from src.utils.logging import get_logger, log_security_event
from src.utils.metrics import metrics_collector

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for applying rate limits to incoming requests."""

    def __init__(
        self,
        app: Any,
        excluded_paths: Optional[List[str]] = None,
        enable_user_limits: bool = True,
        enable_ip_limits: bool = True,
        enable_global_limits: bool = False,
    ) -> None:
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        self.enable_user_limits = enable_user_limits
        self.enable_ip_limits = enable_ip_limits
        self.enable_global_limits = enable_global_limits

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Apply rate limiting to incoming requests."""
        start_time = time.time()

        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            response: Response = await call_next(request)
            return response

        # Get client information
        client_ip = self._get_client_ip(request)
        user_id = self._get_user_id(request)

        # Prepare rate limit checks
        checks = []

        # IP-based rate limiting
        if self.enable_ip_limits:
            checks.append({"type": "ip", "identifier": client_ip, "name": "ip_limit"})

        # User-based rate limiting (if authenticated)
        if self.enable_user_limits and user_id:
            checks.append({"type": "user", "identifier": user_id, "name": "user_limit"})

        # Global rate limiting
        if self.enable_global_limits:
            checks.append({"type": "global", "name": "global_limit"})

        # Endpoint-specific rate limiting
        endpoint_config = self._get_endpoint_config(request)
        if endpoint_config:
            identifier = user_id or client_ip
            checks.append(
                {
                    "type": "endpoint",
                    "endpoint": endpoint_config["name"],
                    "identifier": identifier,
                    "limit": endpoint_config["limit"],
                    "window_seconds": endpoint_config.get("window_seconds", 60),
                    "name": "endpoint_limit",
                }
            )

        try:
            # Check multiple rate limits concurrently
            results = await rate_limiter.check_multiple_limits(checks)

            # Find the most restrictive limit that was exceeded
            most_restrictive_info = None
            most_restrictive_type = None

            for check_name, (allowed, rate_limit_info) in results.items():
                if not allowed and rate_limit_info:
                    # Priority: endpoint > user > ip > global
                    if check_name == "endpoint_limit":
                        most_restrictive_info = rate_limit_info
                        most_restrictive_type = "endpoint"
                        break
                    elif check_name == "user_limit" and most_restrictive_type != "endpoint":
                        most_restrictive_info = rate_limit_info
                        most_restrictive_type = "user"
                    elif check_name == "ip_limit" and most_restrictive_type not in [
                        "endpoint",
                        "user",
                    ]:
                        most_restrictive_info = rate_limit_info
                        most_restrictive_type = "ip"
                    elif check_name == "global_limit" and most_restrictive_type is None:
                        most_restrictive_info = rate_limit_info
                        most_restrictive_type = "global"

            # If any limit was exceeded, return rate limit response
            if most_restrictive_info:
                # Log rate limit violation
                log_security_event(
                    logger,
                    "rate_limit_exceeded",
                    user_id=user_id,
                    client_ip=client_ip,
                    details={
                        "limit_type": most_restrictive_type,
                        "path": request.url.path,
                        "method": request.method,
                        "limit": most_restrictive_info.limit,
                        "window_seconds": most_restrictive_info.window_seconds,
                    },
                )

                # Record metrics
                metrics_collector.record_rate_limit_block(most_restrictive_type or "unknown")

                return self._create_rate_limit_response(
                    most_restrictive_info, most_restrictive_type or "unknown"
                )

            # Add rate limit headers to response
            response = await call_next(request)

            # Add the most restrictive rate limit info to headers
            if results:
                most_restrictive = min(
                    (info for allowed, info in results.values() if allowed and info),
                    key=lambda x: x.remaining,
                    default=None,
                )

                if most_restrictive:
                    self._add_rate_limit_headers(response, most_restrictive)

            # Log successful request
            duration = time.time() - start_time
            logger.debug(
                "Request processed with rate limiting",
                user_id=user_id,
                client_ip=client_ip,
                path=request.url.path,
                method=request.method,
                duration=duration,
            )

            return response

        except Exception as e:
            logger.error(
                "Rate limiting error",
                error=str(e),
                client_ip=client_ip,
                user_id=user_id,
                path=request.url.path,
            )

            # On error, allow the request to proceed (fail open)
            error_response: Response = await call_next(request)
            return error_response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"

    def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from authenticated request."""
        if hasattr(request.state, "user") and request.state.user:
            return str(request.state.user.uid)
        return None

    def _get_endpoint_config(self, request: Request) -> Optional[dict]:
        """Get endpoint-specific rate limit configuration."""
        # Define endpoint-specific limits
        endpoint_limits = {
            "/auth/login": {"name": "login", "limit": 10, "window_seconds": 60},
            "/auth/register": {"name": "register", "limit": 5, "window_seconds": 60},
            "/search": {"name": "search", "limit": 100, "window_seconds": 60},
            "/upload": {"name": "upload", "limit": 20, "window_seconds": 60},
        }

        path = request.url.path
        for endpoint_path, config in endpoint_limits.items():
            if path.startswith(endpoint_path):
                return config

        return None

    def _create_rate_limit_response(
        self, rate_limit_info: RateLimitInfo, limit_type: str
    ) -> JSONResponse:
        """Create rate limit exceeded response."""
        response_data = RateLimitResponse(
            status=ResponseStatus.ERROR,
            message=f"Rate limit exceeded for {limit_type}",
            error_code="RATE_LIMIT_EXCEEDED",
            rate_limit=rate_limit_info,
        )

        headers = {
            "X-RateLimit-Limit": str(rate_limit_info.limit),
            "X-RateLimit-Remaining": str(rate_limit_info.remaining),
            "X-RateLimit-Reset": str(int(rate_limit_info.reset_time.timestamp())),
            "X-RateLimit-Window": str(rate_limit_info.window_seconds),
            "Retry-After": str(rate_limit_info.window_seconds),
        }

        return JSONResponse(status_code=429, content=response_data.dict(), headers=headers)

    def _add_rate_limit_headers(self, response: Response, rate_limit_info: Any) -> None:
        """Add rate limit headers to successful response."""
        response.headers["X-RateLimit-Limit"] = str(rate_limit_info.limit)
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_info.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(rate_limit_info.reset_time.timestamp()))
        response.headers["X-RateLimit-Window"] = str(rate_limit_info.window_seconds)


class AdaptiveRateLimitMiddleware(RateLimitMiddleware):
    """Advanced rate limiting with adaptive limits based on user behavior."""

    def __init__(self, app: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(app, **kwargs)
        self.suspicious_ips: set[str] = set()  # Track suspicious IPs
        self.trusted_users: set[str] = set()  # Track trusted users
        pass

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Apply adaptive rate limiting."""
        client_ip = self._get_client_ip(request)
        user_id = self._get_user_id(request)

        # Adjust limits based on reputation
        if await self._is_suspicious_request(request, client_ip, user_id):
            # Apply stricter limits for suspicious requests
            self._apply_strict_limits(request)
        elif await self._is_trusted_request(request, client_ip, user_id):
            # Apply relaxed limits for trusted requests
            self._apply_relaxed_limits(request)

        return await super().dispatch(request, call_next)

    async def _is_suspicious_request(
        self, request: Request, client_ip: str, user_id: Optional[str]
    ) -> bool:
        """Check if request appears suspicious."""
        # Check for suspicious patterns
        suspicious_patterns = [
            client_ip in self.suspicious_ips,
            self._has_suspicious_user_agent(request),
            self._has_suspicious_headers(request),
            await self._has_recent_failures(client_ip, user_id),
        ]

        return any(suspicious_patterns)

    async def _is_trusted_request(
        self, request: Request, client_ip: str, user_id: Optional[str]
    ) -> bool:
        """Check if request is from a trusted source."""
        if user_id and user_id in self.trusted_users:
            return True

        # Check for internal/admin networks
        if self._is_internal_ip(client_ip):
            return True

        return False

    def _has_suspicious_user_agent(self, request: Request) -> bool:
        """Check for suspicious user agent patterns."""
        user_agent = request.headers.get("user-agent", "").lower()

        suspicious_patterns = ["bot", "crawler", "scraper", "curl", "wget", "python-requests"]

        return any(pattern in user_agent for pattern in suspicious_patterns)

    def _has_suspicious_headers(self, request: Request) -> bool:
        """Check for suspicious request headers."""
        # Missing common browser headers
        if not request.headers.get("accept"):
            return True

        # Unusual accept-language
        accept_lang = request.headers.get("accept-language", "")
        if not accept_lang or len(accept_lang) > 100:
            return True

        return False

    async def _has_recent_failures(self, client_ip: str, user_id: Optional[str]) -> bool:
        """Check for recent authentication or validation failures."""
        # This would typically check a failure tracking system
        # For now, return False as a placeholder
        return False

    def _is_internal_ip(self, ip: str) -> bool:
        """Check if IP is from internal/admin networks."""
        # Define internal IP ranges
        internal_ranges = ["10.", "172.", "192.168.", "127.0.0.1", "localhost"]

        return any(ip.startswith(prefix) for prefix in internal_ranges)

    def _apply_strict_limits(self, request: Request) -> None:
        """Apply stricter rate limits."""
        # Modify request state to indicate strict limits should be used
        request.state.rate_limit_multiplier = 0.5  # Reduce limits by 50%

    def _apply_relaxed_limits(self, request: Request) -> None:
        """Apply relaxed rate limits."""
        # Modify request state to indicate relaxed limits should be used
        request.state.rate_limit_multiplier = 2.0  # Increase limits by 100%


# Utility functions for manual rate limit checks
async def check_rate_limit_dependency(
    request: Request, limit: int = 100, window: int = 60
) -> Dict[str, Any]:
    """Dependency function for manual rate limit checks in endpoints."""
    client_ip = request.client.host if request.client else "unknown"
    user_id = (
        getattr(request.state, "user", {}).get("uid") if hasattr(request.state, "user") else None
    )

    identifier = user_id or client_ip
    allowed, rate_limit_info = await rate_limiter.check_endpoint_rate_limit(
        request.url.path, identifier, limit, window
    )

    if not allowed:
        raise RateLimitExceeded(
            "endpoint",
            rate_limit_info.limit,
            rate_limit_info.window_seconds,
            rate_limit_info.reset_time,
        )

    return rate_limit_info.dict()
