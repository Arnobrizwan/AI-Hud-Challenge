"""
Request logging middleware
"""

import json
import logging
import time
from typing import Any, Dict

from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from config.settings import get_settings

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware:
    """Request logging middleware for observability"""

    def __init__(self, app):
        self.app = app
        self.settings = get_settings()

    async def __call__(self, scope, receive, send) -> Dict[str, Any]:
        if scope["type"] == "http":
            request = Request(scope, receive)

            # Start timing
            start_time = time.time()

            # Log request
            await self._log_request(request)

            # Process request
            response_data = {}
            status_code = 200

            async def send_wrapper(message) -> Dict[str, Any]:
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    response_data["status_code"] = status_code
                    response_data["headers"] = dict(message.get("headers", []))
                await send(message)

            await self.app(scope, receive, send_wrapper)

            # Log response
            end_time = time.time()
            duration = end_time - start_time

            await self._log_response(request, status_code, duration, response_data)

        else:
            await self.app(scope, receive, send)

    async def _log_request(self, request: Request) -> Dict[str, Any]:
        """Log incoming request"""
        try:
            # Extract request information
            request_data = {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
                "timestamp": time.time(),
            }

            # Add user information if available
            if hasattr(request.state, "user_id"):
                request_data["user_id"] = request.state.user_id
                request_data["user_roles"] = getattr(request.state, "user_roles", [])

            # Log request
            logger.info(
                f"Request: {request.method} {request.url.path}",
                extra={"request_data": request_data, "event_type": "request_start"},
            )

        except Exception as e:
            logger.error(f"Failed to log request: {str(e)}")

    async def _log_response(
        self, request: Request, status_code: int, duration: float, response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Log response"""

        try:
            # Extract response information
            response_log_data = {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration * 1000, 2),
                "response_headers": response_data.get("headers", {}),
                "timestamp": time.time(),
            }

            # Add user information if available
            if hasattr(request.state, "user_id"):
                response_log_data["user_id"] = request.state.user_id

            # Determine log level based on status code
            if status_code >= 500:
                log_level = "error"
            elif status_code >= 400:
                log_level = "warning"
            else:
                log_level = "info"

            # Log response
            getattr(logger, log_level)(
                f"Response: {request.method} {request.url.path} -> {status_code} ({duration:.3f}s)",
                extra={"response_data": response_log_data, "event_type": "request_complete"},
            )

            # Log slow requests
            if duration > 5.0:  # Requests taking more than 5 seconds
                logger.warning(
                    f"Slow request: {request.method} {request.url.path} took {duration:.3f}s",
                    extra={
                        "slow_request": True,
                        "duration": duration,
                        "path": request.url.path,
                        "method": request.method,
                    },
                )

        except Exception as e:
            logger.error(f"Failed to log response: {str(e)}")

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""

        # Check for forwarded IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get the first IP in the chain
            return forwarded_for.split(",")[0].strip()

        # Check for real IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"

    def _should_log_request(self, request: Request) -> bool:
        """Determine if request should be logged"""

        # Skip logging for health checks and metrics
        skip_paths = ["/health", "/metrics", "/favicon.ico"]

        return not any(request.url.path.startswith(path) for path in skip_paths)

    def _mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in logs"""
        sensitive_keys = ["authorization", "x-api-key", "password", "token", "secret", "key"]

        masked_data = data.copy()

        for key, value in masked_data.items():
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                masked_data[key] = "***MASKED***"

        return masked_data


class StructuredRequestLoggingMiddleware(RequestLoggingMiddleware):
    """Structured request logging middleware with JSON output"""

    async def _log_request(self, request: Request) -> Dict[str, Any]:
        """Log incoming request with structured data"""
        try:
            # Extract request information
            request_data = {
                "event_type": "request_start",
                "timestamp": time.time(),
                "request": {
                    "method": request.method,
                    "url": str(request.url),
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "headers": self._mask_sensitive_data(dict(request.headers)),
                    "client_ip": self._get_client_ip(request),
                    "user_agent": request.headers.get("user-agent", ""),
                },
            }

            # Add user information if available
            if hasattr(request.state, "user_id"):
                request_data["user"] = {
                    "user_id": request.state.user_id,
                    "roles": getattr(request.state, "user_roles", []),
                }

            # Log as structured JSON
            logger.info("Request started", extra={"structured_data": request_data})

        except Exception as e:
            logger.error(f"Failed to log structured request: {str(e)}")

    async def _log_response(
        self, request: Request, status_code: int, duration: float, response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Log response with structured data"""

        try:
            # Extract response information
            response_log_data = {
                "event_type": "request_complete",
                "timestamp": time.time(),
                "request": {
                    "method": request.method,
                    "url": str(request.url),
                    "path": request.url.path,
                },
                "response": {
                    "status_code": status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "headers": self._mask_sensitive_data(response_data.get("headers", {})),
                },
            }

            # Add user information if available
            if hasattr(request.state, "user_id"):
                response_log_data["user"] = {"user_id": request.state.user_id}

            # Add performance metrics
            response_log_data["performance"] = {
                "duration_seconds": duration,
                "is_slow": duration > 5.0,
                "status_category": self._get_status_category(status_code),
            }

            # Determine log level
            log_level = self._get_log_level(status_code, duration)

            # Log as structured JSON
            getattr(logger, log_level)("Request completed", extra={"structured_data": response_log_data})

        except Exception as e:
            logger.error(f"Failed to log structured response: {str(e)}")

    def _get_status_category(self, status_code: int) -> str:
        """Get status code category"""

        if 200 <= status_code < 300:
            return "success"
        elif 300 <= status_code < 400:
            return "redirect"
        elif 400 <= status_code < 500:
            return "client_error"
        elif 500 <= status_code < 600:
            return "server_error"
        else:
            return "unknown"

    def _get_log_level(self, status_code: int, duration: float) -> str:
        """Get appropriate log level based on status code and duration"""

        if status_code >= 500:
            return "error"
        elif status_code >= 400 or duration > 10.0:
            return "warning"
        else:
            return "info"
