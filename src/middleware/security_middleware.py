"""
Security middleware for adding security headers and CORS handling.
"""

import uuid
from typing import Dict, List, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from src.config.settings import settings
from src.utils.logging import get_logger, set_correlation_id

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers to responses."""

    def __init__(
        self,
        app: ASGIApp,
        enable_hsts: bool = True,
        enable_csp: bool = True,
        enable_frame_options: bool = True,
        enable_content_type_options: bool = True,
        enable_xss_protection: bool = True,
        enable_referrer_policy: bool = True,
        custom_headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(app)
        self.enable_hsts = enable_hsts
        self.enable_csp = enable_csp and settings.ENABLE_SECURITY_HEADERS
        self.enable_frame_options = enable_frame_options
        self.enable_content_type_options = enable_content_type_options
        self.enable_xss_protection = enable_xss_protection
        self.enable_referrer_policy = enable_referrer_policy
        self.custom_headers = custom_headers or {}

    async def dispatch(self, request: Request, call_next) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # Add security headers
        if self.enable_hsts and request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        if self.enable_csp:
            response.headers["Content-Security-Policy"] = settings.CSP_POLICY

        if self.enable_frame_options:
            response.headers["X-Frame-Options"] = "DENY"

        if self.enable_content_type_options:
            response.headers["X-Content-Type-Options"] = "nosniff"

        if self.enable_xss_protection:
            response.headers["X-XSS-Protection"] = "1; mode=block"

        if self.enable_referrer_policy:
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Add Permissions Policy (formerly Feature Policy)
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=(), "
            "accelerometer=(), ambient-light-sensor=(), "
            "encrypted-media=(), picture-in-picture=()"
        )

        # Add custom headers
        for header, value in self.custom_headers.items():
            response.headers[header] = value

        # Add server information (generic for security)
        response.headers["Server"] = settings.APP_NAME

        return response


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware for handling request correlation IDs."""

    def __init__(self, app: ASGIApp, header_name: str = "X-Correlation-ID"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process correlation ID for request tracking."""
        # Get correlation ID from header or generate new one
        correlation_id = request.headers.get(self.header_name) or str(uuid.uuid4())

        # Set correlation ID in context
        set_correlation_id(correlation_id)

        # Add to request state for access in handlers
        request.state.correlation_id = correlation_id

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers[self.header_name] = correlation_id

        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for limiting request body size."""

    def __init__(self, app: ASGIApp, max_size: int = None):
        super().__init__(app)
        self.max_size = max_size or settings.MAX_REQUEST_SIZE

    async def dispatch(self, request: Request, call_next) -> Response:
        """Check request size limits."""
        # Check Content-Length header
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size:
                    logger.warning(
                        "Request size limit exceeded",
                        content_length=size,
                        max_size=self.max_size,
                        client_ip=request.client.host if request.client else "unknown",
                        path=request.url.path,
                    )

                    return Response(
                        content='{"error": "Request entity too large"}',
                        status_code=413,
                        headers={"content-type": "application/json"},
                    )
            except ValueError:
                logger.warning("Invalid Content-Length header", content_length=content_length)

        return await call_next(request)


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware for handling request timeouts."""

    def __init__(self, app: ASGIApp, timeout_seconds: int = None):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds or settings.REQUEST_TIMEOUT

    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply request timeout."""
        import asyncio

        try:
            response = await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
            return response

        except asyncio.TimeoutError:
            logger.warning(
                "Request timeout",
                timeout=self.timeout_seconds,
                path=request.url.path,
                method=request.method,
                client_ip=request.client.host if request.client else "unknown",
            )

            return Response(
                content='{"error": "Request timeout"}',
                status_code=408,
                headers={"content-type": "application/json"},
            )


class ContentValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for validating request content types."""

    def __init__(
        self,
        app: ASGIApp,
        allowed_content_types: List[str] = None,
        require_content_type: List[str] = None,
    ):
        super().__init__(app)
        self.allowed_content_types = allowed_content_types or [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
        ]
        self.require_content_type = require_content_type or ["POST", "PUT", "PATCH"]

    async def dispatch(self, request: Request, call_next) -> Response:
        """Validate request content type."""
        method = request.method
        content_type = request.headers.get("content-type", "").split(";")[0].strip()

        # Check if content type is required for this method
        if method in self.require_content_type and not content_type:
            logger.warning(
                "Missing content-type header",
                method=method,
                path=request.url.path,
                client_ip=request.client.host if request.client else "unknown",
            )

            return Response(
                content='{"error": "Content-Type header required"}',
                status_code=400,
                headers={"content-type": "application/json"},
            )

        # Check if content type is allowed
        if content_type and content_type not in self.allowed_content_types:
            logger.warning(
                "Unsupported content type",
                content_type=content_type,
                method=method,
                path=request.url.path,
                client_ip=request.client.host if request.client else "unknown",
            )

            return Response(
                content='{"error": "Unsupported content type"}',
                status_code=415,
                headers={"content-type": "application/json"},
            )

        return await call_next(request)


def create_cors_middleware() -> CORSMiddleware:
    """Create CORS middleware with configured settings."""
    return CORSMiddleware(
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_CREDENTIALS,
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
        expose_headers=[
            "X-Correlation-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-RateLimit-Window",
        ],
    )


class SecurityMiddlewareStack:
    """Stack of security middlewares for easy application."""

    @staticmethod
    def create_stack(app: ASGIApp, custom_config: Optional[Dict] = None) -> ASGIApp:
        """Create a complete security middleware stack."""
        config = custom_config or {}

        # Apply middlewares in reverse order (last applied = first executed)

        # 1. Security headers (outermost)
        app = SecurityHeadersMiddleware(
            app,
            enable_hsts=config.get("enable_hsts", True),
            enable_csp=config.get("enable_csp", settings.ENABLE_SECURITY_HEADERS),
            custom_headers=config.get("custom_headers", {}),
        )

        # 2. CORS handling
        if config.get("enable_cors", True):
            app = create_cors_middleware()(app)

        # 3. Request validation
        app = ContentValidationMiddleware(
            app,
            allowed_content_types=config.get("allowed_content_types"),
            require_content_type=config.get("require_content_type"),
        )

        # 4. Request size limiting
        app = RequestSizeLimitMiddleware(app, max_size=config.get("max_request_size", settings.MAX_REQUEST_SIZE))

        # 5. Request timeout
        app = RequestTimeoutMiddleware(app, timeout_seconds=config.get("request_timeout", settings.REQUEST_TIMEOUT))

        # 6. Correlation ID (innermost)
        app = CorrelationIdMiddleware(app, header_name=config.get("correlation_header", "X-Correlation-ID"))

        return app
