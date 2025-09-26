"""
Authentication middleware for FastAPI.
"""

import time
from typing import Any, Dict, Optional

from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.config.settings import settings
from src.models.auth import TokenType, TokenValidationRequest
from src.models.common import ErrorResponse, ResponseStatus
from src.services.auth_service import (
    AuthenticationError,
    TokenExpiredError,
    TokenInvalidError,
    auth_service,
)
from src.utils.logging import get_correlation_id, get_logger, log_security_event, set_correlation_id
from src.utils.metrics import metrics_collector

logger = get_logger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for handling authentication across all requests."""

    def __init__(self, app, excluded_paths: list = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/auth/login",
        ]

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process authentication for incoming requests."""
        start_time = time.time()

        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path)
               for path in self.excluded_paths):
            return await call_next(request)

        # Extract authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header:
            return self._create_unauthorized_response(
                "Missing authorization header")

        # Parse Bearer token
        try:
            scheme, token = auth_header.split(" ", 1)
            if scheme.lower() != "bearer":
                return self._create_unauthorized_response(
                    "Invalid authorization scheme")
        except ValueError:
            return self._create_unauthorized_response(
                "Invalid authorization header format")

        # Get client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent")

        # Validate token
        try:
            validation_request = TokenValidationRequest(
                token=token, token_type=TokenType.ACCESS)

            validation_response = await auth_service.validate_request_token(
                validation_request, client_ip, user_agent
            )

            if not validation_response.valid or not validation_response.user:
                log_security_event(
                    logger,
                    "invalid_token_attempt",
                    client_ip=client_ip,
                    details={"error": validation_response.error},
                )
                return self._create_unauthorized_response(
                    validation_response.error or "Invalid token"
                )

            # Add user to request state
            request.state.user = validation_response.user
            request.state.authenticated = True

            # Set correlation ID if not already set
            if not get_correlation_id():
                set_correlation_id(validation_response.user.uid)

            # Log successful authentication
            logger.info(
                "Request authenticated",
                user_id=validation_response.user.uid,
                client_ip=client_ip,
                path=request.url.path,
                method=request.method,
            )

            # Record metrics
            duration = time.time() - start_time
            metrics_collector.record_auth_attempt("middleware", True, duration)

            return await call_next(request)

        except (TokenExpiredError, TokenInvalidError) as e:
            log_security_event(
                logger,
                "authentication_failed",
                client_ip=client_ip,
                details={"error": str(e), "path": request.url.path},
            )

            duration = time.time() - start_time
            metrics_collector.record_auth_attempt(
                "middleware", False, duration)

            return self._create_unauthorized_response(str(e))

        except AuthenticationError as e:
            logger.error(
                "Authentication error",
                error=str(e),
                client_ip=client_ip,
                path=request.url.path)

            duration = time.time() - start_time
            metrics_collector.record_auth_attempt(
                "middleware", False, duration)

            return self._create_unauthorized_response("Authentication failed")

        except Exception as e:
            logger.error(
                "Unexpected authentication error",
                error=str(e),
                client_ip=client_ip,
                path=request.url.path,
            )

            duration = time.time() - start_time
            metrics_collector.record_auth_attempt(
                "middleware", False, duration)

            return self._create_unauthorized_response(
                "Authentication service unavailable")

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (load balancer/proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback to direct connection
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"

    def _create_unauthorized_response(self, message: str) -> Response:
        """Create standardized unauthorized response."""
        error_response = ErrorResponse(
            status=ResponseStatus.ERROR,
            message="Unauthorized",
            errors=[message],
            error_code="AUTHENTICATION_REQUIRED",
        )

        return Response(
            content=error_response.json(),
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"content-type": "application/json"},
        )


class OptionalAuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for optional authentication (sets user if token is valid)."""

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process optional authentication."""
        # Try to authenticate if authorization header is present
        auth_header = request.headers.get("authorization")

        if auth_header:
            try:
                scheme, token = auth_header.split(" ", 1)
                if scheme.lower() == "bearer":
                    client_ip = self._get_client_ip(request)
                    user_agent = request.headers.get("user-agent")

                    validation_request = TokenValidationRequest(
                        token=token, token_type=TokenType.ACCESS
                    )

                    validation_response = await auth_service.validate_request_token(
                        validation_request, client_ip, user_agent
                    )

                    if validation_response.valid and validation_response.user:
                        request.state.user = validation_response.user
                        request.state.authenticated = True
                    else:
                        request.state.authenticated = False
                else:
                    request.state.authenticated = False
            except Exception:
                request.state.authenticated = False
        else:
            request.state.authenticated = False

        return await call_next(request)

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


# Security utilities for dependency injection
security = HTTPBearer()


async def get_current_user(
        credentials: HTTPAuthorizationCredentials = security) -> Dict[str, Any]:
    """Dependency to get current authenticated user."""
    try:
        validation_request = TokenValidationRequest(
            token=credentials.credentials, token_type=TokenType.ACCESS
        )

        validation_response = await auth_service.validate_request_token(validation_request)

        if not validation_response.valid or not validation_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=validation_response.error or "Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return validation_response.user

    except Exception as e:
        logger.error("Failed to get current user", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(request: Request) -> Optional:
    """Dependency to get optional authenticated user."""
    return getattr(request.state, "user", None)


def require_roles(*required_roles):
    """Decorator to require specific user roles."""

    def decorator(func):
        async def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # Get user from request state or dependencies
            user = None
            for arg in args:
                if hasattr(arg, "state") and hasattr(arg.state, "user"):
                    user = arg.state.user
                    break

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required")

            if not any(role in user.roles for role in required_roles):
                log_security_event(
                    logger,
                    "insufficient_permissions",
                    user_id=user.uid,
                    details={
                        "required_roles": list(required_roles),
                        "user_roles": user.roles},
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions")

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_permissions(*required_permissions):
    """Decorator to require specific permissions."""

    def decorator(func):
        async def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # Get user from request state or dependencies
            user = None
            for arg in args:
                if hasattr(arg, "state") and hasattr(arg.state, "user"):
                    user = arg.state.user
                    break

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required")

            if not all(
                    perm in user.permissions for perm in required_permissions):
                log_security_event(
                    logger,
                    "insufficient_permissions",
                    user_id=user.uid,
                    details={
                        "required_permissions": list(required_permissions),
                        "user_permissions": user.permissions,
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions")

            return await func(*args, **kwargs)

        return wrapper

    return decorator
