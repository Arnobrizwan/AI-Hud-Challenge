"""
Authentication middleware
"""

import logging
from typing import Optional

import jwt
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import get_settings

logger = logging.getLogger(__name__)

security = HTTPBearer()


class AuthMiddleware:
    """Authentication middleware for API requests"""

    def __init__(self, app):
        self.app = app
        self.settings = get_settings()

    async def __call__(self, scope, receive, send) -> Dict[str, Any]:
    if scope["type"] == "http":
            request = Request(scope, receive)

            # Skip auth for health checks and public endpoints
            if self._is_public_endpoint(request.url.path):
    await self.app(scope, receive, send)
                return

            # Extract and validate token
            try:
    await self._validate_request(request)
            except HTTPException as e:
                response = self._create_error_response(e.status_code, e.detail)
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)

    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public and doesn't require authentication"""

        public_paths = [
            "/",
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json"]

        return any(path.startswith(public_path)
                   for public_path in public_paths)

    async def _validate_request(self, request: Request) -> Dict[str, Any]:
    """Validate request authentication"""
        # Extract authorization header
        authorization = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header missing")

        # Extract token
        try:
            scheme, token = authorization.split(" ")
            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format",
            )

        # Validate token
        try:
            payload = jwt.decode(
                token, self.settings.jwt_secret, algorithms=[
                    self.settings.jwt_algorithm])

            # Add user info to request state
            request.state.user_id = payload.get("user_id")
            request.state.user_roles = payload.get("roles", [])

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token")

    def _create_error_response(self, status_code: int, detail: str):
        """Create error response"""

        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status_code, content={
                "detail": detail})


def get_current_user(request: Request) -> Optional[dict]:
    """Get current user from request state"""

    return {
        "user_id": getattr(request.state, "user_id", None),
        "roles": getattr(request.state, "user_roles", []),
    }


def require_role(required_role: str):
    """Decorator to require specific role"""

    def decorator(func):
        async def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # This would be implemented in the actual endpoint
            # For now, just pass through
            return await func(*args, **kwargs)

        return wrapper

    return decorator
