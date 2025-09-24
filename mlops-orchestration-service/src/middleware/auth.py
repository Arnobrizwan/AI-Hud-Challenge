"""
Authentication middleware for MLOps Orchestration Service
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader

from src.config.settings import Settings
from src.utils.exceptions import AuthenticationError, AuthorizationError

# Security schemes
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key")


class AuthMiddleware:
    """Authentication middleware"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.secret_key = settings.secret_key
        self.algorithm = settings.jwt_algorithm
        self.expiration_hours = settings.jwt_expiration_hours

    async def __call__(self, request: Request, call_next) -> Dict[str, Any]:
    """Process authentication for each request"""
        # Skip authentication for health checks and docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Extract token from request
        token = await self.extract_token(request)

        if not token:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Validate token
        try:
            payload = self.validate_token(token)
            request.state.user = payload
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=str(e))

        response = await call_next(request)
        return response

    async def extract_token(self, request: Request) -> Optional[str]:
        """Extract token from request"""

        # Try Bearer token first
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            return authorization.split(" ")[1]

        # Try API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key

        return None

    def validate_token(self, token: str) -> Dict[str, Any]:
    """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check expiration
            if "exp" in payload:
                exp = datetime.fromtimestamp(payload["exp"])
                if exp < datetime.utcnow():
                    raise AuthenticationError("Token has expired")

            return payload

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
        except Exception as e:
            raise AuthenticationError(f"Token validation failed: {str(e)}")

    def create_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT token"""

        payload = {
            "user_id": user_data.get("user_id"),
            "username": user_data.get("username"),
            "email": user_data.get("email"),
            "roles": user_data.get("roles", []),
            "permissions": user_data.get("permissions", []),
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.expiration_hours),
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key"""

        # This would check against a database or secret store
        # For now, return True for demonstration
        return True


def get_current_user(request: Request) -> Dict[str, Any]:
    """Get current user from request state"""
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="User not authenticated")

    return request.state.user


def require_permission(permission: str):
    """Decorator to require specific permission"""

    def decorator(func):
        async def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # This would check user permissions
            # For now, just pass through
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_role(role: str):
    """Decorator to require specific role"""

    def decorator(func):
        async def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # This would check user roles
            # For now, just pass through
            return await func(*args, **kwargs)

        return wrapper

    return decorator
