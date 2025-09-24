"""Authentication middleware for Content Enrichment Service."""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
import structlog
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import settings

logger = structlog.get_logger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


class AuthMiddleware:
    """Authentication middleware for API access."""

    def __init__(self):
        """Initialize the auth middleware."""
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm
        self.token_expire_minutes = settings.access_token_expire_minutes

    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create a JWT access token."""
        try:
            to_encode = data.copy()
            expire = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)
            to_encode.update({"exp": expire})

            encoded_jwt = jwt.encode(
                to_encode,
                self.secret_key,
                algorithm=self.algorithm)
            return encoded_jwt

        except Exception as e:
            logger.error("Token creation failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail="Token creation failed")

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token, self.secret_key, algorithms=[
                    self.algorithm])
            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.JWTError as e:
            logger.warning("Token verification failed", error=str(e))
            return None
        except Exception as e:
            logger.error("Token verification error", error=str(e))
            return None

    async def get_current_user(
        self, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> Optional[Dict[str, Any]]:
        """Get current user from token."""
        if not credentials:
            return None

        token = credentials.credentials
        payload = self.verify_token(token)

        if payload is None:
            return None

        return {
            "user_id": payload.get("sub"),
            "username": payload.get("username"),
            "email": payload.get("email"),
            "roles": payload.get("roles", []),
            "permissions": payload.get("permissions", []),
        }

    async def require_auth(
        self, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> Dict[str, Any]:
    """Require authentication for protected endpoints."""
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user = await self.get_current_user(credentials)
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return user

    async def require_permission(self, permission: str) -> Dict[str, Any]:
    """Require specific permission for endpoint access."""
        async def permission_checker(
            user: Dict[str, Any] = Depends(self.require_auth),
        ) -> Dict[str, Any]:
            user_permissions = user.get("permissions", [])

            if permission not in user_permissions:
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission '{permission}' required")

            return user

        return permission_checker

    async def require_role(self, role: str) -> Dict[str, Any]:
    """Require specific role for endpoint access."""
        async def role_checker(user: Dict[str, Any] = Depends(
                self.require_auth)) -> Dict[str, Any]:
            user_roles = user.get("roles", [])

            if role not in user_roles:
                raise HTTPException(
                    status_code=403,
                    detail=f"Role '{role}' required")

            return user

        return role_checker

    def create_api_key_token(
            self,
            api_key: str,
            permissions: list = None) -> str:
        """Create a token for API key authentication."""
        data = {
            "sub": api_key,
            "type": "api_key",
            "permissions": permissions or ["enrich:read", "enrich:write"],
        }
        return self.create_access_token(data)

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key."""
        # In a real implementation, you would check against a database
        # For now, we'll use a simple validation
        valid_api_keys = {
            "test-api-key": ["enrich:read", "enrich:write"],
            "readonly-api-key": ["enrich:read"],
            "admin-api-key": ["enrich:read", "enrich:write", "admin:all"],
        }

        return api_key in valid_api_keys

    async def get_api_key_permissions(self, api_key: str) -> list:
        """Get permissions for an API key."""
        valid_api_keys = {
            "test-api-key": ["enrich:read", "enrich:write"],
            "readonly-api-key": ["enrich:read"],
            "admin-api-key": ["enrich:read", "enrich:write", "admin:all"],
        }

        return valid_api_keys.get(api_key, [])

    async def authenticate_request(
            self, request: Request) -> Optional[Dict[str, Any]]:
        """Authenticate a request and return user info."""
        try:
            # Check for API key in headers
            api_key = request.headers.get("X-API-Key")
            if api_key:
                if await self.validate_api_key(api_key):
                    permissions = await self.get_api_key_permissions(api_key)
                    return {
                        "user_id": api_key,
                        "type": "api_key",
                        "permissions": permissions}
        else:
                    return None

            # Check for Bearer token
            authorization = request.headers.get("Authorization")
            if authorization and authorization.startswith("Bearer "):
                token = authorization.split(" ")[1]
                payload = self.verify_token(token)
                if payload:
                    return {
                        "user_id": payload.get("sub"),
                        "username": payload.get("username"),
                        "email": payload.get("email"),
                        "roles": payload.get("roles", []),
                        "permissions": payload.get("permissions", []),
                        "type": "jwt",
                    }

            return None

        except Exception as e:
            logger.error("Request authentication failed", error=str(e))
            return None

    def create_service_token(self, service_name: str) -> str:
        """Create a token for service-to-service authentication."""
        data = {
            "sub": service_name,
            "type": "service",
            "permissions": ["enrich:read", "enrich:write"],
        }
        return self.create_access_token(data)

    async def validate_service_token(self, token: str) -> bool:
        """Validate a service token."""
        try:
            payload = self.verify_token(token)
            if payload and payload.get("type") == "service":
                return True
            return False
        except Exception:
            return False


# Global auth middleware instance
auth_middleware = AuthMiddleware()


# Dependency functions for FastAPI
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """Get current user dependency."""
    return await auth_middleware.get_current_user(credentials)


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """Require authentication dependency."""
    return await auth_middleware.require_auth(credentials)


def require_permission(permission: str):
    """Require permission dependency factory."""
    return auth_middleware.require_permission(permission)


def require_role(role: str):
    """Require role dependency factory."""
    return auth_middleware.require_role(role)
