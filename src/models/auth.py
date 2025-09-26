"""
Authentication and authorization models.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class UserRole(str, Enum):
    """User roles for authorization."""

    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"
    SYSTEM = "system"


class AuthProvider(str, Enum):
    """Authentication provider types."""

    FIREBASE = "firebase"
    API_KEY = "api_key"
    SYSTEM = "system"


class TokenType(str, Enum):
    """Token types."""

    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


class UserClaims(BaseModel):
    """User claims from JWT token."""

    uid: str = Field(description="User ID")
    email: Optional[str] = Field(default=None, description="User email")
    email_verified: bool = Field(default=False, description="Email verification status")
    name: Optional[str] = Field(default=None, description="User display name")
    picture: Optional[str] = Field(default=None, description="User profile picture URL")
    roles: List[UserRole] = Field(default=[UserRole.USER], description="User roles")
    permissions: List[str] = Field(default=[], description="User permissions")
    provider: AuthProvider = Field(default=AuthProvider.FIREBASE, description="Auth provider")

    @validator("email")
    def validate_email(cls, v: str) -> str:
        """Validate email format."""
        if v and "@" not in v:
            raise ValueError("Invalid email format")
        return v


class AuthenticatedUser(UserClaims):
    """Authenticated user with additional context."""

    token_type: TokenType = Field(description="Token type")
    issued_at: datetime = Field(description="Token issued timestamp")
    expires_at: datetime = Field(description="Token expiration timestamp")
    client_ip: Optional[str] = Field(default=None, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="Client user agent")

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.utcnow() > self.expires_at

    @property
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return UserRole.ADMIN in self.roles

    def has_role(self, role: UserRole) -> bool:
        """Check if user has specific role."""
        return role in self.roles

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions


class LoginRequest(BaseModel):
    """Login request model."""

    token: str = Field(description="Firebase ID token or API key")
    provider: AuthProvider = Field(default=AuthProvider.FIREBASE, description="Auth provider")


class LoginResponse(BaseModel):
    """Login response model."""

    access_token: str = Field(description="Access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(description="Token expiration in seconds")
    user: UserClaims = Field(description="User information")


class TokenValidationRequest(BaseModel):
    """Token validation request."""

    token: str = Field(description="Token to validate")
    token_type: TokenType = Field(default=TokenType.ACCESS, description="Token type")


class TokenValidationResponse(BaseModel):
    """Token validation response."""

    valid: bool = Field(description="Token validity status")
    user: Optional[AuthenticatedUser] = Field(default=None, description="User information if valid")
    error: Optional[str] = Field(default=None, description="Error message if invalid")


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""

    refresh_token: str = Field(description="Refresh token")


class LogoutRequest(BaseModel):
    """Logout request."""

    token: str = Field(description="Access token to revoke")
