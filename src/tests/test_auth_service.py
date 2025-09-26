"""
Unit tests for the authentication service.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import firebase_admin.auth as firebase_auth
import pytest
import pytest_asyncio

from src.models.auth import (
    AuthenticatedUser,
    AuthProvider,
    LoginRequest,
    LoginResponse,
    TokenType,
    TokenValidationRequest,
    UserClaims,
)
from src.services.auth_service import (
    AuthenticationError,
    AuthService,
    FirebaseAuthService,
    JWTAuthService,
    TokenExpiredError,
    TokenInvalidError,
)


@pytest.mark.unit
class TestFirebaseAuthService:
    """Test Firebase authentication service."""

    @pytest_asyncio.fixture
    async def firebase_service(self) -> FirebaseAuthService:
        """Create Firebase auth service for testing."""
        with patch("firebase_admin.initialize_app"), patch("firebase_admin._apps", {}):
            service = FirebaseAuthService()
            yield service

    @pytest.mark.asyncio
    async def test_validate_token_success(
            self, firebase_service, mock_firebase_auth) -> None:
        """Test successful token validation."""
        token = "valid-firebase-token"
        client_ip = "127.0.0.1"
        user_agent = "test-agent"

        result = await firebase_service.validate_token(token, client_ip, user_agent)

        assert isinstance(result, AuthenticatedUser)
        assert result.uid == "test-user-123"
        assert result.email == "test@example.com"
        assert result.email_verified is True
        assert result.name == "Test User"
        assert result.provider == AuthProvider.FIREBASE
        assert result.token_type == TokenType.ACCESS
        assert result.client_ip == client_ip
        assert result.user_agent == user_agent

        mock_firebase_auth.assert_called_once_with(
            token, check_revoked=True, app=firebase_service.app
        )

    @pytest.mark.asyncio
    async def test_validate_token_expired(self, firebase_service) -> None:
        """Test token validation with expired token."""
        token = "expired-firebase-token"

        with patch("firebase_admin.auth.verify_id_token") as mock_verify:
            mock_verify.side_effect = firebase_auth.ExpiredIdTokenError(
                "Token expired", cause=Exception("Token expired"))

            with pytest.raises(TokenExpiredError, match="Token has expired"):
                await firebase_service.validate_token(token, "127.0.0.1", "test-agent")

    @pytest.mark.asyncio
    async def test_validate_token_invalid(self, firebase_service) -> None:
        """Test token validation with invalid token."""
        token = "invalid-firebase-token"

        with patch("firebase_admin.auth.verify_id_token") as mock_verify:
            mock_verify.side_effect = firebase_auth.InvalidIdTokenError(
                "Invalid token", cause=Exception("Invalid token"))

            with pytest.raises(TokenInvalidError, match="Invalid token"):
                await firebase_service.validate_token(token, "127.0.0.1", "test-agent")

    @pytest.mark.asyncio
    async def test_validate_token_revoked(self, firebase_service) -> None:
        """Test token validation with revoked token."""
        token = "revoked-firebase-token"

        with patch("firebase_admin.auth.verify_id_token") as mock_verify:
            mock_verify.side_effect = firebase_auth.RevokedIdTokenError(
                "Token revoked")

            with pytest.raises(TokenInvalidError, match="Invalid token"):
                await firebase_service.validate_token(token, "127.0.0.1", "test-agent")

    @pytest.mark.asyncio
    async def test_get_user_by_uid_success(self, firebase_service) -> None:
        """Test getting user by UID successfully."""
        uid = "test-user-123"

        mock_user_record = Mock()
        mock_user_record.uid = uid
        mock_user_record.email = "test@example.com"
        mock_user_record.email_verified = True
        mock_user_record.display_name = "Test User"
        mock_user_record.photo_url = "https://example.com/avatar.jpg"

        with patch("firebase_admin.auth.get_user") as mock_get_user:
            mock_get_user.return_value = mock_user_record

            result = await firebase_service.get_user_by_uid(uid)

            assert isinstance(result, UserClaims)
            assert result.uid == uid
            assert result.email == "test@example.com"
            assert result.email_verified is True
            assert result.name == "Test User"
            assert result.picture == "https://example.com/avatar.jpg"

    @pytest.mark.asyncio
    async def test_get_user_by_uid_not_found(self, firebase_service) -> None:
        """Test getting user by UID when user not found."""
        uid = "non-existent-user"

        with patch("firebase_admin.auth.get_user") as mock_get_user:
            mock_get_user.side_effect = firebase_auth.UserNotFoundError(
                "User not found", cause=Exception("User not found"))

            result = await firebase_service.get_user_by_uid(uid)
            assert result is None

    @pytest.mark.asyncio
    async def test_revoke_tokens_success(self, firebase_service) -> None:
        """Test successful token revocation."""
        uid = "test-user-123"

        with patch("firebase_admin.auth.revoke_refresh_tokens") as mock_revoke:
            result = await firebase_service.revoke_tokens(uid)

            assert result is True
            mock_revoke.assert_called_once_with(uid, app=firebase_service.app)

    @pytest.mark.asyncio
    async def test_revoke_tokens_failure(self, firebase_service) -> None:
        """Test token revocation failure."""
        uid = "test-user-123"

        with patch("firebase_admin.auth.revoke_refresh_tokens") as mock_revoke:
            mock_revoke.side_effect = Exception("Revocation failed")

            result = await firebase_service.revoke_tokens(uid)
            assert result is False

    def test_health_check_success(self, firebase_service):
        """Test health check success."""
        with patch("firebase_admin._apps", {"default": Mock()}):
            result = firebase_service.health_check()
            assert result is True

    def test_health_check_failure(self, firebase_service):
        """Test health check failure."""
        firebase_service.app = None
        result = firebase_service.health_check()
        assert result is False


@pytest.mark.unit
class TestJWTAuthService:
    """Test JWT authentication service."""

    @pytest.fixture
    def jwt_service(self):
        """Create JWT auth service for testing."""
        return JWTAuthService()

    def test_create_access_token(self, jwt_service):
        """Test access token creation."""
        data = {"sub": "test-user-123", "email": "test@example.com"}

        token = jwt_service.create_access_token(data)

        assert isinstance(token, str)
        assert len(token.split(".")) == 3  # JWT has 3 parts

    def test_create_access_token_with_custom_expiry(self, jwt_service):
        """Test access token creation with custom expiry."""
        data = {"sub": "test-user-123"}
        expires_delta = timedelta(hours=2)

        token = jwt_service.create_access_token(data, expires_delta)

        # Decode to verify expiry
        import jwt

        from src.config.settings import settings

        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[
                settings.JWT_ALGORITHM])

        # Check that expiry is approximately 2 hours from now
        exp_time = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        # Use the same timezone-aware approach as the JWT service
        expected_time = datetime.now(timezone.utc) + expires_delta
        time_diff = abs((exp_time - expected_time).total_seconds())

        assert time_diff < 300  # Within 5 minutes tolerance

    def test_verify_token_success(self, jwt_service):
        """Test successful token verification."""
        data = {"sub": "test-user-123", "email": "test@example.com"}
        token = jwt_service.create_access_token(data)

        payload = jwt_service.verify_token(token)

        assert payload["sub"] == "test-user-123"
        assert payload["email"] == "test@example.com"
        assert "exp" in payload
        assert "iat" in payload
        assert payload["type"] == TokenType.ACCESS

    def test_verify_token_expired(self, jwt_service):
        """Test token verification with expired token."""
        # Create an already expired token
        data = {
            "sub": "test-user-123",
            "exp": int(
                datetime.utcnow().timestamp()) -
            3600}

        import jwt

        from src.config.settings import settings

        expired_token = jwt.encode(
            data,
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM)

        with pytest.raises(TokenExpiredError, match="Token has expired"):
            jwt_service.verify_token(expired_token)

    def test_verify_token_invalid(self, jwt_service):
        """Test token verification with invalid token."""
        invalid_token = "invalid.jwt.token"

        with pytest.raises(TokenInvalidError, match="Invalid token"):
            jwt_service.verify_token(invalid_token)


@pytest.mark.unit
class TestAuthService:
    """Test main authentication service."""

    @pytest.fixture
    def auth_service(self):
        """Create auth service for testing."""
        with (
            patch("src.services.auth_service.FirebaseAuthService"),
            patch("src.services.auth_service.JWTAuthService"),
        ):
            service = AuthService()
            yield service

    @pytest.mark.asyncio
    async def test_authenticate_firebase_success(
            self, auth_service, authenticated_user) -> None:
        """Test successful Firebase authentication."""
        request = LoginRequest(
            token="valid-firebase-token",
            provider=AuthProvider.FIREBASE)

        # Mock the Firebase validation
        auth_service.firebase_auth.validate_token = AsyncMock(
            return_value=authenticated_user)

        result = await auth_service.authenticate(request, "127.0.0.1", "test-agent")

        assert isinstance(result, LoginResponse)
        assert result.access_token == "firebase-token"
        assert result.token_type == "bearer"
        assert result.user.uid == authenticated_user.uid
        assert result.user.email == authenticated_user.email

    @pytest.mark.asyncio
    async def test_authenticate_api_key_success(self, auth_service) -> None:
        """Test successful API key authentication."""
        request = LoginRequest(
            token="valid-api-key-token",
            provider=AuthProvider.API_KEY)

        # Mock JWT validation
        auth_service.jwt_auth.verify_token = Mock(
            return_value={
                "uid": "api-user-123",
                "email": "api@example.com",
                "name": "API User"})

        result = await auth_service.authenticate(request, "127.0.0.1", "test-agent")

        assert isinstance(result, LoginResponse)
        assert result.access_token == "valid-api-key-token"
        assert result.user.uid == "api-user-123"
        assert result.user.provider == AuthProvider.API_KEY

    @pytest.mark.asyncio
    async def test_authenticate_unsupported_provider(self, auth_service) -> None:
        """Test authentication with unsupported provider."""
        request = LoginRequest(
            token="some-token",
            provider=AuthProvider.FIREBASE)

        # Mock the service to raise an error for unsupported provider
        with patch.object(auth_service, 'firebase_auth') as mock_firebase:
            mock_firebase.validate_token.side_effect = AuthenticationError("Unsupported provider")
            
            with pytest.raises(AuthenticationError, match="Unsupported provider"):
                await auth_service.authenticate(request, "127.0.0.1", "test-agent")

    @pytest.mark.asyncio
    async def test_validate_request_token_firebase(
            self, auth_service, authenticated_user) -> None:
        """Test token validation for Firebase token."""
        request = TokenValidationRequest(
            token="valid-firebase-token",
            token_type=TokenType.ACCESS)

        auth_service.firebase_auth.validate_token = AsyncMock(
            return_value=authenticated_user)

        result = await auth_service.validate_request_token(request, "127.0.0.1", "test-agent")

        assert result.valid is True
        assert result.user == authenticated_user
        assert result.error is None

    @pytest.mark.asyncio
    async def test_validate_request_token_jwt_fallback(self, auth_service) -> None:
        """Test token validation falling back to JWT after Firebase fails."""
        request = TokenValidationRequest(
            token="valid-jwt-token",
            token_type=TokenType.ACCESS)

        # Firebase validation fails
        auth_service.firebase_auth.validate_token = AsyncMock(
            side_effect=TokenInvalidError("Firebase validation failed")
        )

        # JWT validation succeeds
        auth_service.jwt_auth.verify_token = Mock(
            return_value={
                "uid": "jwt-user-123",
                "email": "jwt@example.com",
                "name": "JWT User",
                "iat": int(datetime.utcnow().timestamp()),
                "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            }
        )

        result = await auth_service.validate_request_token(request, "127.0.0.1", "test-agent")

        assert result.valid is True
        assert result.user.uid == "jwt-user-123"
        assert result.user.provider == AuthProvider.API_KEY

    @pytest.mark.asyncio
    async def test_validate_request_token_invalid(self, auth_service) -> None:
        """Test token validation with completely invalid token."""
        request = TokenValidationRequest(
            token="invalid-token",
            token_type=TokenType.ACCESS)

        # Both Firebase and JWT validation fail
        auth_service.firebase_auth.validate_token = AsyncMock(
            side_effect=TokenInvalidError("Firebase validation failed")
        )
        auth_service.jwt_auth.verify_token = Mock(
            side_effect=TokenInvalidError("JWT validation failed")
        )

        result = await auth_service.validate_request_token(request, "127.0.0.1", "test-agent")

        assert result.valid is False
        assert result.user is None
        assert "JWT validation failed" in result.error

    @pytest.mark.asyncio
    async def test_logout_success(self, auth_service) -> None:
        """Test successful logout."""
        user_id = "test-user-123"

        auth_service.firebase_auth.revoke_tokens = AsyncMock(return_value=True)

        result = await auth_service.logout(user_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_logout_failure(self, auth_service) -> None:
        """Test logout failure."""
        user_id = "test-user-123"

        auth_service.firebase_auth.revoke_tokens = AsyncMock(
            return_value=False)

        result = await auth_service.logout(user_id)
        assert result is False

    def test_health_check(self, auth_service):
        """Test auth service health check."""
        auth_service.firebase_auth.health_check = Mock(return_value=True)

        result = auth_service.health_check()
        assert result is True
