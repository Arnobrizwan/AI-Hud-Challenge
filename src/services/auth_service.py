"""
Firebase Authentication service with JWT validation and user management.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import firebase_admin
import jwt
from firebase_admin import auth, credentials
from firebase_admin.auth import ExpiredIdTokenError, InvalidIdTokenError, RevokedIdTokenError

from src.config.settings import settings
from src.models.auth import (
    AuthenticatedUser,
    AuthProvider,
    LoginRequest,
    LoginResponse,
    TokenType,
    TokenValidationRequest,
    TokenValidationResponse,
    UserClaims,
)
from src.services.circuit_breaker import CircuitBreakerConfig, get_circuit_breaker
from src.utils.logging import get_logger, log_security_event
from src.utils.metrics import metrics_collector

logger = get_logger(__name__)


class AuthenticationError(Exception):
    """Authentication related errors."""

    pass


class AuthorizationError(Exception):
    """Authorization related errors."""

    pass


class TokenExpiredError(AuthenticationError):
    """Token has expired."""

    pass


class TokenInvalidError(AuthenticationError):
    """Token is invalid."""

    pass


class FirebaseAuthService:
    """Firebase Authentication service."""

    def __init__(self):
        self.app: Optional[firebase_admin.App] = None
        self._circuit_breaker = get_circuit_breaker(
            "firebase_auth",
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30.0,
                expected_exception=(
                    InvalidIdTokenError,
                    ExpiredIdTokenError,
                    RevokedIdTokenError,
                    Exception,
                ),
            ),
        )
        self._initialize_firebase()

    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK."""
        try:
            if not firebase_admin._apps:
                if settings.FIREBASE_CREDENTIALS_PATH:
                    # Load from file
                    cred = credentials.Certificate(
                        settings.FIREBASE_CREDENTIALS_PATH)
                elif settings.FIREBASE_CREDENTIALS_JSON:
                    # Load from JSON string
                    cred_dict = json.loads(settings.FIREBASE_CREDENTIALS_JSON)
                    cred = credentials.Certificate(cred_dict)
        else:
                    # Use default credentials (for GCP environments)
                    cred = credentials.ApplicationDefault()

                self.app = firebase_admin.initialize_app(
                    cred, {"projectId": settings.FIREBASE_PROJECT_ID}
                )

                logger.info("Firebase Admin SDK initialized successfully")
            else:
                self.app = firebase_admin.get_app()

        except Exception as e:
            logger.error(
                "Failed to initialize Firebase Admin SDK",
                error=str(e))
            raise AuthenticationError(
                f"Firebase initialization failed: {str(e)}")

    async def validate_token(
        self, token: str, client_ip: str = None, user_agent: str = None
    ) -> AuthenticatedUser:
        """Validate Firebase ID token and return user information."""
        start_time = time.time()

        try:
            async with self._circuit_breaker:
                # Verify the token with Firebase
                decoded_token = auth.verify_id_token(
                    token, check_revoked=True, app=self.app)

                # Extract user information
                user_claims = UserClaims(
                    uid=decoded_token.get("uid"),
                    email=decoded_token.get("email"),
                    email_verified=decoded_token.get("email_verified", False),
                    name=decoded_token.get("name"),
                    picture=decoded_token.get("picture"),
                    provider=AuthProvider.FIREBASE,
                )

                # Create authenticated user
                authenticated_user = AuthenticatedUser(
                    **user_claims.dict(),
                    token_type=TokenType.ACCESS,
                    issued_at=datetime.fromtimestamp(
                        decoded_token.get(
                            "iat",
                            time.time())),
                    expires_at=datetime.fromtimestamp(
                        decoded_token.get(
                            "exp",
                            time.time())),
                    client_ip=client_ip,
                    user_agent=user_agent,
                )

                # Record successful authentication
                duration = time.time() - start_time
                metrics_collector.record_auth_attempt(
                    "firebase", True, duration)

                logger.info(
                    "Token validation successful",
                    user_id=user_claims.uid,
                    email=user_claims.email,
                    client_ip=client_ip,
                    duration=duration,
                )

                return authenticated_user

        except ExpiredIdTokenError as e:
            duration = time.time() - start_time
            metrics_collector.record_auth_attempt("firebase", False, duration)
            log_security_event(
                logger,
                "token_expired",
                client_ip=client_ip,
                details={
                    "error": str(e)})
            raise TokenExpiredError("Token has expired")

        except (InvalidIdTokenError, RevokedIdTokenError) as e:
            duration = time.time() - start_time
            metrics_collector.record_auth_attempt("firebase", False, duration)
            log_security_event(
                logger,
                "invalid_token",
                client_ip=client_ip,
                details={
                    "error": str(e)})
            raise TokenInvalidError(f"Invalid token: {str(e)}")

        except Exception as e:
            duration = time.time() - start_time
            metrics_collector.record_auth_attempt("firebase", False, duration)
            logger.error(
                "Token validation failed",
                error=str(e),
                client_ip=client_ip,
                duration=duration)
            raise AuthenticationError(f"Authentication failed: {str(e)}")

    async def get_user_by_uid(self, uid: str) -> Optional[UserClaims]:
        """Get user information by UID."""
        try:
            async with self._circuit_breaker:
                user_record = auth.get_user(uid, app=self.app)

                return UserClaims(
                    uid=user_record.uid,
                    email=user_record.email,
                    email_verified=user_record.email_verified,
                    name=user_record.display_name,
                    picture=user_record.photo_url,
                    provider=AuthProvider.FIREBASE,
                )

        except auth.UserNotFoundError:
            logger.warning(f"User not found: {uid}")
            return None

        except Exception as e:
            logger.error(f"Failed to get user {uid}", error=str(e))
            raise AuthenticationError(f"Failed to get user: {str(e)}")

    async def revoke_tokens(self, uid: str) -> bool:
        """Revoke all tokens for a user."""
        try:
            async with self._circuit_breaker:
                auth.revoke_refresh_tokens(uid, app=self.app)
                logger.info(f"Revoked all tokens for user: {uid}")
                return True

        except Exception as e:
            logger.error(f"Failed to revoke tokens for {uid}", error=str(e))
            return False

    async def set_custom_user_claims(
            self, uid: str, claims: Dict[str, Any]) -> bool:
        """Set custom claims for a user."""
        try:
            async with self._circuit_breaker:
                auth.set_custom_user_claims(uid, claims, app=self.app)
                logger.info(
                    f"Set custom claims for user: {uid}",
                    claims=claims)
                return True

        except Exception as e:
            logger.error(
                f"Failed to set custom claims for {uid}",
                error=str(e))
            return False

    async def create_custom_token(
            self, uid: str, claims: Dict[str, Any] = None) -> str:
        """Create a custom token for a user."""
        try:
            async with self._circuit_breaker:
                custom_token = auth.create_custom_token(
                    uid, developer_claims=claims, app=self.app)
                return custom_token.decode("utf-8")

        except Exception as e:
            logger.error(
                f"Failed to create custom token for {uid}",
                error=str(e))
            raise AuthenticationError(
                f"Failed to create custom token: {str(e)}")

    def health_check(self) -> bool:
        """Check if Firebase Auth service is healthy."""
        try:
            # Simple check to see if we can access the service
            return self.app is not None and bool(firebase_admin._apps)
        except Exception:
            return False


class JWTAuthService:
    """JWT token service for API keys and internal tokens."""

    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.expire_minutes = settings.JWT_EXPIRE_MINUTES

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.expire_minutes)

        to_encode.update({"exp": expire,
                          "iat": datetime.utcnow(),
                          "type": TokenType.ACCESS})

        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Dict[str, Any]:
    """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(
                token, self.secret_key, algorithms=[
                    self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise TokenInvalidError(f"Invalid token: {str(e)}")


class AuthService:
    """Main authentication service combining Firebase and JWT."""

    def __init__(self):
        self.firebase_auth = FirebaseAuthService()
        self.jwt_auth = JWTAuthService()

    async def authenticate(
            self,
            request: LoginRequest,
            client_ip: str = None,
            user_agent: str = None) -> LoginResponse:
        """Authenticate user and return tokens."""
        if request.provider == AuthProvider.FIREBASE:
            user = await self.firebase_auth.validate_token(request.token, client_ip, user_agent)

            # Create response
            return LoginResponse(
                access_token=request.token,  # Use Firebase token directly
                token_type="bearer",
                expires_in=int(
                    (user.expires_at - user.issued_at).total_seconds()),
                user=UserClaims(**user.dict()),
            )

        elif request.provider == AuthProvider.API_KEY:
            # Handle API key authentication
            payload = self.jwt_auth.verify_token(request.token)

            user_claims = UserClaims(
                uid=payload.get("uid"),
                email=payload.get("email"),
                name=payload.get("name"),
                provider=AuthProvider.API_KEY,
            )

            return LoginResponse(
                access_token=request.token,
                token_type="bearer",
                expires_in=self.jwt_auth.expire_minutes * 60,
                user=user_claims,
            )

        else:
            raise AuthenticationError(
                f"Unsupported provider: {request.provider}")

    async def validate_request_token(
            self,
            request: TokenValidationRequest,
            client_ip: str = None,
            user_agent: str = None) -> TokenValidationResponse:
        """Validate token from request."""
        try:
            if request.token_type == TokenType.ACCESS:
                # Try Firebase first, then JWT
                try:
                    user = await self.firebase_auth.validate_token(
                        request.token, client_ip, user_agent
                    )
                    return TokenValidationResponse(valid=True, user=user)
                except (TokenExpiredError, TokenInvalidError):
                    # Try JWT validation
                    payload = self.jwt_auth.verify_token(request.token)

                    user = AuthenticatedUser(
                        uid=payload.get("uid"),
                        email=payload.get("email"),
                        name=payload.get("name"),
                        provider=AuthProvider.API_KEY,
                        token_type=TokenType.ACCESS,
                        issued_at=datetime.fromtimestamp(payload.get("iat")),
                        expires_at=datetime.fromtimestamp(payload.get("exp")),
                        client_ip=client_ip,
                        user_agent=user_agent,
                    )

                    return TokenValidationResponse(valid=True, user=user)

            return TokenValidationResponse(
                valid=False, error="Unsupported token type")

        except (TokenExpiredError, TokenInvalidError) as e:
            return TokenValidationResponse(valid=False, error=str(e))

        except Exception as e:
            logger.error("Token validation error", error=str(e))
            return TokenValidationResponse(
                valid=False, error="Authentication failed")

    async def logout(self, user_id: str) -> bool:
        """Logout user by revoking tokens."""
        return await self.firebase_auth.revoke_tokens(user_id)

    def health_check(self) -> bool:
        """Check authentication service health."""
        return self.firebase_auth.health_check()


# Global authentication service instance
auth_service = AuthService()
