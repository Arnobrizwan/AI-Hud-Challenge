"""
pytest configuration and fixtures for testing the Foundations & Guards service.
"""

import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, patch

import fakeredis
import pytest
import pytest_asyncio
import redis
from fastapi.testclient import TestClient

from src.config.settings import settings
from src.main import app
from src.models.auth import AuthenticatedUser, AuthProvider, TokenType, UserClaims
from src.services.auth_service import auth_service
from src.services.rate_limiter import rate_limiter


# Test settings override
@pytest.fixture(scope="session", autouse=True)
def override_settings():
    """Override settings for testing."""
    settings.ENVIRONMENT = "testing"
    settings.DEBUG = True
    settings.LOG_LEVEL = "DEBUG"
    settings.REDIS_URL = "redis://localhost:6379/15"  # Use different DB for tests
    settings.FIREBASE_PROJECT_ID = "test-project"
    settings.SECRET_KEY = "test-secret-key-for-testing-only"
    settings.ENABLE_METRICS = False
    settings.HEALTH_CHECK_DEPENDENCIES = False


# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test client fixture
@pytest.fixture(scope="function")
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


# Mock Redis fixture
@pytest.fixture(scope="function")
def mock_redis():
    """Create a mock Redis client using fakeredis."""
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    with patch("aioredis.from_url") as mock_from_url:
        mock_from_url.return_value = fake_redis
        yield fake_redis


# Mock Firebase Auth fixture
@pytest.fixture(scope="function")
def mock_firebase_auth():
    """Mock Firebase authentication."""
    with patch("firebase_admin.auth.verify_id_token") as mock_verify:
        mock_verify.return_value = {
            "uid": "test-user-123",
            "email": "test@example.com",
            "email_verified": True,
            "name": "Test User",
            "picture": "https://example.com/avatar.jpg",
            "iat": int(datetime.utcnow().timestamp()),
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
        }
        yield mock_verify


# Test user fixture
@pytest.fixture(scope="function")
def test_user() -> UserClaims:
    """Create a test user."""
    return UserClaims(
        uid="test-user-123",
        email="test@example.com",
        email_verified=True,
        name="Test User",
        picture="https://example.com/avatar.jpg",
        provider=AuthProvider.FIREBASE,
    )


# Authenticated test user fixture
@pytest.fixture(scope="function")
def authenticated_user(test_user) -> AuthenticatedUser:
    """Create an authenticated test user."""
    return AuthenticatedUser(
        **test_user.dict(),
        token_type=TokenType.ACCESS,
        issued_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(hours=1),
        client_ip="127.0.0.1",
        user_agent="pytest",
    )


# Valid JWT token fixture
@pytest.fixture(scope="function")
def valid_jwt_token() -> str:
    """Create a valid JWT token for testing."""
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0LXVzZXItMTIzIiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzAzMTY5NjAwfQ.test-signature"


# Invalid JWT token fixture
@pytest.fixture(scope="function")
def invalid_jwt_token() -> str:
    """Create an invalid JWT token for testing."""
    return "invalid.jwt.token"


# Expired JWT token fixture
@pytest.fixture(scope="function")
def expired_jwt_token() -> str:
    """Create an expired JWT token for testing."""
    return (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0LXVzZXItMTIzIiwiZXhwIjoxNTc3ODM2ODAwfQ.expired-signature"
    )


# Mock rate limiter fixture
@pytest.fixture(scope="function")
async def mock_rate_limiter() -> Dict[str, Any]:
    """Mock rate limiter for testing."""
    with (
        patch.object(rate_limiter, "check_user_rate_limit") as mock_user_limit,
        patch.object(rate_limiter, "check_ip_rate_limit") as mock_ip_limit,
    ):

        # Default to allowing requests
        mock_user_limit.return_value = (True, Mock())
        mock_ip_limit.return_value = (True, Mock())

        yield {"user_limit": mock_user_limit, "ip_limit": mock_ip_limit}


# Database fixtures (if using database)
@pytest.fixture(scope="function")
async def db_session() -> Dict[str, Any]:
    """Create a database session for testing."""
    # This would be implemented if using a database like PostgreSQL
    # For now, we're using Redis only
    pass


# Mock external services
@pytest.fixture(scope="function")
def mock_external_services():
    """Mock external services for testing."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}

        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

        yield mock_client


# Test data fixtures
@pytest.fixture(scope="function")
def sample_request_data():
    """Sample request data for testing."""
    return {
        "name": "Test News Source",
        "url": "https://example.com/news",
        "description": "Test news source",
        "category": "technology",
        "language": "en",
        "country": "US",
        "enabled": True,
    }


@pytest.fixture(scope="function")
def sample_filter_data():
    """Sample filter data for testing."""
    return {
        "keywords": ["technology", "ai"],
        "exclude_keywords": ["politics"],
        "categories": ["tech", "science"],
        "min_score": 0.8,
        "languages": ["en"],
        "date_from": "2023-01-01",
        "date_to": "2023-12-31",
    }


# Performance testing fixtures
@pytest.fixture(scope="function")
def performance_test_config():
    """Configuration for performance testing."""
    return {"concurrent_requests": 100, "total_requests": 1000, "timeout": 30.0, "rate_limit": 1000}


# Security testing fixtures
@pytest.fixture(scope="function")
def security_test_payloads():
    """Security test payloads for testing."""
    return {
        "sql_injection": ["'; DROP TABLE users; --", "1' OR '1'='1"],
        "xss": ["<script>alert('xss')</script>", "javascript:alert('xss')"],
        "path_traversal": ["../../../etc/passwd", "..\\windows\\system32"],
        "command_injection": ["; ls -la", "| whoami", "&& cat /etc/passwd"],
    }


# Cleanup fixtures
@pytest.fixture(scope="function", autouse=True)
async def cleanup_redis(mock_redis) -> Dict[str, Any]:
    """Clean up Redis after each test."""
    yield
    mock_redis.flushdb()


@pytest.fixture(scope="function", autouse=True)
def cleanup_metrics():
    """Clean up metrics after each test."""
    yield
    # Reset any global metrics state if needed


# Mock environment variables
@pytest.fixture(scope="function")
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "ENVIRONMENT": "testing",
        "SECRET_KEY": "test-secret-key",
        "FIREBASE_PROJECT_ID": "test-project",
        "REDIS_URL": "redis://localhost:6379/15",
        "LOG_LEVEL": "DEBUG",
    }

    with patch.dict("os.environ", env_vars):
        yield env_vars


# Async test utilities
class AsyncTestCase:
    """Base class for async test utilities."""

    @staticmethod
    async def wait_for_condition(condition_func, timeout=5.0, interval=0.1) -> Dict[str, Any]:
    """Wait for a condition to be true."""
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(interval)

        return False

    @staticmethod
    async def assert_eventually(assertion_func, timeout=5.0, interval=0.1) -> Dict[str, Any]:
    """Assert that a condition becomes true within timeout."""
        success = await AsyncTestCase.wait_for_condition(assertion_func, timeout, interval)
        if not success:
            raise AssertionError(f"Condition not met within {timeout} seconds")


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "security: mark test as a security test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
