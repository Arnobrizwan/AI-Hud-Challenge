"""
pytest configuration and fixtures for testing the Foundations & Guards service.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, Generator
from unittest.mock import Mock, patch

import fakeredis
import pytest
from fastapi.testclient import TestClient

from src.config.settings import settings
from src.main import app
from src.models.auth import AuthenticatedUser, AuthProvider, TokenType, UserClaims

# from src.services.auth_service import auth_service  # Unused import
from src.services.rate_limiter import rate_limiter


# Test settings override
@pytest.fixture(scope="session", autouse=True)
def override_settings() -> None:
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
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
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
def mock_redis() -> Generator[fakeredis.FakeRedis, None, None]:
    """Create a mock Redis client using fakeredis."""
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    with patch("aioredis.create_redis_pool") as mock_create_pool:
        mock_create_pool.return_value = fake_redis
        yield fake_redis


# Mock Firebase Auth fixture
@pytest.fixture(scope="function")
def mock_firebase_auth() -> Generator[Mock, None, None]:
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
def authenticated_user(test_user: Any) -> AuthenticatedUser:
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
    return (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiJ0ZXN0LXVzZXItMTIzIiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIiwi"
        "ZXhwIjoxNzAzMTY5NjAwfQ."
        "test-signature"
    )


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
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiJ0ZXN0LXVzZXItMTIzIiwiZXhwIjoxNTc3ODM2ODAwfQ."
        "expired-signature"
    )


# Mock rate limiter fixture
@pytest.fixture(scope="function")
async def mock_rate_limiter() -> AsyncGenerator[Dict[str, Any], None]:
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
    return {"status": "no_database"}


# Mock external services
@pytest.fixture(scope="function")
def mock_external_services() -> Generator[Mock, None, None]:
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
def sample_request_data() -> Dict[str, Any]:
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
def sample_filter_data() -> Dict[str, Any]:
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
def performance_test_config() -> Dict[str, Any]:
    """Configuration for performance testing."""
    return {"concurrent_requests": 100, "total_requests": 1000, "timeout": 30.0, "rate_limit": 1000}


# Security testing fixtures
@pytest.fixture(scope="function")
def security_test_payloads() -> Dict[str, Any]:
    """Security test payloads for testing."""
    return {
        "sql_injection": ["'; DROP TABLE users; --", "1' OR '1'='1"],
        "xss": ["<script>alert('xss')</script>", "javascript:alert('xss')"],
        "path_traversal": ["../../../etc/passwd", "..\\windows\\system32"],
        "command_injection": ["; ls -la", "| whoami", "&& cat /etc/passwd"],
    }


# Cleanup fixtures
@pytest.fixture(scope="function", autouse=True)
async def cleanup_redis(mock_redis: Any) -> AsyncGenerator[None, None]:
    """Clean up Redis after each test."""
    yield
    mock_redis.flushdb()


@pytest.fixture(scope="function", autouse=True)
def cleanup_circuit_breakers() -> Generator[None, None, None]:
    """Clean up circuit breakers after each test."""
    yield
    from src.services.circuit_breaker import circuit_breaker_registry

    circuit_breaker_registry.reset_all()


@pytest.fixture(scope="function", autouse=True)
def setup_health_checker() -> Generator[None, None, None]:
    """Initialize health checker for tests."""
    import src.main
    from src.monitoring.metrics import (
        HealthChecker,
        RankingMetricsCollector,
        SystemMetricsCollector,
    )

    # Initialize health checker if not already initialized
    if src.main.health_checker is None:
        ranking_collector = RankingMetricsCollector()
        system_collector = SystemMetricsCollector()
        src.main.health_checker = HealthChecker(ranking_collector, system_collector)

    yield


@pytest.fixture(scope="function", autouse=True)
def cleanup_metrics() -> Generator[None, None, None]:
    """Clean up metrics after each test."""
    yield
    # Reset any global metrics state if needed


# Mock environment variables
@pytest.fixture(scope="function")
def mock_env_vars() -> Generator[Dict[str, str], None, None]:
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
    async def wait_for_condition(
        condition_func: Any, timeout: float = 5.0, interval: float = 0.1
    ) -> bool:
        """Wait for a condition to be true."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(interval)

        return False

    @staticmethod
    async def assert_eventually(
        assertion_func: Any, timeout: float = 5.0, interval: float = 0.1
    ) -> None:
        """Assert that a condition becomes true within timeout."""
        success = await AsyncTestCase.wait_for_condition(assertion_func, timeout, interval)
        if not success:
            raise AssertionError(f"Condition not met within {timeout} seconds")


# Test markers
def pytest_configure(config: Any) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "security: mark test as a security test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
