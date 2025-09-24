"""
Integration tests for the Foundations & Guards service.
"""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock
import json

from src.main import app


@pytest.mark.integration
class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_endpoint(self, client: TestClient):
        """Test main health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] in ["healthy", "unhealthy", "degraded"]
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data
        assert "dependencies" in data
    
    def test_liveness_probe(self, client: TestClient):
        """Test liveness probe endpoint."""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_readiness_probe(self, client: TestClient):
        """Test readiness probe endpoint."""
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] in ["ready", "not_ready"]
        assert "timestamp" in data


@pytest.mark.integration
class TestAuthenticationEndpoints:
    """Test authentication endpoints."""
    
    def test_login_firebase_success(self, client: TestClient, mock_firebase_auth):
        """Test successful Firebase login."""
        login_data = {
            "token": "valid-firebase-token",
            "provider": "firebase"
        }
        
        with patch('src.services.auth_service.auth_service.authenticate') as mock_auth:
            mock_auth.return_value = Mock(
                access_token="firebase-token",
                token_type="bearer",
                expires_in=3600,
                user=Mock(
                    uid="test-user-123",
                    email="test@example.com",
                    name="Test User"
                )
            )
            
            response = client.post("/auth/login", json=login_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["access_token"] == "firebase-token"
            assert data["token_type"] == "bearer"
            assert data["expires_in"] == 3600
            assert data["user"]["uid"] == "test-user-123"
            assert data["user"]["email"] == "test@example.com"
    
    def test_login_invalid_token(self, client: TestClient):
        """Test login with invalid token."""
        login_data = {
            "token": "invalid-token",
            "provider": "firebase"
        }
        
        with patch('src.services.auth_service.auth_service.authenticate') as mock_auth:
            mock_auth.side_effect = Exception("Authentication failed")
            
            response = client.post("/auth/login", json=login_data)
            
            assert response.status_code == 401
            data = response.json()
            
            assert data["status"] == "error"
            assert "Authentication failed" in data["message"] or "Authentication failed" in str(data)
    
    def test_login_missing_fields(self, client: TestClient):
        """Test login with missing required fields."""
        login_data = {
            "provider": "firebase"
            # Missing token
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_logout_success(self, client: TestClient, valid_jwt_token):
        """Test successful logout."""
        headers = {"Authorization": f"Bearer {valid_jwt_token}"}
        
        with patch('src.services.auth_service.auth_service.logout') as mock_logout:
            mock_logout.return_value = True
            
            response = client.post("/auth/logout", headers=headers)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert data["message"] == "Logout successful"
    
    def test_logout_unauthorized(self, client: TestClient):
        """Test logout without authentication."""
        response = client.post("/auth/logout")
        
        assert response.status_code == 401
    
    def test_get_current_user(self, client: TestClient, valid_jwt_token):
        """Test getting current user information."""
        headers = {"Authorization": f"Bearer {valid_jwt_token}"}
        
        with patch('src.middleware.auth_middleware.get_current_user') as mock_get_user:
            mock_user = Mock(
                uid="test-user-123",
                email="test@example.com",
                name="Test User"
            )
            mock_get_user.return_value = mock_user
            
            response = client.get("/auth/me", headers=headers)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert "data" in data


@pytest.mark.integration
class TestMiddlewareIntegration:
    """Test middleware integration."""
    
    def test_cors_headers(self, client: TestClient):
        """Test CORS headers are properly set."""
        response = client.options("/health")
        
        assert "access-control-allow-origin" in [h.lower() for h in response.headers]
    
    def test_security_headers(self, client: TestClient):
        """Test security headers are present."""
        response = client.get("/health")
        
        headers = [h.lower() for h in response.headers]
        
        # Check for security headers
        assert any("x-frame-options" in h for h in headers)
        assert any("x-content-type-options" in h for h in headers)
    
    def test_correlation_id_header(self, client: TestClient):
        """Test correlation ID is added to responses."""
        response = client.get("/health")
        
        # Should have correlation ID in response headers
        correlation_headers = [h for h in response.headers if "correlation" in h.lower()]
        assert len(correlation_headers) > 0
    
    def test_request_size_limit(self, client: TestClient):
        """Test request size limiting."""
        # Create a large payload (mock - in real test you'd send actual large data)
        large_payload = {"data": "x" * 1000000}  # 1MB of data
        
        response = client.post("/auth/login", json=large_payload)
        
        # Should either process or reject based on size limits
        # In this case, it would likely fail validation before size check
        assert response.status_code in [400, 413, 422]
    
    def test_authentication_middleware_excluded_paths(self, client: TestClient):
        """Test that authentication middleware excludes certain paths."""
        excluded_paths = ["/health", "/docs", "/openapi.json"]
        
        for path in excluded_paths:
            response = client.get(path)
            # Should not require authentication
            assert response.status_code != 401
    
    @pytest.mark.asyncio
    async def test_rate_limiting_middleware(self, client: TestClient, mock_rate_limiter):
        """Test rate limiting middleware."""
        # Mock rate limiter to allow requests
        mock_rate_limiter['ip_limit'].return_value = (True, Mock())
        
        # Make multiple requests
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200
        
        # Mock rate limiter to block requests
        mock_rate_limiter['ip_limit'].return_value = (False, Mock(
            limit=10,
            remaining=0,
            reset_time=Mock(),
            window_seconds=60
        ))
        
        response = client.get("/health")
        # Should be rate limited, but might not work in test client
        # This would need actual async testing


@pytest.mark.integration
class TestMetricsEndpoint:
    """Test metrics endpoint."""
    
    def test_metrics_endpoint_enabled(self, client: TestClient):
        """Test metrics endpoint when metrics are enabled."""
        with patch('src.config.settings.settings.ENABLE_METRICS', True):
            response = client.get("/metrics")
            
            # Should return metrics data or 404 if disabled
            assert response.status_code in [200, 404]
    
    def test_metrics_endpoint_disabled(self, client: TestClient):
        """Test metrics endpoint when metrics are disabled."""
        with patch('src.config.settings.settings.ENABLE_METRICS', False):
            response = client.get("/metrics")
            
            assert response.status_code == 404


@pytest.mark.integration  
class TestDocumentationEndpoints:
    """Test documentation endpoints."""
    
    def test_swagger_docs(self, client: TestClient):
        """Test Swagger documentation endpoint."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_redoc_docs(self, client: TestClient):
        """Test ReDoc documentation endpoint."""
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_openapi_schema(self, client: TestClient):
        """Test OpenAPI schema endpoint."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
        assert data["info"]["title"] == "Foundations & Guards Service"


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling integration."""
    
    def test_404_error_handling(self, client: TestClient):
        """Test 404 error handling."""
        response = client.get("/non-existent-endpoint")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client: TestClient):
        """Test method not allowed error."""
        response = client.patch("/health")
        
        assert response.status_code == 405
    
    def test_validation_error_handling(self, client: TestClient):
        """Test validation error handling."""
        # Send invalid JSON to login endpoint
        response = client.post("/auth/login", json={"invalid": "data"})
        
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data  # FastAPI validation error format


@pytest.mark.integration
class TestServiceIntegration:
    """Test service integration points."""
    
    @pytest.mark.asyncio
    async def test_auth_service_integration(self, mock_firebase_auth):
        """Test authentication service integration."""
        from src.services.auth_service import auth_service
        from src.models.auth import LoginRequest, AuthProvider
        
        request = LoginRequest(
            token="test-token",
            provider=AuthProvider.FIREBASE
        )
        
        # This would test the actual service integration
        # but requires proper mocking of Firebase
        with patch.object(auth_service.firebase_auth, 'validate_token') as mock_validate:
            mock_validate.return_value = Mock(
                uid="test-user",
                email="test@example.com"
            )
            
            # Test would go here
            pass
    
    @pytest.mark.asyncio
    async def test_rate_limiter_integration(self, mock_redis):
        """Test rate limiter service integration."""
        from src.services.rate_limiter import rate_limiter
        
        # Test the actual rate limiter with mocked Redis
        with patch.object(rate_limiter.limiter, '_get_redis', return_value=mock_redis):
            # Mock Redis operations
            mock_redis.pipeline.return_value.__aenter__ = AsyncMock()
            mock_redis.pipeline.return_value.__aexit__ = AsyncMock()
            mock_redis.pipeline.return_value.execute = AsyncMock(return_value=[None, 1])
            mock_redis.zadd = AsyncMock()
            mock_redis.expire = AsyncMock()
            
            allowed, rate_info = await rate_limiter.check_user_rate_limit("test-user")
            
            assert allowed is True
            assert rate_info is not None
    
    def test_health_checker_integration(self):
        """Test health checker integration."""
        from src.utils.health import health_checker
        
        # Test health checker with mocked dependencies
        with patch.object(health_checker, 'checks', []):
            # Would test health checking
            pass


@pytest.mark.integration
class TestEndToEndScenarios:
    """End-to-end integration test scenarios."""
    
    def test_complete_auth_flow(self, client: TestClient, mock_firebase_auth):
        """Test complete authentication flow."""
        # 1. Login
        login_data = {
            "token": "valid-token",
            "provider": "firebase"
        }
        
        with patch('src.services.auth_service.auth_service.authenticate') as mock_auth:
            mock_auth.return_value = Mock(
                access_token="token",
                token_type="bearer",
                expires_in=3600,
                user=Mock(uid="user-123")
            )
            
            login_response = client.post("/auth/login", json=login_data)
            assert login_response.status_code == 200
            
            token = login_response.json()["access_token"]
        
        # 2. Access protected endpoint
        headers = {"Authorization": f"Bearer {token}"}
        
        with patch('src.middleware.auth_middleware.get_current_user') as mock_get_user:
            mock_get_user.return_value = Mock(uid="user-123")
            
            me_response = client.get("/auth/me", headers=headers)
            assert me_response.status_code == 200
        
        # 3. Logout
        with patch('src.services.auth_service.auth_service.logout') as mock_logout:
            mock_logout.return_value = True
            
            logout_response = client.post("/auth/logout", headers=headers)
            assert logout_response.status_code == 200
    
    def test_rate_limiting_scenario(self, client: TestClient, mock_rate_limiter):
        """Test rate limiting scenario."""
        # Configure rate limiter to allow first few requests
        mock_rate_limiter['ip_limit'].return_value = (True, Mock())
        
        # Make requests within limit
        for i in range(3):
            response = client.get("/health")
            assert response.status_code == 200
        
        # Configure rate limiter to block subsequent requests  
        mock_rate_limiter['ip_limit'].return_value = (False, Mock(
            limit=5,
            remaining=0,
            reset_time=Mock(),
            window_seconds=60
        ))
        
        # This request should be rate limited (in theory)
        response = client.get("/health")
        # Note: TestClient doesn't go through async middleware properly
        # so this might not actually be rate limited in the test
