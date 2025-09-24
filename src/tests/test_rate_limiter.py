"""
Unit tests for the rate limiting service.
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import time

from src.services.rate_limiter import (
    SlidingWindowRateLimiter, DistributedRateLimiter, 
    RateLimitExceeded, rate_limiter
)
from src.models.common import RateLimitInfo


@pytest.mark.unit
class TestSlidingWindowRateLimiter:
    """Test sliding window rate limiter."""
    
    @pytest_asyncio.fixture
    async def limiter(self, mock_redis):
        """Create rate limiter for testing."""
        limiter = SlidingWindowRateLimiter()
        limiter.redis_pool = mock_redis
        yield limiter
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_within_limit(self, limiter, mock_redis):
        """Test rate limit check when within limit."""
        key = "test-user"
        limit = 10
        window = 60
        
        # Mock Redis operations
        mock_redis.pipeline.return_value.__aenter__ = AsyncMock()
        mock_redis.pipeline.return_value.__aexit__ = AsyncMock()
        mock_redis.pipeline.return_value.execute = AsyncMock(return_value=[None, 5])
        mock_redis.zadd = AsyncMock()
        mock_redis.expire = AsyncMock()
        
        allowed, rate_limit_info = await limiter.check_rate_limit(key, limit, window)
        
        assert allowed is True
        assert isinstance(rate_limit_info, RateLimitInfo)
        assert rate_limit_info.limit == limit
        assert rate_limit_info.remaining == 4  # limit - current_count - 1
        assert rate_limit_info.window_seconds == window
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, limiter, mock_redis):
        """Test rate limit check when limit is exceeded."""
        key = "test-user"
        limit = 10
        window = 60
        
        # Mock Redis operations to return count equal to limit
        mock_redis.pipeline.return_value.__aenter__ = AsyncMock()
        mock_redis.pipeline.return_value.__aexit__ = AsyncMock()
        mock_redis.pipeline.return_value.execute = AsyncMock(return_value=[None, 10])
        
        allowed, rate_limit_info = await limiter.check_rate_limit(key, limit, window)
        
        assert allowed is False
        assert rate_limit_info.remaining == 0
        assert isinstance(rate_limit_info.reset_time, datetime)
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_redis_failure(self, limiter, mock_redis):
        """Test rate limit check when Redis fails."""
        key = "test-user"
        limit = 10
        window = 60
        
        # Mock Redis to raise exception
        mock_redis.pipeline.return_value.__aenter__ = AsyncMock(side_effect=Exception("Redis error"))
        
        allowed, rate_limit_info = await limiter.check_rate_limit(key, limit, window)
        
        # Should fail open (allow request) when Redis is unavailable
        assert allowed is True
        assert rate_limit_info.limit == limit
        assert rate_limit_info.remaining == limit - 1
    
    @pytest.mark.asyncio
    async def test_reset_rate_limit_success(self, limiter, mock_redis):
        """Test successful rate limit reset."""
        key = "test-user"
        
        mock_redis.delete = AsyncMock(return_value=1)
        
        result = await limiter.reset_rate_limit(key)
        
        assert result is True
        mock_redis.delete.assert_called_once_with(f"rate_limit:{key}")
    
    @pytest.mark.asyncio
    async def test_reset_rate_limit_not_found(self, limiter, mock_redis):
        """Test rate limit reset when key doesn't exist."""
        key = "non-existent-user"
        
        mock_redis.delete = AsyncMock(return_value=0)
        
        result = await limiter.reset_rate_limit(key)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_rate_limit_info(self, limiter, mock_redis):
        """Test getting rate limit information."""
        key = "test-user"
        limit = 10
        window = 60
        
        # Mock Redis operations
        mock_redis.pipeline.return_value.__aenter__ = AsyncMock()
        mock_redis.pipeline.return_value.__aexit__ = AsyncMock()
        mock_redis.pipeline.return_value.execute = AsyncMock(return_value=[None, 3])
        
        rate_limit_info = await limiter.get_rate_limit_info(key, limit, window)
        
        assert isinstance(rate_limit_info, RateLimitInfo)
        assert rate_limit_info.limit == limit
        assert rate_limit_info.remaining == 7  # limit - current_count
        assert rate_limit_info.window_seconds == window


@pytest.mark.unit
class TestDistributedRateLimiter:
    """Test distributed rate limiter."""
    
    @pytest_asyncio.fixture
    async def distributed_limiter(self, mock_redis):
        """Create distributed rate limiter for testing."""
        limiter = DistributedRateLimiter()
        limiter.limiter.redis_pool = mock_redis
        yield limiter
    
    @pytest.mark.asyncio
    async def test_check_user_rate_limit(self, distributed_limiter, mock_redis):
        """Test user-specific rate limiting."""
        user_id = "test-user-123"
        
        # Mock the underlying limiter
        with patch.object(distributed_limiter.limiter, 'check_rate_limit') as mock_check:
            mock_rate_limit_info = RateLimitInfo(
                limit=1000,
                remaining=999,
                reset_time=datetime.utcnow() + timedelta(minutes=1),
                window_seconds=60
            )
            mock_check.return_value = (True, mock_rate_limit_info)
            
            allowed, rate_limit_info = await distributed_limiter.check_user_rate_limit(user_id)
            
            assert allowed is True
            assert rate_limit_info == mock_rate_limit_info
            mock_check.assert_called_once_with(f"user:{user_id}", 1000, 60)
    
    @pytest.mark.asyncio
    async def test_check_ip_rate_limit(self, distributed_limiter):
        """Test IP-specific rate limiting."""
        ip_address = "192.168.1.1"
        
        with patch.object(distributed_limiter.limiter, 'check_rate_limit') as mock_check:
            mock_rate_limit_info = RateLimitInfo(
                limit=10000,
                remaining=9999,
                reset_time=datetime.utcnow() + timedelta(minutes=1),
                window_seconds=60
            )
            mock_check.return_value = (True, mock_rate_limit_info)
            
            allowed, rate_limit_info = await distributed_limiter.check_ip_rate_limit(ip_address)
            
            assert allowed is True
            assert rate_limit_info == mock_rate_limit_info
            mock_check.assert_called_once_with(f"ip:{ip_address}", 10000, 60)
    
    @pytest.mark.asyncio
    async def test_check_endpoint_rate_limit(self, distributed_limiter):
        """Test endpoint-specific rate limiting."""
        endpoint = "login"
        identifier = "user-123"
        limit = 10
        window = 60
        
        with patch.object(distributed_limiter.limiter, 'check_rate_limit') as mock_check:
            mock_rate_limit_info = RateLimitInfo(
                limit=limit,
                remaining=9,
                reset_time=datetime.utcnow() + timedelta(minutes=1),
                window_seconds=window
            )
            mock_check.return_value = (True, mock_rate_limit_info)
            
            allowed, rate_limit_info = await distributed_limiter.check_endpoint_rate_limit(
                endpoint, identifier, limit, window
            )
            
            assert allowed is True
            mock_check.assert_called_once_with(f"endpoint:{endpoint}:{identifier}", limit, window)
    
    @pytest.mark.asyncio
    async def test_check_multiple_limits_all_allowed(self, distributed_limiter):
        """Test checking multiple rate limits when all are allowed."""
        checks = [
            {
                'type': 'user',
                'identifier': 'user-123',
                'name': 'user_check'
            },
            {
                'type': 'ip',
                'identifier': '192.168.1.1',
                'name': 'ip_check'
            }
        ]
        
        # Mock all limit checks to return allowed
        with patch.object(distributed_limiter, 'check_user_rate_limit') as mock_user, \
             patch.object(distributed_limiter, 'check_ip_rate_limit') as mock_ip:
            
            mock_user.return_value = (True, Mock())
            mock_ip.return_value = (True, Mock())
            
            results = await distributed_limiter.check_multiple_limits(checks)
            
            assert len(results) == 2
            assert results['user_check'][0] is True
            assert results['ip_check'][0] is True
    
    @pytest.mark.asyncio
    async def test_check_multiple_limits_with_failure(self, distributed_limiter):
        """Test checking multiple rate limits when some fail."""
        checks = [
            {
                'type': 'user',
                'identifier': 'user-123',
                'name': 'user_check'
            },
            {
                'type': 'ip',
                'identifier': '192.168.1.1',
                'name': 'ip_check'
            }
        ]
        
        # Mock user limit to fail, IP limit to succeed
        with patch.object(distributed_limiter, 'check_user_rate_limit') as mock_user, \
             patch.object(distributed_limiter, 'check_ip_rate_limit') as mock_ip:
            
            mock_user.return_value = (False, Mock())
            mock_ip.return_value = (True, Mock())
            
            results = await distributed_limiter.check_multiple_limits(checks)
            
            assert results['user_check'][0] is False
            assert results['ip_check'][0] is True
    
    @pytest.mark.asyncio
    async def test_reset_user_rate_limit(self, distributed_limiter):
        """Test resetting user rate limit."""
        user_id = "test-user-123"
        
        with patch.object(distributed_limiter.limiter, 'reset_rate_limit') as mock_reset:
            mock_reset.return_value = True
            
            result = await distributed_limiter.reset_user_rate_limit(user_id)
            
            assert result is True
            mock_reset.assert_called_once_with(f"user:{user_id}")
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, distributed_limiter, mock_redis):
        """Test health check success."""
        mock_redis.ping = AsyncMock()
        
        with patch.object(distributed_limiter.limiter, '_get_redis') as mock_get_redis:
            mock_get_redis.return_value = mock_redis
            
            result = await distributed_limiter.health_check()
            
            assert result is True
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, distributed_limiter, mock_redis):
        """Test health check failure."""
        mock_redis.ping = AsyncMock(side_effect=Exception("Redis error"))
        
        with patch.object(distributed_limiter.limiter, '_get_redis') as mock_get_redis:
            mock_get_redis.return_value = mock_redis
            
            result = await distributed_limiter.health_check()
            
            assert result is False


@pytest.mark.integration
class TestRateLimiterIntegration:
    """Integration tests for rate limiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_integration_with_redis(self, mock_redis):
        """Test rate limiter integration with Redis."""
        limiter = SlidingWindowRateLimiter()
        limiter.redis_pool = mock_redis
        
        key = "integration-test"
        limit = 5
        window = 60
        
        # Mock Redis pipeline operations
        mock_redis.pipeline.return_value.__aenter__ = AsyncMock()
        mock_redis.pipeline.return_value.__aexit__ = AsyncMock()
        
        # Simulate successful requests within limit
        for i in range(5):
            mock_redis.pipeline.return_value.execute = AsyncMock(return_value=[None, i])
            mock_redis.zadd = AsyncMock()
            mock_redis.expire = AsyncMock()
            
            allowed, rate_limit_info = await limiter.check_rate_limit(key, limit, window)
            
            assert allowed is True
            assert rate_limit_info.remaining == limit - i - 1
        
        # Next request should be blocked
        mock_redis.pipeline.return_value.execute = AsyncMock(return_value=[None, 5])
        
        allowed, rate_limit_info = await limiter.check_rate_limit(key, limit, window)
        
        assert allowed is False
        assert rate_limit_info.remaining == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_rate_limit_checks(self, mock_redis):
        """Test concurrent rate limit checks."""
        import asyncio
        
        limiter = SlidingWindowRateLimiter()
        limiter.redis_pool = mock_redis
        
        key = "concurrent-test"
        limit = 10
        window = 60
        
        # Mock Redis operations
        mock_redis.pipeline.return_value.__aenter__ = AsyncMock()
        mock_redis.pipeline.return_value.__aexit__ = AsyncMock()
        mock_redis.pipeline.return_value.execute = AsyncMock(return_value=[None, 1])
        mock_redis.zadd = AsyncMock()
        mock_redis.expire = AsyncMock()
        
        # Run multiple checks concurrently
        tasks = []
        for _ in range(5):
            task = limiter.check_rate_limit(key, limit, window)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should be allowed since we're mocking low count
        for allowed, rate_limit_info in results:
            assert allowed is True
            assert isinstance(rate_limit_info, RateLimitInfo)


@pytest.mark.performance
class TestRateLimiterPerformance:
    """Performance tests for rate limiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_performance(self, mock_redis):
        """Test rate limiter performance under load."""
        import asyncio
        import time
        
        limiter = SlidingWindowRateLimiter()
        limiter.redis_pool = mock_redis
        
        # Mock fast Redis operations
        mock_redis.pipeline.return_value.__aenter__ = AsyncMock()
        mock_redis.pipeline.return_value.__aexit__ = AsyncMock()
        mock_redis.pipeline.return_value.execute = AsyncMock(return_value=[None, 1])
        mock_redis.zadd = AsyncMock()
        mock_redis.expire = AsyncMock()
        
        # Measure performance
        start_time = time.time()
        
        tasks = []
        for i in range(100):
            task = limiter.check_rate_limit(f"perf-test-{i}", 1000, 60)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete 100 checks in reasonable time (< 1 second)
        assert duration < 1.0
        assert len(results) == 100
        
        # All should succeed
        for allowed, rate_limit_info in results:
            assert allowed is True
