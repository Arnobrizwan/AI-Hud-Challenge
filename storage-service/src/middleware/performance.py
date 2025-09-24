"""
Performance Monitoring Middleware
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Monitor and log performance metrics for requests"""
    
    def __init__(self, app, slow_request_threshold: float = 1.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timing
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        duration = time.time() - start_time
        
        # Log slow requests
        if duration > self.slow_request_threshold:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path} "
                f"duration={duration:.3f}s threshold={self.slow_request_threshold}s"
            )
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        response.headers["X-Request-ID"] = getattr(request.state, "request_id", "unknown")
        
        return response
