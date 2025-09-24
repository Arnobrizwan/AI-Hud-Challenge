"""
Rate Limiting Middleware
"""

import time
import logging
from typing import Callable, Dict
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting"""
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        if self._is_rate_limited(client_ip):
            logger.warning(f"Rate limit exceeded for client: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Record request
        self._record_request(client_ip)
        
        # Process request
        response = await call_next(request)
        
        return response
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit"""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if req_time > minute_ago
            ]
        else:
            self.requests[client_ip] = []
        
        # Check if limit exceeded
        return len(self.requests[client_ip]) >= self.requests_per_minute
    
    def _record_request(self, client_ip: str):
        """Record a request for the client"""
        now = time.time()
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        self.requests[client_ip].append(now)
