"""Middleware for evaluation engine."""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class EvaluationMiddleware(BaseHTTPMiddleware):
    """Middleware for evaluation service."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request through middleware."""
        response = await call_next(request)
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Error handling middleware."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request through middleware."""
        response = await call_next(request)
        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Performance monitoring middleware."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request through middleware."""
        response = await call_next(request)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request through middleware."""
        response = await call_next(request)
        return response