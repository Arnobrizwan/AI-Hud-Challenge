"""
Middleware for Evaluation Suite Microservice
"""

import time
import logging
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests"""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
        
        # Add process time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring"""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        process_time = time.time() - start_time
        
        # Log performance metrics
        logger.info(f"Performance: {request.method} {request.url.path} - {process_time:.3f}s")
        
        # Add performance headers
        response.headers["X-Response-Time"] = str(process_time)
        response.headers["X-Request-ID"] = str(hash(request.url.path + str(start_time)))
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for error handling"""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Error processing request {request.method} {request.url.path}: {str(e)}")
            
            # Return error response
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An error occurred while processing the request",
                    "request_id": str(hash(request.url.path + str(time.time())))
                }
            )
