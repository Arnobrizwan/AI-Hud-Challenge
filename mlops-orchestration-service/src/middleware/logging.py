"""
Logging middleware for MLOps Orchestration Service
"""

import time
import uuid
from typing import Any, Dict

from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class LoggingMiddleware:
    """Request/response logging middleware"""

    def __init__(self):
        self.logger = logger

    async def __call__(self, request: Request, call_next) -> Dict[str, Any]:
        """Log request and response"""

        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Log request
        await self.log_request(request, request_id)

        # Process request
        start_time = time.time()

        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response
            await self.log_response(request, response, process_time, request_id)

            return response

        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time

            # Log error
            await self.log_error(request, e, process_time, request_id)

            raise

    async def log_request(self, request: Request, request_id: str) -> None:
        """Log incoming request"""

        # Extract request information
        client_ip = self.get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")

        # Log request
        self.logger.info(
            f"Request started: {request.method} {request.url.path}",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            client_ip=client_ip,
            user_agent=user_agent,
            headers=self.get_safe_headers(request.headers),
        )

    async def log_response(
            self,
            request: Request,
            response: Response,
            process_time: float,
            request_id: str) -> None:
        """Log outgoing response"""

        # Extract response information
        status_code = response.status_code
        content_length = self.get_content_length(response)

        # Log response
        self.logger.info(
            f"Request completed: {request.method} {request.url.path}",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            process_time_ms=round(process_time * 1000, 2),
            content_length=content_length,
        )

    async def log_error(
            self,
            request: Request,
            error: Exception,
            process_time: float,
            request_id: str) -> None:
        """Log request error"""

        # Log error
        self.logger.error(
            f"Request failed: {request.method} {request.url.path}",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            error_type=type(error).__name__,
            error_message=str(error),
            process_time_ms=round(process_time * 1000, 2),
        )

    def get_client_ip(self, request: Request) -> str:
        """Get client IP address"""

        # Check for forwarded IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check for real IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to client host
        return request.client.host if request.client else "unknown"

    def get_safe_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Get headers with sensitive information removed"""

        safe_headers = {}
        sensitive_headers = {
            "authorization",
            "cookie",
            "x-api-key",
            "x-auth-token"}

        for key, value in headers.items():
            if key.lower() not in sensitive_headers:
                safe_headers[key] = value
            else:
                safe_headers[key] = "***"

        return safe_headers

    def get_content_length(self, response: Response) -> int:
        """Get response content length"""

        if hasattr(
                response,
                "headers") and "content-length" in response.headers:
            return int(response.headers["content-length"])

        if hasattr(response, "body") and response.body:
            return len(response.body)

        return 0


class StructuredLoggingMiddleware:
    """Structured logging middleware with detailed request/response logging"""

    def __init__(self):
        self.logger = logger

    async def __call__(self, request: Request, call_next) -> Dict[str, Any]:
        """Log structured request and response data"""

        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Create structured log data
        log_data = {
            "request_id": request_id,
            "timestamp": time.time(),
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": self.get_client_ip(request),
            "user_agent": request.headers.get("user-agent", "unknown"),
            "content_type": request.headers.get("content-type", "unknown"),
            "content_length": request.headers.get("content-length", "0"),
        }

        # Log request
        self.logger.info("Request received", **log_data)

        # Process request
        start_time = time.time()

        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Update log data with response information
            log_data.update(
                {
                    "status_code": response.status_code,
                    "process_time_ms": round(
                        process_time * 1000,
                        2),
                    "response_content_length": self.get_content_length(response),
                    "response_content_type": response.headers.get(
                        "content-type",
                        "unknown"),
                })

            # Log response
            self.logger.info("Request completed", **log_data)

            return response

        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time

            # Update log data with error information
            log_data.update(
                {
                    "status_code": 500,
                    "process_time_ms": round(process_time * 1000, 2),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            )

            # Log error
            self.logger.error("Request failed", **log_data)

            raise

    def get_client_ip(self, request: Request) -> str:
        """Get client IP address"""

        # Check for forwarded IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check for real IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to client host
        return request.client.host if request.client else "unknown"

    def get_content_length(self, response: Response) -> int:
        """Get response content length"""

        if hasattr(
                response,
                "headers") and "content-length" in response.headers:
            return int(response.headers["content-length"])

        if hasattr(response, "body") and response.body:
            return len(response.body)

        return 0


class MLPipelineLoggingMiddleware:
    """Specialized logging middleware for ML pipeline operations"""

    def __init__(self):
        self.logger = logger

    async def __call__(self, request: Request, call_next) -> Dict[str, Any]:
        """Log ML pipeline specific operations"""

        # Check if this is an ML pipeline related request
        if not self.is_ml_pipeline_request(request):
            return await call_next(request)

        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Extract ML pipeline context
        pipeline_context = self.extract_pipeline_context(request)

        # Log pipeline operation start
        self.logger.info(
            f"ML Pipeline operation started: {request.method} {request.url.path}",
            request_id=request_id,
            pipeline_context=pipeline_context,
        )

        # Process request
        start_time = time.time()

        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log pipeline operation completion
            self.logger.info(
                f"ML Pipeline operation completed: {request.method} {request.url.path}",
                request_id=request_id,
                pipeline_context=pipeline_context,
                status_code=response.status_code,
                process_time_ms=round(
                    process_time * 1000,
                    2),
            )

            return response

        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time

            # Log pipeline operation error
            self.logger.error(
                f"ML Pipeline operation failed: {request.method} {request.url.path}",
                request_id=request_id,
                pipeline_context=pipeline_context,
                error_type=type(e).__name__,
                error_message=str(e),
                process_time_ms=round(
                    process_time * 1000,
                    2),
            )

            raise

    def is_ml_pipeline_request(self, request: Request) -> bool:
        """Check if request is ML pipeline related"""

        ml_pipeline_paths = [
            "/api/v1/pipelines/",
            "/api/v1/training/",
            "/api/v1/deployment/",
            "/api/v1/monitoring/",
            "/api/v1/features/",
            "/api/v1/retraining/",
        ]

        return any(request.url.path.startswith(path)
                   for path in ml_pipeline_paths)

    def extract_pipeline_context(self, request: Request) -> Dict[str, Any]:
        """Extract ML pipeline context from request"""
        context = {"path": request.url.path, "method": request.method}

        # Extract pipeline ID from path
        path_parts = request.url.path.split("/")
        if "pipelines" in path_parts:
            pipeline_idx = path_parts.index("pipelines")
            if pipeline_idx + 1 < len(path_parts):
                context["pipeline_id"] = path_parts[pipeline_idx + 1]

        # Extract model name from query params or body
        if "model_name" in request.query_params:
            context["model_name"] = request.query_params["model_name"]

        return context
