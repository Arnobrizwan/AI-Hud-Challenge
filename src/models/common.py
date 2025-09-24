"""
Common request/response models and validators.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ResponseStatus(str, Enum):
    """Response status types."""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class RequestMethod(str, Enum):
    """HTTP request methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class BaseResponse(BaseModel):
    """Base response model for all API responses."""

    status: ResponseStatus = Field(description="Response status")
    message: Optional[str] = Field(default=None, description="Response message")
    data: Optional[Any] = Field(default=None, description="Response data")
    errors: Optional[List[str]] = Field(default=None, description="Error messages")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(default=None, description="Request correlation ID")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ErrorResponse(BaseResponse):
    """Error response model."""

    status: ResponseStatus = Field(default=ResponseStatus.ERROR)
    error_code: Optional[str] = Field(default=None, description="Error code")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")


class ValidationErrorDetail(BaseModel):
    """Validation error detail."""

    field: str = Field(description="Field name")
    message: str = Field(description="Error message")
    value: Any = Field(description="Invalid value")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response."""

    error_code: str = Field(default="VALIDATION_ERROR")
    validation_errors: List[ValidationErrorDetail] = Field(description="Validation error details")


class HealthCheckStatus(str, Enum):
    """Health check status types."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class DependencyStatus(BaseModel):
    """Dependency health status."""

    name: str = Field(description="Dependency name")
    status: HealthCheckStatus = Field(description="Dependency status")
    response_time: Optional[float] = Field(default=None, description="Response time in seconds")
    error: Optional[str] = Field(default=None, description="Error message if unhealthy")


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: HealthCheckStatus = Field(description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(description="Application version")
    uptime: float = Field(description="Uptime in seconds")
    dependencies: List[DependencyStatus] = Field(description="Dependency statuses")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class RateLimitInfo(BaseModel):
    """Rate limit information."""

    limit: int = Field(description="Rate limit threshold")
    remaining: int = Field(description="Remaining requests")
    reset_time: datetime = Field(description="Rate limit reset time")
    window_seconds: int = Field(description="Rate limit window in seconds")


class RateLimitResponse(ErrorResponse):
    """Rate limit exceeded response."""

    error_code: str = Field(default="RATE_LIMIT_EXCEEDED")
    rate_limit: RateLimitInfo = Field(description="Rate limit information")


class MetricsData(BaseModel):
    """Metrics data model."""

    metric_name: str = Field(description="Metric name")
    value: Union[int, float] = Field(description="Metric value")
    labels: Dict[str, str] = Field(default={}, description="Metric labels")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RequestContext(BaseModel):
    """Request context information."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = Field(default=None)
    client_ip: str = Field(description="Client IP address")
    user_agent: Optional[str] = Field(default=None)
    method: RequestMethod = Field(description="HTTP method")
    path: str = Field(description="Request path")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PaginationRequest(BaseModel):
    """Pagination request parameters."""

    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(default=None, description="Sort field")
    sort_order: Optional[str] = Field(default="asc", regex="^(asc|desc)$", description="Sort order")


class PaginationResponse(BaseModel):
    """Pagination response metadata."""

    page: int = Field(description="Current page")
    page_size: int = Field(description="Items per page")
    total_items: int = Field(description="Total number of items")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Has next page")
    has_previous: bool = Field(description="Has previous page")


class PaginatedResponse(BaseResponse):
    """Paginated response model."""

    data: List[Any] = Field(description="Response data items")
    pagination: PaginationResponse = Field(description="Pagination metadata")
