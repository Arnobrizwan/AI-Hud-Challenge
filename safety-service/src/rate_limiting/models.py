"""
Rate Limiting Models
Data models for rate limiting system
"""

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, Field


class BaseRateLimitModel(BaseModel):
    """Base model for rate limiting"""

    model_config = {"arbitrary_types_allowed": True}

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RateLimitRequest(BaseRateLimitModel):
    """Request for rate limiting check"""

    user_id: str
    endpoint: str
    ip_address: str
    request_size: int = 1
    current_load: float = 0.0


class RateLimitResult(BaseRateLimitModel):
    """Result of rate limiting check"""

    user_id: str
    endpoint: str
    is_rate_limited: bool
    triggered_limits: List[str]
    remaining_capacity: int
    retry_after: int
    check_timestamp: datetime


class DynamicRateLimitConfig(BaseRateLimitModel):
    """Dynamic rate limiting configuration"""

    user_id: str
    base_limits: Dict[str, int]
    ttl_seconds: int
    reputation_multiplier: float = 1.0
    abuse_penalty: float = 1.0
