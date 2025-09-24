"""
Monitoring and metrics middleware
"""

import logging
import time
from typing import Callable

from fastapi import Request, Response

logger = logging.getLogger(__name__)


class MonitoringMiddleware:
    """Middleware for monitoring and metrics collection"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send) -> Dict[str, Any]:
    if scope["type"] == "http":
            request = Request(scope, receive)
            start_time = time.time()

            # Process request
            await self.app(scope, receive, send)

            # Calculate metrics
            process_time = time.time() - start_time

            # Log metrics
            logger.info(
                "Request processed",
                method=request.method,
                path=request.url.path,
                process_time=process_time,
            )

            # In production, this would send metrics to Prometheus, etc.
        else:
    await self.app(scope, receive, send)
