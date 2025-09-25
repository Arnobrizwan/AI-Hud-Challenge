"""
FastAPI application for intelligent notification decisioning service.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

import structlog
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from .api import router as api_router
from .config import get_settings
from .database import init_db
from .decision_engine import NotificationDecisionEngine
from .monitoring import setup_monitoring
from .redis_client import get_redis_client

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
NOTIFICATION_DECISIONS_TOTAL = Counter(
    "notification_decisions_total",
    "Total number of notification decisions made",
    ["decision", "notification_type"],
)

NOTIFICATION_DECISION_DURATION = Histogram(
    "notification_decision_duration_seconds",
    "Time spent making notification decisions",
    ["notification_type"],
)

NOTIFICATION_DELIVERIES_TOTAL = Counter(
    "notification_deliveries_total",
    "Total number of notification deliveries",
    ["channel", "status"],
)

# Global instances
decision_engine: NotificationDecisionEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Dict[str, Any]:
    """Application lifespan manager."""
    global decision_engine

    # Startup
    logger.info("Starting notification decisioning service")

    # Initialize database
    await init_db()

    # Initialize Redis
    redis_client = await get_redis_client()

    # Initialize decision engine
    decision_engine = NotificationDecisionEngine(redis_client)
    await decision_engine.initialize()

    # Setup monitoring
    setup_monitoring()

    logger.info("Notification decisioning service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down notification decisioning service")
    if decision_engine:
        await decision_engine.cleanup()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Notification Decisioning Service",
        description="Intelligent notification decisioning with ML optimization",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router, prefix="/api/v1")

    # Health check endpoint
    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        return {"status": "healthy", "service": "notification-decisioning", "version": "1.0.0"}

    # Metrics endpoint
    @app.get("/metrics")
    async def metrics() -> Dict[str, Any]:
        """Prometheus metrics endpoint."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


# Create app instance
app = create_app()


def get_decision_engine() -> NotificationDecisionEngine:
    """Dependency to get decision engine instance."""
    if decision_engine is None:
        raise HTTPException(status_code=503, detail="Decision engine not initialized")
    return decision_engine


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
