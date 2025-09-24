"""
Feedback & Human-in-the-Loop Service
Main FastAPI application with WebSocket support
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

import structlog
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .api.routes import analytics_router, annotation_router, editorial_router, feedback_router
from .config.settings import get_settings
from .database.connection import close_db, init_db
from .middleware.auth import AuthMiddleware
from .middleware.monitoring import MonitoringMiddleware
from .models.schemas import ErrorResponse
from .realtime.websocket_manager import WebSocketManager

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

logger = structlog.get_logger(__name__)

# Global WebSocket manager
websocket_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> Dict[str, Any]:
    """Application lifespan manager"""
    settings = get_settings()

    # Startup
    logger.info("Starting Feedback Service", version="1.0.0")

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Initialize Redis connection
    # await init_redis()
    # logger.info("Redis initialized")

    # Start background tasks
    asyncio.create_task(websocket_manager.start_cleanup_task())
    logger.info("Background tasks started")

    yield

    # Shutdown
    logger.info("Shutting down Feedback Service")
    await close_db()
    await websocket_manager.cleanup()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Feedback & Human-in-the-Loop Service",
    description="Comprehensive feedback collection, editorial workflows, and ML model improvement",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(AuthMiddleware)
app.add_middleware(MonitoringMiddleware)

# Include routers
app.include_router(
    feedback_router,
    prefix="/api/v1/feedback",
    tags=["feedback"])
app.include_router(
    editorial_router,
    prefix="/api/v1/editorial",
    tags=["editorial"])
app.include_router(
    annotation_router,
    prefix="/api/v1/annotation",
    tags=["annotation"])
app.include_router(
    analytics_router,
    prefix="/api/v1/analytics",
    tags=["analytics"])


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "feedback-service",
        "version": "1.0.0",
        "timestamp": "2024-01-01T00:00:00Z",
    }


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """Prometheus metrics endpoint"""
    # This would be implemented with prometheus_client
    return {"message": "Metrics endpoint - implement with prometheus_client"}


@app.websocket("/ws/feedback/{user_id}")
async def websocket_feedback_endpoint(websocket: WebSocket, user_id: str) -> Dict[str, Any]:
    """WebSocket endpoint for real-time feedback processing"""
    await websocket_manager.connect(websocket, user_id)

    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()

            # Process feedback in real-time
            result = await websocket_manager.process_realtime_feedback(user_id, data)

            # Send response back to client
            await websocket.send_json(
                {
                    "status": "processed",
                    "feedback_id": result.get("feedback_id"),
                    "timestamp": result.get("timestamp"),
                }
            )

    except WebSocketDisconnect:
        websocket_manager.disconnect(user_id)
        logger.info("WebSocket disconnected", user_id=user_id)
    except Exception as e:
        logger.error("WebSocket error", user_id=user_id, error=str(e))
        await websocket.send_json({"status": "error", "message": str(e)})


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc) -> Dict[str, Any]:
    """Global HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc) -> Dict[str, Any]:
    """Global exception handler"""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(status_code=500, content=ErrorResponse(
        error="Internal server error", status_code=500).dict(), )


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info")
