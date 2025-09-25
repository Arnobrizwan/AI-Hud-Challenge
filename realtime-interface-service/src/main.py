"""
Main FastAPI application for the Realtime Interfaces microservice.
Provides HUD-facing APIs for real-time news updates and interactions.
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_client.core import CollectorRegistry

from src.config.settings import settings
from src.models.api import (
    FeedRequest,
    FeedResponse,
    ClusterDetailResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthCheckResponse,
    LiveUpdateResponse,
    MetricsResponse,
    WebSocketMessage,
)
from src.models.content import ContentType, ProcessingStatus
from src.services.feed_service import FeedService
from src.services.cluster_service import ClusterService
from src.services.feedback_service import FeedbackService
from src.services.websocket_manager import WebSocketManager
from src.services.cache_service import CacheService
from src.services.ranking_service import RankingService

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global services
feed_service: Optional[FeedService] = None
cluster_service: Optional[ClusterService] = None
feedback_service: Optional[FeedbackService] = None
websocket_manager: Optional[WebSocketManager] = None
cache_service: Optional[CacheService] = None
ranking_service: Optional[RankingService] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Dict[str, Any]:
    """Application lifespan manager."""
    global (
        feed_service,
        cluster_service,
        feedback_service,
        websocket_manager,
        cache_service,
        ranking_service,
    )

    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    try:
        # Initialize services
        cache_service = CacheService()
        ranking_service = RankingService()
        feed_service = FeedService(cache_service=cache_service, ranking_service=ranking_service)
        cluster_service = ClusterService(cache_service=cache_service)
        feedback_service = FeedbackService()
        websocket_manager = WebSocketManager()

        # Initialize services
        await cache_service.initialize()
        await ranking_service.initialize()
        await feed_service.initialize()
        await cluster_service.initialize()
        await feedback_service.initialize()
        await websocket_manager.initialize()

        logger.info("Application startup complete")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down application")

    try:
        if websocket_manager:
            await websocket_manager.shutdown()

        if feed_service:
            await feed_service.shutdown()

        if cluster_service:
            await cluster_service.shutdown()

        if feedback_service:
            await feedback_service.shutdown()

        if cache_service:
            await cache_service.shutdown()

        if ranking_service:
            await ranking_service.shutdown()

        logger.info("Application shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Realtime Interfaces microservice for HUD-facing APIs",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get services
def get_feed_service() -> FeedService:
    if not feed_service:
        raise HTTPException(status_code=503, detail="Feed service not available")
    return feed_service


def get_cluster_service() -> ClusterService:
    if not cluster_service:
        raise HTTPException(status_code=503, detail="Cluster service not available")
    return cluster_service


def get_feedback_service() -> FeedbackService:
    if not feedback_service:
        raise HTTPException(status_code=503, detail="Feedback service not available")
    return feedback_service


def get_websocket_manager() -> WebSocketManager:
    if not websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not available")
    return websocket_manager


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["Monitoring"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Check service dependencies
        dependencies = {
            "feed_service": "healthy" if feed_service else "unhealthy",
            "cluster_service": "healthy" if cluster_service else "unhealthy",
            "feedback_service": "healthy" if feedback_service else "unhealthy",
            "websocket_manager": "healthy" if websocket_manager else "unhealthy",
            "cache_service": "healthy" if cache_service else "unhealthy",
            "ranking_service": "healthy" if ranking_service else "unhealthy",
        }

        # Check overall status
        overall_status = "healthy" if all(status == "healthy" for status in dependencies.values()) else "degraded"

        return HealthCheckResponse(
            status=overall_status,
            version=settings.APP_VERSION,
            uptime_seconds=time.time() - time.time(),  # Placeholder
            dependencies=dependencies,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            version=settings.APP_VERSION,
            uptime_seconds=0,
            dependencies={"error": str(e)},
        )


# Metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def metrics() -> Dict[str, Any]:
    """Prometheus metrics endpoint."""
    try:
        return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        raise HTTPException(status_code=500, detail="Error generating metrics")


# Feed endpoints
@app.get("/feed", response_model=FeedResponse, tags=["Feed"])
async def get_feed(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(20, ge=1, le=100, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    topics: Optional[str] = Query(None, description="Comma-separated list of topics"),
    sources: Optional[str] = Query(None, description="Comma-separated list of sources"),
    content_types: Optional[str] = Query(None, description="Comma-separated list of content types"),
    time_range: Optional[str] = Query("24h", description="Time range (1h, 24h, 7d, 30d)"),
    personalization: bool = Query(True, description="Enable personalization"),
    feed_svc: FeedService = Depends(get_feed_service),
) -> Dict[str, Any]:
    """Get personalized news feed for user."""
    try:
        start_time = time.time()

        # Parse query parameters
        topic_list = topics.split(",") if topics else None
        source_list = sources.split(",") if sources else None
        content_type_list = content_types.split(",") if content_types else None

        # Get feed
        feed = await feed_svc.get_personalized_feed(
            user_id=user_id,
            limit=limit,
            offset=offset,
            topics=topic_list,
            sources=source_list,
            content_types=content_type_list,
            time_range=time_range,
            personalization=personalization,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return FeedResponse(
            user_id=user_id,
            items=feed.items,
            total_count=feed.total_count,
            limit=limit,
            offset=offset,
            processing_time_ms=processing_time_ms,
            personalization_enabled=personalization,
            cache_hit=feed.cache_hit,
            etag=feed.etag,
        )

    except Exception as e:
        logger.error(f"Error getting feed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting feed: {str(e)}")


# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.DEBUG,
    )