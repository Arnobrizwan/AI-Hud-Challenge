"""
Main FastAPI application for the Ingestion & Normalization microservice.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_client.core import CollectorRegistry

from src.config.load_sources import source_config_loader
from src.config.settings import settings
from src.models.api import (
    APIResponse,
    BatchStatusResponse,
    ContentSearchRequest,
    ContentSearchResponse,
    DuplicateDetectionRequest,
    DuplicateDetectionResponse,
    ErrorResponse,
    HealthCheckResponse,
    IngestionRequest,
    IngestionResponse,
    MetricsResponse,
    SourceConfigRequest,
    SourceListResponse,
    SourceStatsResponse,
)
from src.models.content import ContentType, ProcessingStatus, SourceType
from src.services.firestore_service import FirestoreService
from src.services.ingestion_service import IngestionService
from src.services.pubsub_service import PubSubService
from src.utils.content_parser import content_parser
from src.utils.date_utils import date_utils
from src.utils.http_client import get_http_client
from src.utils.url_utils import url_utils

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global services
ingestion_service: Optional[IngestionService] = None
pubsub_service: Optional[PubSubService] = None
firestore_service: Optional[FirestoreService] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Dict[str, Any]:
    """Application lifespan manager."""
    global ingestion_service, pubsub_service, firestore_service

    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    try:
        # Initialize services
        ingestion_service = IngestionService(
            http_client=get_http_client(),
            content_parser=content_parser,
            url_utils=url_utils,
            date_utils=date_utils,
        )

        pubsub_service = PubSubService()
        firestore_service = FirestoreService()

        # Load source configurations
        sources = source_config_loader.load_sources()
        logger.info(f"Loaded {len(sources)} source configurations")

        # Create Pub/Sub topics if they don't exist
        await pubsub_service.create_topic(settings.PUBSUB_TOPIC_INGESTION)
        await pubsub_service.create_topic(settings.PUBSUB_TOPIC_NORMALIZATION)
        await pubsub_service.create_subscription(
            settings.PUBSUB_TOPIC_INGESTION, settings.PUBSUB_SUBSCRIPTION_INGESTION
        )

        logger.info("Application startup complete")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down application")

    try:
        if ingestion_service:
            await ingestion_service.shutdown()

        http_client = get_http_client()
        if http_client:
            await http_client.close()

        logger.info("Application shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Scalable Ingestion & Normalization microservice for news content",
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
def get_ingestion_service() -> IngestionService:
    if not ingestion_service:
        raise HTTPException(status_code=503, detail="Ingestion service not available")
    return ingestion_service


def get_pubsub_service() -> PubSubService:
    if not pubsub_service:
        raise HTTPException(status_code=503, detail="Pub/Sub service not available")
    return pubsub_service


def get_firestore_service() -> FirestoreService:
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore service not available")
    return firestore_service


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["Monitoring"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Check service dependencies
        dependencies = {
            "ingestion_service": "healthy" if ingestion_service else "unhealthy",
            "pubsub_service": "healthy" if pubsub_service else "unhealthy",
            "firestore_service": "healthy" if firestore_service else "unhealthy",
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


# Source management endpoints
@app.get("/sources", response_model=SourceListResponse, tags=["Sources"])
async def list_sources() -> Dict[str, Any]:
    """List all configured sources."""
    try:
        sources = source_config_loader.load_sources()
        enabled_sources = [s for s in sources if s.enabled]
        disabled_sources = [s for s in sources if not s.enabled]

        # Convert to API response format
        sources_data = []
        for source in sources:
            source_info = {
                "id": source.id,
                "name": source.name,
                "type": source.type.value,
                "url": source.url,
                "enabled": source.enabled,
                "priority": source.priority,
                "rate_limit": source.rate_limit,
                "timeout": source.timeout,
                "last_checked": source.last_checked,
                "last_success": source.last_success,
                "error_count": source.error_count,
                "success_count": source.success_count,
            }
            sources_data.append(source_info)

        return SourceListResponse(
            sources=sources_data,
            total_count=len(sources),
            enabled_count=len(enabled_sources),
            disabled_count=len(disabled_sources),
        )

    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        raise HTTPException(status_code=500, detail="Error listing sources")


@app.get("/sources/{source_id}", tags=["Sources"])
async def get_source(source_id: str) -> Dict[str, Any]:
    """Get source configuration by ID."""
    try:
        source = source_config_loader.get_source_by_id(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        return {
            "id": source.id,
            "name": source.name,
            "type": source.type.value,
            "url": source.url,
            "enabled": source.enabled,
            "priority": source.priority,
            "rate_limit": source.rate_limit,
            "timeout": source.timeout,
            "retry_attempts": source.retry_attempts,
            "backoff_factor": source.backoff_factor,
            "user_agent": source.user_agent,
            "headers": source.headers,
            "auth": source.auth,
            "filters": source.filters,
            "last_checked": source.last_checked,
            "last_success": source.last_success,
            "error_count": source.error_count,
            "success_count": source.success_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting source {source_id}: {e}")
        raise HTTPException(status_code=500, detail="Error getting source")


@app.put("/sources/{source_id}", tags=["Sources"])
async def update_source(source_id: str, request: SourceConfigRequest) -> Dict[str, Any]:
    """Update source configuration."""
    try:
        success = source_config_loader.update_source_config(source_id, request.model_dump(exclude_unset=True))
        if not success:
            raise HTTPException(status_code=404, detail="Source not found")

        return {"message": "Source updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating source {source_id}: {e}")
        raise HTTPException(status_code=500, detail="Error updating source")


# Ingestion endpoints
@app.post("/ingest", response_model=IngestionResponse, tags=["Ingestion"])
async def ingest_content(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    ingestion_svc: IngestionService = Depends(get_ingestion_service),
):
    """Start content ingestion for a source."""
    try:
        # Get source configuration
        source = source_config_loader.get_source_by_id(request.source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        if not source.enabled:
            raise HTTPException(status_code=400, detail="Source is disabled")

        # Start ingestion in background
        batch = await ingestion_svc.process_source(source)

        return IngestionResponse(
            batch_id=batch.batch_id,
            source_id=batch.source_id,
            articles_processed=batch.total_count,
            articles_successful=batch.processed_count,
            articles_failed=batch.failed_count,
            articles_duplicates=batch.duplicate_count,
            processing_time_ms=(int(batch.processing_time_seconds * 1000) if batch.processing_time_seconds else 0),
            status=batch.status,
            errors=[batch.error_message] if batch.error_message else [],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting ingestion: {e}")
        raise HTTPException(status_code=500, detail="Error starting ingestion")


@app.post("/ingest/all", tags=["Ingestion"])
async def ingest_all_sources(
    background_tasks: BackgroundTasks,
    ingestion_svc: IngestionService = Depends(get_ingestion_service),
):
    """Start content ingestion for all enabled sources."""
    try:
        sources = source_config_loader.get_enabled_sources()
        if not sources:
            raise HTTPException(status_code=400, detail="No enabled sources found")

        # Process all sources
        batches = await ingestion_svc.process_sources(sources)

        return {
            "message": f"Started ingestion for {len(sources)} sources",
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "source_id": batch.source_id,
                    "status": batch.status.value,
                    "total_count": batch.total_count,
                }
                for batch in batches
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting bulk ingestion: {e}")
        raise HTTPException(status_code=500, detail="Error starting bulk ingestion")


# Batch status endpoints
@app.get("/batches/{batch_id}", response_model=BatchStatusResponse, tags=["Batches"])
async def get_batch_status(batch_id: str, ingestion_svc: IngestionService = Depends(get_ingestion_service)):
    """Get status of a processing batch."""
    try:
        batch = await ingestion_svc.get_batch_status(batch_id)
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")

        progress_percentage = (batch.processed_count / batch.total_count * 100) if batch.total_count > 0 else 0

        return BatchStatusResponse(
            batch_id=batch.batch_id,
            status=batch.status,
            progress_percentage=progress_percentage,
            total_items=batch.total_count,
            processed_items=batch.processed_count,
            failed_items=batch.failed_count,
            duplicate_items=batch.duplicate_count,
            started_at=batch.started_at,
            completed_at=batch.completed_at,
            estimated_completion=None,  # Could be calculated based on processing rate
            errors=[batch.error_message] if batch.error_message else [],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch status: {e}")
        raise HTTPException(status_code=500, detail="Error getting batch status")


@app.get("/batches", tags=["Batches"])
async def list_batches(ingestion_svc: IngestionService = Depends(get_ingestion_service)):
    """List all active processing batches."""
    try:
        batches = await ingestion_svc.get_active_batches()

        return {
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "source_id": batch.source_id,
                    "status": batch.status.value,
                    "total_count": batch.total_count,
                    "processed_count": batch.processed_count,
                    "failed_count": batch.failed_count,
                    "duplicate_count": batch.duplicate_count,
                    "created_at": batch.created_at,
                    "started_at": batch.started_at,
                    "completed_at": batch.completed_at,
                }
                for batch in batches
            ]
        }

    except Exception as e:
        logger.error(f"Error listing batches: {e}")
        raise HTTPException(status_code=500, detail="Error listing batches")


# Content search endpoints
@app.post("/search", response_model=ContentSearchResponse, tags=["Content"])
async def search_content(
    request: ContentSearchRequest, firestore_svc: FirestoreService = Depends(get_firestore_service)
):
    """Search for content articles."""
    try:
        start_time = time.time()

        # Convert request to search parameters
        articles = await firestore_svc.search_articles(
            query_text=request.query or "",
            limit=request.limit,
            source_id=request.source_ids[0] if request.source_ids else None,
            content_type=request.content_types[0].value if request.content_types else None,
            language=request.languages[0] if request.languages else None,
        )

        query_time_ms = int((time.time() - start_time) * 1000)

        # Convert articles to response format
        articles_data = []
        for article in articles:
            article_data = {
                "id": article.id,
                "title": article.title,
                "url": article.url,
                "summary": article.summary,
                "author": article.author,
                "source": article.source,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "language": article.language,
                "content_type": article.content_type.value,
                "word_count": article.word_count,
                "reading_time": article.reading_time,
                "tags": article.tags,
            }
            articles_data.append(article_data)

        return ContentSearchResponse(
            articles=articles_data,
            total_count=len(articles_data),
            limit=request.limit,
            offset=request.offset,
            query_time_ms=query_time_ms,
        )

    except Exception as e:
        logger.error(f"Error searching content: {e}")
        raise HTTPException(status_code=500, detail="Error searching content")


# Metrics endpoints
@app.get("/metrics/sources", tags=["Metrics"])
async def get_source_metrics(
    source_id: str = Query(None, description="Source ID to get metrics for"),
    ingestion_svc: IngestionService = Depends(get_ingestion_service),
):
    """Get processing metrics for sources."""
    try:
        if source_id:
            metrics = await ingestion_svc.get_processing_metrics(source_id)
            return metrics
        else:
            metrics = await ingestion_svc.get_processing_metrics()
            return metrics

    except Exception as e:
        logger.error(f"Error getting source metrics: {e}")
        raise HTTPException(status_code=500, detail="Error getting source metrics")


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
