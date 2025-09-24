"""FastAPI application for the deduplication service."""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, List

import redis.asyncio as redis
import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .algorithms.lsh.lsh_index import LSHIndexManager
from .algorithms.lsh.minhash import MinHashGenerator
from .algorithms.similarity.combined import CombinedSimilarityCalculator, SimilarityThresholdManager
from .algorithms.similarity.semantic import (
    ContentSimilarityCalculator,
    SemanticSimilarityCalculator,
)
from .clustering.event_grouping import EventGroupingEngine
from .config.settings import settings
from .deduplication.pipeline import DeduplicationPipeline, DeduplicationService
from .models.schemas import (
    APIError,
    DeduplicationRequest,
    DeduplicationResponse,
    EventGroupingRequest,
    EventGroupingResponse,
    HealthCheck,
    SystemMetrics,
)
from .monitoring.metrics import MetricsCollector
from .utils.cache import CacheManager
from .utils.database import DatabaseManager

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


# Global services
deduplication_service: DeduplicationService = None
event_grouping_engine: EventGroupingEngine = None
metrics_collector: MetricsCollector = None
database_manager: DatabaseManager = None
cache_manager: CacheManager = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Dict[str, Any]:
    """Application lifespan manager."""
    global deduplication_service, event_grouping_engine, metrics_collector
    global database_manager, cache_manager

    logger.info("Starting deduplication service")

    try:
        # Initialize Redis connection
        redis_client = redis.from_url(settings.redis_url)
        await redis_client.ping()
        logger.info("Connected to Redis")

        # Initialize database manager
        database_manager = DatabaseManager(settings.database_url)
        await database_manager.initialize()
        logger.info("Connected to database")

        # Initialize cache manager
        cache_manager = CacheManager(redis_client)
        await cache_manager.initialize()
        logger.info("Cache manager initialized")

        # Initialize metrics collector
        metrics_collector = MetricsCollector(redis_client)
        await metrics_collector.initialize()
        logger.info("Metrics collector initialized")

        # Initialize LSH index manager
        lsh_manager = LSHIndexManager(redis_client)

        # Initialize similarity calculators
        semantic_calc = SemanticSimilarityCalculator(
            model_name=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension)

        content_calc = ContentSimilarityCalculator(
            MinHashGenerator(num_perm=settings.lsh_num_perm))

        combined_calc = CombinedSimilarityCalculator(
            semantic_calc,
            content_calc,
            title_weight=settings.title_weight,
            content_weight=settings.content_weight,
            entity_weight=settings.entity_weight,
        )

        threshold_manager = SimilarityThresholdManager(
            default_threshold=settings.similarity_threshold,
            lsh_threshold=settings.lsh_threshold,
            content_threshold=settings.content_similarity_threshold,
            title_threshold=settings.title_similarity_threshold,
        )

        # Initialize deduplication pipeline
        pipeline = DeduplicationPipeline(
            redis_client=redis_client,
            lsh_index_manager=lsh_manager,
            similarity_calculator=combined_calc,
            threshold_manager=threshold_manager,
        )
        await pipeline.initialize()

        deduplication_service = DeduplicationService(pipeline)
        logger.info("Deduplication service initialized")

        # Initialize event grouping engine
        from .clustering.incremental_dbscan import IncrementalDBSCAN

        clustering_engine = IncrementalDBSCAN(
            eps=settings.clustering_eps,
            min_samples=settings.clustering_min_samples,
            max_cluster_size=settings.max_cluster_size,
            temporal_decay_half_life_hours=settings.temporal_decay_half_life_hours,
        )

        event_grouping_engine = EventGroupingEngine(
            clustering_engine=clustering_engine,
            semantic_calculator=semantic_calc,
            time_window_hours=24,
            min_cluster_size=settings.clustering_min_samples,
            max_cluster_size=settings.max_cluster_size,
        )
        logger.info("Event grouping engine initialized")

        logger.info("Deduplication service started successfully")

        yield

    except Exception as e:
        logger.error("Failed to start deduplication service", error=str(e))
        raise
    finally:
        # Cleanup
        logger.info("Shutting down deduplication service")
        if database_manager:
    await database_manager.close()
        if cache_manager:
    await cache_manager.close()
        logger.info("Deduplication service shut down")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="High-performance deduplication and event grouping microservice",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Dependency injection
async def get_deduplication_service() -> DeduplicationService:
    """Get deduplication service dependency."""
    if deduplication_service is None:
        raise HTTPException(status_code=503,
                            detail="Deduplication service not available")
    return deduplication_service


async def get_event_grouping_engine() -> EventGroupingEngine:
    """Get event grouping engine dependency."""
    if event_grouping_engine is None:
        raise HTTPException(status_code=503,
                            detail="Event grouping engine not available")
    return event_grouping_engine


async def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector dependency."""
    if metrics_collector is None:
        raise HTTPException(status_code=503,
                            detail="Metrics collector not available")
    return metrics_collector


# API Routes


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running"}


@app.get("/health", response_model=HealthCheck)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Check service health
        service_health = await deduplication_service.health_check()

        # Get system metrics
        metrics = await metrics_collector.get_system_metrics()

        return HealthCheck(
            status="healthy" if service_health["status"] == "healthy" else "unhealthy",
            timestamp=time.time(),
            version=settings.app_version,
            uptime=time.time(),  # Would track actual uptime
            dependencies={
                "redis": "healthy" if service_health.get("redis_connected") else "unhealthy",
                "database": "healthy",  # Would check actual DB connection
                "lsh_index": (
                    "healthy" if service_health.get(
                        "lsh_index_size", 0) > 0 else "unhealthy"
                ),
            },
            metrics=metrics,
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthCheck(
            status="unhealthy",
            timestamp=time.time(),
            version=settings.app_version,
            uptime=0,
            dependencies={"error": str(e)},
        )


@app.post("/deduplicate", response_model=DeduplicationResponse)
async def deduplicate_articles(
    request: DeduplicationRequest,
    service: DeduplicationService = Depends(get_deduplication_service),
    metrics: MetricsCollector = Depends(get_metrics_collector),
):
    """Deduplicate articles."""
    start_time = time.time()

    try:
        # Process articles
        if len(request.articles) == 1:
            results = [await service.deduplicate_article(request.articles[0])]
    else:
            results = await service.deduplicate_batch(request.articles)

        # Update metrics
        processing_time = time.time() - start_time
        await metrics.increment_counter("articles_processed", len(request.articles))
        await metrics.increment_counter(
            "duplicates_found", sum(1 for r in results if r.is_duplicate)
        )
        await metrics.record_histogram("processing_time", processing_time)

        return DeduplicationResponse(
            batch_id=request.batch_id,
            results=results,
            processing_time=processing_time,
            articles_processed=len(request.articles),
            duplicates_found=sum(1 for r in results if r.is_duplicate),
            clusters_created=0,  # Would track actual clusters
        )

    except Exception as e:
        logger.error(
            "Deduplication failed",
            error=str(e),
            batch_id=request.batch_id)
        await metrics.increment_counter("deduplication_errors")
        raise HTTPException(status_code=500,
                            detail=f"Deduplication failed: {str(e)}")


@app.post("/group-events", response_model=EventGroupingResponse)
async def group_articles_into_events(
    request: EventGroupingRequest,
    engine: EventGroupingEngine = Depends(get_event_grouping_engine),
    metrics: MetricsCollector = Depends(get_metrics_collector),
):
    """Group articles into news events."""
    start_time = time.time()

    try:
        # Group articles into events
        events = await engine.group_into_events(request.articles)

        # Separate clustered and unclustered articles
        clustered_article_ids = set()
        for event in events:
            clustered_article_ids.update(
                article.id for article in event.articles)

        unclustered_articles = [
            article for article in request.articles if article.id not in clustered_article_ids]

        # Update metrics
        processing_time = time.time() - start_time
        await metrics.increment_counter("events_created", len(events))
        await metrics.increment_counter(
            "articles_clustered", len(request.articles) - len(unclustered_articles)
        )
        await metrics.record_histogram("event_grouping_time", processing_time)

        return EventGroupingResponse(
            events=events,
            unclustered_articles=unclustered_articles,
            processing_time=processing_time,
            events_created=len(events),
            articles_clustered=len(
                request.articles) -
            len(unclustered_articles),
        )

    except Exception as e:
        logger.error("Event grouping failed", error=str(e))
        await metrics.increment_counter("event_grouping_errors")
        raise HTTPException(status_code=500,
                            detail=f"Event grouping failed: {str(e)}")


@app.get("/similarity/{article_id}")
async def find_similar_articles(
    article_id: str,
    threshold: float = 0.85,
    max_results: int = 10,
    service: DeduplicationService = Depends(get_deduplication_service),
):
    """Find similar articles for a given article."""
    try:
        # This would require getting the article from database
        # For now, return placeholder response
        return {
            "article_id": article_id,
            "similar_articles": [],
            "threshold": threshold,
            "max_results": max_results,
        }
    except Exception as e:
        logger.error(
            "Similarity search failed",
            error=str(e),
            article_id=article_id)
        raise HTTPException(status_code=500,
                            detail=f"Similarity search failed: {str(e)}")


@app.get("/metrics", response_model=SystemMetrics)
async def get_metrics(
        metrics: MetricsCollector = Depends(get_metrics_collector)):
    """Get system metrics."""
    try:
        return await metrics.get_system_metrics()
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500,
                            detail=f"Failed to get metrics: {str(e)}")


@app.get("/stats")
async def get_stats(service: DeduplicationService = Depends(
        get_deduplication_service)):
    """Get service statistics."""
    try:
        return await service.get_service_stats()
    except Exception as e:
        logger.error("Failed to get stats", error=str(e))
        raise HTTPException(status_code=500,
                            detail=f"Failed to get stats: {str(e)}")


@app.post("/reindex")
async def rebuild_indexes(
    background_tasks: BackgroundTasks,
    service: DeduplicationService = Depends(get_deduplication_service),
):
    """Rebuild LSH indexes."""
    try:
        # This would trigger a background task to rebuild indexes
        background_tasks.add_task(service.rebuild_indexes)
        return {"message": "Index rebuild started"}
    except Exception as e:
        logger.error("Index rebuild failed", error=str(e))
        raise HTTPException(status_code=500,
                            detail=f"Index rebuild failed: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc) -> Dict[str, Any]:
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=APIError(
            error=exc.detail,
            message=exc.detail,
            timestamp=time.time()).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc) -> Dict[str, Any]:
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content=APIError(
            error="Internal server error",
            message="An unexpected error occurred",
            timestamp=time.time(),
        ).dict(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
