"""Main FastAPI application for Content Enrichment Service."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import structlog
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .enrichment import ContentEnrichmentPipeline
from .middleware.auth import AuthMiddleware
from .middleware.rate_limiter import RateLimiter
from .models import (
    BatchEnrichmentRequest,
    BatchEnrichmentResponse,
    EnrichmentRequest,
    EnrichmentResponse,
    ExtractedContent,
)
from .monitoring.health import HealthChecker
from .monitoring.metrics import MetricsCollector

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

# Global pipeline instance
enrichment_pipeline = None
rate_limiter = None
metrics_collector = None
health_checker = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Dict[str, Any]:
    """Application lifespan manager."""
    global enrichment_pipeline, rate_limiter, metrics_collector, health_checker

    # Startup
    logger.info("Starting Content Enrichment Service")

    try:
        # Initialize components
        enrichment_pipeline = ContentEnrichmentPipeline()
        rate_limiter = RateLimiter()
        metrics_collector = MetricsCollector()
        health_checker = HealthChecker()

        # Initialize health checker
        await health_checker.initialize()

        logger.info("Content Enrichment Service started successfully")

        yield

    except Exception as e:
        logger.error(
            "Failed to start Content Enrichment Service",
            error=str(e))
        raise

    finally:
        # Shutdown
        logger.info("Shutting down Content Enrichment Service")

        if health_checker:
    await health_checker.cleanup()

        logger.info("Content Enrichment Service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered Content Enrichment Service with NER, Topic Classification, Sentiment Analysis, and Trustworthiness Scoring",
    lifespan=lifespan,
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

# Add custom middleware
if settings.enable_metrics:
    app.add_middleware(metrics_collector.middleware_class())


# Rate limiting dependency
async def get_rate_limiter() -> Dict[str, Any]:
    """Get rate limiter instance."""
    return rate_limiter


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        if not health_checker:
            raise HTTPException(
                status_code=503,
                detail="Health checker not initialized")

        health_status = await health_checker.check_health()

        if health_status["status"] == "healthy":
            return JSONResponse(status_code=200, content=health_status)
        else:
            return JSONResponse(status_code=503, content=health_status)

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)})


# Metrics endpoint
@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get service metrics."""
    try:
        if not metrics_collector:
            raise HTTPException(status_code=503,
                                detail="Metrics collector not initialized")

        metrics = await metrics_collector.get_metrics()
        return JSONResponse(content=metrics)

    except Exception as e:
        logger.error("Metrics retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with service information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "enrich": "/api/v1/enrich",
            "enrich-batch": "/api/v1/enrich/batch",
            "docs": "/docs",
        },
    }


# Single content enrichment endpoint
@app.post("/api/v1/enrich", response_model=EnrichmentResponse)
async def enrich_content(
    request: EnrichmentRequest,
    background_tasks: BackgroundTasks,
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
):
    """Enrich a single piece of content."""
    try:
        # Rate limiting
        if not await rate_limiter.check_rate_limit(request.content.id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Validate content
        if not request.content.content.strip():
            raise HTTPException(
                status_code=400,
                detail="Content cannot be empty")

        # Enrich content
        start_time = asyncio.get_event_loop().time()

        enriched_content = await enrichment_pipeline.enrich_content(
            content=request.content,
            processing_mode=request.processing_mode,
            include_entities=request.include_entities,
            include_topics=request.include_topics,
            include_sentiment=request.include_sentiment,
            include_signals=request.include_signals,
            include_trust_score=request.include_trust_score,
            language_hint=request.language_hint,
        )

        processing_time = int(
            (asyncio.get_event_loop().time() - start_time) * 1000)

        # Record metrics
        if metrics_collector:
            background_tasks.add_task(
                metrics_collector.record_enrichment,
                processing_time=processing_time,
                success=True,
                entities_count=len(enriched_content.entities),
                topics_count=len(enriched_content.topics),
            )

        return EnrichmentResponse(
            success=True,
            enriched_content=enriched_content,
            processing_time_ms=processing_time,
            model_versions_used={
                name: version.version for name,
                version in enriched_content.model_versions.items()},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Content enrichment failed",
            content_id=request.content.id,
            error=str(e),
            exc_info=True)

        # Record error metrics
        if metrics_collector:
            background_tasks.add_task(
                metrics_collector.record_enrichment,
                processing_time=0,
                success=False,
                entities_count=0,
                topics_count=0,
            )

        raise HTTPException(status_code=500,
                            detail=f"Content enrichment failed: {str(e)}")


# Batch content enrichment endpoint
@app.post("/api/v1/enrich/batch", response_model=BatchEnrichmentResponse)
async def enrich_content_batch(
    request: BatchEnrichmentRequest,
    background_tasks: BackgroundTasks,
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
):
    """Enrich multiple pieces of content in batch."""
    try:
        # Rate limiting for batch requests
        batch_id = f"batch_{len(request.contents)}"
        if not await rate_limiter.check_rate_limit(batch_id):
            raise HTTPException(status_code=429,
                                detail="Rate limit exceeded for batch request")

        # Validate contents
        if not request.contents:
            raise HTTPException(
                status_code=400,
                detail="No content provided for batch processing")

        if len(request.contents) > 100:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Batch size too large. Maximum 100 items per batch.")

        # Enrich contents
        start_time = asyncio.get_event_loop().time()

        enriched_contents = await enrichment_pipeline.enrich_batch(
            contents=request.contents,
            processing_mode=request.processing_mode,
            include_entities=request.include_entities,
            include_topics=request.include_topics,
            include_sentiment=request.include_sentiment,
            include_signals=request.include_signals,
            include_trust_score=request.include_trust_score,
        )

        total_processing_time = int(
            (asyncio.get_event_loop().time() - start_time) * 1000)

        # Record metrics
        if metrics_collector:
            background_tasks.add_task(
                metrics_collector.record_batch_enrichment,
                batch_size=len(request.contents),
                processing_time=total_processing_time,
                success_count=len(enriched_contents),
            )

        return BatchEnrichmentResponse(
            success=True,
            enriched_contents=enriched_contents,
            total_processing_time_ms=total_processing_time,
            model_versions_used=(
                {
                    name: version.version
                    for name, version in enriched_contents[0].model_versions.items()
                }
                if enriched_contents
                else {}
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Batch content enrichment failed",
            batch_size=len(request.contents),
            error=str(e),
            exc_info=True,
        )

        # Record error metrics
        if metrics_collector:
            background_tasks.add_task(
                metrics_collector.record_batch_enrichment,
                batch_size=len(request.contents),
                processing_time=0,
                success_count=0,
            )

        raise HTTPException(
            status_code=500,
            detail=f"Batch content enrichment failed: {str(e)}")


# Service information endpoint
@app.get("/api/v1/info")
async def get_service_info() -> Dict[str, Any]:
    """Get service information and capabilities."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "capabilities": {
            "named_entity_recognition": True,
            "topic_classification": True,
            "sentiment_analysis": True,
            "content_signals": True,
            "trustworthiness_scoring": True,
            "entity_linking": True,
            "emotion_detection": True,
            "bias_detection": True,
            "readability_scoring": True,
            "fact_checking": True,
        },
        "supported_languages": [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ru",
            "zh",
            "ja",
            "ko"],
        "processing_modes": [
            "realtime",
            "batch"],
        "max_content_length": settings.max_text_length,
        "rate_limits": {
            "requests_per_hour": settings.rate_limit_requests,
            "max_batch_size": 100},
    }


# Model information endpoint
@app.get("/api/v1/models")
async def get_model_info() -> Dict[str, Any]:
    """Get information about loaded models."""
    try:
        if not enrichment_pipeline:
            raise HTTPException(status_code=503,
                                detail="Enrichment pipeline not initialized")

        model_info = {}
        for name, version in enrichment_pipeline.model_versions.items():
            model_info[name] = {
                "name": version.name,
                "version": version.version,
                "created_at": version.created_at.isoformat(),
                "performance_metrics": version.performance_metrics,
            }

        return model_info

    except Exception as e:
        logger.error("Model info retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info",
        reload=settings.debug,
    )
