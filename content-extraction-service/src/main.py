"""
Main FastAPI application for the Content Extraction & Cleanup microservice.
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

from src.config.settings import settings
from src.models.api import (
    ContentExtractionRequest,
    ContentExtractionResponse,
    ExtractionBatchRequest,
    ExtractionBatchResponse,
    HealthCheckResponse,
    MetricsResponse,
    QualityAnalysisRequest,
    QualityAnalysisResponse,
)
from src.models.content import ContentType, ProcessingStatus
from src.services.content_extraction_service import ContentExtractionService
from src.services.cache_service import CacheService
from src.services.cloud_tasks_service import CloudTasksService
from src.services.document_ai_service import DocumentAIService
from src.utils.html_cleaner import HTMLCleaner
from src.utils.image_processor import ImageProcessor
from src.utils.metadata_extractor import MetadataExtractor
from src.utils.quality_analyzer import QualityAnalyzer
from src.utils.readability_extractor import ReadabilityExtractor

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global services
content_extraction_service: Optional[ContentExtractionService] = None
cache_service: Optional[CacheService] = None
cloud_tasks_service: Optional[CloudTasksService] = None
document_ai_service: Optional[DocumentAIService] = None
html_cleaner: Optional[HTMLCleaner] = None
image_processor: Optional[ImageProcessor] = None
metadata_extractor: Optional[MetadataExtractor] = None
quality_analyzer: Optional[QualityAnalyzer] = None
readability_extractor: Optional[ReadabilityExtractor] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Dict[str, Any]:
    """Application lifespan manager."""
    global (
        content_extraction_service,
        cache_service,
        cloud_tasks_service,
        document_ai_service,
        html_cleaner,
        image_processor,
        metadata_extractor,
        quality_analyzer,
        readability_extractor,
    )

    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    try:
        # Initialize utility services
        html_cleaner = HTMLCleaner()
        image_processor = ImageProcessor()
        metadata_extractor = MetadataExtractor()
        quality_analyzer = QualityAnalyzer()
        readability_extractor = ReadabilityExtractor()

        # Initialize core services
        cache_service = CacheService()
        cloud_tasks_service = CloudTasksService()
        document_ai_service = DocumentAIService()

        # Initialize main service
        content_extraction_service = ContentExtractionService(
            cache_service=cache_service,
            cloud_tasks_service=cloud_tasks_service,
            document_ai_service=document_ai_service,
            html_cleaner=html_cleaner,
            image_processor=image_processor,
            metadata_extractor=metadata_extractor,
            quality_analyzer=quality_analyzer,
            readability_extractor=readability_extractor,
        )

        # Initialize services
        await cache_service.initialize()
        await cloud_tasks_service.initialize()
        await document_ai_service.initialize()
        await content_extraction_service.initialize()

        logger.info("Application startup complete")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down application")

    try:
        if content_extraction_service:
            await content_extraction_service.shutdown()

        if cache_service:
            await cache_service.shutdown()

        if cloud_tasks_service:
            await cloud_tasks_service.shutdown()

        if document_ai_service:
            await document_ai_service.shutdown()

        logger.info("Application shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered content extraction and cleanup microservice",
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
def get_content_extraction_service() -> ContentExtractionService:
    if not content_extraction_service:
        raise HTTPException(status_code=503, detail="Content extraction service not available")
    return content_extraction_service


def get_cache_service() -> CacheService:
    if not cache_service:
        raise HTTPException(status_code=503, detail="Cache service not available")
    return cache_service


def get_quality_analyzer() -> QualityAnalyzer:
    if not quality_analyzer:
        raise HTTPException(status_code=503, detail="Quality analyzer not available")
    return quality_analyzer


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["Monitoring"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Check service dependencies
        dependencies = {
            "content_extraction_service": "healthy" if content_extraction_service else "unhealthy",
            "cache_service": "healthy" if cache_service else "unhealthy",
            "cloud_tasks_service": "healthy" if cloud_tasks_service else "unhealthy",
            "document_ai_service": "healthy" if document_ai_service else "unhealthy",
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


# Content extraction endpoints
@app.post("/extract", response_model=ContentExtractionResponse, tags=["Extraction"])
async def extract_content(
    request: ContentExtractionRequest,
    background_tasks: BackgroundTasks,
    extraction_svc: ContentExtractionService = Depends(get_content_extraction_service),
) -> Dict[str, Any]:
    """Extract and clean content from raw HTML or text."""
    try:
        start_time = time.time()

        # Extract content
        result = await extraction_svc.extract_content(
            url=request.url,
            html_content=request.html_content,
            text_content=request.text_content,
            content_type=request.content_type,
            language=request.language,
            extract_images=request.extract_images,
            extract_metadata=request.extract_metadata,
            quality_threshold=request.quality_threshold,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return ContentExtractionResponse(
            content_id=result.content_id,
            title=result.title,
            content=result.content,
            summary=result.summary,
            language=result.language,
            content_type=result.content_type,
            word_count=result.word_count,
            reading_time=result.reading_time,
            quality_score=result.quality_score,
            images=result.images,
            metadata=result.metadata,
            processing_time_ms=processing_time_ms,
            status=result.status,
        )

    except Exception as e:
        logger.error(f"Error extracting content: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting content: {str(e)}")


@app.post("/extract/batch", response_model=ExtractionBatchResponse, tags=["Extraction"])
async def extract_content_batch(
    request: ExtractionBatchRequest,
    background_tasks: BackgroundTasks,
    extraction_svc: ContentExtractionService = Depends(get_content_extraction_service),
) -> Dict[str, Any]:
    """Extract content from multiple sources in batch."""
    try:
        start_time = time.time()

        # Process batch
        batch_result = await extraction_svc.extract_content_batch(
            items=request.items,
            batch_id=request.batch_id,
            parallel_workers=request.parallel_workers,
            quality_threshold=request.quality_threshold,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return ExtractionBatchResponse(
            batch_id=batch_result.batch_id,
            total_items=batch_result.total_items,
            processed_items=batch_result.processed_items,
            failed_items=batch_result.failed_items,
            processing_time_ms=processing_time_ms,
            status=batch_result.status,
            results=batch_result.results,
            errors=batch_result.errors,
        )

    except Exception as e:
        logger.error(f"Error processing batch extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")


@app.post("/analyze/quality", response_model=QualityAnalysisResponse, tags=["Analysis"])
async def analyze_content_quality(
    request: QualityAnalysisRequest,
    analyzer: QualityAnalyzer = Depends(get_quality_analyzer),
) -> Dict[str, Any]:
    """Analyze content quality and readability."""
    try:
        start_time = time.time()

        # Analyze quality
        analysis = await analyzer.analyze_content(
            content=request.content,
            title=request.title,
            language=request.language,
            content_type=request.content_type,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return QualityAnalysisResponse(
            content_id=analysis.content_id,
            quality_score=analysis.quality_score,
            readability_score=analysis.readability_score,
            word_count=analysis.word_count,
            sentence_count=analysis.sentence_count,
            paragraph_count=analysis.paragraph_count,
            avg_sentence_length=analysis.avg_sentence_length,
            flesch_kincaid_grade=analysis.flesch_kincaid_grade,
            sentiment_score=analysis.sentiment_score,
            language_confidence=analysis.language_confidence,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Error analyzing content quality: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing quality: {str(e)}")


@app.get("/extract/{content_id}", tags=["Extraction"])
async def get_extracted_content(
    content_id: str,
    extraction_svc: ContentExtractionService = Depends(get_content_extraction_service),
) -> Dict[str, Any]:
    """Get previously extracted content by ID."""
    try:
        content = await extraction_svc.get_extracted_content(content_id)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        return {
            "content_id": content.content_id,
            "title": content.title,
            "content": content.content,
            "summary": content.summary,
            "language": content.language,
            "content_type": content.content_type.value,
            "word_count": content.word_count,
            "reading_time": content.reading_time,
            "quality_score": content.quality_score,
            "images": content.images,
            "metadata": content.metadata,
            "created_at": content.created_at.isoformat() if content.created_at else None,
            "updated_at": content.updated_at.isoformat() if content.updated_at else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting content {content_id}: {e}")
        raise HTTPException(status_code=500, detail="Error getting content")


@app.delete("/extract/{content_id}", tags=["Extraction"])
async def delete_extracted_content(
    content_id: str,
    extraction_svc: ContentExtractionService = Depends(get_content_extraction_service),
) -> Dict[str, Any]:
    """Delete extracted content by ID."""
    try:
        success = await extraction_svc.delete_extracted_content(content_id)
        if not success:
            raise HTTPException(status_code=404, detail="Content not found")

        return {"message": "Content deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting content {content_id}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting content")


@app.get("/extract", tags=["Extraction"])
async def list_extracted_content(
    limit: int = Query(20, ge=1, le=100, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    content_type: Optional[ContentType] = Query(None, description="Filter by content type"),
    language: Optional[str] = Query(None, description="Filter by language"),
    min_quality: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum quality score"),
    extraction_svc: ContentExtractionService = Depends(get_content_extraction_service),
) -> Dict[str, Any]:
    """List extracted content with optional filters."""
    try:
        results = await extraction_svc.list_extracted_content(
            limit=limit,
            offset=offset,
            content_type=content_type,
            language=language,
            min_quality=min_quality,
        )

        return {
            "items": [
                {
                    "content_id": item.content_id,
                    "title": item.title,
                    "summary": item.summary,
                    "language": item.language,
                    "content_type": item.content_type.value,
                    "word_count": item.word_count,
                    "quality_score": item.quality_score,
                    "created_at": item.created_at.isoformat() if item.created_at else None,
                }
                for item in results.items
            ],
            "total_count": results.total_count,
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.error(f"Error listing content: {e}")
        raise HTTPException(status_code=500, detail="Error listing content")


# Cache management endpoints
@app.get("/cache/stats", tags=["Cache"])
async def get_cache_stats(cache_svc: CacheService = Depends(get_cache_service)) -> Dict[str, Any]:
    """Get cache statistics."""
    try:
        stats = await cache_svc.get_stats()
        return {
            "hit_count": stats.hit_count,
            "miss_count": stats.miss_count,
            "total_requests": stats.total_requests,
            "hit_rate": stats.hit_rate,
            "cache_size": stats.cache_size,
            "memory_usage": stats.memory_usage,
        }

    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail="Error getting cache stats")


@app.post("/cache/clear", tags=["Cache"])
async def clear_cache(cache_svc: CacheService = Depends(get_cache_service)) -> Dict[str, Any]:
    """Clear all cache entries."""
    try:
        await cache_svc.clear_all()
        return {"message": "Cache cleared successfully"}

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Error clearing cache")


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
