"""
AI-Powered Summarization & Headline Generation Microservice
FastAPI application with Vertex AI and advanced NLP models
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from headline_generation.generator import HeadlineGenerator
from pydantic import BaseModel, Field
from quality_validation.validator import SummaryQualityValidator
from summarization.engine import ContentSummarizationEngine
from summarization.models import (
    BatchSummarizationRequest,
    SummarizationRequest,
    SummarizationResponse,
    SummaryResult,
)

from config.settings import Settings
from monitoring.metrics import MetricsCollector
from optimization.cache import SummaryCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
settings = Settings()
summarization_engine = None
headline_generator = None
quality_validator = None
metrics_collector = None
summary_cache = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Dict[str, Any]:
    """Application lifespan management"""
    global summarization_engine, headline_generator, quality_validator, metrics_collector, summary_cache

    logger.info("Starting Summarization Service...")

    # Initialize components
    try:
        summarization_engine = ContentSummarizationEngine()
        headline_generator = HeadlineGenerator()
        quality_validator = SummaryQualityValidator()
        metrics_collector = MetricsCollector()
        summary_cache = SummaryCache()

        # Warm up models
        await summarization_engine.warm_up()
        await headline_generator.warm_up()

        logger.info("Service initialization completed successfully")

    except Exception as e:
        logger.error(f"Failed to initialize service: {str(e)}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down Summarization Service...")
    if summarization_engine:
    await summarization_engine.cleanup()
    if headline_generator:
    await headline_generator.cleanup()


# Create FastAPI application
app = FastAPI(
    title="AI Summarization & Headline Generation Service",
    description="Advanced AI-powered content summarization with quality validation and multi-language support",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Health check endpoints
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "summarization-service",
        "version": "1.0.0"}


@app.get("/health/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check endpoint"""
    if not all([summarization_engine, headline_generator, quality_validator]):
        raise HTTPException(status_code=503, detail="Service not ready")

    return {"status": "ready"}


# Main summarization endpoint
@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_content(
        request: SummarizationRequest,
        background_tasks: BackgroundTasks):
     -> Dict[str, Any]:"""
    Generate AI-powered summary and headlines for content

    Supports:
    - Extractive and abstractive summarization
    - Multiple summary lengths (50, 120, 300 words)
    - Dynamic headline generation with A/B testing
    - Quality validation and bias detection
    - Multi-language support
    """
    try:
        # Check cache first
        cache_key = summary_cache.generate_key(request)
        cached_result = await summary_cache.get(cache_key)
        if cached_result:
            metrics_collector.increment_cache_hit()
            return cached_result

        # Generate summary
        result = await summarization_engine.generate_summary(request)

        # Cache result
        await summary_cache.set(cache_key, result, ttl=3600)

        # Record metrics
        metrics_collector.record_summarization_metrics(result)

        # Background quality validation
        background_tasks.add_task(
            quality_validator.validate_summary_quality_async,
            request.content.text,
            result.summary.text,
        )

        return SummarizationResponse(
            success=True,
            result=result,
            processing_time=result.processing_stats.get("total_time", 0),
        )

    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        metrics_collector.increment_error()
        raise HTTPException(status_code=500,
                            detail=f"Summarization failed: {str(e)}")


@app.post("/summarize/batch", response_model=List[SummarizationResponse])
async def batch_summarize(
        request: BatchSummarizationRequest,
        background_tasks: BackgroundTasks):
     -> Dict[str, Any]:"""
    Batch summarization for multiple content pieces

    Optimized for processing multiple articles simultaneously
    with parallel processing and GPU batching
    """
    try:
        # Process in parallel with batching
        tasks = []
        for content_request in request.requests:
            task = summarization_engine.generate_summary(content_request)
            tasks.append(task)

        # Execute with controlled concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {i} failed: {str(result)}")
                responses.append(
                    SummarizationResponse(
                        success=False,
                        error=str(result)))
            else:
                responses.append(
                    SummarizationResponse(
                        success=True, result=result))

        # Background metrics collection
        background_tasks.add_task(
            metrics_collector.record_batch_metrics,
            len(request.requests),
            len([r for r in responses if r.success]),
        )

        return responses

    except Exception as e:
        logger.error(f"Batch summarization failed: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Batch processing failed: {str(e)}")


@app.post("/headlines/generate")
async def generate_headlines(
        content: str,
        styles: List[str] = [
            "news",
            "engaging",
            "neutral"],
        num_variants: int = 5):
     -> Dict[str, Any]:"""
    Generate multiple headline variants with different styles

    Styles: news, engaging, question, neutral, urgent
    """
    try:
        headlines = await headline_generator.generate_headlines(
            content=content, styles=styles, num_variants=num_variants
        )

        metrics_collector.increment_headline_generation()

        return {"headlines": headlines, "count": len(headlines)}

    except Exception as e:
        logger.error(f"Headline generation failed: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Headline generation failed: {str(e)}")


@app.post("/quality/validate")
async def validate_quality(original: str, summary: str) -> Dict[str, Any]:
    """
    Validate summary quality with comprehensive metrics

    Returns ROUGE scores, BERTScore, factual consistency, and more
    """
    try:
        quality_metrics = await quality_validator.validate_summary_quality(original, summary)

        return {"quality_metrics": quality_metrics,
                "overall_score": quality_metrics.overall_score}

    except Exception as e:
        logger.error(f"Quality validation failed: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Quality validation failed: {str(e)}")


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get service metrics and performance statistics"""
    try:
        metrics = await metrics_collector.get_metrics()
        return metrics

    except Exception as e:
        logger.error(f"Failed to retrieve metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve metrics")


@app.get("/models/status")
async def get_models_status() -> Dict[str, Any]:
    """Get status of loaded ML models"""
    try:
        status = {
            "summarization_engine":
    await summarization_engine.get_status(),
            "headline_generator":
    await headline_generator.get_status(),
            "quality_validator":
    await quality_validator.get_status(),
        }
        return status

    except Exception as e:
        logger.error(f"Failed to get model status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get model status")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info")
