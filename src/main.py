"""FastAPI application for the ranking microservice."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import structlog
import uvicorn

from .schemas import (
    RankingRequest, RankedResults, Article, UserProfile, 
    ABTestExperiment, RankingMetrics, HealthStatus
)
from .ranking.engine import ContentRankingEngine
from .optimization.cache import CacheManager, FeatureCache, RankingCache
from .monitoring.metrics import (
    RankingMetricsCollector, SystemMetricsCollector, 
    HealthChecker, MetricsExporter
)
from .testing.ab_framework import ABTestingFramework

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
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global variables for dependency injection
cache_manager: Optional[CacheManager] = None
ranking_engine: Optional[ContentRankingEngine] = None
metrics_collector: Optional[RankingMetricsCollector] = None
system_collector: Optional[SystemMetricsCollector] = None
health_checker: Optional[HealthChecker] = None
ab_framework: Optional[ABTestingFramework] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global cache_manager, ranking_engine, metrics_collector, system_collector, health_checker, ab_framework
    
    # Startup
    logger.info("Starting ranking microservice")
    
    try:
        # Initialize cache manager
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        cache_manager = CacheManager(redis_url=redis_url)
        
        # Initialize metrics collectors
        metrics_collector = RankingMetricsCollector()
        system_collector = SystemMetricsCollector()
        health_checker = HealthChecker(metrics_collector, system_collector)
        
        # Initialize A/B testing framework
        ab_framework = ABTestingFramework(cache_manager)
        
        # Initialize ranking engine
        ranking_engine = ContentRankingEngine(cache_manager)
        
        # Start metrics export
        metrics_exporter = MetricsExporter(metrics_collector, system_collector)
        
        # Start Prometheus metrics server
        prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8001"))
        start_http_server(prometheus_port)
        logger.info("Prometheus metrics server started", port=prometheus_port)
        
        logger.info("Ranking microservice started successfully")
        
    except Exception as e:
        logger.error("Failed to start ranking microservice", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down ranking microservice")
    
    try:
        if cache_manager:
            await cache_manager.close()
        logger.info("Ranking microservice shutdown complete")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


# Create FastAPI application
app = FastAPI(
    title="Ranking Microservice",
    description="High-performance content ranking with ML and personalization",
    version="1.0.0",
    lifespan=lifespan
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


# Dependency injection
async def get_cache_manager() -> CacheManager:
    if not cache_manager:
        raise HTTPException(status_code=500, detail="Cache manager not initialized")
    return cache_manager


async def get_ranking_engine() -> ContentRankingEngine:
    if not ranking_engine:
        raise HTTPException(status_code=500, detail="Ranking engine not initialized")
    return ranking_engine


async def get_metrics_collector() -> RankingMetricsCollector:
    if not metrics_collector:
        raise HTTPException(status_code=500, detail="Metrics collector not initialized")
    return metrics_collector


async def get_ab_framework() -> ABTestingFramework:
    if not ab_framework:
        raise HTTPException(status_code=500, detail="A/B testing framework not initialized")
    return ab_framework


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Ranking Microservice",
        "version": "1.0.0",
        "status": "running",
        "description": "High-performance content ranking with ML and personalization"
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint."""
    try:
        if not health_checker:
            raise HTTPException(status_code=500, detail="Health checker not initialized")
        
        health_status = await health_checker.check_health()
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }


@app.post("/rank", response_model=RankedResults)
async def rank_content(
    request: RankingRequest,
    background_tasks: BackgroundTasks,
    engine: ContentRankingEngine = Depends(get_ranking_engine),
    metrics: RankingMetricsCollector = Depends(get_metrics_collector)
):
    """Rank content based on request parameters."""
    try:
        logger.info("Ranking request received", user_id=request.user_id, limit=request.limit)
        
        # Perform ranking
        results = await engine.rank_content(request)
        
        # Log ranking decision for analysis
        background_tasks.add_task(
            log_ranking_decision,
            request.user_id,
            request.dict(),
            results.dict()
        )
        
        logger.info("Ranking completed", 
                   user_id=request.user_id, 
                   article_count=len(results.articles),
                   processing_time_ms=results.processing_time_ms)
        
        return results
        
    except Exception as e:
        logger.error("Ranking failed", error=str(e), user_id=request.user_id)
        await metrics.record_error("ranking_failed")
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")


@app.get("/articles/{article_id}", response_model=Article)
async def get_article(article_id: str):
    """Get article by ID."""
    try:
        # In production, this would query a content database
        # For now, return a dummy article
        from .schemas import Article, Source, Author, ContentType
        
        article = Article(
            id=article_id,
            title=f"Article {article_id}",
            content=f"This is the content for article {article_id}",
            url=f"https://example.com/articles/{article_id}",
            published_at="2024-01-01T00:00:00Z",
            source=Source(
                id="source_1",
                name="Example Source",
                domain="example.com"
            ),
            author=Author(
                id="author_1",
                name="Example Author"
            ),
            word_count=500,
            reading_time=2,
            quality_score=0.8
        )
        
        return article
        
    except Exception as e:
        logger.error("Failed to get article", error=str(e), article_id=article_id)
        raise HTTPException(status_code=404, detail="Article not found")


@app.get("/users/{user_id}/profile", response_model=UserProfile)
async def get_user_profile(
    user_id: str,
    cache: CacheManager = Depends(get_cache_manager)
):
    """Get user profile for personalization."""
    try:
        # Get user profile from cache
        profile_data = await cache.get(f"user_profile:{user_id}")
        if not profile_data:
            # Create default profile
            from .schemas import UserProfile
            profile = UserProfile(
                user_id=user_id,
                topic_preferences={},
                source_preferences={},
                reading_patterns={},
                content_preferences={},
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z"
            )
        else:
            profile = UserProfile(**profile_data)
        
        return profile
        
    except Exception as e:
        logger.error("Failed to get user profile", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Failed to get user profile")


@app.put("/users/{user_id}/profile", response_model=Dict[str, str])
async def update_user_profile(
    user_id: str,
    profile: UserProfile,
    cache: CacheManager = Depends(get_cache_manager)
):
    """Update user profile."""
    try:
        # Update profile in cache
        await cache.set(f"user_profile:{user_id}", profile.dict(), ttl=3600)
        
        logger.info("User profile updated", user_id=user_id)
        return {"message": "Profile updated successfully"}
        
    except Exception as e:
        logger.error("Failed to update user profile", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Failed to update user profile")


@app.get("/experiments", response_model=List[Dict[str, Any]])
async def get_experiments(ab_framework: ABTestingFramework = Depends(get_ab_framework)):
    """Get all A/B test experiments."""
    try:
        experiments = await ab_framework.get_all_experiments()
        return experiments
        
    except Exception as e:
        logger.error("Failed to get experiments", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get experiments")


@app.get("/experiments/{experiment_id}/stats", response_model=Dict[str, Any])
async def get_experiment_stats(
    experiment_id: str,
    ab_framework: ABTestingFramework = Depends(get_ab_framework)
):
    """Get experiment statistics."""
    try:
        stats = await ab_framework.get_experiment_stats(experiment_id)
        return stats
        
    except Exception as e:
        logger.error("Failed to get experiment stats", error=str(e), experiment_id=experiment_id)
        raise HTTPException(status_code=500, detail="Failed to get experiment stats")


@app.post("/experiments", response_model=Dict[str, str])
async def create_experiment(
    experiment: ABTestExperiment,
    ab_framework: ABTestingFramework = Depends(get_ab_framework)
):
    """Create a new A/B test experiment."""
    try:
        success = await ab_framework.create_experiment(experiment)
        if success:
            return {"message": "Experiment created successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to create experiment")
        
    except Exception as e:
        logger.error("Failed to create experiment", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create experiment")


@app.get("/metrics/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    time_window: int = Query(60, ge=1, le=1440, description="Time window in minutes"),
    metrics: RankingMetricsCollector = Depends(get_metrics_collector)
):
    """Get performance metrics."""
    try:
        summary = metrics.get_performance_summary(time_window)
        return summary
        
    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")


@app.get("/metrics/algorithm-comparison", response_model=Dict[str, Any])
async def get_algorithm_comparison(
    metrics: RankingMetricsCollector = Depends(get_metrics_collector)
):
    """Get algorithm performance comparison."""
    try:
        comparison = metrics.get_algorithm_comparison()
        return comparison
        
    except Exception as e:
        logger.error("Failed to get algorithm comparison", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get algorithm comparison")


@app.get("/metrics/system", response_model=Dict[str, Any])
async def get_system_metrics(
    time_window: int = Query(60, ge=1, le=1440, description="Time window in minutes"),
    system_collector: SystemMetricsCollector = Depends(lambda: system_collector)
):
    """Get system metrics."""
    try:
        if not system_collector:
            raise HTTPException(status_code=500, detail="System collector not initialized")
        
        summary = system_collector.get_system_summary(time_window)
        return summary
        
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get system metrics")


@app.get("/cache/stats", response_model=Dict[str, Any])
async def get_cache_stats(cache: CacheManager = Depends(get_cache_manager)):
    """Get cache statistics."""
    try:
        stats = cache.get_stats()
        return stats
        
    except Exception as e:
        logger.error("Failed to get cache stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get cache stats")


@app.post("/cache/clear", response_model=Dict[str, str])
async def clear_cache(cache: CacheManager = Depends(get_cache_manager)):
    """Clear cache."""
    try:
        # In production, this would be more selective
        # For now, just clear stats
        await cache.clear_stats()
        
        logger.info("Cache cleared")
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error("Failed to clear cache", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to clear cache")


# Background tasks

async def log_ranking_decision(user_id: str, request: Dict[str, Any], results: Dict[str, Any]):
    """Log ranking decision for analysis."""
    try:
        # In production, this would log to a data warehouse
        logger.info("Ranking decision logged", 
                   user_id=user_id,
                   request_id=request.get("request_id"),
                   algorithm_variant=results.get("algorithm_variant"),
                   article_count=len(results.get("articles", [])))
    except Exception as e:
        logger.error("Failed to log ranking decision", error=str(e))


# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.error("HTTP exception", status_code=exc.status_code, detail=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    # Start server
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False
    )