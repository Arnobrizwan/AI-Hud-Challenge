"""
Storage, Indexing & Retrieval Microservice
High-performance polyglot persistence with vector search and intelligent caching
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from storage_orchestrator import StorageOrchestrator
from cache_management import CacheCoordinator
from data_lifecycle import DataLifecycleManager
from query_optimization import QueryOptimizer
from models import (
    Article, StorageResult, RetrievedArticle, SearchRequest, SearchResult,
    SimilaritySearchParams, SimilaritySearchResult, CacheConfig, CacheResult,
    GDPRRequest, GDPRResponse, RetentionResult, HealthCheck
)
from middleware import (
    RequestLoggingMiddleware, PerformanceMiddleware, 
    ErrorHandlingMiddleware, RateLimitMiddleware
)
from monitoring import MetricsCollector, HealthChecker
from config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
storage_orchestrator: Optional[StorageOrchestrator] = None
cache_coordinator: Optional[CacheCoordinator] = None
data_lifecycle_manager: Optional[DataLifecycleManager] = None
query_optimizer: Optional[QueryOptimizer] = None
metrics_collector: Optional[MetricsCollector] = None
health_checker: Optional[HealthChecker] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global storage_orchestrator, cache_coordinator, data_lifecycle_manager
    global query_optimizer, metrics_collector, health_checker
    
    logger.info("Starting Storage Service...")
    
    try:
        # Initialize core components
        settings = Settings()
        
        storage_orchestrator = StorageOrchestrator()
        await storage_orchestrator.initialize()
        
        cache_coordinator = CacheCoordinator()
        await cache_coordinator.initialize()
        
        data_lifecycle_manager = DataLifecycleManager()
        await data_lifecycle_manager.initialize()
        
        query_optimizer = QueryOptimizer()
        await query_optimizer.initialize()
        
        metrics_collector = MetricsCollector()
        await metrics_collector.initialize()
        
        health_checker = HealthChecker()
        await health_checker.initialize()
        
        # Start background tasks
        asyncio.create_task(data_lifecycle_manager.start_retention_scheduler())
        asyncio.create_task(metrics_collector.start_metrics_collection())
        
        logger.info("Storage Service started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Storage Service: {e}")
        raise
    finally:
        logger.info("Shutting down Storage Service...")
        
        # Cleanup resources
        if storage_orchestrator:
            await storage_orchestrator.cleanup()
        if cache_coordinator:
            await cache_coordinator.cleanup()
        if data_lifecycle_manager:
            await data_lifecycle_manager.cleanup()
        if query_optimizer:
            await query_optimizer.cleanup()
        if metrics_collector:
            await metrics_collector.cleanup()
        if health_checker:
            await health_checker.cleanup()
        
        logger.info("Storage Service shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Storage, Indexing & Retrieval Service",
    description="High-performance polyglot persistence with vector search and intelligent caching",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(PerformanceMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RateLimitMiddleware)

# Dependency injection
async def get_storage_orchestrator() -> StorageOrchestrator:
    if not storage_orchestrator:
        raise HTTPException(status_code=503, detail="Storage orchestrator not available")
    return storage_orchestrator

async def get_cache_coordinator() -> CacheCoordinator:
    if not cache_coordinator:
        raise HTTPException(status_code=503, detail="Cache coordinator not available")
    return cache_coordinator

async def get_data_lifecycle_manager() -> DataLifecycleManager:
    if not data_lifecycle_manager:
        raise HTTPException(status_code=503, detail="Data lifecycle manager not available")
    return data_lifecycle_manager

async def get_query_optimizer() -> QueryOptimizer:
    if not query_optimizer:
        raise HTTPException(status_code=503, detail="Query optimizer not available")
    return query_optimizer

# Health check endpoints
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not available")
    
    health_status = await health_checker.check_health()
    return health_status

@app.get("/health/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    if not storage_orchestrator or not cache_coordinator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready"}

@app.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes"""
    return {"status": "alive"}

# Storage endpoints
@app.post("/storage/articles", response_model=StorageResult)
async def store_article(
    article: Article,
    background_tasks: BackgroundTasks,
    storage: StorageOrchestrator = Depends(get_storage_orchestrator)
):
    """Store article across multiple data stores"""
    try:
        result = await storage.store_article(article)
        
        # Schedule background tasks
        background_tasks.add_task(
            storage.update_search_indexes, article.id
        )
        background_tasks.add_task(
            storage.update_cache_warmup, article.id
        )
        
        return result
    except Exception as e:
        logger.error(f"Failed to store article {article.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/storage/articles/{article_id}", response_model=RetrievedArticle)
async def retrieve_article(
    article_id: str,
    use_cache: bool = True,
    update_cache: bool = True,
    storage: StorageOrchestrator = Depends(get_storage_orchestrator)
):
    """Retrieve article from optimal data stores"""
    try:
        from models import RetrievalOptions
        
        retrieval_options = RetrievalOptions(
            use_cache=use_cache,
            update_cache=update_cache
        )
        
        article = await storage.retrieve_article(article_id, retrieval_options)
        return article
    except Exception as e:
        logger.error(f"Failed to retrieve article {article_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search endpoints
@app.post("/search/fulltext", response_model=SearchResult)
async def fulltext_search(
    search_request: SearchRequest,
    storage: StorageOrchestrator = Depends(get_storage_orchestrator)
):
    """Advanced full-text search"""
    try:
        result = await storage.search_articles(search_request)
        return result
    except Exception as e:
        logger.error(f"Full-text search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/similarity", response_model=SimilaritySearchResult)
async def similarity_search(
    search_params: SimilaritySearchParams,
    storage: StorageOrchestrator = Depends(get_storage_orchestrator)
):
    """Vector similarity search"""
    try:
        result = await storage.similarity_search(search_params)
        return result
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache management endpoints
@app.post("/cache/invalidate")
async def invalidate_cache(
    cache_keys: list[str],
    cache: CacheCoordinator = Depends(get_cache_coordinator)
):
    """Invalidate cache entries"""
    try:
        from models import CacheInvalidationRequest
        
        invalidation_request = CacheInvalidationRequest(
            key_patterns=cache_keys
        )
        
        result = await cache.invalidate_cache(invalidation_request)
        return result
    except Exception as e:
        logger.error(f"Cache invalidation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats")
async def cache_stats(
    cache: CacheCoordinator = Depends(get_cache_coordinator)
):
    """Get cache statistics"""
    try:
        stats = await cache.get_cache_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data lifecycle endpoints
@app.post("/lifecycle/retention/apply")
async def apply_retention_policies(
    background_tasks: BackgroundTasks,
    lifecycle: DataLifecycleManager = Depends(get_data_lifecycle_manager)
):
    """Apply data retention policies"""
    try:
        result = await lifecycle.apply_retention_policies()
        return result
    except Exception as e:
        logger.error(f"Retention policy application failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/lifecycle/gdpr", response_model=GDPRResponse)
async def handle_gdpr_request(
    gdpr_request: GDPRRequest,
    background_tasks: BackgroundTasks,
    lifecycle: DataLifecycleManager = Depends(get_data_lifecycle_manager)
):
    """Handle GDPR data requests"""
    try:
        result = await lifecycle.handle_gdpr_request(gdpr_request)
        
        # Log GDPR request for audit
        background_tasks.add_task(
            lifecycle.log_gdpr_request, gdpr_request
        )
        
        return result
    except Exception as e:
        logger.error(f"GDPR request handling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Query optimization endpoints
@app.post("/query/optimize")
async def optimize_query(
    query: Dict[str, Any],
    optimizer: QueryOptimizer = Depends(get_query_optimizer)
):
    """Optimize query across data stores"""
    try:
        from models import MultiStoreQuery
        
        multi_store_query = MultiStoreQuery(**query)
        optimized_query = await optimizer.optimize_multi_store_query(multi_store_query)
        return optimized_query
    except Exception as e:
        logger.error(f"Query optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Metrics endpoints
@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector not available")
    
    try:
        metrics = await metrics_collector.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
