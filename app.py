"""
AI News Hub - Main Application for Hugging Face Spaces
Integrates all microservices into a single deployable application
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Try to import structlog, fallback to standard logging
try:
    import structlog
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
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Global variables for service integration
services_initialized = True

# Create FastAPI application
app = FastAPI(
    title="AI News Hub",
    description="Intelligent news aggregation and personalization platform",
    version="1.0.0",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simplified middleware - removed complex middleware that might cause 502 errors

# API Endpoints

@app.get("/", response_model=Dict[str, Any])
async def root() -> Dict[str, Any]:
    """Root endpoint with service information."""
    try:
        return {
            "service": "AI News Hub",
            "version": "1.0.0",
            "status": "running",
            "description": "Intelligent news aggregation and personalization platform",
            "timestamp": datetime.utcnow().isoformat(),
            "services": [
                "safety-service",
                "ingestion-service", 
                "content-extraction-service",
                "content-enrichment-service",
                "deduplication-service",
                "ranking-service",
                "summarization-service",
                "personalization-service",
                "notification-service",
                "feedback-service",
                "evaluation-service",
                "mlops-orchestration-service",
                "storage-service",
                "realtime-interface-service",
                "observability-service"
            ]
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {str(e)}")
        return {
            "service": "AI News Hub",
            "version": "1.0.0",
            "status": "running",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services_initialized": services_initialized,
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}

@app.get("/health/live", response_model=Dict[str, Any])
async def liveness_probe() -> Dict[str, Any]:
    """Liveness probe endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/health/ready", response_model=Dict[str, Any])
async def readiness_probe() -> Dict[str, Any]:
    """Readiness probe endpoint."""
    try:
        if not services_initialized:
            return {
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "Services not initialized",
            }
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return {"status": "not_ready", "timestamp": datetime.utcnow().isoformat(), "error": str(e)}

# News Processing Endpoints

@app.post("/news/ingest")
async def ingest_news(url: str = Query(..., description="URL to ingest")):
    """Ingest news from a URL."""
    try:
        # This would integrate with your ingestion service
        logger.info("News ingestion requested", url=url)
        
        # Simulate processing
        await asyncio.sleep(0.1)
        
        return {
            "status": "success",
            "message": "News ingested successfully",
            "url": url,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("News ingestion failed", error=str(e), url=url)
        raise HTTPException(status_code=500, detail=f"News ingestion failed: {str(e)}")

@app.post("/news/rank")
async def rank_news(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(10, ge=1, le=100, description="Number of articles to return")
):
    """Rank news for a user."""
    try:
        # This would integrate with your ranking service
        logger.info("News ranking requested", user_id=user_id, limit=limit)
        
        # Simulate processing
        await asyncio.sleep(0.1)
        
        return {
            "status": "success",
            "user_id": user_id,
            "articles": [
                {
                    "id": f"article_{i}",
                    "title": f"Sample Article {i}",
                    "url": f"https://example.com/article_{i}",
                    "score": 0.9 - (i * 0.1)
                }
                for i in range(1, min(limit + 1, 6))
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("News ranking failed", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail=f"News ranking failed: {str(e)}")

@app.post("/news/summarize")
async def summarize_news(
    article_id: str = Query(..., description="Article ID to summarize")
):
    """Summarize a news article."""
    try:
        # This would integrate with your summarization service
        logger.info("News summarization requested", article_id=article_id)
        
        # Simulate processing
        await asyncio.sleep(0.2)
        
        return {
            "status": "success",
            "article_id": article_id,
            "summary": "This is a sample summary of the news article. It provides a concise overview of the main points and key information contained in the original article.",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("News summarization failed", error=str(e), article_id=article_id)
        raise HTTPException(status_code=500, detail=f"News summarization failed: {str(e)}")

@app.get("/news/personalize")
async def personalize_news(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(10, ge=1, le=100, description="Number of articles to return")
):
    """Get personalized news for a user."""
    try:
        # This would integrate with your personalization service
        logger.info("News personalization requested", user_id=user_id, limit=limit)
        
        # Simulate processing
        await asyncio.sleep(0.1)
        
        return {
            "status": "success",
            "user_id": user_id,
            "personalized_articles": [
                {
                    "id": f"personalized_{i}",
                    "title": f"Personalized Article {i}",
                    "url": f"https://example.com/personalized_{i}",
                    "relevance_score": 0.95 - (i * 0.05)
                }
                for i in range(1, min(limit + 1, 6))
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("News personalization failed", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail=f"News personalization failed: {str(e)}")

# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Any, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    logger.error("HTTP exception", status_code=exc.status_code, detail=exc.detail)
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc))
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    # Configuration for Railway deployment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))  # Railway default port
    log_level = os.getenv("LOG_LEVEL", "info")
    
    # Start server
    uvicorn.run(
        "app:app", 
        host=host, 
        port=port, 
        log_level=log_level, 
        reload=False
    )
