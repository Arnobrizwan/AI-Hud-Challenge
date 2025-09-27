"""
AI News Hub - Simplified Version for Railway
"""

import os
from datetime import datetime
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, Any]:
    """Root endpoint with service information."""
    return {
        "service": "AI News Hub",
        "version": "1.0.0",
        "status": "running",
        "description": "Intelligent news aggregation and personalization platform",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health/live", response_model=Dict[str, Any])
async def liveness_probe() -> Dict[str, Any]:
    """Liveness probe endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/health/ready", response_model=Dict[str, Any])
async def readiness_probe() -> Dict[str, Any]:
    """Readiness probe endpoint."""
    return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}

# News Processing Endpoints

@app.post("/news/ingest")
async def ingest_news(url: str = Query(..., description="URL to ingest")):
    """Ingest news from a URL."""
    try:
        return {
            "status": "success",
            "message": "News ingested successfully",
            "url": url,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News ingestion failed: {str(e)}")

@app.post("/news/rank")
async def rank_news(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(10, ge=1, le=100, description="Number of articles to return")
):
    """Rank news for a user."""
    try:
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
        raise HTTPException(status_code=500, detail=f"News ranking failed: {str(e)}")

@app.post("/news/summarize")
async def summarize_news(
    article_id: str = Query(..., description="Article ID to summarize")
):
    """Summarize a news article."""
    try:
        return {
            "status": "success",
            "article_id": article_id,
            "summary": "This is a sample summary of the news article. It provides a concise overview of the main points and key information contained in the original article.",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News summarization failed: {str(e)}")

@app.get("/news/personalize")
async def personalize_news(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(10, ge=1, le=100, description="Number of articles to return")
):
    """Get personalized news for a user."""
    try:
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
        raise HTTPException(status_code=500, detail=f"News personalization failed: {str(e)}")

# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Any, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
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
        "app_simple:app", 
        host=host, 
        port=port, 
        log_level=log_level, 
        reload=False
    )
