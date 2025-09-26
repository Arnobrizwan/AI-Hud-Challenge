"""FastAPI application optimized for Vercel deployment."""

from datetime import datetime
from typing import Any, Callable, Dict

import structlog
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

# Configure structured logging for Vercel
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

# Create FastAPI application
app = FastAPI(
    title="AI Hub Challenge - News Ranking API",
    description="AI-powered news aggregation pipeline with ranking and personalization",
    version="1.0.0",
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


# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next: Callable) -> Response:
    response: Response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# Add correlation ID middleware
@app.middleware("http")
async def add_correlation_id(request: Request, call_next: Callable) -> Response:
    import uuid

    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    response: Response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


# API Endpoints


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, Any]:
    """Root endpoint with service information."""
    return {
        "service": "AI Hub Challenge - News Ranking API",
        "version": "1.0.0",
        "status": "running",
        "description": "AI-powered news aggregation pipeline with ranking and personalization",
        "deployment": "Vercel Serverless",
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "AI Hub Challenge API",
            "version": "1.0.0",
            "environment": "vercel",
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@app.get("/health/live", response_model=Dict[str, Any])
async def liveness_probe() -> Dict[str, Any]:
    """Liveness probe endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/health/ready", response_model=Dict[str, Any])
async def readiness_probe() -> Dict[str, Any]:
    """Readiness probe endpoint."""
    try:
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return {"status": "not_ready", "timestamp": datetime.utcnow().isoformat(), "error": str(e)}


# Authentication Endpoints (Simplified for Vercel)


@app.post("/auth/login", response_model=Dict[str, Any])
async def login(request: Dict[str, Any]) -> Dict[str, Any]:
    """Login endpoint for authentication."""
    try:
        # Simplified authentication for demo purposes
        return {
            "status": "success",
            "message": "Login successful",
            "token": "demo-token-123",
            "user_id": request.get("username", "demo-user"),
        }
    except Exception as e:
        logger.error("Login failed", error=str(e))
        raise HTTPException(status_code=401, detail="Authentication failed")


@app.post("/auth/logout", response_model=Dict[str, str])
async def logout() -> Dict[str, str]:
    """Logout endpoint."""
    return {"status": "success", "message": "Logout successful"}


@app.get("/auth/me", response_model=Dict[str, Any])
async def get_current_user() -> Dict[str, Any]:
    """Get current user information."""
    return {
        "status": "success",
        "data": {"uid": "demo-user", "email": "demo@example.com", "name": "Demo User"},
    }


# Ranking Endpoints (Simplified for Vercel)


@app.post("/rank", response_model=Dict[str, Any])
async def rank_content(request: Dict[str, Any]) -> Dict[str, Any]:
    """Rank content based on request parameters."""
    try:
        logger.info("Ranking request received", user_id=request.get("user_id"))

        # Simplified ranking logic for demo
        articles = [
            {
                "id": f"article_{i}",
                "title": f"News Article {i}",
                "content": f"This is the content for article {i}",
                "url": f"https://example.com/articles/{i}",
                "score": 0.9 - (i * 0.1),
                "rank": i + 1,
            }
            for i in range(1, 6)
        ]

        return {
            "status": "success",
            "articles": articles,
            "processing_time_ms": 50,
            "algorithm": "simplified_demo",
        }
    except Exception as e:
        logger.error("Ranking failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")


@app.get("/articles/{article_id}", response_model=Dict[str, Any])
async def get_article(article_id: str) -> Dict[str, Any]:
    """Get article by ID."""
    try:
        article = {
            "id": article_id,
            "title": f"Article {article_id}",
            "content": f"This is the content for article {article_id}",
            "url": f"https://example.com/articles/{article_id}",
            "published_at": datetime.utcnow().isoformat(),
            "source": {"id": "source_1", "name": "Example Source", "domain": "example.com"},
            "author": {"id": "author_1", "name": "Example Author"},
            "word_count": 500,
            "reading_time": 2,
            "quality_score": 0.8,
        }
        return article
    except Exception as e:
        logger.error("Failed to get article", error=str(e), article_id=article_id)
        raise HTTPException(status_code=404, detail="Article not found")


# User Profile Endpoints (Simplified for Vercel)


@app.get("/users/{user_id}/profile", response_model=Dict[str, Any])
async def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Get user profile for personalization."""
    try:
        profile = {
            "user_id": user_id,
            "topic_preferences": {},
            "source_preferences": {},
            "reading_patterns": {},
            "content_preferences": {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        return profile
    except Exception as e:
        logger.error("Failed to get user profile", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Failed to get user profile")


@app.put("/users/{user_id}/profile", response_model=Dict[str, str])
async def update_user_profile(user_id: str, profile: Dict[str, Any]) -> Dict[str, str]:
    """Update user profile."""
    try:
        logger.info("User profile updated", user_id=user_id)
        return {"message": "Profile updated successfully"}
    except Exception as e:
        logger.error("Failed to update user profile", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Failed to update user profile")


# Metrics Endpoints (Simplified for Vercel)


@app.get("/metrics/performance", response_model=Dict[str, Any])
async def get_performance_metrics(time_window: int = Query(60, ge=1, le=1440)) -> Dict[str, Any]:
    """Get performance metrics."""
    try:
        return {
            "status": "success",
            "metrics": {
                "total_requests": 1000,
                "avg_response_time_ms": 150,
                "success_rate": 0.99,
                "time_window_minutes": time_window,
            },
        }
    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")


# Error handlers


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    logger.error("HTTP exception", status_code=exc.status_code, detail=exc.detail)
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc))
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error", "status_code": 500}
    )


# Vercel entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
