"""
Vercel entry point for the AI Hub Challenge application.
Simplified version for serverless deployment.
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# Configure simple logging for Vercel
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="AI Hub Challenge - News Ranking API",
    description="AI-powered news aggregation pipeline with ranking and personalization",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Hub Challenge API",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ai-hub-challenge"
    }

@app.get("/api/health")
async def api_health():
    """API health check."""
    return {"status": "ok", "message": "API is running"}

@app.post("/rank")
async def rank_content():
    """Content ranking endpoint (simplified)."""
    return {
        "message": "Ranking endpoint - simplified for Vercel",
        "status": "success",
        "results": []
    }

@app.get("/api/docs")
async def get_docs():
    """API documentation."""
    return {
        "message": "API Documentation",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Root endpoint"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/api/health", "method": "GET", "description": "API health check"},
            {"path": "/rank", "method": "POST", "description": "Content ranking"},
            {"path": "/api/docs", "method": "GET", "description": "This documentation"}
        ]
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500}
    )

# Export for Vercel
handler = app