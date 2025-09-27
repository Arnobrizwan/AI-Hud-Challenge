"""
Dashboard Service - Main Application
Enterprise-grade dashboard for AI/ML pipeline management
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from .api.dashboard_api import router as dashboard_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="AI Pipeline Dashboard",
    description="Enterprise-grade dashboard for AI/ML pipeline management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(dashboard_router)

# Serve static files (React build)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_dashboard():
    """Serve the React dashboard"""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    else:
        return {
            "message": "AI Pipeline Dashboard API",
            "version": "1.0.0",
            "endpoints": {
                "overview": "/api/v1/dashboard/overview",
                "pipelines": "/api/v1/dashboard/pipelines",
                "executions": "/api/v1/dashboard/executions",
                "health": "/api/v1/dashboard/health",
                "metrics": "/api/v1/dashboard/metrics/trends",
                "alerts": "/api/v1/dashboard/alerts"
            }
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "dashboard",
        "timestamp": "2024-01-01T00:00:00Z"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
