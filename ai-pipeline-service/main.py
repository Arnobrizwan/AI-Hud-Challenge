"""
AI Pipeline Service - Main Application
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .src.api.pipeline_api import router as pipeline_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="AI Pipeline Service",
    description="AI/ML Pipeline Management Service",
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
app.include_router(pipeline_router)

@app.get("/")
async def root():
    return {
        "service": "AI Pipeline Service",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
