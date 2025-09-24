"""
Evaluation Suite Microservice - Main FastAPI Application
Comprehensive evaluation system for ML pipelines with offline evaluation,
online A/B testing, business impact analysis, and real-time monitoring.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from evaluation_engine.cache import init_cache
from evaluation_engine.config import get_settings
from evaluation_engine.core import EvaluationEngine
from evaluation_engine.database import init_database
from evaluation_engine.dependencies import get_evaluation_engine
from evaluation_engine.middleware import (
    ErrorHandlingMiddleware,
    PerformanceMonitoringMiddleware,
    RequestLoggingMiddleware,
)
from evaluation_engine.monitoring import EvaluationMonitoring
from evaluation_engine.routers import (
    business_impact_router,
    dashboard_router,
    drift_detection_router,
    evaluation_router,
    monitoring_router,
    offline_evaluation_router,
    online_evaluation_router,
)
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for services
evaluation_engine: EvaluationEngine = None
monitoring_service: EvaluationMonitoring = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    global evaluation_engine, monitoring_service

    # Startup
    logger.info("Starting Evaluation Suite Microservice...")

    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized successfully")

        # Initialize cache
        await init_cache()
        logger.info("Cache initialized successfully")

        # Initialize evaluation engine
        evaluation_engine = EvaluationEngine()
        await evaluation_engine.initialize()
        logger.info("Evaluation engine initialized successfully")

        # Initialize monitoring service
        monitoring_service = EvaluationMonitoring()
        await monitoring_service.initialize()
        logger.info("Monitoring service initialized successfully")

        # Start background monitoring task
        asyncio.create_task(monitoring_service.start_monitoring())
        logger.info("Background monitoring started")

        logger.info("Evaluation Suite Microservice started successfully")

    except Exception as e:
        logger.error(f"Failed to start Evaluation Suite Microservice: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Evaluation Suite Microservice...")

    try:
        if monitoring_service:
            await monitoring_service.stop_monitoring()
            logger.info("Monitoring service stopped")

        if evaluation_engine:
            await evaluation_engine.cleanup()
            logger.info("Evaluation engine cleaned up")

        logger.info("Evaluation Suite Microservice shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    settings = get_settings()

    app = FastAPI(
        title="Evaluation Suite Microservice",
        description="Comprehensive evaluation system for ML pipelines with offline evaluation, "
        "online A/B testing, business impact analysis, and real-time monitoring",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
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

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(PerformanceMonitoringMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)

    # Include routers
    app.include_router(evaluation_router, prefix="/api/v1/evaluation", tags=["evaluation"])

    app.include_router(
        offline_evaluation_router, prefix="/api/v1/offline-evaluation", tags=["offline-evaluation"]
    )

    app.include_router(
        online_evaluation_router, prefix="/api/v1/online-evaluation", tags=["online-evaluation"]
    )

    app.include_router(
        business_impact_router, prefix="/api/v1/business-impact", tags=["business-impact"]
    )

    app.include_router(
        drift_detection_router, prefix="/api/v1/drift-detection", tags=["drift-detection"]
    )

    app.include_router(monitoring_router, prefix="/api/v1/monitoring", tags=["monitoring"])

    app.include_router(dashboard_router, prefix="/api/v1/dashboard", tags=["dashboard"])

    # Health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "evaluation-suite",
            "version": "1.0.0",
            "timestamp": "2024-01-01T00:00:00Z",
        }

    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with service information"""
        return {
            "service": "Evaluation Suite Microservice",
            "description": "Comprehensive evaluation system for ML pipelines",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create the application instance
app = create_app()


# Dependency injection for evaluation engine
async def get_evaluation_engine_dependency() -> EvaluationEngine:
    """Get evaluation engine instance"""
    if evaluation_engine is None:
        raise HTTPException(status_code=503, detail="Evaluation engine not initialized")
    return evaluation_engine


# Override the dependency function
app.dependency_overrides[get_evaluation_engine] = get_evaluation_engine_dependency


if __name__ == "__main__":
    settings = get_settings()

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )
