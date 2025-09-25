"""
MLOps Orchestration Service - Main FastAPI Application
Production-grade MLOps pipeline orchestration with Airflow, MLflow, and Vertex AI
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.health import health_router
from src.api.v1 import deployment, features, monitoring, pipelines, retraining, training
from src.config.settings import Settings
from src.deployment.deployment_manager import ModelDeploymentManager
from src.feature_store.feature_store_manager import FeatureStoreManager
from src.infrastructure.resource_manager import ResourceManager
from src.middleware.auth import AuthMiddleware
from src.middleware.logging import LoggingMiddleware
from src.middleware.rate_limiting import RateLimitMiddleware
from src.monitoring.monitoring_service import ModelMonitoringService
from src.orchestration.pipeline_orchestrator import MLOpsPipelineOrchestrator
from src.registry.model_registry import ModelRegistry
from src.retraining.retraining_manager import AutomatedRetrainingManager
from src.training.training_orchestrator import ModelTrainingOrchestrator
from src.utils.logging_config import setup_logging

# Global services
orchestrator: MLOpsPipelineOrchestrator = None
training_orchestrator: ModelTrainingOrchestrator = None
deployment_manager: ModelDeploymentManager = None
monitoring_service: ModelMonitoringService = None
feature_store_manager: FeatureStoreManager = None
retraining_manager: AutomatedRetrainingManager = None
model_registry: ModelRegistry = None
resource_manager: ResourceManager = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Dict[str, Any]:
    """Application lifespan manager for startup and shutdown"""
    global orchestrator, training_orchestrator, deployment_manager
    global monitoring_service, feature_store_manager, retraining_manager
    global model_registry, resource_manager

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting MLOps Orchestration Service...")

        # Initialize core services
        settings = Settings()
        resource_manager = ResourceManager(settings)
        model_registry = ModelRegistry(settings)

        # Initialize orchestrators
        orchestrator = MLOpsPipelineOrchestrator(settings)
        training_orchestrator = ModelTrainingOrchestrator(settings)
        deployment_manager = ModelDeploymentManager(settings)
        monitoring_service = ModelMonitoringService(settings)
        feature_store_manager = FeatureStoreManager(settings)
        retraining_manager = AutomatedRetrainingManager(settings)

        # Initialize services
        await orchestrator.initialize()
        await training_orchestrator.initialize()
        await deployment_manager.initialize()
        await monitoring_service.initialize()
        await feature_store_manager.initialize()
        await retraining_manager.initialize()

        # Start background tasks
        asyncio.create_task(retraining_manager.start_trigger_monitoring())
        asyncio.create_task(monitoring_service.start_performance_monitoring())

        logger.info("MLOps Orchestration Service started successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to start MLOps Orchestration Service: {str(e)}")
        raise
    finally:
        logger.info("Shutting down MLOps Orchestration Service...")

        # Cleanup resources
        if retraining_manager:
            await retraining_manager.stop_trigger_monitoring()
        if monitoring_service:
            await monitoring_service.stop_performance_monitoring()

        logger.info("MLOps Orchestration Service shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    app = FastAPI(
        title="MLOps Orchestration Service",
        description="Production-grade MLOps pipeline orchestration with Airflow, MLflow, and Vertex AI",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
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
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(LoggingMiddleware)

    # Add Prometheus metrics
    Instrumentator().instrument(app).expose(app)

    # Include routers
    app.include_router(health_router, prefix="/health", tags=["health"])
    app.include_router(pipelines.router, prefix="/api/v1/pipelines", tags=["pipelines"])
    app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
    app.include_router(deployment.router, prefix="/api/v1/deployment", tags=["deployment"])
    app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["monitoring"])
    app.include_router(features.router, prefix="/api/v1/features", tags=["features"])
    app.include_router(retraining.router, prefix="/api/v1/retraining", tags=["retraining"])

    return app


# Create app instance
app = create_app()


# Dependency injection
def get_orchestrator() -> MLOpsPipelineOrchestrator:
    return orchestrator


def get_training_orchestrator() -> ModelTrainingOrchestrator:
    return training_orchestrator


def get_deployment_manager() -> ModelDeploymentManager:
    return deployment_manager


def get_monitoring_service() -> ModelMonitoringService:
    return monitoring_service


def get_feature_store_manager() -> FeatureStoreManager:
    return feature_store_manager


def get_retraining_manager() -> AutomatedRetrainingManager:
    return retraining_manager


def get_model_registry() -> ModelRegistry:
    return model_registry


def get_resource_manager() -> ResourceManager:
    return resource_manager


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
