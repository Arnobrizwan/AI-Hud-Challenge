"""
Observability & Runbooks Microservice
Comprehensive monitoring, alerting, and automated incident response
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn

from config.settings import get_settings
from monitoring.health import HealthChecker
from monitoring.metrics import MetricsCollector
from monitoring.tracing import TraceManager
from monitoring.logging import LogAggregator
from monitoring.alerting import AlertingSystem
from monitoring.runbooks import RunbookEngine
from monitoring.slo import SLOMonitor
from monitoring.incidents import IncidentManager
from monitoring.cost import CostMonitor
from monitoring.chaos import ChaosEngine
from api.routes import health, metrics, alerts, runbooks, incidents, slo, cost, chaos
from middleware.auth import AuthMiddleware
from middleware.rate_limiting import RateLimitMiddleware
from middleware.request_logging import RequestLoggingMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global observability components
observability_engine = None
health_checker = None
metrics_collector = None
trace_manager = None
log_aggregator = None
alerting_system = None
runbook_engine = None
slo_monitor = None
incident_manager = None
cost_monitor = None
chaos_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global observability_engine, health_checker, metrics_collector
    global trace_manager, log_aggregator, alerting_system
    global runbook_engine, slo_monitor, incident_manager, cost_monitor, chaos_engine
    
    settings = get_settings()
    
    try:
        logger.info("Initializing Observability Service...")
        
        # Initialize core components
        health_checker = HealthChecker()
        metrics_collector = MetricsCollector()
        trace_manager = TraceManager()
        log_aggregator = LogAggregator()
        alerting_system = AlertingSystem()
        runbook_engine = RunbookEngine()
        slo_monitor = SLOMonitor()
        incident_manager = IncidentManager()
        cost_monitor = CostMonitor()
        chaos_engine = ChaosEngine()
        
        # Initialize observability engine
        from monitoring.observability_engine import ObservabilityEngine
        observability_engine = ObservabilityEngine()
        
        # Initialize all components
        await observability_engine.initialize_observability(settings.observability_config)
        
        # Start background monitoring tasks
        asyncio.create_task(observability_engine.start_monitoring_loops())
        
        logger.info("Observability Service initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize Observability Service: {str(e)}")
        raise
    finally:
        logger.info("Shutting down Observability Service...")
        
        # Cleanup resources
        if observability_engine:
            await observability_engine.cleanup()


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Observability & Runbooks Service",
        description="Comprehensive monitoring, alerting, and automated incident response",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
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
    app.add_middleware(RequestLoggingMiddleware)
    
    # Add Prometheus instrumentation
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app)
    
    # Include routers
    app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
    app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["metrics"])
    app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])
    app.include_router(runbooks.router, prefix="/api/v1/runbooks", tags=["runbooks"])
    app.include_router(incidents.router, prefix="/api/v1/incidents", tags=["incidents"])
    app.include_router(slo.router, prefix="/api/v1/slo", tags=["slo"])
    app.include_router(cost.router, prefix="/api/v1/cost", tags=["cost"])
    app.include_router(chaos.router, prefix="/api/v1/chaos", tags=["chaos"])
    
    return app


app = create_app()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Observability & Runbooks Service",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/v1/status")
async def get_service_status():
    """Get comprehensive service status"""
    if not observability_engine:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        status = await observability_engine.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get service status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get service status")


@app.post("/api/v1/emergency/trigger")
async def trigger_emergency_response(
    incident_type: str,
    severity: str,
    description: str,
    background_tasks: BackgroundTasks
):
    """Trigger emergency response procedures"""
    if not incident_manager:
        raise HTTPException(status_code=503, detail="Incident manager not available")
    
    try:
        # Create emergency incident
        incident = await incident_manager.create_emergency_incident(
            incident_type=incident_type,
            severity=severity,
            description=description
        )
        
        # Trigger automated response
        background_tasks.add_task(
            incident_manager.trigger_emergency_response,
            incident.id
        )
        
        return {
            "incident_id": incident.id,
            "status": "emergency_response_triggered",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger emergency response: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to trigger emergency response")


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
