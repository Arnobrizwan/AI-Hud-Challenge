"""
Safety Service - Comprehensive Drift, Abuse, and Safety Monitoring
Main FastAPI application with real-time monitoring and automated response
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from safety_engine.config import get_settings
from safety_engine.core import SafetyMonitoringEngine
from safety_engine.database import get_db_session, init_database
from safety_engine.middleware import RateLimitMiddleware, SafetyMiddleware
from safety_engine.models import (
    AbuseDetectionRequest,
    ComplianceRequest,
    ContentModerationRequest,
    DriftDetectionRequest,
    IncidentResponse,
    RateLimitRequest,
    SafetyMonitoringRequest,
    SafetyStatus,
)
from safety_engine.monitoring import get_metrics, setup_monitoring

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Prometheus metrics
safety_checks_total = Counter(
    "safety_checks_total", "Total safety checks performed", [
        "check_type", "status"])
safety_check_duration = Histogram(
    "safety_check_duration_seconds",
    "Time spent on safety checks",
    ["check_type"])
active_incidents = Gauge(
    "active_incidents_total", "Number of active safety incidents", ["severity"]
)
drift_detection_score = Gauge(
    "drift_detection_score",
    "Current drift detection score")
abuse_detection_score = Gauge(
    "abuse_detection_score",
    "Current abuse detection score")

# Global safety engine instance
safety_engine: Optional[SafetyMonitoringEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Dict[str, Any]:
    """Application lifespan manager"""
    global safety_engine

    # Initialize database
    await init_database()
    logger.info("Database initialized")

    # Initialize safety engine
    safety_engine = SafetyMonitoringEngine()
    await safety_engine.initialize()
    logger.info("Safety monitoring engine initialized")

    # Start background monitoring tasks
    asyncio.create_task(background_monitoring())
    asyncio.create_task(incident_cleanup())

    yield

    # Cleanup
    if safety_engine:
    await safety_engine.cleanup()
    logger.info("Safety service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Safety Service",
    description="Comprehensive Drift, Abuse, and Safety Monitoring Microservice",
    version="1.0.0",
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

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(SafetyMiddleware)
app.add_middleware(RateLimitMiddleware)

# Setup monitoring
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "safety-service",
        "version": "1.0.0",
    }


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """Prometheus metrics endpoint"""
    return generate_latest()


@app.post("/safety/monitor", response_model=SafetyStatus)
async def monitor_system_safety(
    request: SafetyMonitoringRequest, background_tasks: BackgroundTasks
):
     -> Dict[str, Any]:"""Comprehensive system safety monitoring endpoint"""
    try:
        safety_checks_total.labels(
            check_type="comprehensive",
            status="started").inc()

        with safety_check_duration.labels(check_type="comprehensive").time():
            safety_status = await safety_engine.monitor_system_safety(request)

        # Update metrics
        drift_detection_score.set(
            safety_status.drift_status.overall_drift_score if safety_status.drift_status else 0.0)
        abuse_detection_score.set(
            safety_status.abuse_status.abuse_score if safety_status.abuse_status else 0.0)

        safety_checks_total.labels(
            check_type="comprehensive",
            status="completed").inc()

        # Trigger background response if needed
        if safety_status.requires_intervention:
            background_tasks.add_task(
                handle_safety_intervention, safety_status)

        return safety_status

    except Exception as e:
        safety_checks_total.labels(
            check_type="comprehensive",
            status="error").inc()
        logger.error(f"Safety monitoring failed: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Safety monitoring failed: {str(e)}")


@app.post("/safety/drift/detect")
async def detect_drift(request: DriftDetectionRequest) -> Dict[str, Any]:
    """Data and concept drift detection endpoint"""
    try:
        safety_checks_total.labels(check_type="drift", status="started").inc()

        with safety_check_duration.labels(check_type="drift").time():
            drift_result = await safety_engine.drift_detector.detect_comprehensive_drift(request)

        safety_checks_total.labels(
            check_type="drift",
            status="completed").inc()
        return drift_result

    except Exception as e:
        safety_checks_total.labels(check_type="drift", status="error").inc()
        logger.error(f"Drift detection failed: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Drift detection failed: {str(e)}")


@app.post("/safety/abuse/detect")
async def detect_abuse(request: AbuseDetectionRequest) -> Dict[str, Any]:
    """Abuse detection and prevention endpoint"""
    try:
        safety_checks_total.labels(check_type="abuse", status="started").inc()

        with safety_check_duration.labels(check_type="abuse").time():
            abuse_result = await safety_engine.abuse_detector.detect_abuse(request)

        safety_checks_total.labels(
            check_type="abuse",
            status="completed").inc()
        return abuse_result

    except Exception as e:
        safety_checks_total.labels(check_type="abuse", status="error").inc()
        logger.error(f"Abuse detection failed: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Abuse detection failed: {str(e)}")


@app.post("/safety/content/moderate")
async def moderate_content(request: ContentModerationRequest) -> Dict[str, Any]:
    """Content moderation and safety endpoint"""
    try:
        safety_checks_total.labels(
            check_type="content",
            status="started").inc()

        with safety_check_duration.labels(check_type="content").time():
            moderation_result = await safety_engine.content_moderator.moderate_content(
                request.content
            )

        safety_checks_total.labels(
            check_type="content",
            status="completed").inc()
        return moderation_result

    except Exception as e:
        safety_checks_total.labels(check_type="content", status="error").inc()
        logger.error(f"Content moderation failed: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Content moderation failed: {str(e)}")


@app.post("/safety/rate-limit/check")
async def check_rate_limits(request: RateLimitRequest) -> Dict[str, Any]:
    """Rate limiting check endpoint"""
    try:
        safety_checks_total.labels(
            check_type="rate_limit",
            status="started").inc()

        with safety_check_duration.labels(check_type="rate_limit").time():
            rate_limit_result = await safety_engine.rate_limiter.check_rate_limits(request)

        safety_checks_total.labels(
            check_type="rate_limit",
            status="completed").inc()
        return rate_limit_result

    except Exception as e:
        safety_checks_total.labels(
            check_type="rate_limit",
            status="error").inc()
        logger.error(f"Rate limiting check failed: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Rate limiting check failed: {str(e)}")


@app.post("/safety/compliance/check")
async def check_compliance(request: ComplianceRequest) -> Dict[str, Any]:
    """Compliance monitoring endpoint"""
    try:
        safety_checks_total.labels(
            check_type="compliance",
            status="started").inc()

        with safety_check_duration.labels(check_type="compliance").time():
            compliance_result = await safety_engine.compliance_monitor.check_compliance(request)

        safety_checks_total.labels(
            check_type="compliance",
            status="completed").inc()
        return compliance_result

    except Exception as e:
        safety_checks_total.labels(
            check_type="compliance",
            status="error").inc()
        logger.error(f"Compliance check failed: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Compliance check failed: {str(e)}")


@app.get("/safety/incidents")
async def get_active_incidents() -> Dict[str, Any]:
    """Get active safety incidents"""
    try:
        incidents = await safety_engine.incident_manager.get_active_incidents()
        return {"incidents": incidents}
    except Exception as e:
        logger.error(f"Failed to get incidents: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Failed to get incidents: {str(e)}")


@app.post("/safety/incidents/{incident_id}/resolve")
async def resolve_incident(incident_id: str) -> Dict[str, Any]:
    """Resolve a safety incident"""
    try:
        result = await safety_engine.incident_manager.resolve_incident(incident_id)
        return result
    except Exception as e:
        logger.error(f"Failed to resolve incident {incident_id}: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Failed to resolve incident: {str(e)}")


@app.get("/safety/audit/logs")
async def get_audit_logs(limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    """Get audit logs"""
    try:
        logs = await safety_engine.audit_logger.get_audit_logs(limit=limit, offset=offset)
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Failed to get audit logs: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Failed to get audit logs: {str(e)}")


async def background_monitoring() -> Dict[str, Any]:
    """Background monitoring task"""
    while True:
        try:
            # Perform periodic safety checks
            await safety_engine.periodic_safety_check()
            await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Background monitoring error: {str(e)}")
            await asyncio.sleep(60)


async def incident_cleanup() -> Dict[str, Any]:
    """Cleanup resolved incidents"""
    while True:
        try:
    await safety_engine.incident_manager.cleanup_resolved_incidents()
            await asyncio.sleep(3600)  # Cleanup every hour
        except Exception as e:
            logger.error(f"Incident cleanup error: {str(e)}")
            await asyncio.sleep(3600)


async def handle_safety_intervention(safety_status: SafetyStatus) -> Dict[str, Any]:
    """Handle safety intervention in background"""
    try:
    await safety_engine.trigger_safety_response(safety_status)
        logger.info(
            f"Safety intervention triggered for status: {safety_status.overall_score}")
    except Exception as e:
        logger.error(f"Safety intervention failed: {str(e)}")


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info")
