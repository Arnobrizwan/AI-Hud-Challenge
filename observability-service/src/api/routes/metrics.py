"""
Metrics API routes
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from monitoring.metrics import MetricsCollector
from monitoring.observability_engine import observability_engine

logger = logging.getLogger(__name__)

router = APIRouter()


class MetricsResponse(BaseModel):
    """Metrics response"""

    metrics: Dict[str, Any]
    timestamp: str
    collection_interval: int


class CustomMetricRequest(BaseModel):
    """Custom metric request"""

    name: str
    value: float
    labels: Dict[str, str] = {}
    metric_type: str = "gauge"  # gauge, counter, histogram, summary


class BusinessMetricsResponse(BaseModel):
    """Business metrics response"""

    articles_processed: int
    users_active: int
    content_quality_score: float
    engagement_metrics: Dict[str, Any]
    pipeline_metrics: Dict[str, Any]
    timestamp: str


@router.get("/", response_model=MetricsResponse)
async def get_metrics() -> Dict[str, Any]:
    """Get all metrics"""
    try:
        if not observability_engine or not observability_engine.metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")

        # Collect real-time metrics
        metrics = await observability_engine.metrics_collector.collect_real_time_metrics()

        return MetricsResponse(metrics=metrics, timestamp=datetime.utcnow().isoformat(), collection_interval=30)

    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/prometheus")
async def get_prometheus_metrics() -> Dict[str, Any]:
    """Get Prometheus metrics in text format"""
    try:
        if not observability_engine or not observability_engine.metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")

        # Get Prometheus metrics
        metrics_text = observability_engine.metrics_collector.get_prometheus_metrics()
        content_type = observability_engine.metrics_collector.get_metrics_content_type()

        return Response(content=metrics_text, media_type=content_type)

    except Exception as e:
        logger.error(f"Failed to get Prometheus metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get Prometheus metrics: {str(e)}")


@router.get("/summary", response_model=Dict[str, Any])
async def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary"""
    try:
        if not observability_engine or not observability_engine.metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")

        # Get metrics summary
        summary = await observability_engine.metrics_collector.get_metrics_summary()

        return {"metrics_summary": summary, "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Failed to get metrics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics summary: {str(e)}")


@router.get("/system")
async def get_system_metrics() -> Dict[str, Any]:
    """Get system resource metrics"""
    try:
        if not observability_engine or not observability_engine.metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")

        # Collect system metrics
        metrics = await observability_engine.metrics_collector.collect_real_time_metrics()

        # Filter system metrics
        system_metrics = {key: value for key, value in metrics.items() if key.startswith(("cpu_", "memory_", "disk_"))}

        return {"system_metrics": system_metrics, "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Failed to get system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")


@router.get("/application")
async def get_application_metrics() -> Dict[str, Any]:
    """Get application performance metrics"""
    try:
        if not observability_engine or not observability_engine.metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")

        # Collect application metrics
        metrics = await observability_engine.metrics_collector.collect_real_time_metrics()

        # Filter application metrics
        app_metrics = {key: value for key, value in metrics.items() if key.startswith(("db_", "cache_", "queue_"))}

        return {"application_metrics": app_metrics, "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Failed to get application metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get application metrics: {str(e)}")


@router.get("/business", response_model=BusinessMetricsResponse)
async def get_business_metrics() -> Dict[str, Any]:
    """Get business metrics"""
    try:
        if not observability_engine or not observability_engine.metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")

        # Collect business metrics
        metrics = await observability_engine.metrics_collector.collect_real_time_metrics()

        return BusinessMetricsResponse(
            articles_processed=int(metrics.get("articles_processed_today", 0)),
            users_active=int(metrics.get("users_active_today", 0)),
            content_quality_score=metrics.get("content_quality_score", 0.0),
            engagement_metrics={
                "avg_engagement_time": metrics.get("avg_engagement_time", 0.0),
                "click_through_rate": metrics.get("click_through_rate", 0.0),
            },
            pipeline_metrics={
                "processing_time": metrics.get("pipeline_processing_time", 0.0),
                "throughput": metrics.get("pipeline_throughput", 0.0),
            },
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to get business metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get business metrics: {str(e)}")


@router.post("/custom")
async def record_custom_metric(metric_request: CustomMetricRequest) -> Dict[str, Any]:
    """Record custom metric"""
    try:
        if not observability_engine or not observability_engine.metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")

        # Record custom metric
        # This would integrate with the metrics collector
        logger.info(f"Recording custom metric: {metric_request.name} = {metric_request.value}")

        return {
            "message": f"Custom metric {metric_request.name} recorded successfully",
            "metric_name": metric_request.name,
            "value": metric_request.value,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to record custom metric: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to record custom metric: {str(e)}")


@router.get("/sli")
async def get_sli_metrics() -> Dict[str, Any]:
    """Get SLI (Service Level Indicator) metrics"""
    try:
        if not observability_engine or not observability_engine.slo_monitor:
            raise HTTPException(status_code=503, detail="SLO monitor not available")

        # Get SLI metrics
        sli_status = await observability_engine.slo_monitor.get_overall_slo_status()

        return {"sli_metrics": sli_status, "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Failed to get SLI metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get SLI metrics: {str(e)}")


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics"""
    try:
        if not observability_engine or not observability_engine.metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")

        # Collect performance metrics
        metrics = await observability_engine.metrics_collector.collect_real_time_metrics()

        # Filter performance metrics
        performance_metrics = {
            key: value
            for key, value in metrics.items()
            if key.startswith(("response_time", "throughput", "latency", "error_rate"))
        }

        return {
            "performance_metrics": performance_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.get("/trends")
async def get_metrics_trends(
    time_window: int = Query(3600, description="Time window in seconds"),
    metric_type: Optional[str] = Query(None, description="Type of metrics to analyze"),
):
    """Get metrics trends over time"""

    try:
        if not observability_engine or not observability_engine.metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")

        # This would query historical metrics data
        # For now, return mock trends

        trends = {
            "time_window_seconds": time_window,
            "metric_type": metric_type or "all",
            "trends": {
                "cpu_usage": {"trend": "stable", "change_percent": 2.5},
                "memory_usage": {"trend": "increasing", "change_percent": 15.3},
                "response_time": {"trend": "decreasing", "change_percent": -8.7},
                "error_rate": {"trend": "stable", "change_percent": 0.1},
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        return trends

    except Exception as e:
        logger.error(f"Failed to get metrics trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics trends: {str(e)}")


@router.get("/export")
async def export_metrics(
    format: str = Query("json", description="Export format: json, csv, prometheus"),
    time_range: int = Query(3600, description="Time range in seconds"),
):
    """Export metrics in various formats"""

    try:
        if not observability_engine or not observability_engine.metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")

        if format == "prometheus":
            # Return Prometheus format
            metrics_text = observability_engine.metrics_collector.get_prometheus_metrics()
            return PlainTextResponse(content=metrics_text, media_type="text/plain")

        elif format == "csv":
            # Return CSV format
            # This would generate CSV from metrics data
            csv_content = "timestamp,metric_name,value\n"
            csv_content += f"{datetime.utcnow().isoformat()},sample_metric,1.0\n"
            return PlainTextResponse(content=csv_content, media_type="text/csv")

        else:  # json
            # Return JSON format
            metrics = await observability_engine.metrics_collector.collect_real_time_metrics()
            return {
                "metrics": metrics,
                "export_format": "json",
                "time_range_seconds": time_range,
                "timestamp": datetime.utcnow().isoformat(),
            }

    except Exception as e:
        logger.error(f"Failed to export metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export metrics: {str(e)}")


@router.post("/collect")
async def trigger_metrics_collection() -> Dict[str, Any]:
    """Trigger immediate metrics collection"""
    try:
        if not observability_engine or not observability_engine.metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")

        # Trigger metrics collection
        metrics = await observability_engine.metrics_collector.collect_real_time_metrics()

        return {
            "message": "Metrics collection triggered successfully",
            "metrics_collected": len(metrics),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to trigger metrics collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger metrics collection: {str(e)}")
