"""
Monitoring API endpoints - REST API for model monitoring management
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

from src.monitoring.monitoring_service import ModelMonitoringService
from src.models.monitoring_models import (
    MonitoringConfig, ModelHealth, PerformanceReport, AlertRule, AlertSeverity
)
from src.utils.exceptions import MonitoringError, ValidationError

router = APIRouter()

# Request/Response models
class SetupMonitoringRequest(BaseModel):
    config: MonitoringConfig

class SetupMonitoringResponse(BaseModel):
    monitoring_id: str
    status: str
    message: str

class ModelHealthResponse(BaseModel):
    model_name: str
    health: ModelHealth

@router.post("/setup", response_model=SetupMonitoringResponse)
async def setup_model_monitoring(
    request: SetupMonitoringRequest,
    background_tasks: BackgroundTasks,
    monitoring_service: ModelMonitoringService = Depends()
):
    """Setup monitoring for a model"""
    
    try:
        await monitoring_service.setup_model_monitoring(
            model_name=request.config.model_name,
            monitoring_config=request.config
        )
        
        return SetupMonitoringResponse(
            monitoring_id=f"monitoring_{request.config.model_name}",
            status="active",
            message=f"Monitoring setup completed for model '{request.config.model_name}'"
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except MonitoringError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health/{model_name}", response_model=ModelHealthResponse)
async def get_model_health(
    model_name: str,
    monitoring_service: ModelMonitoringService = Depends()
):
    """Get model health status"""
    
    try:
        health = await monitoring_service.get_model_health(model_name)
        
        return ModelHealthResponse(
            model_name=model_name,
            health=health
        )
        
    except MonitoringError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model health: {str(e)}")

@router.get("/report/{model_name}", response_model=PerformanceReport)
async def get_performance_report(
    model_name: str,
    start_time: Optional[datetime] = Query(None, description="Start time for report"),
    end_time: Optional[datetime] = Query(None, description="End time for report"),
    monitoring_service: ModelMonitoringService = Depends()
):
    """Get performance report for a model"""
    
    try:
        # Default to last 24 hours if no time range specified
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.utcnow()
        
        report = await monitoring_service.generate_performance_report(
            model_name=model_name,
            start_time=start_time,
            end_time=end_time
        )
        
        return report
        
    except MonitoringError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate performance report: {str(e)}")

@router.post("/drift/{model_name}/detect")
async def detect_data_drift(
    model_name: str,
    reference_data_path: str = Query(..., description="Path to reference data"),
    current_data_path: str = Query(..., description="Path to current data"),
    monitoring_service: ModelMonitoringService = Depends()
):
    """Detect data drift for a model"""
    
    try:
        # This would load actual data and detect drift
        # For now, return mock response
        drift_result = {
            "model_name": model_name,
            "overall_drift_score": 0.15,
            "is_drift_detected": True,
            "feature_drift_scores": {
                "feature_1": 0.12,
                "feature_2": 0.18,
                "feature_3": 0.08
            },
            "detected_at": datetime.utcnow().isoformat()
        }
        
        return drift_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect data drift: {str(e)}")

@router.get("/alerts")
async def list_alerts(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    severity: Optional[AlertSeverity] = Query(None, description="Filter by severity"),
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    monitoring_service: ModelMonitoringService = Depends()
):
    """List monitoring alerts"""
    
    try:
        # This would integrate with actual alerting system
        alerts = [
            {
                "id": "alert_1",
                "model_name": "customer_churn_prediction",
                "alert_type": "performance",
                "severity": "high",
                "message": "Accuracy dropped below threshold",
                "status": "active",
                "triggered_at": datetime.utcnow().isoformat()
            },
            {
                "id": "alert_2",
                "model_name": "fraud_detection",
                "alert_type": "drift",
                "severity": "medium",
                "message": "Data drift detected",
                "status": "acknowledged",
                "triggered_at": datetime.utcnow().isoformat()
            }
        ]
        
        # Apply filters
        if model_name:
            alerts = [alert for alert in alerts if alert["model_name"] == model_name]
        if severity:
            alerts = [alert for alert in alerts if alert["severity"] == severity.value]
        if status:
            alerts = [alert for alert in alerts if alert["status"] == status]
        
        # Pagination
        total = len(alerts)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_alerts = alerts[start_idx:end_idx]
        
        return {
            "alerts": paginated_alerts,
            "total": total,
            "page": page,
            "page_size": page_size
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list alerts: {str(e)}")

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str = Query(..., description="User who acknowledged the alert"),
    monitoring_service: ModelMonitoringService = Depends()
):
    """Acknowledge an alert"""
    
    try:
        # This would integrate with actual alerting system
        return {
            "alert_id": alert_id,
            "status": "acknowledged",
            "acknowledged_by": acknowledged_by,
            "acknowledged_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolved_by: str = Query(..., description="User who resolved the alert"),
    monitoring_service: ModelMonitoringService = Depends()
):
    """Resolve an alert"""
    
    try:
        # This would integrate with actual alerting system
        return {
            "alert_id": alert_id,
            "status": "resolved",
            "resolved_by": resolved_by,
            "resolved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")

@router.get("/metrics/{model_name}")
async def get_model_metrics(
    model_name: str,
    start_time: Optional[datetime] = Query(None, description="Start time for metrics"),
    end_time: Optional[datetime] = Query(None, description="End time for metrics"),
    monitoring_service: ModelMonitoringService = Depends()
):
    """Get model metrics"""
    
    try:
        # This would integrate with actual metrics system
        metrics = {
            "model_name": model_name,
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "latency_ms": 120.5,
            "throughput_rps": 150.3,
            "error_rate": 0.02,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model metrics: {str(e)}")

@router.get("/dashboard/{model_name}")
async def get_monitoring_dashboard(
    model_name: str,
    monitoring_service: ModelMonitoringService = Depends()
):
    """Get monitoring dashboard for a model"""
    
    try:
        # This would integrate with actual dashboard system
        dashboard = {
            "model_name": model_name,
            "dashboard_url": f"http://grafana:3000/d/mlops-{model_name}",
            "panels": [
                {
                    "title": "Model Performance",
                    "type": "graph",
                    "metrics": ["accuracy", "precision", "recall", "f1_score"]
                },
                {
                    "title": "Request Metrics",
                    "type": "graph",
                    "metrics": ["latency_ms", "throughput_rps", "error_rate"]
                },
                {
                    "title": "Data Drift",
                    "type": "table",
                    "metrics": ["drift_score", "feature_drift"]
                }
            ],
            "refresh_interval": 30
        }
        
        return dashboard
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring dashboard: {str(e)}")
