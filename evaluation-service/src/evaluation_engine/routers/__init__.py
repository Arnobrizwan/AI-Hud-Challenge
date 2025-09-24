"""
API Routers Module
"""

from .business_impact import business_impact_router
from .dashboard import dashboard_router
from .drift_detection import drift_detection_router
from .evaluation import evaluation_router
from .monitoring import monitoring_router
from .offline_evaluation import offline_evaluation_router
from .online_evaluation import online_evaluation_router

__all__ = [
    "evaluation_router",
    "offline_evaluation_router",
    "online_evaluation_router",
    "business_impact_router",
    "drift_detection_router",
    "monitoring_router",
    "dashboard_router",
]
