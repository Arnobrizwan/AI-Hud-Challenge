"""
Simple tests for evaluation service components.
"""

import pytest


def test_evaluation_engine_initialization():
    """Test evaluation engine initialization."""
    from src.evaluation_engine.core import EvaluationEngine
    
    engine = EvaluationEngine()
    assert engine is not None


def test_evaluation_cache():
    """Test evaluation cache functionality."""
    from src.evaluation_engine.cache import EvaluationCache
    
    cache = EvaluationCache()
    assert cache is not None
    
    # Test cache operations
    cache.set("test_key", "test_value")
    assert cache.get("test_key") == "test_value"
    
    cache.clear()
    assert cache.get("test_key") is None


def test_evaluation_monitoring():
    """Test evaluation monitoring."""
    from src.evaluation_engine.monitoring import EvaluationMonitoring
    
    monitoring = EvaluationMonitoring()
    assert monitoring is not None
    
    # Test metrics
    metrics = monitoring.get_metrics()
    assert isinstance(metrics, dict)
    assert "evaluations" in metrics


def test_evaluation_dependencies():
    """Test evaluation dependencies."""
    from src.evaluation_engine.dependencies import get_evaluation_engine
    
    engine = get_evaluation_engine()
    assert engine is not None


def test_evaluation_config():
    """Test evaluation configuration."""
    from src.evaluation_engine.config import get_settings
    
    settings = get_settings()
    assert settings is not None
    assert settings.service_name == "evaluation-service"


def test_evaluation_middleware():
    """Test evaluation middleware."""
    from src.evaluation_engine.middleware import (
        ErrorHandlingMiddleware,
        PerformanceMonitoringMiddleware,
        RequestLoggingMiddleware
    )
    
    # Test middleware classes exist
    assert ErrorHandlingMiddleware is not None
    assert PerformanceMonitoringMiddleware is not None
    assert RequestLoggingMiddleware is not None


def test_evaluation_routers():
    """Test evaluation routers."""
    from src.evaluation_engine.routers import (
        evaluation_router,
        dashboard_router,
        drift_detection_router,
        business_impact_router,
        performance_router,
        monitoring_router
    )
    
    # Test routers exist
    assert evaluation_router is not None
    assert dashboard_router is not None
    assert drift_detection_router is not None
    assert business_impact_router is not None
    assert performance_router is not None
    assert monitoring_router is not None


def test_evaluation_engine_methods():
    """Test evaluation engine methods."""
    from src.evaluation_engine.core import EvaluationEngine
    
    engine = EvaluationEngine()
    
    # Test evaluate_model method exists
    assert hasattr(engine, 'evaluate_model')
    assert hasattr(engine, 'compare_models')


def test_evaluation_database():
    """Test evaluation database functionality."""
    from src.evaluation_engine.database import init_database, get_evaluation_results
    
    # Test functions exist
    assert init_database is not None
    assert get_evaluation_results is not None
    
    # Test database operations
    results = get_evaluation_results()
    assert isinstance(results, dict)
    assert "results" in results
