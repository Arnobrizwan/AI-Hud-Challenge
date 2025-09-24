"""
Test cases for Evaluation Suite Microservice
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.main import app
from src.evaluation_engine.models import EvaluationConfig, EvaluationStatus

client = TestClient(app)


class TestEvaluationAPI:
    """Test cases for evaluation API endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "evaluation-suite"
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Evaluation Suite Microservice"
        assert "docs" in data
        assert "health" in data
    
    @patch('src.evaluation_engine.core.EvaluationEngine.run_comprehensive_evaluation')
    def test_create_comprehensive_evaluation(self, mock_evaluate):
        """Test comprehensive evaluation creation"""
        # Mock evaluation result
        mock_evaluation = Mock()
        mock_evaluation.evaluation_id = "test_eval_123"
        mock_evaluation.status = EvaluationStatus.COMPLETED
        mock_evaluate.return_value = mock_evaluation
        
        # Test request
        request_data = {
            "config": {
                "include_offline": True,
                "include_online": True,
                "include_business_impact": True,
                "include_drift_analysis": True,
                "include_causal_analysis": True,
                "models": [{"name": "test_model", "type": "classification"}],
                "datasets": [{"name": "test_dataset", "type": "classification"}],
                "metrics": {"metric_types": ["classification"]}
            }
        }
        
        response = client.post("/api/v1/evaluation/comprehensive", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["evaluation_id"] == "test_eval_123"
        assert data["status"] == "completed"
        assert "message" in data
    
    @patch('src.evaluation_engine.core.EvaluationEngine.run_offline_evaluation')
    def test_create_offline_evaluation(self, mock_evaluate):
        """Test offline evaluation creation"""
        # Mock evaluation result
        mock_evaluation = Mock()
        mock_evaluation.evaluation_id = "test_offline_123"
        mock_evaluation.status = EvaluationStatus.COMPLETED
        mock_evaluate.return_value = mock_evaluation
        
        # Test request
        request_data = {
            "models": [{"name": "test_model", "type": "classification"}],
            "datasets": [{"name": "test_dataset", "type": "classification"}],
            "metrics": {"metric_types": ["classification"]}
        }
        
        response = client.post("/api/v1/evaluation/offline", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["evaluation_id"] == "test_offline_123"
        assert data["status"] == "completed"
    
    @patch('src.evaluation_engine.core.EvaluationEngine.run_online_evaluation')
    def test_create_online_evaluation(self, mock_evaluate):
        """Test online evaluation creation"""
        # Mock evaluation result
        mock_evaluation = Mock()
        mock_evaluation.evaluation_id = "test_online_123"
        mock_evaluation.status = EvaluationStatus.COMPLETED
        mock_evaluate.return_value = mock_evaluation
        
        # Test request
        request_data = {
            "experiments": [{"id": "exp_1", "type": "ab_test"}],
            "evaluation_period": {"start": "2024-01-01", "end": "2024-01-31"}
        }
        
        response = client.post("/api/v1/evaluation/online", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["evaluation_id"] == "test_online_123"
        assert data["status"] == "completed"
    
    @patch('src.evaluation_engine.core.EvaluationEngine.get_evaluation_status')
    def test_get_evaluation(self, mock_get):
        """Test get evaluation endpoint"""
        # Mock evaluation result
        mock_evaluation = Mock()
        mock_evaluation.evaluation_id = "test_eval_123"
        mock_evaluation.status = EvaluationStatus.COMPLETED
        mock_get.return_value = mock_evaluation
        
        response = client.get("/api/v1/evaluation/test_eval_123")
        assert response.status_code == 200
        
        data = response.json()
        assert "evaluation" in data
        assert data["evaluation"]["evaluation_id"] == "test_eval_123"
    
    @patch('src.evaluation_engine.core.EvaluationEngine.get_evaluation_status')
    def test_get_evaluation_not_found(self, mock_get):
        """Test get evaluation when not found"""
        mock_get.return_value = None
        
        response = client.get("/api/v1/evaluation/nonexistent")
        assert response.status_code == 404
    
    @patch('src.evaluation_engine.core.EvaluationEngine.list_evaluations')
    def test_list_evaluations(self, mock_list):
        """Test list evaluations endpoint"""
        # Mock evaluations list
        mock_evaluations = [
            Mock(evaluation_id="eval_1", status=EvaluationStatus.COMPLETED),
            Mock(evaluation_id="eval_2", status=EvaluationStatus.RUNNING)
        ]
        mock_list.return_value = mock_evaluations
        
        response = client.get("/api/v1/evaluation/")
        assert response.status_code == 200
        
        data = response.json()
        assert "evaluations" in data
        assert len(data["evaluations"]) == 2
    
    @patch('src.evaluation_engine.core.EvaluationEngine.cancel_evaluation')
    def test_cancel_evaluation(self, mock_cancel):
        """Test cancel evaluation endpoint"""
        mock_cancel.return_value = True
        
        response = client.delete("/api/v1/evaluation/test_eval_123")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "cancelled" in data["message"]


class TestOfflineEvaluationAPI:
    """Test cases for offline evaluation API endpoints"""
    
    @patch('src.evaluation_engine.core.EvaluationEngine.offline_evaluator.evaluate')
    def test_evaluate_models(self, mock_evaluate):
        """Test evaluate models endpoint"""
        # Mock evaluation result
        mock_result = {
            "evaluation_id": "offline_123",
            "models": [{"name": "test_model", "overall_score": 0.85}],
            "overall_summary": {"total_models": 1, "average_score": 0.85}
        }
        mock_evaluate.return_value = mock_result
        
        # Test request
        request_data = {
            "models": [{"name": "test_model", "type": "classification"}],
            "datasets": [{"name": "test_dataset", "type": "classification"}],
            "metrics": {"metric_types": ["classification"]}
        }
        
        response = client.post("/api/v1/offline-evaluation/evaluate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "results" in data
        assert data["results"]["evaluation_id"] == "offline_123"
    
    def test_get_available_metrics(self):
        """Test get available metrics endpoint"""
        response = client.get("/api/v1/offline-evaluation/metrics/classification")
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_type"] == "classification"
        assert "available_metrics" in data
        assert "accuracy" in data["available_metrics"]
        assert "precision" in data["available_metrics"]


class TestOnlineEvaluationAPI:
    """Test cases for online evaluation API endpoints"""
    
    @patch('src.evaluation_engine.core.EvaluationEngine.ab_tester.create_experiment')
    def test_create_experiment(self, mock_create):
        """Test create experiment endpoint"""
        # Mock experiment result
        mock_experiment = Mock()
        mock_experiment.id = "exp_123"
        mock_experiment.name = "Test Experiment"
        mock_experiment.status = "draft"
        mock_create.return_value = mock_experiment
        
        # Test request
        request_data = {
            "name": "Test Experiment",
            "hypothesis": "Test hypothesis",
            "variants": ["control", "treatment"],
            "traffic_allocation": {"control": 0.5, "treatment": 0.5},
            "primary_metric": "conversion_rate",
            "minimum_detectable_effect": 0.05,
            "alpha": 0.05,
            "power": 0.8,
            "baseline_rate": 0.12,
            "start_date": "2024-01-01T00:00:00Z"
        }
        
        response = client.post("/api/v1/online-evaluation/experiments", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["experiment_id"] == "exp_123"
    
    @patch('src.evaluation_engine.core.EvaluationEngine.ab_tester.start_experiment')
    def test_start_experiment(self, mock_start):
        """Test start experiment endpoint"""
        mock_start.return_value = True
        
        response = client.post("/api/v1/online-evaluation/experiments/exp_123/start")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["experiment_id"] == "exp_123"
    
    @patch('src.evaluation_engine.core.EvaluationEngine.ab_tester.stop_experiment')
    def test_stop_experiment(self, mock_stop):
        """Test stop experiment endpoint"""
        mock_stop.return_value = True
        
        response = client.post("/api/v1/online-evaluation/experiments/exp_123/stop")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["experiment_id"] == "exp_123"
    
    @patch('src.evaluation_engine.core.EvaluationEngine.ab_tester.analyze_experiment')
    def test_analyze_experiment(self, mock_analyze):
        """Test analyze experiment endpoint"""
        # Mock analysis result
        mock_analysis = Mock()
        mock_analysis.dict.return_value = {
            "experiment_id": "exp_123",
            "analysis_type": "frequentist",
            "statistical_results": {"variant_results": {}},
            "significance_results": {"overall_significant": True},
            "recommendations": []
        }
        mock_analyze.return_value = mock_analysis
        
        response = client.post("/api/v1/online-evaluation/experiments/exp_123/analyze")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["experiment_id"] == "exp_123"
        assert "analysis" in data


class TestBusinessImpactAPI:
    """Test cases for business impact API endpoints"""
    
    @patch('src.evaluation_engine.core.EvaluationEngine.business_impact_analyzer.analyze')
    def test_analyze_business_impact(self, mock_analyze):
        """Test analyze business impact endpoint"""
        # Mock analysis result
        mock_analysis = Mock()
        mock_analysis.dict.return_value = {
            "intervention_date": "2024-01-01T00:00:00Z",
            "metric_impacts": {"revenue": {"revenue_change": 100000}},
            "overall_roi": 1.5,
            "statistical_significance": {"overall": True},
            "confidence_intervals": {}
        }
        mock_analyze.return_value = mock_analysis
        
        # Test request
        request_data = {
            "business_metrics": ["revenue", "engagement"],
            "evaluation_period": {"start": "2024-01-01", "end": "2024-01-31"}
        }
        
        response = client.post("/api/v1/business-impact/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "analysis" in data
    
    def test_get_business_metrics(self):
        """Test get business metrics endpoint"""
        response = client.get("/api/v1/business-impact/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "business_metrics" in data
        assert "revenue" in data["business_metrics"]
        assert "engagement" in data["business_metrics"]


class TestDriftDetectionAPI:
    """Test cases for drift detection API endpoints"""
    
    @patch('src.evaluation_engine.core.EvaluationEngine.drift_detector.analyze_drift')
    def test_analyze_drift(self, mock_analyze):
        """Test analyze drift endpoint"""
        # Mock analysis result
        mock_analysis = Mock()
        mock_analysis.dict.return_value = {
            "model_name": "test_model",
            "drift_results": {"data_drift": {"overall_drift_score": 0.3}},
            "drift_severity": 0.3,
            "recommendations": []
        }
        mock_analyze.return_value = mock_analysis
        
        # Test request
        request_data = {
            "models": [{"name": "test_model", "type": "classification"}],
            "drift_config": {
                "reference_period": {"start": "2024-01-01", "end": "2024-01-15"},
                "analysis_period": {"start": "2024-01-16", "end": "2024-01-31"},
                "significance_level": 0.05,
                "alert_threshold": 0.7
            }
        }
        
        response = client.post("/api/v1/drift-detection/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "analysis" in data
    
    def test_detect_data_drift(self):
        """Test detect data drift endpoint"""
        request_data = {
            "model_name": "test_model",
            "reference_data": {"features": [1, 2, 3]},
            "current_data": {"features": [1, 2, 3]},
            "significance_level": 0.05
        }
        
        response = client.post("/api/v1/drift-detection/data-drift", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "drift_results" in data
        assert data["drift_results"]["model_name"] == "test_model"


class TestMonitoringAPI:
    """Test cases for monitoring API endpoints"""
    
    def test_get_monitoring_status(self):
        """Test get monitoring status endpoint"""
        response = client.get("/api/v1/monitoring/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "monitoring_status" in data
    
    def test_get_monitoring_metrics(self):
        """Test get monitoring metrics endpoint"""
        response = client.get("/api/v1/monitoring/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "metrics" in data
    
    def test_get_alerts(self):
        """Test get alerts endpoint"""
        response = client.get("/api/v1/monitoring/alerts")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "alerts" in data


class TestDashboardAPI:
    """Test cases for dashboard API endpoints"""
    
    def test_get_dashboard_overview(self):
        """Test get dashboard overview endpoint"""
        response = client.get("/api/v1/dashboard/overview")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "overview" in data
    
    def test_get_experiments_dashboard(self):
        """Test get experiments dashboard endpoint"""
        response = client.get("/api/v1/dashboard/experiments")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "experiments" in data
    
    def test_get_metrics_trends(self):
        """Test get metrics trends endpoint"""
        response = client.get("/api/v1/dashboard/metrics/trends?metric_name=conversion_rate")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "trends" in data
        assert data["trends"]["metric_name"] == "conversion_rate"


if __name__ == "__main__":
    pytest.main([__file__])
