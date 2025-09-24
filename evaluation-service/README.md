# Evaluation Suite Microservice

A comprehensive evaluation system for ML pipelines with offline evaluation, online A/B testing, business impact analysis, and real-time monitoring.

## Features

### 🧪 Offline Model Evaluation
- Comprehensive metrics for ranking, classification, regression, recommendation, and clustering models
- Cross-validation and feature importance analysis
- Statistical significance testing with confidence intervals
- Performance by segment analysis

### 🎯 Online A/B Testing
- Advanced A/B testing framework with statistical rigor
- Multi-armed bandit testing
- Sequential testing with early stopping
- Bayesian testing methods
- Real-time experiment monitoring

### 💼 Business Impact Analysis
- ROI calculation and revenue impact analysis
- User engagement and retention impact measurement
- Causal inference using difference-in-differences
- Content consumption impact analysis

### 📊 Model Drift Detection
- Data drift detection using statistical tests
- Prediction drift monitoring
- Performance drift analysis
- Concept drift detection
- Automated alerting system

### 📈 Real-time Monitoring
- Live experiment monitoring
- Anomaly detection and alerting
- Performance metrics tracking
- Dashboard integration with Grafana

## Architecture

```
src/
├── main.py                 # FastAPI application
├── evaluation_engine/     # Core evaluation orchestration
├── offline_evaluation/    # Offline model evaluation
├── online_evaluation/     # A/B testing and online evaluation
├── business_impact/       # Business impact analysis
├── drift_detection/       # Model drift detection
├── statistical_testing/   # Statistical methods and tests
├── causal_inference/      # Causal impact analysis
├── monitoring/            # Real-time monitoring and alerting
└── visualization/         # Dashboard and reporting
```

## Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- PostgreSQL
- Redis
- MLflow

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd evaluation-service
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run with Docker Compose:
```bash
docker-compose up -d
```

5. Access the service:
- API Documentation: http://localhost:8000/docs
- Grafana Dashboard: http://localhost:3000
- MLflow UI: http://localhost:5000

## API Usage

### Create a Comprehensive Evaluation

```python
import requests

# Create evaluation
response = requests.post("http://localhost:8000/api/v1/evaluation/comprehensive", json={
    "config": {
        "include_offline": True,
        "include_online": True,
        "include_business_impact": True,
        "include_drift_analysis": True,
        "include_causal_analysis": True,
        "models": [
            {
                "name": "ranking_model",
                "type": "ranking",
                "version": "1.0.0"
            }
        ],
        "datasets": [
            {
                "name": "test_dataset",
                "type": "ranking",
                "n_samples": 1000
            }
        ],
        "metrics": {
            "metric_types": ["ranking"],
            "segments": ["user_segment_1", "user_segment_2"]
        }
    }
})

evaluation_id = response.json()["evaluation_id"]
```

### Create an A/B Test Experiment

```python
# Create experiment
response = requests.post("http://localhost:8000/api/v1/online-evaluation/experiments", json={
    "name": "Homepage Redesign Test",
    "hypothesis": "New homepage design will increase conversion rate",
    "variants": ["control", "treatment"],
    "traffic_allocation": {"control": 0.5, "treatment": 0.5},
    "primary_metric": "conversion_rate",
    "secondary_metrics": ["engagement_rate", "bounce_rate"],
    "minimum_detectable_effect": 0.05,
    "alpha": 0.05,
    "power": 0.8,
    "baseline_rate": 0.12,
    "start_date": "2024-01-01T00:00:00Z"
})

experiment_id = response.json()["experiment_id"]

# Start experiment
requests.post(f"http://localhost:8000/api/v1/online-evaluation/experiments/{experiment_id}/start")

# Record events
requests.post(f"http://localhost:8000/api/v1/online-evaluation/experiments/{experiment_id}/events", json={
    "user_id": "user_123",
    "variant": "treatment",
    "event_type": "conversion",
    "value": 1.0
})

# Analyze experiment
analysis = requests.post(f"http://localhost:8000/api/v1/online-evaluation/experiments/{experiment_id}/analyze")
```

### Monitor Model Drift

```python
# Analyze drift
response = requests.post("http://localhost:8000/api/v1/drift-detection/analyze", json={
    "models": [
        {
            "name": "ranking_model",
            "type": "ranking"
        }
    ],
    "drift_config": {
        "reference_period": {
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-01-15T23:59:59Z"
        },
        "analysis_period": {
            "start": "2024-01-16T00:00:00Z",
            "end": "2024-01-31T23:59:59Z"
        },
        "significance_level": 0.05,
        "alert_threshold": 0.7
    }
})
```

## Configuration

### Environment Variables

```bash
# Application
DEBUG=false
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/evaluation_db

# Cache
REDIS_URL=redis://localhost:6379

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_REGISTRY_URI=http://localhost:5000

# BigQuery
BIGQUERY_PROJECT_ID=your-project-id
BIGQUERY_DATASET_ID=evaluation_dataset

# Vertex AI
VERTEX_AI_PROJECT_ID=your-project-id
VERTEX_AI_LOCATION=us-central1

# Monitoring
PROMETHEUS_ENDPOINT=http://localhost:9090
GRAFANA_ENDPOINT=http://localhost:3000
```

## Development

### Running Tests

```bash
pytest tests/ -v --cov=src/
```

### Code Formatting

```bash
black src/
isort src/
flake8 src/
```

### Type Checking

```bash
mypy src/
```

## Performance Requirements

- Process 1M+ evaluation events per hour
- Real-time dashboard updates < 1 second
- Statistical test computation < 10 seconds
- Support 100+ concurrent experiments

## Monitoring and Alerting

The service includes comprehensive monitoring and alerting:

- **Prometheus** for metrics collection
- **Grafana** for visualization dashboards
- **MLflow** for experiment tracking
- **Real-time alerts** for anomalies and drift

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please open an issue in the GitHub repository.
