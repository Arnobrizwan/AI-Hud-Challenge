# MLOps Orchestration Service

A production-grade MLOps & Orchestration microservice built with FastAPI, Airflow, MLflow, Vertex AI Pipelines, and Kubernetes. This service manages the complete ML lifecycle including model training, deployment, monitoring, and automated retraining with comprehensive pipeline orchestration.

## 🚀 Features

### Core Orchestration
- **Complete ML Pipeline Orchestration** with Airflow/Vertex AI Pipelines
- **Automated Model Training** with hyperparameter optimization
- **Model Versioning and Registry Management** with MLflow
- **Automated Model Deployment** with blue-green, canary, and rolling strategies
- **A/B Testing Integration** for model experiments
- **Data Pipeline Management** and validation
- **Feature Store Integration** and management
- **Model Monitoring** and performance tracking
- **Automated Retraining Triggers** and scheduling
- **CI/CD Integration** for ML workflows

### Advanced Capabilities
- **Multi-Orchestrator Support**: Airflow, Vertex AI Pipelines, Kubeflow
- **Advanced Deployment Strategies**: Blue-Green, Canary, Rolling, A/B Testing
- **Comprehensive Monitoring**: Real-time metrics, drift detection, alerting
- **Feature Store Management**: Online/offline feature serving
- **Automated Retraining**: Performance, drift, scheduled, and data volume triggers
- **Resource Management**: Auto-scaling, resource optimization
- **Security**: Authentication, authorization, encryption
- **Observability**: Logging, metrics, tracing, dashboards

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLOps Orchestration Service                 │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Application Layer                                     │
│  ├── REST API Endpoints                                        │
│  ├── Authentication & Authorization                            │
│  ├── Rate Limiting & Middleware                               │
│  └── API Documentation                                         │
├─────────────────────────────────────────────────────────────────┤
│  Core Orchestration Layer                                      │
│  ├── Pipeline Orchestrator                                     │
│  ├── Training Orchestrator                                     │
│  ├── Deployment Manager                                        │
│  ├── Feature Store Manager                                     │
│  ├── Retraining Manager                                        │
│  └── Monitoring Service                                        │
├─────────────────────────────────────────────────────────────────┤
│  Integration Layer                                             │
│  ├── Airflow Client                                            │
│  ├── Vertex AI Client                                          │
│  ├── MLflow Client                                             │
│  ├── Kubernetes Client                                         │
│  └── Prometheus Client                                         │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                          │
│  ├── Kubernetes                                                │
│  ├── Docker Containers                                         │
│  ├── Redis Cache                                               │
│  ├── PostgreSQL Database                                       │
│  └── Cloud Storage                                             │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

### System Requirements
- Python 3.9+
- Docker 20.10+
- Kubernetes 1.20+
- PostgreSQL 13+
- Redis 6.0+

### Cloud Requirements
- Google Cloud Platform (for Vertex AI)
- MLflow Tracking Server
- Airflow Instance
- Prometheus & Grafana (optional)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd mlops-orchestration-service
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### 3. Docker Compose (Development)
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f mlops-service
```

### 4. Kubernetes Deployment (Production)
```bash
# Create namespace
kubectl create namespace mlops

# Apply configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n mlops
kubectl get services -n mlops
```

### 5. Verify Installation
```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

## 📖 Usage

### Creating ML Pipelines

```python
from src.models.pipeline_models import MLPipelineConfig, PipelineType

# Define pipeline configuration
pipeline_config = MLPipelineConfig(
    name="customer_churn_prediction",
    description="Predict customer churn using ML",
    pipeline_type=PipelineType.TRAINING,
    orchestrator="airflow",
    
    # Data configuration
    data_sources=[
        {
            "type": "bigquery",
            "query": "SELECT * FROM customer_data",
            "project_id": "my-project"
        }
    ],
    
    # Model configuration
    model_class="sklearn.ensemble.RandomForestClassifier",
    model_params={"n_estimators": 100, "random_state": 42},
    
    # Feature engineering
    include_feature_engineering=True,
    feature_definitions=[
        {
            "name": "customer_age",
            "type": "numerical",
            "transformation": "normalize"
        }
    ],
    
    # Training configuration
    include_training=True,
    enable_hyperparameter_tuning=True,
    hyperparameter_space={
        "n_estimators": {"type": "int", "low": 50, "high": 200},
        "max_depth": {"type": "int", "low": 3, "high": 20}
    },
    
    # Deployment configuration
    include_deployment=True,
    deployment_strategy="canary",
    
    # Monitoring configuration
    include_monitoring=True,
    monitoring_metrics=["accuracy", "precision", "recall", "f1"]
)

# Create pipeline
response = requests.post(
    "http://localhost:8000/api/v1/pipelines/",
    json={"config": pipeline_config.dict()}
)
```

### Triggering Pipeline Execution

```python
# Trigger pipeline execution
execution_params = {
    "data_version": "v1.2.3",
    "experiment_name": "churn_prediction_v2",
    "triggered_by": "scheduled"
}

response = requests.post(
    f"http://localhost:8000/api/v1/pipelines/{pipeline_id}/execute",
    json={"execution_params": execution_params}
)
```

### Model Deployment

```python
from src.models.deployment_models import DeploymentConfig, DeploymentStrategy

# Configure deployment
deployment_config = DeploymentConfig(
    model_name="customer_churn_prediction",
    model_version="v1.2.3",
    strategy=DeploymentStrategy.CANARY,
    environment="production",
    
    # Canary configuration
    canary_config={
        "initial_traffic_percentage": 5,
        "traffic_stages": [5, 10, 25, 50, 75, 100],
        "stage_duration_minutes": 10
    },
    
    # Infrastructure
    machine_type="n1-standard-4",
    instance_count=2,
    min_instances=1,
    max_instances=10
)

# Deploy model
response = requests.post(
    "http://localhost:8000/api/v1/deployment/deploy",
    json=deployment_config.dict()
)
```

### Setting Up Monitoring

```python
from src.models.monitoring_models import MonitoringConfig, AlertRule, AlertSeverity

# Configure monitoring
monitoring_config = MonitoringConfig(
    model_name="customer_churn_prediction",
    monitoring_interval_seconds=60,
    drift_detection_enabled=True,
    
    # Alert rules
    alert_rules=[
        AlertRule(
            name="accuracy_drop",
            alert_type="performance",
            severity=AlertSeverity.HIGH,
            metric_name="accuracy",
            operator="less_than",
            threshold=0.85,
            duration_minutes=5
        ),
        AlertRule(
            name="data_drift",
            alert_type="drift",
            severity=AlertSeverity.MEDIUM,
            metric_name="drift_score",
            operator="greater_than",
            threshold=0.1,
            duration_minutes=10
        )
    ]
)

# Setup monitoring
response = requests.post(
    "http://localhost:8000/api/v1/monitoring/setup",
    json=monitoring_config.dict()
)
```

### Automated Retraining

```python
from src.models.retraining_models import RetrainingTriggerConfig

# Configure retraining triggers
trigger_config = RetrainingTriggerConfig(
    # Performance trigger
    performance_threshold=0.8,
    performance_metric="accuracy",
    evaluation_window=60,
    
    # Data drift trigger
    data_drift_threshold=0.15,
    drift_monitoring_features=["customer_age", "transaction_amount"],
    
    # Scheduled trigger
    retraining_schedule="0 2 * * 0",  # Weekly on Sunday at 2 AM
    
    # Data volume trigger
    new_data_threshold=10000
)

# Setup retraining triggers
response = requests.post(
    f"http://localhost:8000/api/v1/retraining/setup/{model_name}",
    json=trigger_config.dict()
)
```

## 🔧 Configuration

### Environment Variables

```bash
# Application
APP_NAME="MLOps Orchestration Service"
ENVIRONMENT="production"
DEBUG=false
LOG_LEVEL="INFO"

# Database
DATABASE_URL="postgresql://user:password@db:5432/mlops"
REDIS_URL="redis://redis:6379"

# Airflow
AIRFLOW_WEBSERVER_URL="http://airflow-webserver:8080"
AIRFLOW_API_URL="http://airflow-webserver:8080/api/v1"

# Vertex AI
VERTEX_AI_PROJECT_ID="my-project-id"
VERTEX_AI_REGION="us-central1"
VERTEX_AI_SERVICE_ACCOUNT="path/to/service-account.json"

# MLflow
MLFLOW_TRACKING_URI="http://mlflow-server:5000"
MLFLOW_REGISTRY_URI="http://mlflow-server:5000"

# Monitoring
PROMETHEUS_URL="http://prometheus:9090"
GRAFANA_URL="http://grafana:3000"

# Security
SECRET_KEY="your-secret-key-here"
JWT_ALGORITHM="HS256"
JWT_EXPIRATION_HOURS=24
```

### Kubernetes Configuration

The service is configured with:
- **Horizontal Pod Autoscaler**: Auto-scales based on CPU and memory usage
- **Resource Limits**: CPU and memory limits for optimal performance
- **Health Checks**: Liveness and readiness probes
- **Security**: Non-root user, security contexts, RBAC
- **Persistence**: PVCs for logs and model storage
- **Networking**: Service and ingress configuration

## 📊 Monitoring and Observability

### Metrics
- **Application Metrics**: Request count, latency, error rate
- **ML Metrics**: Model accuracy, precision, recall, F1-score
- **Resource Metrics**: CPU, memory, disk usage
- **Business Metrics**: Prediction count, model performance

### Dashboards
- **Grafana Dashboards**: Pre-built dashboards for monitoring
- **Custom Dashboards**: Configurable monitoring dashboards
- **Real-time Metrics**: Live monitoring of ML pipelines

### Alerting
- **Performance Alerts**: Model performance degradation
- **Drift Alerts**: Data drift detection
- **System Alerts**: Infrastructure and application issues
- **Custom Alerts**: Configurable alert rules

## 🧪 Testing

### Unit Tests
```bash
# Run unit tests
pytest tests/unit/

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

### Integration Tests
```bash
# Run integration tests
pytest tests/integration/

# Run with test database
pytest tests/integration/ --database-url=postgresql://test:test@localhost/test_db
```

### End-to-End Tests
```bash
# Run E2E tests
pytest tests/e2e/

# Run with Docker Compose
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## 🚀 Deployment

### Development
```bash
# Using Docker Compose
docker-compose up -d

# Using local Python
pip install -r requirements.txt
python -m uvicorn src.main:app --reload
```

### Staging
```bash
# Deploy to staging
kubectl apply -f k8s/staging/

# Check deployment
kubectl get pods -n mlops-staging
```

### Production
```bash
# Deploy to production
kubectl apply -f k8s/production/

# Verify deployment
kubectl get pods -n mlops
kubectl get services -n mlops
```

## 🔒 Security

### Authentication
- JWT-based authentication
- API key authentication
- OAuth2 integration

### Authorization
- Role-based access control (RBAC)
- Resource-level permissions
- API endpoint protection

### Data Protection
- Encryption at rest and in transit
- Secure secret management
- Data anonymization

## 📈 Performance

### Scalability
- Horizontal pod autoscaling
- Load balancing
- Resource optimization

### Performance Requirements
- **Pipeline Execution Latency**: < 30 seconds overhead
- **Model Deployment Time**: < 5 minutes
- **Concurrent Experiments**: 100+ supported
- **Auto-scaling**: Based on resource utilization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
black src/
isort src/
flake8 src/
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [API Documentation](http://localhost:8000/docs)
- **Issues**: [GitHub Issues](https://github.com/your-org/mlops-orchestration-service/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/mlops-orchestration-service/discussions)

## 🙏 Acknowledgments

- FastAPI for the web framework
- Apache Airflow for workflow orchestration
- MLflow for experiment tracking
- Google Vertex AI for cloud ML services
- Kubernetes for container orchestration
