"""
Configuration settings for MLOps Orchestration Service
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class DatabaseType(str, Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"

class OrchestratorType(str, Enum):
    AIRFLOW = "airflow"
    VERTEX_AI = "vertex_ai"
    KUBEFLOW = "kubeflow"

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = Field(default="MLOps Orchestration Service", env="APP_NAME")
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_type: DatabaseType = Field(default=DatabaseType.POSTGRESQL, env="DATABASE_TYPE")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Airflow
    airflow_webserver_url: str = Field(default="http://localhost:8080", env="AIRFLOW_WEBSERVER_URL")
    airflow_api_url: str = Field(default="http://localhost:8080/api/v1", env="AIRFLOW_API_URL")
    airflow_username: str = Field(default="admin", env="AIRFLOW_USERNAME")
    airflow_password: str = Field(default="admin", env="AIRFLOW_PASSWORD")
    airflow_dag_folder: str = Field(default="/opt/airflow/dags", env="AIRFLOW_DAG_FOLDER")
    
    # Vertex AI
    vertex_ai_project_id: str = Field(..., env="VERTEX_AI_PROJECT_ID")
    vertex_ai_region: str = Field(default="us-central1", env="VERTEX_AI_REGION")
    vertex_ai_staging_bucket: str = Field(..., env="VERTEX_AI_STAGING_BUCKET")
    vertex_ai_service_account: str = Field(..., env="VERTEX_AI_SERVICE_ACCOUNT")
    
    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_registry_uri: str = Field(default="http://localhost:5000", env="MLFLOW_REGISTRY_URI")
    mlflow_experiment_prefix: str = Field(default="mlops", env="MLFLOW_EXPERIMENT_PREFIX")
    
    # Kubernetes
    kubernetes_namespace: str = Field(default="mlops", env="KUBERNETES_NAMESPACE")
    kubernetes_config_path: Optional[str] = Field(default=None, env="KUBECONFIG")
    
    # Feature Store
    feature_store_project_id: str = Field(..., env="FEATURE_STORE_PROJECT_ID")
    feature_store_location: str = Field(default="us-central1", env="FEATURE_STORE_LOCATION")
    feature_store_online_serving_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "fixed_node_count": 1,
            "scaling": {
                "min_node_count": 1,
                "max_node_count": 10
            }
        },
        env="FEATURE_STORE_ONLINE_SERVING_CONFIG"
    )
    
    # Model Registry
    model_registry_type: str = Field(default="mlflow", env="MODEL_REGISTRY_TYPE")
    model_storage_bucket: str = Field(..., env="MODEL_STORAGE_BUCKET")
    model_artifact_prefix: str = Field(default="models", env="MODEL_ARTIFACT_PREFIX")
    
    # Monitoring
    monitoring_enabled: bool = Field(default=True, env="MONITORING_ENABLED")
    prometheus_url: str = Field(default="http://localhost:9090", env="PROMETHEUS_URL")
    grafana_url: str = Field(default="http://localhost:3000", env="GRAFANA_URL")
    alert_manager_url: str = Field(default="http://localhost:9093", env="ALERT_MANAGER_URL")
    
    # Retraining
    retraining_check_interval: int = Field(default=300, env="RETRAINING_CHECK_INTERVAL")  # 5 minutes
    retraining_max_concurrent: int = Field(default=3, env="RETRAINING_MAX_CONCURRENT")
    retraining_timeout: int = Field(default=3600, env="RETRAINING_TIMEOUT")  # 1 hour
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Resource Management
    max_concurrent_pipelines: int = Field(default=10, env="MAX_CONCURRENT_PIPELINES")
    max_concurrent_training_jobs: int = Field(default=5, env="MAX_CONCURRENT_TRAINING_JOBS")
    max_concurrent_deployments: int = Field(default=3, env="MAX_CONCURRENT_DEPLOYMENTS")
    
    # Pipeline Orchestration
    default_orchestrator: OrchestratorType = Field(default=OrchestratorType.AIRFLOW, env="DEFAULT_ORCHESTRATOR")
    pipeline_execution_timeout: int = Field(default=7200, env="PIPELINE_EXECUTION_TIMEOUT")  # 2 hours
    pipeline_retry_attempts: int = Field(default=3, env="PIPELINE_RETRY_ATTEMPTS")
    
    # A/B Testing
    ab_testing_enabled: bool = Field(default=True, env="AB_TESTING_ENABLED")
    ab_test_traffic_split_min: float = Field(default=0.05, env="AB_TEST_TRAFFIC_SPLIT_MIN")
    ab_test_traffic_split_max: float = Field(default=0.5, env="AB_TEST_TRAFFIC_SPLIT_MAX")
    
    # Data Validation
    data_validation_enabled: bool = Field(default=True, env="DATA_VALIDATION_ENABLED")
    data_quality_threshold: float = Field(default=0.8, env="DATA_QUALITY_THRESHOLD")
    
    # Feature Engineering
    feature_engineering_enabled: bool = Field(default=True, env="FEATURE_ENGINEERING_ENABLED")
    feature_cache_ttl: int = Field(default=3600, env="FEATURE_CACHE_TTL")  # 1 hour
    
    # Model Deployment
    deployment_strategies: List[str] = Field(
        default=["blue_green", "canary", "rolling", "standard"],
        env="DEPLOYMENT_STRATEGIES"
    )
    deployment_health_check_timeout: int = Field(default=300, env="DEPLOYMENT_HEALTH_CHECK_TIMEOUT")
    deployment_rollback_enabled: bool = Field(default=True, env="DEPLOYMENT_ROLLBACK_ENABLED")
    
    # Performance Requirements
    pipeline_execution_latency_threshold: int = Field(default=30, env="PIPELINE_EXECUTION_LATENCY_THRESHOLD")
    model_deployment_time_threshold: int = Field(default=300, env="MODEL_DEPLOYMENT_TIME_THRESHOLD")  # 5 minutes
    max_concurrent_experiments: int = Field(default=100, env="MAX_CONCURRENT_EXPERIMENTS")
    
    @validator('environment')
    def validate_environment(cls, v):
        if v not in [e.value for e in Environment]:
            raise ValueError(f"Invalid environment: {v}")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()
    
    @validator('deployment_strategies')
    def validate_deployment_strategies(cls, v):
        valid_strategies = ['blue_green', 'canary', 'rolling', 'standard']
        for strategy in v:
            if strategy not in valid_strategies:
                raise ValueError(f"Invalid deployment strategy: {strategy}")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        validate_assignment = True

# Global settings instance
settings = Settings()
