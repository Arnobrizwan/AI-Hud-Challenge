"""
Deployment Models - Data models for model deployment and serving
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DeploymentStrategy(str, Enum):
    STANDARD = "standard"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    AB_TEST = "ab_test"


class DeploymentStatus(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class CanaryConfig(BaseModel):
    """Configuration for canary deployment"""

    initial_traffic_percentage: int = 5
    traffic_stages: List[int] = Field(default_factory=lambda: [5, 10, 25, 50, 75, 100])
    stage_duration_minutes: int = 10
    success_threshold: float = 0.95
    failure_threshold: float = 0.05
    monitoring_metrics: List[str] = Field(default_factory=lambda: ["latency", "error_rate", "accuracy"])


class BlueGreenConfig(BaseModel):
    """Configuration for blue-green deployment"""

    switch_timeout: int = 300  # 5 minutes
    health_check_interval: int = 30  # 30 seconds
    health_check_timeout: int = 60  # 1 minute


class RollingConfig(BaseModel):
    """Configuration for rolling deployment"""

    batch_size: int = 1
    batch_interval_seconds: int = 60
    max_unavailable: int = 0
    max_surge: int = 1


class ABTestConfig(BaseModel):
    """Configuration for A/B test deployment"""

    control_version: str
    treatment_version: str
    control_traffic_percentage: int = 50
    treatment_traffic_percentage: int = 50
    test_duration_days: int = 7
    success_metric: str = "conversion_rate"
    success_threshold: float = 0.05  # 5% improvement


class DeploymentConfig(BaseModel):
    """Configuration for model deployment"""

    model_name: str
    model_version: str
    strategy: DeploymentStrategy = DeploymentStrategy.STANDARD
    environment: Environment = Environment.PRODUCTION

    # Deployment-specific configurations
    canary_config: Optional[CanaryConfig] = None
    blue_green_config: Optional[BlueGreenConfig] = None
    rolling_config: Optional[RollingConfig] = None
    ab_test_config: Optional[ABTestConfig] = None

    # Infrastructure configuration
    machine_type: str = "n1-standard-4"
    instance_count: int = 1
    min_instances: int = 1
    max_instances: int = 10
    accelerator_type: Optional[str] = None
    accelerator_count: int = 0

    # Environment variables
    environment_variables: Dict[str, str] = Field(default_factory=dict)

    # Monitoring configuration
    monitoring_config: Optional[Dict[str, Any]] = None

    # Health check configuration
    health_check_path: str = "/health"
    health_check_interval: int = 30
    health_check_timeout: int = 10
    health_check_retries: int = 3

    # Traffic configuration
    traffic_percentage: int = 100
    stable_endpoint: Optional[str] = None

    # Resource limits
    cpu_limit: str = "2"
    memory_limit: str = "4Gi"
    cpu_request: str = "1"
    memory_request: str = "2Gi"


class ModelDeployment(BaseModel):
    """Model deployment instance"""

    id: str
    model_name: str
    model_version: str
    deployment_strategy: DeploymentStrategy
    target_environment: Environment
    status: DeploymentStatus = DeploymentStatus.PENDING

    # Configuration
    config: DeploymentConfig

    # Endpoint information
    endpoint_url: Optional[str] = None
    endpoint_name: Optional[str] = None

    # Deployment details
    deployment_info: Optional[Dict[str, Any]] = None

    # Timestamps
    created_at: datetime
    deployed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Error handling
    error_message: Optional[str] = None

    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    created_by: Optional[str] = None


class DeploymentInfo(BaseModel):
    """Deployment information"""

    endpoint_url: str
    deployment_type: str
    traffic_percentage: int
    deployment_timestamp: datetime
    health_status: str = "healthy"

    # Additional deployment-specific info
    ab_test_info: Optional[Dict[str, Any]] = None
    canary_info: Optional[Dict[str, Any]] = None
    blue_green_info: Optional[Dict[str, Any]] = None
    rolling_info: Optional[Dict[str, Any]] = None


class DeploymentResult(BaseModel):
    """Result of deployment operation"""

    deployment_id: str
    endpoint_url: str
    status: DeploymentStatus
    deployment_info: DeploymentInfo
    created_at: datetime

    # Additional result information
    warnings: List[str] = Field(default_factory=list)
    rollback_available: bool = True


class HealthCheck(BaseModel):
    """Health check result"""

    is_healthy: bool
    response_time_ms: float
    error_rate: float
    failure_reason: Optional[str] = None
    checked_at: datetime = Field(default_factory=datetime.utcnow)


class TrafficSplit(BaseModel):
    """Traffic split configuration"""

    stable: int
    canary: int
    control: int = 0
    treatment: int = 0

    @property
    def total_percentage(self) -> int:
        return self.stable + self.canary + self.control + self.treatment

    def is_valid(self) -> bool:
        return self.total_percentage == 100


class DeploymentMetrics(BaseModel):
    """Deployment metrics"""

    deployment_id: str
    endpoint_url: str
    request_count: int
    success_count: int
    error_count: int
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    throughput_rps: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.success_count / self.request_count
