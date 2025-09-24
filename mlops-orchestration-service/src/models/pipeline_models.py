"""
Pipeline Models - Data models for ML pipeline orchestration
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field


class PipelineType(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    BATCH_PREDICTION = "batch_prediction"
    DATA_VALIDATION = "data_validation"
    FEATURE_ENGINEERING = "feature_engineering"


class PipelineStatus(str, Enum):
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    CANCELLED = "cancelled"


class ComponentType(str, Enum):
    DATA_VALIDATION = "data_validation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_MONITORING = "model_monitoring"


class MLPipelineConfig(BaseModel):
    """Configuration for ML pipeline"""

    name: str
    description: Optional[str] = None
    pipeline_type: PipelineType
    orchestrator: str = "airflow"  # airflow, vertex_ai, kubeflow

    # Component flags
    include_data_validation: bool = True
    include_feature_engineering: bool = True
    include_training: bool = True
    include_validation: bool = True
    include_deployment: bool = False
    include_monitoring: bool = False

    # Data configuration
    data_sources: List[Dict[str, Any]] = Field(default_factory=list)
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    data_quality_threshold: float = 0.8

    # Feature engineering
    feature_definitions: List[Dict[str, Any]] = Field(default_factory=list)
    transformation_pipeline: List[Dict[str, Any]] = Field(default_factory=list)
    feature_store_config: Optional[Dict[str, Any]] = None

    # Training configuration
    model_class: Optional[str] = None
    model_params: Dict[str, Any] = Field(default_factory=dict)
    training_data_config: Dict[str, Any] = Field(default_factory=dict)
    hyperparameter_tuning: bool = False
    experiment_config: Dict[str, Any] = Field(default_factory=dict)

    # Validation configuration
    validation_metrics: List[str] = Field(default_factory=list)
    model_quality_threshold: float = 0.8
    validation_data_config: Dict[str, Any] = Field(default_factory=dict)

    # Deployment configuration
    deployment_strategy: str = "standard"
    deployment_config: Dict[str, Any] = Field(default_factory=dict)
    environment_config: Dict[str, Any] = Field(default_factory=dict)

    # Monitoring configuration
    monitoring_metrics: List[str] = Field(default_factory=list)
    alerting_config: Dict[str, Any] = Field(default_factory=dict)
    monitoring_dashboard: Dict[str, Any] = Field(default_factory=dict)

    # Scheduling
    schedule_interval: Optional[str] = None
    max_retries: int = 3
    retry_delay: int = 300  # seconds

    # Timeouts
    data_validation_timeout: int = 1800  # 30 minutes
    feature_engineering_timeout: int = 3600  # 1 hour
    training_timeout: int = 7200  # 2 hours
    validation_timeout: int = 1800  # 30 minutes
    deployment_timeout: int = 1800  # 30 minutes
    monitoring_timeout: int = 300  # 5 minutes

    # Parameters and outputs
    parameters: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)

    # Monitoring flags
    monitoring_enabled: bool = True


class PipelineComponent(BaseModel):
    """Individual pipeline component"""

    id: str
    name: str
    component_type: ComponentType
    config: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    timeout: int = 3600  # seconds
    retries: int = 3
    resources: Dict[str, Any] = Field(default_factory=dict)


class MLPipeline(BaseModel):
    """ML Pipeline instance"""

    id: str
    name: str
    description: Optional[str] = None
    pipeline_type: PipelineType
    orchestrator: str
    status: PipelineStatus = PipelineStatus.CREATED

    # Components
    components: List[PipelineComponent] = Field(default_factory=list)

    # Orchestration IDs
    airflow_dag_id: Optional[str] = None
    vertex_pipeline_id: Optional[str] = None
    kubeflow_pipeline_id: Optional[str] = None

    # Configuration
    config: MLPipelineConfig

    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime] = None

    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    created_by: Optional[str] = None


class PipelineExecution(BaseModel):
    """Pipeline execution instance"""

    id: str
    pipeline_id: str
    execution_params: Dict[str, Any] = Field(default_factory=dict)
    status: PipelineStatus = PipelineStatus.RUNNING

    # External execution ID
    external_run_id: Optional[str] = None

    # Timestamps
    started_at: datetime
    completed_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    # Execution details
    triggered_by: str = "manual"
    error_message: Optional[str] = None

    # Results
    outputs: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # Duration
    duration: Optional[int] = None  # seconds

    @property
    def is_running(self) -> bool:
        return self.status == PipelineStatus.RUNNING

    @property
    def is_completed(self) -> bool:
        return self.status == PipelineStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        return self.status == PipelineStatus.FAILED
