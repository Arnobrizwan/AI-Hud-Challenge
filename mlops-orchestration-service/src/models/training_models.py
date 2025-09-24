"""
Training Models - Data models for model training and hyperparameter optimization
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from uuid import UUID
from enum import Enum
from pydantic import BaseModel, Field

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    REINFORCEMENT_LEARNING = "reinforcement_learning"

class OptimizationMetric(str, Enum):
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    ROC_AUC = "roc_auc"
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2_SCORE = "r2_score"

class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class HyperparameterSpace(BaseModel):
    """Hyperparameter search space definition"""
    param_name: str
    param_type: str  # categorical, int, float
    choices: Optional[List[Any]] = None  # for categorical
    low: Optional[float] = None  # for int/float
    high: Optional[float] = None  # for int/float
    step: Optional[float] = None  # for int/float

class TrainingConfig(BaseModel):
    """Configuration for model training"""
    model_name: str
    description: Optional[str] = None
    task_type: TaskType
    
    # Model configuration
    model_class: str
    model_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Data configuration
    data_sources: List[Dict[str, Any]] = Field(default_factory=list)
    feature_store_config: Optional[Dict[str, Any]] = None
    train_split: float = 0.7
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Training parameters
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_loss"
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = False
    hyperparameter_space: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    hyperparameter_trials: int = 100
    hyperparameter_timeout: int = 3600  # 1 hour
    optimization_metric: OptimizationMetric = OptimizationMetric.ACCURACY
    cv_folds: int = 5
    
    # Quality thresholds
    quality_threshold: float = 0.8
    primary_metric: str = "accuracy"
    
    # Resource configuration
    machine_type: str = "n1-standard-4"
    accelerator_type: Optional[str] = None
    accelerator_count: int = 0
    
    # Experiment tracking
    experiment_name: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)

class TrainingData(BaseModel):
    """Training data container"""
    train_set: Any  # Would be a proper dataset class
    validation_set: Any
    test_set: Any
    feature_count: int
    feature_names: List[str]
    target_name: str

class ModelTrainingResult(BaseModel):
    """Result of model training"""
    model: Any  # Trained model object
    model_path: str
    training_duration: datetime
    training_history: Dict[str, List[float]] = Field(default_factory=dict)
    experiment_id: str
    run_id: Optional[str] = None

class ModelEvaluationResult(BaseModel):
    """Result of model evaluation"""
    metrics: Dict[str, float] = Field(default_factory=dict)
    primary_metric: str
    primary_value: float
    meets_threshold: bool
    quality_threshold: float
    evaluation_timestamp: datetime
    
    def meets_quality_threshold(self, threshold: float) -> bool:
        """Check if model meets quality threshold"""
        return self.primary_value >= threshold

class TrainingResult(BaseModel):
    """Complete training result"""
    id: str
    model_name: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    
    # Training data info
    training_data_info: Optional[Dict[str, Any]] = None
    
    # Experiment tracking
    experiment_id: Optional[str] = None
    
    # Hyperparameter optimization
    best_hyperparameters: Optional[Dict[str, Any]] = None
    hyperparameter_trials: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Training results
    model_result: Optional[ModelTrainingResult] = None
    evaluation_result: Optional[ModelEvaluationResult] = None
    
    # Model registry
    registered_model_version: Optional[Dict[str, Any]] = None
    
    # Timestamps
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration: Optional[datetime] = None
    
    # Error handling
    error_message: Optional[str] = None
    
    @property
    def is_completed(self) -> bool:
        return self.status == TrainingStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        return self.status == TrainingStatus.FAILED
