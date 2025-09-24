"""
Retraining Models - Data models for automated retraining
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from uuid import UUID
from enum import Enum
from pydantic import BaseModel, Field

class TriggerType(str, Enum):
    PERFORMANCE = "performance"
    DATA_DRIFT = "data_drift"
    SCHEDULED = "scheduled"
    DATA_VOLUME = "data_volume"
    MANUAL = "manual"

class TriggerStatus(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    FIRED = "fired"
    ERROR = "error"

class RetrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RetrainingTriggerConfig(BaseModel):
    """Configuration for retraining triggers"""
    # Performance trigger
    performance_threshold: Optional[float] = None
    performance_metric: str = "accuracy"
    evaluation_window: int = 60  # minutes
    
    # Data drift trigger
    data_drift_threshold: Optional[float] = None
    drift_monitoring_features: List[str] = Field(default_factory=list)
    
    # Scheduled trigger
    retraining_schedule: Optional[str] = None  # Cron expression
    
    # Data volume trigger
    new_data_threshold: Optional[int] = None
    
    # General settings
    cooldown_period: int = 3600  # 1 hour
    max_retraining_frequency: int = 24  # hours

class RetrainingTrigger(BaseModel):
    """Base retraining trigger"""
    id: str
    model_name: str
    trigger_type: TriggerType
    status: TriggerStatus = TriggerStatus.ACTIVE
    created_at: datetime
    last_fired_at: Optional[datetime] = None
    fire_count: int = 0

class PerformanceTrigger(RetrainingTrigger):
    """Performance degradation trigger"""
    metric: str
    threshold: float
    evaluation_window: int  # minutes

class DataDriftTrigger(RetrainingTrigger):
    """Data drift trigger"""
    drift_threshold: float
    monitoring_features: List[str]

class ScheduledTrigger(RetrainingTrigger):
    """Scheduled retraining trigger"""
    schedule: str  # Cron expression

class DataVolumeTrigger(RetrainingTrigger):
    """Data volume trigger"""
    threshold: int  # Number of new records

class RetrainingResult(BaseModel):
    """Result of retraining operation"""
    id: str
    model_name: str
    trigger_id: str
    trigger_type: TriggerType
    status: RetrainingStatus = RetrainingStatus.PENDING
    
    # Training configuration
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # Training results
    training_result: Optional[Dict[str, Any]] = None
    
    # Model comparison
    ab_test_id: Optional[str] = None
    performance_comparison: Optional[Dict[str, Any]] = None
    
    # Timestamps
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration: Optional[datetime] = None
    
    # Error handling
    error_message: Optional[str] = None
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)

class RetrainingJob(BaseModel):
    """Retraining job instance"""
    id: str
    model_name: str
    trigger: RetrainingTrigger
    status: RetrainingStatus = RetrainingStatus.PENDING
    
    # Job configuration
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution details
    training_job_id: Optional[str] = None
    deployment_job_id: Optional[str] = None
    
    # Results
    result: Optional[RetrainingResult] = None
    
    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Priority and scheduling
    priority: int = 0
    scheduled_at: Optional[datetime] = None
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

class ModelComparison(BaseModel):
    """Model comparison result"""
    model_name: str
    current_version: str
    new_version: str
    
    # Performance metrics
    current_metrics: Dict[str, float] = Field(default_factory=dict)
    new_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Statistical significance
    is_significant: bool = False
    p_value: Optional[float] = None
    confidence_interval: Optional[Dict[str, float]] = None
    
    # Recommendation
    recommendation: str  # keep_current, deploy_new, need_more_data
    confidence: float
    
    # Comparison details
    comparison_timestamp: datetime = Field(default_factory=datetime.utcnow)
    test_duration_hours: float = 0.0

class RetrainingSchedule(BaseModel):
    """Retraining schedule configuration"""
    model_name: str
    schedule: str  # Cron expression
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RetrainingMetrics(BaseModel):
    """Retraining metrics and statistics"""
    model_name: str
    total_retrainings: int = 0
    successful_retrainings: int = 0
    failed_retrainings: int = 0
    average_duration_minutes: float = 0.0
    last_retraining: Optional[datetime] = None
    trigger_fire_counts: Dict[str, int] = Field(default_factory=dict)
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
