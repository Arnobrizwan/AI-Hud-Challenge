"""
Feature Models - Data models for feature store management
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FeatureType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IMAGE = "image"
    EMBEDDING = "embedding"
    TIMESTAMP = "timestamp"


class FeatureStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class TransformationType(str, Enum):
    NORMALIZATION = "normalization"
    SCALING = "scaling"
    ENCODING = "encoding"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"


class FeatureDefinition(BaseModel):
    """Feature definition"""

    name: str
    description: Optional[str] = None
    feature_type: FeatureType
    data_type: str  # int, float, string, bool, etc.
    is_required: bool = True
    default_value: Optional[Any] = None
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)


class FeatureTransformation(BaseModel):
    """Feature transformation definition"""

    type: TransformationType
    config: Dict[str, Any] = Field(default_factory=dict)
    order: int = 0


class FeatureConfig(BaseModel):
    """Configuration for feature set"""

    name: str
    description: Optional[str] = None
    feature_definitions: List[FeatureDefinition]
    source_config: Dict[str, Any] = Field(default_factory=dict)
    update_schedule: Optional[str] = None  # Cron expression
    retention_days: int = 365
    tags: Dict[str, str] = Field(default_factory=dict)


class FeatureSet(BaseModel):
    """Feature set instance"""

    id: str
    name: str
    description: Optional[str] = None
    features: List[FeatureDefinition]
    source_config: Dict[str, Any] = Field(default_factory=dict)
    update_schedule: Optional[str] = None
    status: FeatureStatus = FeatureStatus.ACTIVE

    # Vertex AI integration
    vertex_feature_set_name: Optional[str] = None

    # Pipeline integration
    ingestion_pipeline_id: Optional[str] = None

    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime] = None

    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    created_by: Optional[str] = None


class FeatureServingRequest(BaseModel):
    """Request for feature serving"""

    feature_store_name: str
    feature_set_names: List[str]
    entity_ids: List[str]
    transformations: Optional[List[FeatureTransformation]] = None
    cache_ttl: int = 3600  # 1 hour
    request_time: datetime = Field(default_factory=datetime.utcnow)


class FeatureVector(BaseModel):
    """Feature vector result"""

    entity_ids: List[str]
    feature_values: Dict[str, Any]
    served_at: datetime
    feature_set_names: List[str]
    cache_hit: bool = False


class FeatureValidationResult(BaseModel):
    """Result of feature validation"""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureQualityMetrics(BaseModel):
    """Feature quality metrics"""

    feature_set_id: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureLineage(BaseModel):
    """Feature lineage information"""

    feature_name: str
    source_datasets: List[str] = Field(default_factory=list)
    transformations: List[Dict[str, Any]] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureIngestionJob(BaseModel):
    """Feature ingestion job"""

    id: str
    feature_set_id: str
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    records_processed: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None


class FeatureStatistics(BaseModel):
    """Feature statistics"""

    feature_name: str
    count: int
    null_count: int
    unique_count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    percentiles: Dict[str, float] = Field(default_factory=dict)
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
