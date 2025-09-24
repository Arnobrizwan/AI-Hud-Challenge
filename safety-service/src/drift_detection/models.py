"""
Drift Detection Models
Data models for drift detection system
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field


class BaseDriftModel(BaseModel):
    """Base model for drift detection"""

    model_config = {"arbitrary_types_allowed": True}

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DriftDetectionRequest(BaseDriftModel):
    """Request for drift detection"""

    reference_data: pd.DataFrame
    current_data: pd.DataFrame
    features_to_monitor: List[str]
    reference_labels: Optional[pd.Series] = None
    current_labels: Optional[pd.Series] = None
    reference_predictions: Optional[pd.Series] = None
    current_predictions: Optional[pd.Series] = None
    reference_model: Optional[Any] = None
    current_model: Optional[Any] = None


class StatisticalTestResult(BaseDriftModel):
    """Result of a statistical test"""

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    drift_score: float
    threshold: float


class FeatureDriftResult(BaseDriftModel):
    """Result of drift detection for a single feature"""

    feature_name: str
    test_results: Dict[str, StatisticalTestResult]
    drift_magnitude: float
    is_drifted: bool
    drift_score: float


class DataDriftResult(BaseDriftModel):
    """Result of data drift detection"""

    feature_results: Dict[str, FeatureDriftResult]
    overall_drift_score: float
    drifted_features: List[str]


class ConceptDriftResult(BaseDriftModel):
    """Result of concept drift detection"""

    model_config = {"protected_namespaces": ()}

    drift_score: float
    is_drifted: bool
    feature_importance_changes: Dict[str, float]
    model_performance_change: float


class PredictionDriftResult(BaseDriftModel):
    """Result of prediction drift detection"""

    drift_score: float
    is_drifted: bool
    prediction_distribution_change: float
    accuracy_change: float


class ImportanceDriftResult(BaseDriftModel):
    """Result of feature importance drift detection"""

    drift_score: float
    is_drifted: bool
    importance_changes: Dict[str, float]
    top_changed_features: List[str]


class DriftAnalysisResult(BaseDriftModel):
    """Comprehensive drift analysis result"""

    data_drift: DataDriftResult
    concept_drift: ConceptDriftResult
    prediction_drift: PredictionDriftResult
    importance_drift: ImportanceDriftResult
    overall_severity: float
    requires_action: bool
    analysis_timestamp: datetime
