"""
Data models for Evaluation Suite Microservice
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


class EvaluationStatus(str, Enum):
    """Evaluation status enumeration"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelType(str, Enum):
    """Model type enumeration"""

    RANKING = "ranking"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RECOMMENDATION = "recommendation"
    CLUSTERING = "clustering"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"


class EvaluationType(str, Enum):
    """Evaluation type enumeration"""

    OFFLINE = "offline"
    ONLINE = "online"
    COMPREHENSIVE = "comprehensive"
    DRIFT = "drift"
    BUSINESS_IMPACT = "business_impact"
    CAUSAL = "causal"


class StatisticalTestType(str, Enum):
    """Statistical test type enumeration"""

    T_TEST = "t_test"
    WELCH_T_TEST = "welch_t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    WILCOXON = "wilcoxon"


class ExperimentStatus(str, Enum):
    """Experiment status enumeration"""

    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


# Base models
class BaseEvaluationModel(BaseModel):
    """Base model for evaluation entities"""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True
        validate_assignment = True


# Configuration models
class EvaluationConfig(BaseModel):
    """Configuration for comprehensive evaluation"""

    include_offline: bool = True
    include_online: bool = True
    include_business_impact: bool = True
    include_drift_analysis: bool = True
    include_causal_analysis: bool = True

    # Model and dataset configuration
    models: List[Dict[str, Any]] = Field(default_factory=list)
    datasets: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # Online evaluation configuration
    online_experiments: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_period: Dict[str, Any] = Field(default_factory=dict)

    # Business impact configuration
    business_metrics: List[str] = Field(default_factory=list)

    # Drift detection configuration
    drift_config: Dict[str, Any] = Field(default_factory=dict)

    # Causal analysis configuration
    causal_config: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    created_by: str = Field(default="system")
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class MetricsConfig(BaseModel):
    """Configuration for metrics calculation"""

    metric_types: List[str] = Field(default_factory=list)
    segments: List[str] = Field(default_factory=list)
    confidence_level: float = Field(default=0.95)
    bootstrap_samples: int = Field(default=1000)


class DriftConfig(BaseModel):
    """Configuration for drift detection"""

    reference_period: Dict[str, datetime]
    analysis_period: Dict[str, datetime]
    significance_level: float = Field(default=0.05)
    alert_threshold: float = Field(default=0.7)
    window_size: int = Field(default=1000)


class ExperimentConfig(BaseModel):
    """Configuration for A/B testing experiments"""

    name: str
    hypothesis: str
    variants: List[str] = Field(default_factory=list)
    traffic_allocation: Dict[str, float] = Field(default_factory=dict)
    primary_metric: str
    secondary_metrics: List[str] = Field(default_factory=list)
    guardrail_metrics: List[str] = Field(default_factory=list)
    minimum_detectable_effect: float
    alpha: float = Field(default=0.05)
    power: float = Field(default=0.8)
    baseline_rate: float
    start_date: datetime
    end_date: Optional[datetime] = None


class BusinessImpactConfig(BaseModel):
    """Configuration for business impact analysis"""

    intervention_date: datetime
    metrics: List[str] = Field(default_factory=list)
    pre_period_days: int = Field(default=30)
    post_period_days: int = Field(default=30)


# Result models
class EvaluationResults(BaseModel):
    """Results from comprehensive evaluation"""

    evaluation_id: str
    config: EvaluationConfig
    status: EvaluationStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Results from different evaluation types
    offline_results: Optional[Dict[str, Any]] = None
    online_results: Optional[Dict[str, Any]] = None
    business_impact: Optional[Dict[str, Any]] = None
    drift_analysis: Optional[Dict[str, Any]] = None
    causal_analysis: Optional[Dict[str, Any]] = None

    # Generated recommendations
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)

    # Error handling
    error_message: Optional[str] = None

    # Metadata
    created_by: str = Field(default="system")


class OfflineEvaluationResult(BaseModel):
    """Results from offline model evaluation"""

    model_name: str
    model_version: str
    dataset_name: str
    metrics: Dict[str, Any]
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    confidence_intervals: Dict[str, Any] = Field(default_factory=dict)
    segment_performance: Dict[str, Any] = Field(default_factory=dict)
    evaluation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    overall_score: float = Field(default=0.0)


class RankingMetrics(BaseModel):
    """Metrics for ranking models"""

    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    precision_at_20: float = 0.0

    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0

    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0

    mrr: float = 0.0  # Mean Reciprocal Rank
    map: float = 0.0  # Mean Average Precision

    intra_list_diversity: float = 0.0
    catalog_coverage: float = 0.0
    novelty: float = 0.0


class ClassificationMetrics(BaseModel):
    """Metrics for classification models"""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    confusion_matrix: List[List[int]] = Field(default_factory=list)


class RegressionMetrics(BaseModel):
    """Metrics for regression models"""

    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    mape: float = 0.0  # Mean Absolute Percentage Error


class RecommendationMetrics(BaseModel):
    """Metrics for recommendation models"""

    hit_rate: float = 0.0
    coverage: float = 0.0
    diversity: float = 0.0
    novelty: float = 0.0
    serendipity: float = 0.0


class ClusteringMetrics(BaseModel):
    """Metrics for clustering models"""

    silhouette_score: float = 0.0
    calinski_harabasz_score: float = 0.0
    davies_bouldin_score: float = 0.0
    inertia: float = 0.0


# A/B Testing models
class Experiment(BaseModel):
    """A/B testing experiment"""

    id: str
    name: str
    hypothesis: str
    variants: List[str]
    traffic_allocation: Dict[str, float]
    primary_metric: str
    secondary_metrics: List[str] = Field(default_factory=list)
    guardrail_metrics: List[str] = Field(default_factory=list)
    sample_size_per_variant: int
    start_date: datetime
    estimated_end_date: Optional[datetime] = None
    actual_end_date: Optional[datetime] = None
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_by: str = Field(default="system")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExperimentData(BaseModel):
    """Data collected during experiment"""

    experiment_id: str
    variant_data: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    collected_at: datetime = Field(default_factory=datetime.utcnow)

    def get_variant_data(self, variant_name: str) -> Dict[str, Any]:
        """Get data for specific variant"""
        return self.variant_data.get(variant_name, {})

    def get_metric_values(self, metric_name: str) -> List[float]:
        """Get values for specific metric across all variants"""
        values = []
        for variant_data in self.variant_data.values():
            if metric_name in variant_data:
                values.extend(variant_data[metric_name])
        return values

    def get_conversion_rate(self, metric_name: str) -> float:
        """Get conversion rate for specific metric"""
        values = self.get_metric_values(metric_name)
        if not values:
            return 0.0
        return sum(values) / len(values)


class VariantAnalysis(BaseModel):
    """Analysis results for experiment variant"""

    variant_name: str
    primary_metric_result: Dict[str, Any]
    secondary_metric_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    effect_size: float = 0.0
    confidence_interval: Dict[str, float] = Field(default_factory=dict)
    sample_size: int = 0
    conversion_rate: float = 0.0


class FrequentistAnalysis(BaseModel):
    """Frequentist statistical analysis results"""

    experiment_id: str
    variant_results: Dict[str, VariantAnalysis]
    overall_significance: bool = False
    multiple_testing_correction: Dict[str, Any] = Field(default_factory=dict)


class BayesianAnalysis(BaseModel):
    """Bayesian statistical analysis results"""

    experiment_id: str
    variant_results: Dict[str, Dict[str, Any]]
    posterior_probabilities: Dict[str, float] = Field(default_factory=dict)
    credible_intervals: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class SequentialAnalysis(BaseModel):
    """Sequential statistical analysis results"""

    experiment_id: str
    variant_results: Dict[str, Dict[str, Any]]
    stopping_boundaries: Dict[str, float] = Field(default_factory=dict)
    early_stopping: bool = False


class ExperimentAnalysis(BaseModel):
    """Complete experiment analysis results"""

    experiment_id: str
    analysis_type: str
    statistical_results: Union[FrequentistAnalysis, BayesianAnalysis, SequentialAnalysis]
    significance_results: Dict[str, Any] = Field(default_factory=dict)
    guardrail_analysis: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


# Business Impact models
class BusinessImpactAnalysis(BaseModel):
    """Business impact analysis results"""

    intervention_date: datetime
    metric_impacts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    overall_roi: float = 0.0
    statistical_significance: Dict[str, bool] = Field(default_factory=dict)
    confidence_intervals: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


# Drift Detection models
class FeatureDriftResult(BaseModel):
    """Drift detection result for individual feature"""

    feature_name: str
    test_statistic: float
    p_value: float
    is_drifted: bool
    drift_magnitude: float


class DataDriftResult(BaseModel):
    """Data drift detection results"""

    feature_results: Dict[str, FeatureDriftResult] = Field(default_factory=dict)
    overall_drift_score: float = 0.0
    drifted_features: List[str] = Field(default_factory=list)


class PredictionDriftResult(BaseModel):
    """Prediction drift detection results"""

    test_statistic: float
    p_value: float
    is_drifted: bool
    drift_magnitude: float


class PerformanceDriftResult(BaseModel):
    """Performance drift detection results"""

    metric_name: str
    reference_performance: float
    current_performance: float
    performance_change: float
    is_drifted: bool
    confidence_interval: Dict[str, float] = Field(default_factory=dict)


class ConceptDriftResult(BaseModel):
    """Concept drift detection results"""

    test_statistic: float
    p_value: float
    is_drifted: bool
    drift_magnitude: float


class DriftAnalysis(BaseModel):
    """Complete drift analysis results"""

    model_name: str
    reference_period: Dict[str, datetime]
    analysis_period: Dict[str, datetime]
    drift_results: Dict[str, Any] = Field(default_factory=dict)
    drift_severity: float = 0.0
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


# Statistical Testing models
class StatisticalTestResult(BaseModel):
    """Result from statistical test"""

    test_name: str
    test_statistic: float
    p_value: float
    is_significant: bool
    effect_size: float = 0.0
    confidence_interval: Dict[str, float] = Field(default_factory=dict)
    degrees_of_freedom: Optional[int] = None


class PowerAnalysisResult(BaseModel):
    """Power analysis results"""

    effect_size: float
    alpha: float
    power: float
    sample_size: int
    alternative: str = "two-sided"


# Monitoring models
class MetricAlert(BaseModel):
    """Alert for metric anomaly"""

    alert_id: str
    metric_name: str
    threshold: float
    current_value: float
    severity: str  # low, medium, high, critical
    message: str
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None


class EvaluationDashboard(BaseModel):
    """Dashboard data for evaluation metrics"""

    evaluation_id: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    alerts: List[MetricAlert] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# Request/Response models
class CreateEvaluationRequest(BaseModel):
    """Request to create new evaluation"""

    config: EvaluationConfig
    created_by: str = Field(default="system")


class CreateEvaluationResponse(BaseModel):
    """Response from creating evaluation"""

    evaluation_id: str
    status: EvaluationStatus
    message: str


class GetEvaluationResponse(BaseModel):
    """Response for getting evaluation details"""

    evaluation: EvaluationResults


class ListEvaluationsResponse(BaseModel):
    """Response for listing evaluations"""

    evaluations: List[EvaluationResults]
    total_count: int
    limit: int
    offset: int


class CreateExperimentRequest(BaseModel):
    """Request to create A/B test experiment"""

    config: ExperimentConfig
    created_by: str = Field(default="system")


class CreateExperimentResponse(BaseModel):
    """Response from creating experiment"""

    experiment_id: str
    status: ExperimentStatus
    message: str


class AnalyzeExperimentRequest(BaseModel):
    """Request to analyze experiment"""

    experiment_id: str
    analysis_type: str = "frequentist"


class AnalyzeExperimentResponse(BaseModel):
    """Response from experiment analysis"""

    analysis: ExperimentAnalysis


# Error models
class EvaluationError(BaseModel):
    """Error response model"""

    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExperimentDesignError(Exception):
    """Exception for experiment design validation errors"""

    pass
