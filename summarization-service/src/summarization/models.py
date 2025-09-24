"""
Data models for the Summarization Service
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class SummarizationMethod(str, Enum):
    """Summarization method types"""

    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"


class HeadlineStyle(str, Enum):
    """Headline style variants"""

    NEWS = "news"
    ENGAGING = "engaging"
    QUESTION = "question"
    NEUTRAL = "neutral"
    URGENT = "urgent"


class Language(str, Enum):
    """Supported languages"""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"


class ContentType(str, Enum):
    """Content type categories"""

    NEWS_ARTICLE = "news_article"
    BLOG_POST = "blog_post"
    ACADEMIC_PAPER = "academic_paper"
    SOCIAL_MEDIA = "social_media"
    PRODUCT_DESCRIPTION = "product_description"
    GENERAL = "general"


class ProcessedContent(BaseModel):
    """Processed content ready for summarization"""

    text: str = Field(..., description="Main content text")
    title: Optional[str] = Field(None, description="Content title")
    author: Optional[str] = Field(None, description="Content author")
    source: Optional[str] = Field(None, description="Content source/publication")
    published_at: Optional[datetime] = Field(None, description="Publication date")
    language: Language = Field(default=Language.ENGLISH, description="Content language")
    content_type: ContentType = Field(default=ContentType.GENERAL, description="Content type")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator("text")
    def validate_text_length(cls, v):
        if len(v.strip()) < 50:
            raise ValueError("Content text must be at least 50 characters")
        return v.strip()


class SummarizationRequest(BaseModel):
    """Request model for content summarization"""

    content: ProcessedContent = Field(..., description="Content to summarize")
    target_lengths: List[int] = Field(default=[50, 120, 300], description="Target summary lengths in words")
    methods: List[SummarizationMethod] = Field(
        default=[SummarizationMethod.HYBRID], description="Summarization methods to use"
    )
    headline_styles: List[HeadlineStyle] = Field(
        default=[HeadlineStyle.NEWS, HeadlineStyle.ENGAGING],
        description="Headline styles to generate",
    )
    enable_quality_validation: bool = Field(default=True, description="Enable quality validation")
    enable_bias_detection: bool = Field(default=True, description="Enable bias detection")
    enable_factual_consistency: bool = Field(default=True, description="Enable factual consistency checking")
    custom_prompts: Optional[Dict[str, str]] = Field(None, description="Custom prompts for different methods")

    @validator("target_lengths")
    def validate_target_lengths(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one target length must be specified")
        if any(length < 10 or length > 1000 for length in v):
            raise ValueError("Target lengths must be between 10 and 1000 words")
        return sorted(v)


class Summary(BaseModel):
    """Individual summary variant"""

    text: str = Field(..., description="Summary text")
    method: SummarizationMethod = Field(..., description="Method used")
    length: int = Field(..., description="Target length in words")
    word_count: int = Field(..., description="Actual word count")
    quality_score: Optional[float] = Field(None, description="Quality score (0-1)")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

    @validator("word_count", always=True)
    def set_word_count(cls, v, values):
        if "text" in values:
            return len(values["text"].split())
        return v


class Headline(BaseModel):
    """Generated headline variant"""

    text: str = Field(..., description="Headline text")
    style: HeadlineStyle = Field(..., description="Headline style")
    score: float = Field(..., description="Quality score (0-1)")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Detailed metrics")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class QualityMetrics(BaseModel):
    """Quality assessment metrics"""

    rouge1_f1: float = Field(..., description="ROUGE-1 F1 score")
    rouge2_f1: float = Field(..., description="ROUGE-2 F1 score")
    rougeL_f1: float = Field(..., description="ROUGE-L F1 score")
    bertscore_f1: float = Field(..., description="BERTScore F1 score")
    factual_consistency: float = Field(..., description="Factual consistency score")
    readability: float = Field(..., description="Readability score")
    coverage: float = Field(..., description="Information coverage score")
    abstractiveness: float = Field(..., description="Abstractiveness score")
    overall_score: float = Field(..., description="Overall quality score")

    @validator("overall_score", always=True)
    def calculate_overall_score(cls, v, values):
        if v is not None:
            return v

        # Weighted combination of metrics
        weights = {
            "rouge1_f1": 0.2,
            "rouge2_f1": 0.15,
            "rougeL_f1": 0.15,
            "bertscore_f1": 0.2,
            "factual_consistency": 0.15,
            "readability": 0.1,
            "coverage": 0.05,
        }

        score = sum(values.get(metric, 0) * weight for metric, weight in weights.items())
        return round(score, 3)


class BiasAnalysis(BaseModel):
    """Bias detection analysis"""

    political_bias: float = Field(..., description="Political bias score (0-1)")
    gender_bias: float = Field(..., description="Gender bias score (0-1)")
    racial_bias: float = Field(..., description="Racial bias score (0-1)")
    sentiment_bias: float = Field(..., description="Sentiment bias score (0-1)")
    overall_bias: float = Field(..., description="Overall bias score (0-1)")
    neutrality_score: float = Field(..., description="Neutrality score (0-1)")
    detected_biases: List[str] = Field(default_factory=list, description="Detected bias types")

    @validator("overall_bias", always=True)
    def calculate_overall_bias(cls, v, values):
        if v is not None:
            return v

        bias_scores = [
            values.get("political_bias", 0),
            values.get("gender_bias", 0),
            values.get("racial_bias", 0),
            values.get("sentiment_bias", 0),
        ]
        return round(sum(bias_scores) / len(bias_scores), 3)


class ConsistencyScores(BaseModel):
    """Factual consistency scores"""

    entity_consistency: float = Field(..., description="Entity consistency score")
    numerical_consistency: float = Field(..., description="Numerical consistency score")
    temporal_consistency: float = Field(..., description="Temporal consistency score")
    entailment_score: float = Field(..., description="Natural language entailment score")
    overall_consistency: float = Field(..., description="Overall consistency score")

    @validator("overall_consistency", always=True)
    def calculate_overall_consistency(cls, v, values):
        if v is not None:
            return v

        consistency_scores = [
            values.get("entity_consistency", 0),
            values.get("numerical_consistency", 0),
            values.get("temporal_consistency", 0),
            values.get("entailment_score", 0),
        ]
        return round(sum(consistency_scores) / len(consistency_scores), 3)


class ProcessingStats(BaseModel):
    """Processing statistics"""

    total_time: float = Field(..., description="Total processing time in seconds")
    preprocessing_time: float = Field(..., description="Preprocessing time")
    summarization_time: float = Field(..., description="Summarization time")
    headline_generation_time: float = Field(..., description="Headline generation time")
    quality_validation_time: float = Field(..., description="Quality validation time")
    bias_detection_time: float = Field(..., description="Bias detection time")
    tokens_processed: int = Field(..., description="Number of tokens processed")
    models_used: List[str] = Field(default_factory=list, description="Models used in processing")


class SummaryResult(BaseModel):
    """Complete summarization result"""

    summary: Summary = Field(..., description="Best summary")
    headline: Headline = Field(..., description="Best headline")
    variants: List[Union[Summary, Headline]] = Field(default_factory=list, description="All generated variants")
    quality_metrics: QualityMetrics = Field(..., description="Quality metrics")
    consistency_scores: ConsistencyScores = Field(..., description="Consistency scores")
    bias_analysis: BiasAnalysis = Field(..., description="Bias analysis")
    processing_stats: ProcessingStats = Field(..., description="Processing statistics")
    source_attribution: List[str] = Field(default_factory=list, description="Source attribution and citations")
    language_detected: Language = Field(..., description="Detected language")
    confidence_score: float = Field(..., description="Overall confidence score")


class BatchSummarizationRequest(BaseModel):
    """Batch summarization request"""

    requests: List[SummarizationRequest] = Field(..., description="List of summarization requests")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    max_concurrent: int = Field(default=5, description="Maximum concurrent requests")

    @validator("requests")
    def validate_requests(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one request must be provided")
        if len(v) > 100:
            raise ValueError("Maximum 100 requests allowed per batch")
        return v


class SummarizationResponse(BaseModel):
    """API response model"""

    success: bool = Field(..., description="Whether the operation was successful")
    result: Optional[SummaryResult] = Field(None, description="Summarization result")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: Optional[float] = Field(None, description="Total processing time")
    request_id: Optional[str] = Field(None, description="Unique request identifier")


class ABTestVariant(BaseModel):
    """A/B test variant configuration"""

    variant_id: str = Field(..., description="Variant identifier")
    name: str = Field(..., description="Variant name")
    description: str = Field(..., description="Variant description")
    parameters: Dict[str, Any] = Field(..., description="Variant parameters")
    traffic_percentage: float = Field(..., description="Traffic percentage (0-1)")
    is_active: bool = Field(default=True, description="Whether variant is active")


class ABTestResult(BaseModel):
    """A/B test result"""

    test_id: str = Field(..., description="Test identifier")
    variant_id: str = Field(..., description="Selected variant")
    metrics: Dict[str, float] = Field(..., description="Performance metrics")
    confidence_level: float = Field(..., description="Statistical confidence level")
    sample_size: int = Field(..., description="Sample size")
    duration_days: int = Field(..., description="Test duration in days")
