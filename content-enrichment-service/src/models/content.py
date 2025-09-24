"""Content models for enrichment service."""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class ProcessingMode(str, Enum):
    """Processing mode for content enrichment."""
    REALTIME = "realtime"
    BATCH = "batch"


class ContentType(str, Enum):
    """Type of content being processed."""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    DOCUMENT = "document"
    WEB_PAGE = "web_page"


class ExtractedContent(BaseModel):
    """Input content for enrichment."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    summary: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    content_type: ContentType = ContentType.ARTICLE
    language: str = "en"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('content')
    def validate_content_length(cls, v):
        if len(v) > 50000:  # 50k character limit
            raise ValueError("Content too long")
        return v


class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "GPE"
    DATE = "DATE"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    CARDINAL = "CARDINAL"
    ORDINAL = "ORDINAL"
    QUANTITY = "QUANTITY"
    EVENT = "EVENT"
    FACILITY = "FACILITY"
    LANGUAGE = "LANGUAGE"
    LAW = "LAW"
    NATIONALITY = "NORP"
    PRODUCT = "PRODUCT"
    WORK_OF_ART = "WORK_OF_ART"
    CUSTOM = "CUSTOM"


class Entity(BaseModel):
    """Named entity extracted from content."""
    text: str
    label: EntityType
    start: int
    end: int
    confidence: float = Field(ge=0.0, le=1.0)
    wikidata_id: Optional[str] = None
    canonical_name: Optional[str] = None
    description: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class TopicCategory(str, Enum):
    """Top-level topic categories."""
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    POLITICS = "politics"
    HEALTH = "health"
    SCIENCE = "science"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    WORLD = "world"
    EDUCATION = "education"
    ENVIRONMENT = "environment"
    FINANCE = "finance"
    LIFESTYLE = "lifestyle"
    TRAVEL = "travel"
    FOOD = "food"
    AUTOMOTIVE = "automotive"
    OTHER = "other"


class Topic(BaseModel):
    """Topic classification result."""
    id: str
    name: str
    category: TopicCategory
    confidence: float = Field(ge=0.0, le=1.0)
    hierarchy_path: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class EmotionLabel(str, Enum):
    """Emotion classification labels."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""
    sentiment: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    emotions: Dict[EmotionLabel, float] = Field(default_factory=dict)
    subjectivity: float = Field(ge=0.0, le=1.0)  # 0 = objective, 1 = subjective
    polarity: float = Field(ge=-1.0, le=1.0)  # -1 = negative, 1 = positive


class ContentSignal(BaseModel):
    """Content quality and engagement signals."""
    readability_score: float = Field(ge=0.0, le=1.0)
    factual_claims: int = Field(ge=0)
    citations_count: int = Field(ge=0)
    bias_score: float = Field(ge=-1.0, le=1.0)  # -1 = left bias, 1 = right bias
    political_leaning: Optional[str] = None
    engagement_prediction: float = Field(ge=0.0, le=1.0)
    virality_potential: float = Field(ge=0.0, le=1.0)
    content_freshness: float = Field(ge=0.0, le=1.0)
    authority_score: float = Field(ge=0.0, le=1.0)
    expertise_indicators: List[str] = Field(default_factory=list)


class TrustworthinessScore(BaseModel):
    """Content trustworthiness assessment."""
    overall_score: float = Field(ge=0.0, le=1.0)
    source_reliability: float = Field(ge=0.0, le=1.0)
    fact_checking_score: float = Field(ge=0.0, le=1.0)
    citation_quality: float = Field(ge=0.0, le=1.0)
    author_credibility: float = Field(ge=0.0, le=1.0)
    content_quality: float = Field(ge=0.0, le=1.0)
    bias_indicators: List[str] = Field(default_factory=list)
    warning_flags: List[str] = Field(default_factory=list)


class ModelVersion(BaseModel):
    """Model version information."""
    name: str
    version: str
    created_at: datetime
    performance_metrics: Dict[str, float] = Field(default_factory=dict)


class EnrichedContent(BaseModel):
    """Fully enriched content with all AI insights."""
    id: str
    original_content: ExtractedContent
    entities: List[Entity]
    topics: List[Topic]
    sentiment: SentimentAnalysis
    signals: ContentSignal
    trust_score: TrustworthinessScore
    enrichment_timestamp: datetime
    model_versions: Dict[str, ModelVersion]
    processing_time_ms: int
    language_detected: str
    processing_mode: ProcessingMode
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EnrichmentRequest(BaseModel):
    """Request for content enrichment."""
    content: ExtractedContent
    processing_mode: ProcessingMode = ProcessingMode.REALTIME
    include_entities: bool = True
    include_topics: bool = True
    include_sentiment: bool = True
    include_signals: bool = True
    include_trust_score: bool = True
    language_hint: Optional[str] = None
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)


class EnrichmentResponse(BaseModel):
    """Response from content enrichment."""
    success: bool
    enriched_content: Optional[EnrichedContent] = None
    error_message: Optional[str] = None
    processing_time_ms: int
    model_versions_used: Dict[str, str] = Field(default_factory=dict)


class BatchEnrichmentRequest(BaseModel):
    """Batch enrichment request."""
    contents: List[ExtractedContent]
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    include_entities: bool = True
    include_topics: bool = True
    include_sentiment: bool = True
    include_signals: bool = True
    include_trust_score: bool = True
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)


class BatchEnrichmentResponse(BaseModel):
    """Batch enrichment response."""
    success: bool
    enriched_contents: List[EnrichedContent]
    failed_contents: List[Dict[str, Any]] = Field(default_factory=list)
    total_processing_time_ms: int
    model_versions_used: Dict[str, str] = Field(default_factory=dict)
