"""Summarization models."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Content type enumeration."""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    NEWS = "news"
    PRESS_RELEASE = "press_release"
    OPINION = "opinion"
    REVIEW = "review"
    TUTORIAL = "tutorial"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class Language(str, Enum):
    """Language enumeration."""
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    IT = "it"
    PT = "pt"
    RU = "ru"
    ZH = "zh"
    JA = "ja"
    KO = "ko"
    AR = "ar"
    HI = "hi"
    OTHER = "other"


class ProcessedContent(BaseModel):
    """Processed content model."""
    content: str = Field(..., description="Processed content")
    content_type: ContentType = Field(..., description="Type of content")
    language: Language = Field(..., description="Language of content")
    word_count: int = Field(..., description="Word count")
    processing_time: float = Field(..., description="Processing time in seconds")


class SummarizationRequest(BaseModel):
    """Request for content summarization."""
    content: str = Field(..., description="Content to summarize")
    title: Optional[str] = Field(None, description="Optional title")
    max_length: int = Field(200, description="Maximum summary length")
    style: str = Field("neutral", description="Summary style")


class SummarizationResponse(BaseModel):
    """Response for content summarization."""
    summary: str = Field(..., description="Generated summary")
    original_length: int = Field(..., description="Original content length")
    summary_length: int = Field(..., description="Summary length")
    compression_ratio: float = Field(..., description="Compression ratio")


class BatchSummarizationRequest(BaseModel):
    """Request for batch summarization."""
    contents: List[str] = Field(..., description="List of contents to summarize")
    titles: Optional[List[str]] = Field(None, description="Optional titles")
    max_length: int = Field(200, description="Maximum summary length")


class BatchSummarizationResponse(BaseModel):
    """Response for batch summarization."""
    summaries: List[str] = Field(..., description="Generated summaries")
    total_original_length: int = Field(..., description="Total original length")
    total_summary_length: int = Field(..., description="Total summary length")
    avg_compression_ratio: float = Field(..., description="Average compression ratio")


class SummaryResult(BaseModel):
    """Summary result model."""
    summary: str = Field(..., description="Generated summary")
    quality_score: float = Field(..., description="Quality score")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata")