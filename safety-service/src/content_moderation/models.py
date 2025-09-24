"""
Content Moderation Models
Data models for content moderation system
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

class BaseContentModel(BaseModel):
    """Base model for content moderation"""
    model_config = {"arbitrary_types_allowed": True}
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ContentItem(BaseContentModel):
    """Content item to moderate"""
    id: str
    text_content: Optional[str] = None
    image_urls: Optional[List[str]] = None
    video_urls: Optional[List[str]] = None
    external_urls: Optional[List[str]] = None
    user_id: str
    content_type: str = "text"
    metadata: Optional[Dict[str, Any]] = None

class ContentModerationRequest(BaseContentModel):
    """Request for content moderation"""
    content: ContentItem
    user_id: str
    content_type: str = "text"
    priority: str = "normal"

class TextModerationResult(BaseContentModel):
    """Result of text content moderation"""
    text: str
    toxicity_score: float
    hate_speech_score: float
    spam_score: float
    misinformation_score: float
    external_results: Optional[Dict[str, Any]] = None
    detected_issues: List[str]

class ImageModerationResult(BaseContentModel):
    """Result of image content moderation"""
    image_url: str
    adult_content_score: float
    violence_score: float
    inappropriate_score: float
    detected_objects: List[str]
    detected_text: Optional[str] = None

class VideoModerationResult(BaseContentModel):
    """Result of video content moderation"""
    video_url: str
    adult_content_score: float
    violence_score: float
    inappropriate_score: float
    duration: Optional[float] = None
    detected_scenes: List[str]

class URLSafetyResult(BaseContentModel):
    """Result of URL safety check"""
    url: str
    is_safe: bool
    threat_type: Optional[str] = None
    reputation_score: float
    malware_detected: bool
    phishing_detected: bool

class ModerationResult(BaseContentModel):
    """Comprehensive content moderation result"""
    content_id: str
    overall_safety_score: float
    moderation_results: Dict[str, Any]
    recommended_action: str
    violations: List[str]
    moderation_timestamp: datetime
