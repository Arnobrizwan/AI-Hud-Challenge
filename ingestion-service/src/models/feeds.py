"""
Feed parsing models for RSS, Atom, and JSON feeds.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class FeedType(str, Enum):
    """Feed type enumeration."""

    RSS = "rss"
    ATOM = "atom"
    JSON = "json"
    UNKNOWN = "unknown"


class FeedItem(BaseModel):
    """Individual feed item."""

    title: str = Field(..., description="Item title")
    link: str = Field(..., description="Item URL")
    description: Optional[str] = Field(None, description="Item description")
    content: Optional[str] = Field(None, description="Item content")
    author: Optional[str] = Field(None, description="Item author")
    published: Optional[datetime] = Field(None, description="Publication date")
    updated: Optional[datetime] = Field(None, description="Last update date")
    guid: Optional[str] = Field(None, description="Unique identifier")
    categories: List[str] = Field(default_factory=list, description="Categories/tags")
    image_url: Optional[str] = Field(None, description="Image URL")
    enclosure_url: Optional[str] = Field(None, description="Enclosure URL (media)")
    enclosure_type: Optional[str] = Field(None, description="Enclosure MIME type")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw feed data")

    @validator("link")
    def validate_link(cls, v):
        """Validate link URL."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Link must be a valid URL")
        return v

    @validator("published", "updated")
    def validate_dates(cls, v):
        """Validate date fields."""
        if v and v > datetime.utcnow():
            # Allow future dates but log warning
            pass
        return v


class FeedMetadata(BaseModel):
    """Feed metadata."""

    title: str = Field(..., description="Feed title")
    description: Optional[str] = Field(None, description="Feed description")
    link: str = Field(..., description="Feed URL")
    language: Optional[str] = Field(None, description="Feed language")
    last_build_date: Optional[datetime] = Field(None, description="Last build date")
    generator: Optional[str] = Field(None, description="Feed generator")
    ttl: Optional[int] = Field(None, description="Time to live in minutes")
    image_url: Optional[str] = Field(None, description="Feed image URL")
    web_master: Optional[str] = Field(None, description="Web master email")
    managing_editor: Optional[str] = Field(None, description="Managing editor email")
    copyright: Optional[str] = Field(None, description="Copyright notice")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw feed metadata")


class ParsedFeed(BaseModel):
    """Parsed feed with items and metadata."""

    feed_type: FeedType = Field(..., description="Type of feed")
    metadata: FeedMetadata = Field(..., description="Feed metadata")
    items: List[FeedItem] = Field(..., description="Feed items")
    parsing_errors: List[str] = Field(default_factory=list, description="Parsing errors")
    parsed_at: datetime = Field(default_factory=datetime.utcnow, description="Parsing timestamp")
    source_url: str = Field(..., description="Source feed URL")
    etag: Optional[str] = Field(None, description="HTTP ETag")
    last_modified: Optional[datetime] = Field(None, description="Last-Modified header")
    content_length: Optional[int] = Field(None, description="Content length")

    @validator("items")
    def validate_items(cls, v):
        """Validate items list."""
        if not v:
            raise ValueError("Feed must contain at least one item")
        return v

    @property
    def item_count(self) -> int:
        """Get number of items in feed."""
        return len(self.items)

    @property
    def has_errors(self) -> bool:
        """Check if feed has parsing errors."""
        return len(self.parsing_errors) > 0


class RSSFeedItem(FeedItem):
    """RSS-specific feed item."""

    comments: Optional[str] = Field(None, description="Comments URL")
    source: Optional[str] = Field(None, description="Source name")
    source_url: Optional[str] = Field(None, description="Source URL")
    enclosure_length: Optional[int] = Field(None, description="Enclosure length")


class AtomFeedItem(FeedItem):
    """Atom-specific feed item."""

    summary: Optional[str] = Field(None, description="Item summary")
    rights: Optional[str] = Field(None, description="Rights information")
    source: Optional[str] = Field(None, description="Source name")
    contributors: List[str] = Field(default_factory=list, description="Contributors")


class JSONFeedItem(FeedItem):
    """JSON Feed-specific item."""

    external_url: Optional[str] = Field(None, description="External URL")
    summary: Optional[str] = Field(None, description="Item summary")
    banner_image: Optional[str] = Field(None, description="Banner image URL")
    date_published: Optional[datetime] = Field(None, description="Publication date")
    date_modified: Optional[datetime] = Field(None, description="Modification date")
    authors: List[Dict[str, str]] = Field(default_factory=list, description="Authors")
    tags: List[str] = Field(default_factory=list, description="Tags")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="Attachments")


class FeedParseResult(BaseModel):
    """Result of feed parsing operation."""

    success: bool = Field(..., description="Whether parsing was successful")
    feed: Optional[ParsedFeed] = Field(None, description="Parsed feed if successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    parse_time_ms: int = Field(..., description="Parsing time in milliseconds")
    source_url: str = Field(..., description="Source URL")
    http_status_code: Optional[int] = Field(None, description="HTTP status code")
    content_type: Optional[str] = Field(None, description="Content type")
    content_length: Optional[int] = Field(None, description="Content length")

    @validator("parse_time_ms")
    def validate_parse_time(cls, v):
        """Validate parse time."""
        if v < 0:
            raise ValueError("Parse time must be non-negative")
        return v


class FeedSubscription(BaseModel):
    """WebSub (PubSubHubbub) subscription."""

    topic_url: str = Field(..., description="Topic URL")
    hub_url: str = Field(..., description="Hub URL")
    callback_url: str = Field(..., description="Callback URL")
    secret: Optional[str] = Field(None, description="Verification secret")
    lease_seconds: Optional[int] = Field(None, description="Lease duration in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    verified: bool = Field(default=False, description="Whether subscription is verified")
    last_verification: Optional[datetime] = Field(None, description="Last verification timestamp")

    @validator("topic_url", "hub_url", "callback_url")
    def validate_urls(cls, v):
        """Validate URLs."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class FeedValidationResult(BaseModel):
    """Feed validation result."""

    is_valid: bool = Field(..., description="Whether feed is valid")
    feed_type: Optional[FeedType] = Field(None, description="Detected feed type")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    item_count: int = Field(default=0, description="Number of items found")
    last_item_date: Optional[datetime] = Field(None, description="Date of most recent item")
    first_item_date: Optional[datetime] = Field(None, description="Date of oldest item")
    update_frequency: Optional[str] = Field(None, description="Estimated update frequency")

    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0
