"""
Base adapter class for content ingestion.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.models.content import NormalizedArticle, ProcessingStatus, SourceConfig
from src.models.feeds import FeedParseResult, ParsedFeed
from src.utils.content_parser import ContentParser
from src.utils.date_utils import DateUtils
from src.utils.http_client import HTTPClient, HTTPResponse
from src.utils.url_utils import URLUtils

logger = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """Base class for content ingestion adapters."""

    def __init__(
        self,
        source_config: SourceConfig,
        http_client: HTTPClient = None,
        content_parser: ContentParser = None,
        url_utils: URLUtils = None,
        date_utils: DateUtils = None,
    ):
        self.source_config = source_config
        self.http_client = http_client or HTTPClient()
        self.content_parser = content_parser or ContentParser()
        self.url_utils = url_utils or URLUtils()
        self.date_utils = date_utils or DateUtils()

        # Processing state
        self.is_processing = False
        self.last_processed = None
        self.error_count = 0
        self.success_count = 0

    @abstractmethod
    async def fetch_content(self) -> AsyncGenerator[NormalizedArticle, None]:
        """Fetch content from the source."""
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test connection to the source."""
        pass

    @abstractmethod
    def get_source_info(self) -> Dict[str, Any]:
    """Get information about the source."""
        pass

    async def process(self) -> List[NormalizedArticle]:
        """Process content from the source."""
        if self.is_processing:
            logger.warning(f"Source {self.source_config.id} is already being processed")
            return []

        self.is_processing = True
        articles = []

        try:
            logger.info(f"Starting processing for source {self.source_config.id}")

            async for article in self.fetch_content():
                if article:
                    articles.append(article)

                    # Apply rate limiting
                    await self._apply_rate_limiting()

            self.success_count += 1
            self.last_processed = datetime.utcnow()
            self.error_count = 0

            logger.info(f"Successfully processed {len(articles)} articles from source {self.source_config.id}")

        except Exception as e:
            self.error_count += 1
            logger.error(f"Error processing source {self.source_config.id}: {e}")
            raise

        finally:
            self.is_processing = False

        return articles

    async def _apply_rate_limiting(self) -> None:
        """Apply rate limiting based on source configuration."""
        if self.source_config.rate_limit > 0:
            delay = 60.0 / self.source_config.rate_limit  # Convert to seconds per request
            await asyncio.sleep(delay)

    def _create_ingestion_metadata(self, response: HTTPResponse = None) -> Dict[str, Any]:
    """Create ingestion metadata for articles."""
        metadata = {
            "source_id": self.source_config.id,
            "source_type": self.source_config.type,
            "source_url": self.source_config.url,
            "ingested_at": datetime.utcnow(),
            "retry_count": 0,
            "user_agent": self.http_client.user_agent,
            "robots_txt_respected": True,
        }

        if response:
            metadata.update(
                {
                    "http_status_code": response.status_code,
                    "content_length": response.content_length,
                    "etag": response.etag,
                    "last_modified": response.last_modified,
                    "elapsed": response.elapsed,
                }
            )

        return metadata

    def _normalize_article(
        self,
        title: str,
        url: str,
        content: str = None,
        summary: str = None,
        author: str = None,
        published_at: datetime = None,
        image_url: str = None,
        tags: List[str] = None,
        raw_data: Dict[str, Any] = None,
        ingestion_metadata: Dict[str, Any] = None,
    ) -> NormalizedArticle:
        """Normalize article data into standard format."""

        # Generate unique ID
        article_id = self._generate_article_id(url, title)

        # Normalize URL
        normalized_url = self.url_utils.normalize_url(url)
        canonical_url = self.url_utils.normalize_url(url)

        # Extract domain info
        domain_info = self.url_utils.extract_domain_info(url)
        source_url = domain_info.get("registered_domain", "")

        # Process content
        if content:
            # Extract text from HTML if needed
            if "<" in content and ">" in content:
                content = self.content_parser.extract_text_from_html(content)

            # Calculate reading metrics
            reading_metrics = self.content_parser.calculate_reading_metrics(content)
            word_count = reading_metrics["word_count"]
            reading_time = reading_metrics["reading_time_minutes"]
        else:
            word_count = 0
            reading_time = 0

        # Detect language
        text_for_language = content or summary or title
        language = self.content_parser.detect_language(text_for_language) if text_for_language else "en"

        # Normalize published date
        if published_at:
            published_at = self.date_utils.normalize_date(published_at)

        # Create content hash for duplicate detection
        content_for_hash = f"{title}|{content or ''}|{url}"
        content_hash = self._calculate_content_hash(content_for_hash)

        # Create normalized article
        article = NormalizedArticle(
            id=article_id,
            url=normalized_url,
            canonical_url=canonical_url,
            title=title or "Untitled",
            summary=summary,
            content=content,
            author=author,
            source=self.source_config.name,
            source_url=source_url,
            published_at=published_at or datetime.utcnow(),
            language=language,
            image_url=image_url,
            tags=tags or [],
            word_count=word_count,
            reading_time=reading_time,
            content_hash=content_hash,
            raw_data=raw_data or {},
            ingestion_metadata=ingestion_metadata or {},
            processing_status=ProcessingStatus.PENDING,
        )

        return article

    def _generate_article_id(self, url: str, title: str) -> str:
        """Generate unique article ID."""
        import hashlib

        # Use URL and title to generate consistent ID
        content = f"{url}|{title}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate content hash for duplicate detection."""
        import hashlib

        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _is_valid_article(self, article: NormalizedArticle) -> bool:
        """Check if article meets quality requirements."""
        # Check minimum word count
        if article.word_count < 50:  # Minimum 50 words
            return False

        # Check title length
        if len(article.title) < 10:  # Minimum 10 characters
            return False

        # Check URL validity
        if not self.url_utils.is_valid_url(article.url):
            return False

        # Check if URL should be skipped
        if self.url_utils.should_skip_url(article.url):
            return False

        return True

    def _apply_content_filters(self, article: NormalizedArticle) -> bool:
        """Apply content filters based on source configuration."""
        if not self.source_config.filters:
            return True

        filters = self.source_config.filters

        # Language filter
        if "languages" in filters:
            if article.language not in filters["languages"]:
                return False

        # Word count filter
        if "min_word_count" in filters:
            if article.word_count < filters["min_word_count"]:
                return False

        if "max_word_count" in filters:
            if article.word_count > filters["max_word_count"]:
                return False

        # Title filter
        if "title_keywords" in filters:
            title_lower = article.title.lower()
            if not any(keyword.lower() in title_lower for keyword in filters["title_keywords"]):
                return False

        # Content filter
        if "content_keywords" in filters and article.content:
            content_lower = article.content.lower()
            if not any(keyword.lower() in content_lower for keyword in filters["content_keywords"]):
                return False

        # Exclude keywords
        if "exclude_keywords" in filters:
            text_to_check = f"{article.title} {article.content or ''}".lower()
            if any(keyword.lower() in text_to_check for keyword in filters["exclude_keywords"]):
                return False

        return True

    async def health_check(self) -> Dict[str, Any]:
    """Perform health check on the adapter."""
        try:
            is_connected = await self.test_connection()

            return {
                "adapter_type": self.__class__.__name__,
                "source_id": self.source_config.id,
                "is_connected": is_connected,
                "is_processing": self.is_processing,
                "last_processed": self.last_processed,
                "error_count": self.error_count,
                "success_count": self.success_count,
                "rate_limit": self.source_config.rate_limit,
                "enabled": self.source_config.enabled,
            }
        except Exception as e:
            logger.error(f"Health check failed for adapter {self.source_config.id}: {e}")
            return {
                "adapter_type": self.__class__.__name__,
                "source_id": self.source_config.id,
                "is_connected": False,
                "error": str(e),
            }
