"""
RSS/Atom feed adapter for content ingestion.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import feedparser

from src.adapters.base import BaseAdapter
from src.models.content import NormalizedArticle, SourceConfig
from src.models.feeds import FeedParseResult, FeedType, ParsedFeed
from src.utils.http_client import HTTPResponse

logger = logging.getLogger(__name__)


class RSSAdapter(BaseAdapter):
    """Adapter for RSS and Atom feeds."""

    def __init__(self, source_config: SourceConfig, **kwargs):
        super().__init__(source_config, **kwargs)
        self.feed_type = None
        self.last_etag = None
        self.last_modified = None

    async def fetch_content(self) -> AsyncGenerator[NormalizedArticle, None]:
        """Fetch content from RSS/Atom feed."""
        try:
            # Fetch feed
            feed_result = await self._fetch_feed()

            if not feed_result.success:
                logger.error(f"Failed to fetch feed {self.source_config.url}: {feed_result.error_message}")
                return

            feed = feed_result.feed
            self.feed_type = feed.feed_type

            # Update last modified info
            if feed.etag:
                self.last_etag = feed.etag
            if feed.last_modified:
                self.last_modified = feed.last_modified

            # Process feed items
            for item in feed.items:
                try:
                    article = await self._process_feed_item(item, feed.metadata)

                    if article and self._is_valid_article(article) and self._apply_content_filters(article):
                        yield article

                except Exception as e:
                    logger.warning(f"Error processing feed item: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error fetching RSS feed {self.source_config.url}: {e}")
            raise

    async def _fetch_feed(self) -> FeedParseResult:
        """Fetch and parse RSS/Atom feed."""
        start_time = datetime.utcnow()

        try:
            # Make HTTP request with conditional headers
            headers = {}
            if self.last_etag:
                headers["If-None-Match"] = self.last_etag
            if self.last_modified:
                headers["If-Modified-Since"] = self.last_modified

            response = await self.http_client.get(
                self.source_config.url, headers=headers, timeout=self.source_config.timeout
            )

            # Check if content hasn't changed
            if response.status_code == 304:
                return FeedParseResult(
                    success=True,
                    feed=None,
                    error_message=None,
                    parse_time_ms=0,
                    source_url=self.source_config.url,
                    http_status_code=response.status_code,
                    content_type=response.content_type,
                    content_length=response.content_length,
                )

            # Parse feed
            parsed_feed = self._parse_feed_content(response.text, response)

            parse_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return FeedParseResult(
                success=True,
                feed=parsed_feed,
                error_message=None,
                parse_time_ms=int(parse_time),
                source_url=self.source_config.url,
                http_status_code=response.status_code,
                content_type=response.content_type,
                content_length=response.content_length,
            )

        except Exception as e:
            parse_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return FeedParseResult(
                success=False,
                feed=None,
                error_message=str(e),
                parse_time_ms=int(parse_time),
                source_url=self.source_config.url,
            )

    def _parse_feed_content(self, content: str, response: HTTPResponse) -> ParsedFeed:
        """Parse feed content using feedparser."""
        # Parse with feedparser
        parsed = feedparser.parse(content)

        # Determine feed type
        feed_type = FeedType.UNKNOWN
        if hasattr(parsed, "version") and parsed.version:
            if "rss" in parsed.version.lower():
                feed_type = FeedType.RSS
            elif "atom" in parsed.version.lower():
                feed_type = FeedType.ATOM

        # Extract feed metadata
        feed_metadata = self._extract_feed_metadata(parsed)

        # Extract feed items
        items = []
        parsing_errors = []

        for entry in parsed.entries:
            try:
                item = self._extract_feed_item(entry, feed_type)
                if item:
                    items.append(item)
            except Exception as e:
                parsing_errors.append(f"Error parsing item: {e}")
                continue

        return ParsedFeed(
            feed_type=feed_type,
            metadata=feed_metadata,
            items=items,
            parsing_errors=parsing_errors,
            source_url=self.source_config.url,
            etag=response.etag,
            last_modified=response.last_modified,
            content_length=response.content_length,
        )

    def _extract_feed_metadata(self, parsed) -> Dict[str, Any]:
    """Extract feed metadata from parsed feed."""
        feed_info = parsed.feed

        return {
            "title": getattr(feed_info, "title", ""),
            "description": getattr(feed_info, "description", ""),
            "link": getattr(feed_info, "link", ""),
            "language": getattr(feed_info, "language", ""),
            "last_build_date": getattr(feed_info, "updated_parsed", None),
            "generator": getattr(feed_info, "generator", ""),
            "ttl": getattr(feed_info, "ttl", None),
            "image_url": (getattr(feed_info, "image", {}).get("href", "") if hasattr(feed_info, "image") else ""),
            "web_master": getattr(feed_info, "webmaster", ""),
            "managing_editor": getattr(feed_info, "managingeditor", ""),
            "copyright": getattr(feed_info, "copyright", ""),
            "raw_data": {
                "version": getattr(parsed, "version", ""),
                "encoding": getattr(parsed, "encoding", ""),
                "bozo": getattr(parsed, "bozo", False),
                "bozo_exception": str(getattr(parsed, "bozo_exception", "")),
            },
        }

    def _extract_feed_item(self, entry, feed_type: FeedType) -> Optional[Dict[str, Any]]:
        """Extract feed item from parsed entry."""
        try:
            # Basic fields
            title = getattr(entry, "title", "")
            link = getattr(entry, "link", "")
            description = getattr(entry, "description", "")
            content = getattr(entry, "content", [{}])[0].get("value", "") if hasattr(entry, "content") else ""
            author = getattr(entry, "author", "")

            # Dates
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6])

            # GUID
            guid = getattr(entry, "id", "") or getattr(entry, "guid", "")

            # Categories
            categories = []
            if hasattr(entry, "tags"):
                categories = [tag.get("term", "") for tag in entry.tags if tag.get("term")]
            elif hasattr(entry, "category"):
                categories = [entry.category]

            # Image
            image_url = ""
            if hasattr(entry, "media_content"):
                for media in entry.media_content:
                    if media.get("type", "").startswith("image/"):
                        image_url = media.get("url", "")
                        break

            # Enclosure
            enclosure_url = ""
            enclosure_type = ""
            if hasattr(entry, "enclosures"):
                for enclosure in entry.enclosures:
                    if enclosure.get("type", "").startswith("image/"):
                        enclosure_url = enclosure.get("href", "")
                        enclosure_type = enclosure.get("type", "")
                        break

            return {
                "title": title,
                "link": link,
                "description": description,
                "content": content,
                "author": author,
                "published": published,
                "guid": guid,
                "categories": categories,
                "image_url": image_url,
                "enclosure_url": enclosure_url,
                "enclosure_type": enclosure_type,
                "raw_data": {
                    "feed_type": feed_type.value,
                    "entry_keys": list(entry.keys()) if hasattr(entry, "keys") else [],
                },
            }

        except Exception as e:
            logger.warning(f"Error extracting feed item: {e}")
            return None

    async def _process_feed_item(
        self, item: Dict[str, Any], feed_metadata: Dict[str, Any]
    ) -> Optional[NormalizedArticle]:
        """Process feed item into normalized article."""
        try:
            # Extract basic information
            title = item.get("title", "")
            url = item.get("link", "")
            content = item.get("content", "")
            summary = item.get("description", "")
            author = item.get("author", "")
            published_at = item.get("published")
            image_url = item.get("image_url", "")
            tags = item.get("categories", [])

            # If no content, try to use summary
            if not content and summary:
                content = summary

            # If no summary, try to extract from content
            if not summary and content:
                summary = self.content_parser.extract_description(content)

            # Create ingestion metadata
            ingestion_metadata = self._create_ingestion_metadata()
            ingestion_metadata.update(
                {
                    "feed_type": self.feed_type.value if self.feed_type else "unknown",
                    "feed_title": feed_metadata.get("title", ""),
                    "item_guid": item.get("guid", ""),
                    "item_categories": tags,
                }
            )

            # Create normalized article
            article = self._normalize_article(
                title=title,
                url=url,
                content=content,
                summary=summary,
                author=author,
                published_at=published_at,
                image_url=image_url,
                tags=tags,
                raw_data=item,
                ingestion_metadata=ingestion_metadata,
            )

            return article

        except Exception as e:
            logger.warning(f"Error processing feed item: {e}")
            return None

    async def test_connection(self) -> bool:
        """Test connection to RSS feed."""
        try:
            response = await self.http_client.head(self.source_config.url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"RSS connection test failed: {e}")
            return False

    def get_source_info(self) -> Dict[str, Any]:
    """Get information about the RSS source."""
        return {
            "type": "RSS/Atom Feed",
            "url": self.source_config.url,
            "feed_type": self.feed_type.value if self.feed_type else "unknown",
            "last_etag": self.last_etag,
            "last_modified": self.last_modified,
            "rate_limit": self.source_config.rate_limit,
            "timeout": self.source_config.timeout,
            "enabled": self.source_config.enabled,
        }
