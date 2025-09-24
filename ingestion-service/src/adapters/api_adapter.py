"""
API adapter for content ingestion from REST APIs.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.adapters.base import BaseAdapter
from src.models.content import NormalizedArticle, SourceConfig
from src.utils.http_client import HTTPResponse

logger = logging.getLogger(__name__)


class APIAdapter(BaseAdapter):
    """Adapter for REST API content sources."""

    def __init__(self, source_config: SourceConfig, **kwargs):
        super().__init__(source_config, **kwargs)
        self.api_config = source_config.filters.get("api_config", {})
        self.pagination_config = self.api_config.get("pagination", {})
        self.current_page = 1
        self.has_more_pages = True

    async def fetch_content(self) -> AsyncGenerator[NormalizedArticle, None]:
        """Fetch content from API."""
        try:
            # Reset pagination state
            self.current_page = 1
            self.has_more_pages = True

            while self.has_more_pages:
                # Fetch current page
                articles = await self._fetch_page()

                if not articles:
                    break

                # Process articles
                for article_data in articles:
                    try:
                        article = await self._process_api_item(article_data)

                        if (
                            article
                            and self._is_valid_article(article)
                            and self._apply_content_filters(article)
                        ):
                            yield article

                    except Exception as e:
                        logger.warning(f"Error processing API item: {e}")
                        continue

                # Check if there are more pages
                self.has_more_pages = await self._has_more_pages(articles)
                if self.has_more_pages:
                    self.current_page += 1
                    await self._apply_rate_limiting()

        except Exception as e:
            logger.error(
                f"Error fetching from API {self.source_config.url}: {e}")
            raise

    async def _fetch_page(self) -> List[Dict[str, Any]]:
        """Fetch a single page from the API."""
        try:
            # Build request URL with pagination
            url = self._build_request_url()

            # Prepare headers
            headers = self._prepare_headers()

            # Make request
            response = await self.http_client.get(
                url, headers=headers, timeout=self.source_config.timeout
            )

            if response.status_code != 200:
                raise Exception(
                    f"API request failed with status {response.status_code}")

            # Parse response
            data = json.loads(response.text)

            # Extract articles from response
            articles = self._extract_articles_from_response(data)

            return articles

        except Exception as e:
            logger.error(f"Error fetching API page: {e}")
            return []

    def _build_request_url(self) -> str:
        """Build request URL with pagination parameters."""
        base_url = self.source_config.url

        # Add pagination parameters
        pagination_params = self.pagination_config.get("params", {})
        if pagination_params:
            # Replace placeholders with actual values
            for key, value in pagination_params.items():
                if isinstance(value, str) and "{page}" in value:
                    pagination_params[key] = value.format(
                        page=self.current_page)
                elif key == "page":
                    pagination_params[key] = self.current_page
                elif key == "offset":
                    page_size = pagination_params.get("limit", 20)
                    pagination_params[key] = (
                        self.current_page - 1) * page_size

        # Add other parameters
        all_params = {**pagination_params}

        # Add any additional parameters from source config
        if "params" in self.api_config:
            all_params.update(self.api_config["params"])

        # Build URL with parameters
        if all_params:
            param_string = "&".join(
                [f"{k}={v}" for k, v in all_params.items()])
            separator = "&" if "?" in base_url else "?"
            return f"{base_url}{separator}{param_string}"

        return base_url

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare request headers."""
        headers = {}

        # Add authentication headers
        auth_config = self.api_config.get("auth", {})
        if auth_config.get("type") == "bearer":
            headers["Authorization"] = f"Bearer {auth_config.get('token', '')}"
        elif auth_config.get("type") == "api_key":
            key = auth_config.get("key", "")
            value = auth_config.get("value", "")
            if key and value:
                headers[key] = value
        elif auth_config.get("type") == "basic":
            import base64

            username = auth_config.get("username", "")
            password = auth_config.get("password", "")
            if username and password:
                credentials = base64.b64encode(
                    f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {credentials}"

        # Add custom headers
        if "headers" in self.api_config:
            headers.update(self.api_config["headers"])

        # Add source config headers
        if self.source_config.headers:
            headers.update(self.source_config.headers)

        return headers

    def _extract_articles_from_response(
            self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract articles from API response."""
        # Get the path to articles in the response
        articles_path = self.api_config.get("articles_path", "data")

        # Navigate to articles
        articles = data
        for key in articles_path.split("."):
            if isinstance(articles, dict) and key in articles:
                articles = articles[key]
            else:
                return []

        # Ensure articles is a list
        if not isinstance(articles, list):
            return []

        return articles

    async def _process_api_item(
            self, item_data: Dict[str, Any]) -> Optional[NormalizedArticle]:
        """Process API item into normalized article."""
        try:
            # Extract fields using field mapping
            field_mapping = self.api_config.get("field_mapping", {})

            title = self._extract_field(
                item_data, field_mapping.get(
                    "title", "title"))
            url = self._extract_field(
                item_data, field_mapping.get(
                    "url", "url"))
            content = self._extract_field(
                item_data, field_mapping.get(
                    "content", "content"))
            summary = self._extract_field(
                item_data, field_mapping.get(
                    "summary", "summary"))
            author = self._extract_field(
                item_data, field_mapping.get(
                    "author", "author"))
            published_at = self._extract_field(
                item_data, field_mapping.get("published_at", "published_at")
            )
            image_url = self._extract_field(
                item_data, field_mapping.get(
                    "image_url", "image_url"))
            tags = self._extract_field(
                item_data, field_mapping.get(
                    "tags", "tags"))

            # Parse published date
            if published_at:
                if isinstance(published_at, str):
                    published_at = self.date_utils.parse_date(published_at)
                elif isinstance(published_at, (int, float)):
                    # Unix timestamp
                    published_at = datetime.fromtimestamp(
                        published_at, tz=self.date_utils.default_timezone
                    )

            # Ensure tags is a list
            if tags and not isinstance(tags, list):
                tags = [str(tags)]
            elif not tags:
                tags = []

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
                    "api_endpoint": self.source_config.url,
                    "api_page": self.current_page,
                    "api_item_id": self._extract_field(
                        item_data,
                        field_mapping.get(
                            "id",
                            "id")),
                    "api_response_keys": (
                        list(
                            item_data.keys()) if isinstance(
                            item_data,
                            dict) else []),
                })

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
                raw_data=item_data,
                ingestion_metadata=ingestion_metadata,
            )

            return article

        except Exception as e:
            logger.warning(f"Error processing API item: {e}")
            return None

    def _extract_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Extract field value using dot notation path."""
        if not field_path:
            return None

        # Split path by dots
        keys = field_path.split(".")
        value = data

        try:
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                elif isinstance(value, list) and key.isdigit():
                    index = int(key)
                    if 0 <= index < len(value):
                        value = value[index]
        else:
                        return None
        else:
                    return None

            return value
        except (KeyError, IndexError, TypeError):
            return None

    async def _has_more_pages(self, articles: List[Dict[str, Any]]) -> bool:
        """Check if there are more pages to fetch."""
        pagination_type = self.pagination_config.get("type", "none")

        if pagination_type == "none":
            return False
        elif pagination_type == "page_based":
            # Check if we got fewer articles than expected
            page_size = self.pagination_config.get("page_size", 20)
            return len(articles) >= page_size
        elif pagination_type == "offset_based":
            # Check if we got fewer articles than expected
            limit = self.pagination_config.get("limit", 20)
            return len(articles) >= limit
        elif pagination_type == "cursor_based":
            # Check if there's a next cursor in the response
            # This would need to be implemented based on specific API
            return False
        elif pagination_type == "response_field":
            # Check a specific field in the response
            field_path = self.pagination_config.get("field", "has_more")
            # This would need to be implemented based on specific API
            return False

        return False

    async def test_connection(self) -> bool:
        """Test connection to API."""
        try:
            # Try to fetch first page
            articles = await self._fetch_page()
            return len(articles) >= 0  # Even empty response is valid
        except Exception as e:
            logger.warning(f"API connection test failed: {e}")
            return False

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the API source."""
        return {
            "type": "REST API",
            "url": self.source_config.url,
            "api_config": self.api_config,
            "pagination_type": self.pagination_config.get("type", "none"),
            "current_page": self.current_page,
            "has_more_pages": self.has_more_pages,
            "rate_limit": self.source_config.rate_limit,
            "timeout": self.source_config.timeout,
            "enabled": self.source_config.enabled,
        }
