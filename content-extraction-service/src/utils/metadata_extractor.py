"""
Metadata extraction utilities for OpenGraph, Twitter Cards, and JSON-LD.
"""

import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from loguru import logger

from ..exceptions import ContentProcessingError
from ..models.content import ContentMetadata


class MetadataExtractor:
    """Advanced metadata extraction from HTML content."""

    def __init__(self):
        """Initialize metadata extractor."""
        self.og_properties = [
            "og:title",
            "og:description",
            "og:image",
            "og:type",
            "og:url",
            "og:site_name",
            "og:locale",
            "og:article:author",
            "og:article:published_time",
            "og:article:modified_time",
            "og:article:section",
            "og:article:tag",
        ]
        self.twitter_properties = [
            "twitter:card",
            "twitter:title",
            "twitter:description",
            "twitter:image",
            "twitter:site",
            "twitter:creator",
            "twitter:player",
            "twitter:player:width",
            "twitter:player:height",
        ]

    async def extract_metadata(self, html_content: str, url: str) -> ContentMetadata:
        """
        Extract comprehensive metadata from HTML content.

        Args:
            html_content: Raw HTML content
            url: Source URL

        Returns:
            ContentMetadata object
        """
        try:
            logger.info(f"Extracting metadata from {url}")

            soup = BeautifulSoup(html_content, "html.parser")

            # Extract Open Graph metadata
            og_data = await self._extract_open_graph(soup)

            # Extract Twitter Card metadata
            twitter_data = await self._extract_twitter_cards(soup)

            # Extract JSON-LD structured data
            json_ld_data = await self._extract_json_ld(soup)

            # Extract basic meta tags
            basic_meta = await self._extract_basic_meta(soup)

            # Extract canonical link
            canonical_link = await self._extract_canonical_link(soup, url)

            # Extract favicon
            favicon = await self._extract_favicon(soup, url)

            # Extract site name
            site_name = await self._extract_site_name(soup, og_data)

            # Extract author information
            author = await self._extract_author(soup, og_data, json_ld_data)

            # Extract keywords and categories
            keywords, categories, tags = await self._extract_keywords_and_categories(soup, json_ld_data)

            return ContentMetadata(
                og_title=og_data.get("og:title"),
                og_description=og_data.get("og:description"),
                og_image=og_data.get("og:image"),
                og_type=og_data.get("og:type"),
                og_url=og_data.get("og:url"),
                twitter_card=twitter_data.get("twitter:card"),
                twitter_title=twitter_data.get("twitter:title"),
                twitter_description=twitter_data.get("twitter:description"),
                twitter_image=twitter_data.get("twitter:image"),
                json_ld=json_ld_data,
                canonical_link=canonical_link,
                amp_url=basic_meta.get("amp_url"),
                rss_feed=basic_meta.get("rss_feed"),
                favicon=favicon,
                site_name=site_name,
                robots=basic_meta.get("robots"),
                viewport=basic_meta.get("viewport"),
                charset=basic_meta.get("charset"),
                generator=basic_meta.get("generator"),
                author=author,
                keywords=keywords,
                categories=categories,
                tags=tags,
            )

        except Exception as e:
            logger.error(f"Metadata extraction failed for {url}: {str(e)}")
            raise ContentProcessingError(f"Metadata extraction failed: {str(e)}")

    async def _extract_open_graph(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract Open Graph metadata."""
        og_data = {}

        try:
            # Extract og: properties
            for prop in self.og_properties:
                meta_tag = soup.find("meta", property=prop)
                if meta_tag and meta_tag.get("content"):
                    og_data[prop] = meta_tag["content"]

            # Also check for og: properties in name attribute
            for prop in self.og_properties:
                meta_tag = soup.find("meta", attrs={"name": prop})
                if meta_tag and meta_tag.get("content"):
                    og_data[prop] = meta_tag["content"]

        except Exception as e:
            logger.warning(f"Open Graph extraction failed: {str(e)}")

        return og_data

    async def _extract_twitter_cards(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract Twitter Card metadata."""
        twitter_data = {}

        try:
            # Extract twitter: properties
            for prop in self.twitter_properties:
                meta_tag = soup.find("meta", attrs={"name": prop})
                if meta_tag and meta_tag.get("content"):
                    twitter_data[prop] = meta_tag["content"]

        except Exception as e:
            logger.warning(f"Twitter Cards extraction failed: {str(e)}")

        return twitter_data

    async def _extract_json_ld(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract JSON-LD structured data."""
        json_ld_data = {}

        try:
            # Find all script tags with type="application/ld+json"
            script_tags = soup.find_all("script", type="application/ld+json")

            for i, script in enumerate(script_tags):
                try:
                    content = script.string
                    if content:
                        data = json.loads(content)
                        json_ld_data[f"schema_{i}"] = data
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON-LD parsing failed for script {i}: {str(e)}")
                    continue

        except Exception as e:
            logger.warning(f"JSON-LD extraction failed: {str(e)}")

        return json_ld_data

    async def _extract_basic_meta(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract basic meta tags."""
        basic_meta = {}

        try:
            # Extract title
            title_tag = soup.find("title")
            if title_tag:
                basic_meta["title"] = title_tag.get_text(strip=True)

            # Extract description
            desc_tag = soup.find("meta", attrs={"name": "description"})
            if desc_tag and desc_tag.get("content"):
                basic_meta["description"] = desc_tag["content"]

            # Extract keywords
            keywords_tag = soup.find("meta", attrs={"name": "keywords"})
            if keywords_tag and keywords_tag.get("content"):
                basic_meta["keywords"] = keywords_tag["content"]

            # Extract robots
            robots_tag = soup.find("meta", attrs={"name": "robots"})
            if robots_tag and robots_tag.get("content"):
                basic_meta["robots"] = robots_tag["content"]

            # Extract viewport
            viewport_tag = soup.find("meta", attrs={"name": "viewport"})
            if viewport_tag and viewport_tag.get("content"):
                basic_meta["viewport"] = viewport_tag["content"]

            # Extract charset
            charset_tag = soup.find("meta", attrs={"charset": True})
            if charset_tag:
                basic_meta["charset"] = charset_tag.get("charset")

            # Extract generator
            generator_tag = soup.find("meta", attrs={"name": "generator"})
            if generator_tag and generator_tag.get("content"):
                basic_meta["generator"] = generator_tag["content"]

            # Extract AMP URL
            amp_tag = soup.find("link", attrs={"rel": "amphtml"})
            if amp_tag and amp_tag.get("href"):
                basic_meta["amp_url"] = amp_tag["href"]

            # Extract RSS feed
            rss_tag = soup.find("link", attrs={"type": "application/rss+xml"})
            if rss_tag and rss_tag.get("href"):
                basic_meta["rss_feed"] = rss_tag["href"]

        except Exception as e:
            logger.warning(f"Basic meta extraction failed: {str(e)}")

        return basic_meta

    async def _extract_canonical_link(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract canonical link."""
        try:
            canonical_tag = soup.find("link", attrs={"rel": "canonical"})
            if canonical_tag and canonical_tag.get("href"):
                canonical_url = canonical_tag["href"]
                # Resolve relative URLs
                if canonical_url.startswith("/"):
                    parsed_url = urlparse(url)
                    canonical_url = f"{parsed_url.scheme}://{parsed_url.netloc}{canonical_url}"
                return canonical_url
        except Exception as e:
            logger.warning(f"Canonical link extraction failed: {str(e)}")

        return None

    async def _extract_favicon(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract favicon URL."""
        try:
            # Try different favicon selectors
            favicon_selectors = [
                'link[rel="icon"]',
                'link[rel="shortcut icon"]',
                'link[rel="apple-touch-icon"]',
                'link[rel="apple-touch-icon-precomposed"]',
            ]

            for selector in favicon_selectors:
                favicon_tag = soup.select_one(selector)
                if favicon_tag and favicon_tag.get("href"):
                    favicon_url = favicon_tag["href"]
                    # Resolve relative URLs
                    if favicon_url.startswith("/"):
                        parsed_url = urlparse(url)
                        favicon_url = f"{parsed_url.scheme}://{parsed_url.netloc}{favicon_url}"
                    return favicon_url

            # Fallback to default favicon location
            parsed_url = urlparse(url)
            return f"{parsed_url.scheme}://{parsed_url.netloc}/favicon.ico"

        except Exception as e:
            logger.warning(f"Favicon extraction failed: {str(e)}")

        return None

    async def _extract_site_name(self, soup: BeautifulSoup, og_data: Dict[str, str]) -> Optional[str]:
        """Extract site name."""
        try:
            # Try Open Graph site name first
            if "og:site_name" in og_data:
                return og_data["og:site_name"]

            # Try meta tag
            site_name_tag = soup.find("meta", attrs={"name": "application-name"})
            if site_name_tag and site_name_tag.get("content"):
                return site_name_tag["content"]

            # Try title tag as fallback
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)
                # Extract site name from title (before separator)
                if " - " in title:
                    return title.split(" - ")[-1].strip()
                elif " | " in title:
                    return title.split(" | ")[-1].strip()

        except Exception as e:
            logger.warning(f"Site name extraction failed: {str(e)}")

        return None

    async def _extract_author(
        self, soup: BeautifulSoup, og_data: Dict[str, str], json_ld_data: Dict[str, Any]
    ) -> Optional[str]:
        """Extract author information."""
        try:
            # Try Open Graph author
            if "og:article:author" in og_data:
                return og_data["og:article:author"]

            # Try meta tag
            author_tag = soup.find("meta", attrs={"name": "author"})
            if author_tag and author_tag.get("content"):
                return author_tag["content"]

            # Try JSON-LD author
            for schema_data in json_ld_data.values():
                if isinstance(schema_data, dict):
                    author = schema_data.get("author", {})
                    if isinstance(author, dict):
                        name = author.get("name")
                        if name:
                            return name
                    elif isinstance(author, str):
                        return author

        except Exception as e:
            logger.warning(f"Author extraction failed: {str(e)}")

        return None

    async def _extract_keywords_and_categories(
        self, soup: BeautifulSoup, json_ld_data: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Extract keywords, categories, and tags."""
        keywords = []
        categories = []
        tags = []

        try:
            # Extract keywords from meta tag
            keywords_tag = soup.find("meta", attrs={"name": "keywords"})
            if keywords_tag and keywords_tag.get("content"):
                keywords = [kw.strip() for kw in keywords_tag["content"].split(",") if kw.strip()]

            # Extract from Open Graph tags
            og_tags = []
            for key, value in json_ld_data.items():
                if isinstance(value, dict):
                    if "keywords" in value:
                        if isinstance(value["keywords"], list):
                            keywords.extend(value["keywords"])
                        elif isinstance(value["keywords"], str):
                            keywords.extend([kw.strip() for kw in value["keywords"].split(",")])

                    if "articleSection" in value:
                        categories.append(value["articleSection"])

                    if "articleTag" in value:
                        if isinstance(value["articleTag"], list):
                            tags.extend(value["articleTag"])
                        elif isinstance(value["articleTag"], str):
                            tags.append(value["articleTag"])

            # Extract from JSON-LD
            for schema_data in json_ld_data.values():
                if isinstance(schema_data, dict):
                    if "keywords" in schema_data:
                        if isinstance(schema_data["keywords"], list):
                            keywords.extend(schema_data["keywords"])
                        elif isinstance(schema_data["keywords"], str):
                            keywords.extend([kw.strip() for kw in schema_data["keywords"].split(",")])

                    if "articleSection" in schema_data:
                        categories.append(schema_data["articleSection"])

                    if "articleTag" in schema_data:
                        if isinstance(schema_data["articleTag"], list):
                            tags.extend(schema_data["articleTag"])
                        elif isinstance(schema_data["articleTag"], str):
                            tags.append(schema_data["articleTag"])

            # Remove duplicates and clean
            keywords = list(set([kw.strip() for kw in keywords if kw.strip()]))
            categories = list(set([cat.strip() for cat in categories if cat.strip()]))
            tags = list(set([tag.strip() for tag in tags if tag.strip()]))

        except Exception as e:
            logger.warning(f"Keywords and categories extraction failed: {str(e)}")

        return keywords, categories, tags

    async def extract_structured_data(self, html_content: str) -> Dict[str, Any]:
        """
        Extract all structured data from HTML content.

        Args:
            html_content: Raw HTML content

        Returns:
            Dictionary with all structured data
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            structured_data = {
                "open_graph": await self._extract_open_graph(soup),
                "twitter_cards": await self._extract_twitter_cards(soup),
                "json_ld": await self._extract_json_ld(soup),
                "basic_meta": await self._extract_basic_meta(soup),
            }

            return structured_data

        except Exception as e:
            logger.error(f"Structured data extraction failed: {str(e)}")
            return {}
