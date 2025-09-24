"""
Web scraping adapter for content ingestion using Playwright.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.adapters.base import BaseAdapter
from src.models.content import NormalizedArticle, SourceConfig
from src.utils.http_client import HTTPResponse

logger = logging.getLogger(__name__)


class WebScrapingAdapter(BaseAdapter):
    """Adapter for web scraping content sources."""

    def __init__(self, source_config: SourceConfig, **kwargs):
        super().__init__(source_config, **kwargs)
        self.scraping_config = source_config.filters.get("scraping_config", {})
        self.playwright = None
        self.browser = None
        self.page = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup_browser()

    async def _initialize_browser(self):
        """Initialize Playwright browser."""
        try:
            from playwright.async_api import async_playwright

            self.playwright = await async_playwright().start()

            # Browser options
            browser_options = {
                "headless": True,
                "args": [
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                ],
            }

            # Launch browser
            self.browser = await self.playwright.chromium.launch(**browser_options)

            # Create page
            self.page = await self.browser.new_page()

            # Set user agent
            await self.page.set_extra_http_headers({"User-Agent": self.http_client.user_agent})

            # Set viewport
            await self.page.set_viewport_size({"width": 1920, "height": 1080})

        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise

    async def _cleanup_browser(self):
        """Cleanup browser resources."""
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            logger.warning(f"Error cleaning up browser: {e}")

    async def fetch_content(self) -> AsyncGenerator[NormalizedArticle, None]:
        """Fetch content by web scraping."""
        try:
            # Initialize browser if not already done
            if not self.browser:
                await self._initialize_browser()

            # Get URLs to scrape
            urls = await self._get_urls_to_scrape()

            for url in urls:
                try:
                    # Check robots.txt compliance
                    if not await self._check_robots_compliance(url):
                        logger.info(f"Skipping {url} due to robots.txt")
                        continue

                    # Scrape URL
                    article = await self._scrape_url(url)

                    if (
                        article
                        and self._is_valid_article(article)
                        and self._apply_content_filters(article)
                    ):
                        yield article

                    # Apply rate limiting
                    await self._apply_rate_limiting()

                except Exception as e:
                    logger.warning(f"Error scraping URL {url}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in web scraping: {e}")
            raise

    async def _get_urls_to_scrape(self) -> List[str]:
        """Get URLs to scrape based on configuration."""
        urls = []

        # Get URLs from configuration
        if "urls" in self.scraping_config:
            urls.extend(self.scraping_config["urls"])

        # Get URLs from sitemap
        if "sitemap_url" in self.scraping_config:
            sitemap_urls = await self._extract_urls_from_sitemap(
                self.scraping_config["sitemap_url"]
            )
            urls.extend(sitemap_urls)

        # Get URLs from listing pages
        if "listing_pages" in self.scraping_config:
            for listing_url in self.scraping_config["listing_pages"]:
                listing_urls = await self._extract_urls_from_listing(listing_url)
                urls.extend(listing_urls)

        # Remove duplicates and filter
        urls = list(set(urls))
        urls = [url for url in urls if self.url_utils.is_valid_url(url)]
        urls = [url for url in urls if not self.url_utils.should_skip_url(url)]

        # Limit number of URLs
        max_urls = self.scraping_config.get("max_urls", 100)
        urls = urls[:max_urls]

        return urls

    async def _extract_urls_from_sitemap(self, sitemap_url: str) -> List[str]:
        """Extract URLs from sitemap."""
        try:
            response = await self.http_client.get(sitemap_url)
            if response.status_code != 200:
                return []

            # Parse sitemap XML
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "xml")

            urls = []
            for loc in soup.find_all("loc"):
                url = loc.get_text().strip()
                if self.url_utils.is_valid_url(url):
                    urls.append(url)

            return urls

        except Exception as e:
            logger.warning(f"Error extracting URLs from sitemap {sitemap_url}: {e}")
            return []

    async def _extract_urls_from_listing(self, listing_url: str) -> List[str]:
        """Extract URLs from listing page."""
        try:
            await self.page.goto(listing_url, wait_until="networkidle")

            # Get URLs using CSS selectors
            selectors = self.scraping_config.get("url_selectors", ["a[href]"])
            urls = []

            for selector in selectors:
                elements = await self.page.query_selector_all(selector)
                for element in elements:
                    href = await element.get_attribute("href")
                    if href:
                        # Convert relative URL to absolute
                        absolute_url = self.url_utils.resolve_relative_url(href, listing_url)
                        if self.url_utils.is_valid_url(absolute_url):
                            urls.append(absolute_url)

            return urls

        except Exception as e:
            logger.warning(f"Error extracting URLs from listing {listing_url}: {e}")
            return []

    async def _check_robots_compliance(self, url: str) -> bool:
        """Check if URL can be scraped according to robots.txt."""
        try:
            from src.utils.http_client import robots_checker

            return await robots_checker.can_fetch(url, self.http_client.user_agent)
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}")
            return True  # Allow if we can't check

    async def _scrape_url(self, url: str) -> Optional[NormalizedArticle]:
        """Scrape a single URL for content."""
        try:
            # Navigate to URL
            await self.page.goto(
                url, wait_until="networkidle", timeout=self.source_config.timeout * 1000
            )

            # Wait for content to load
            await self._wait_for_content()

            # Extract content
            content_data = await self._extract_content_from_page()

            if not content_data:
                return None

            # Process extracted content
            article = await self._process_scraped_content(url, content_data)

            return article

        except Exception as e:
            logger.warning(f"Error scraping URL {url}: {e}")
            return None

    async def _wait_for_content(self):
        """Wait for content to load on the page."""
        try:
            # Wait for main content selectors
            content_selectors = self.scraping_config.get(
                "content_selectors", ["article", "main", ".content", ".post", ".article"]
            )

            for selector in content_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=5000)
                    break
                except:
                    continue

            # Wait for any lazy-loaded content
            await self.page.wait_for_timeout(2000)

        except Exception as e:
            logger.debug(f"Error waiting for content: {e}")

    async def _extract_content_from_page(self) -> Optional[Dict[str, Any]]:
        """Extract content from the current page."""
        try:
            # Get page HTML
            html_content = await self.page.content()

            # Extract content using selectors
            content_data = {}

            # Title
            title_selectors = self.scraping_config.get(
                "title_selectors", ["h1", "title", ".title", ".headline"]
            )
            content_data["title"] = await self._extract_text_by_selectors(title_selectors)

            # Content
            content_selectors = self.scraping_config.get(
                "content_selectors",
                ["article", "main", ".content", ".post", ".article", ".entry-content"],
            )
            content_data["content"] = await self._extract_html_by_selectors(content_selectors)

            # Author
            author_selectors = self.scraping_config.get(
                "author_selectors", [".author", ".byline", '[rel="author"]', ".post-author"]
            )
            content_data["author"] = await self._extract_text_by_selectors(author_selectors)

            # Published date
            date_selectors = self.scraping_config.get(
                "date_selectors", ["time", ".date", ".published", ".post-date"]
            )
            content_data["published_at"] = await self._extract_text_by_selectors(date_selectors)

            # Image
            image_selectors = self.scraping_config.get(
                "image_selectors", ["img", ".featured-image img", ".post-image img"]
            )
            content_data["image_url"] = await self._extract_attribute_by_selectors(
                image_selectors, "src"
            )

            # Tags
            tag_selectors = self.scraping_config.get(
                "tag_selectors", [".tags a", ".categories a", ".post-tags a"]
            )
            content_data["tags"] = await self._extract_text_list_by_selectors(tag_selectors)

            # Summary
            summary_selectors = self.scraping_config.get(
                "summary_selectors", [".summary", ".excerpt", ".description"]
            )
            content_data["summary"] = await self._extract_text_by_selectors(summary_selectors)

            # Add raw HTML for further processing
            content_data["raw_html"] = html_content

            return content_data

        except Exception as e:
            logger.warning(f"Error extracting content from page: {e}")
            return None

    async def _extract_text_by_selectors(self, selectors: List[str]) -> str:
        """Extract text using multiple selectors."""
        for selector in selectors:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    text = await element.text_content()
                    if text and text.strip():
                        return text.strip()
            except:
                continue
        return ""

    async def _extract_html_by_selectors(self, selectors: List[str]) -> str:
        """Extract HTML using multiple selectors."""
        for selector in selectors:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    html = await element.inner_html()
                    if html and html.strip():
                        return html.strip()
            except:
                continue
        return ""

    async def _extract_attribute_by_selectors(self, selectors: List[str], attribute: str) -> str:
        """Extract attribute value using multiple selectors."""
        for selector in selectors:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    value = await element.get_attribute(attribute)
                    if value and value.strip():
                        return value.strip()
            except:
                continue
        return ""

    async def _extract_text_list_by_selectors(self, selectors: List[str]) -> List[str]:
        """Extract list of text using multiple selectors."""
        for selector in selectors:
            try:
                elements = await self.page.query_selector_all(selector)
                if elements:
                    texts = []
                    for element in elements:
                        text = await element.text_content()
                        if text and text.strip():
                            texts.append(text.strip())
                    if texts:
                        return texts
            except:
                continue
        return []

    async def _process_scraped_content(
        self, url: str, content_data: Dict[str, Any]
    ) -> Optional[NormalizedArticle]:
        """Process scraped content into normalized article."""
        try:
            # Extract basic information
            title = content_data.get("title", "")
            content = content_data.get("content", "")
            summary = content_data.get("summary", "")
            author = content_data.get("author", "")
            published_at_str = content_data.get("published_at", "")
            image_url = content_data.get("image_url", "")
            tags = content_data.get("tags", [])
            raw_html = content_data.get("raw_html", "")

            # Parse published date
            published_at = None
            if published_at_str:
                published_at = self.date_utils.parse_date(published_at_str)

            # Extract additional information from HTML
            if raw_html:
                # Use content parser to extract additional metadata
                if not title:
                    title = self.content_parser.extract_title(raw_html)
                if not summary:
                    summary = self.content_parser.extract_description(raw_html)
                if not author:
                    author = self.content_parser.extract_author(raw_html)
                if not image_url:
                    image_url = self.content_parser.extract_image_url(raw_html, url)
                if not tags:
                    tags = self.content_parser.extract_tags(raw_html)

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
                    "scraping_method": "playwright",
                    "scraping_config": self.scraping_config,
                    "raw_html_length": len(raw_html) if raw_html else 0,
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
                raw_data=content_data,
                ingestion_metadata=ingestion_metadata,
            )

            return article

        except Exception as e:
            logger.warning(f"Error processing scraped content: {e}")
            return None

    async def test_connection(self) -> bool:
        """Test connection by trying to load a simple page."""
        try:
            if not self.browser:
                await self._initialize_browser()

            await self.page.goto("https://httpbin.org/get", timeout=10000)
            return True
        except Exception as e:
            logger.warning(f"Web scraping connection test failed: {e}")
            return False

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the web scraping source."""
        return {
            "type": "Web Scraping",
            "url": self.source_config.url,
            "scraping_config": self.scraping_config,
            "browser_initialized": self.browser is not None,
            "rate_limit": self.source_config.rate_limit,
            "timeout": self.source_config.timeout,
            "enabled": self.source_config.enabled,
        }
