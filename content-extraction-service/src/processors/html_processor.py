"""
HTML content processor for web pages.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment
from loguru import logger

from ..exceptions import ContentProcessingError
from ..models.content import ContentType, ProcessedImage, VideoMetadata
from ..utils.html_cleaner import HTMLCleaner
from ..utils.metadata_extractor import MetadataExtractor
from ..utils.readability_extractor import ReadabilityExtractor


class HTMLProcessor:
    """HTML content processor with advanced cleaning and extraction capabilities."""

    def __init__(
        self,
        html_cleaner: HTMLCleaner,
        readability_extractor: ReadabilityExtractor,
        metadata_extractor: MetadataExtractor,
    ):
        """Initialize HTML processor."""
        self.html_cleaner = html_cleaner
        self.readability_extractor = readability_extractor
        self.metadata_extractor = metadata_extractor

    async def process_html(
        self,
        html_content: str,
        url: str,
        include_images: bool = True,
        include_videos: bool = True,
        custom_selectors: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Process HTML content and extract clean text, images, and metadata.

        Args:
            html_content: Raw HTML content
            url: Source URL
            include_images: Whether to process images
            include_videos: Whether to process videos
            custom_selectors: Custom CSS selectors for content extraction

        Returns:
            Dictionary with processed content data
        """
        try:
            logger.info(f"Processing HTML content from {url}")

            # Parse HTML
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract metadata
            metadata = await self.metadata_extractor.extract_metadata(html_content, url)

            # Extract main content
            main_content = await self._extract_main_content(soup, url, custom_selectors)

            # Clean and sanitize content
            clean_content = await self.html_cleaner.sanitize_text(main_content)

            # Extract images
            images = []
            if include_images:
                images = await self._extract_images(soup, url)

            # Extract videos
            videos = []
            if include_videos:
                videos = await self._extract_videos(soup, url)

            # Extract additional content elements
            additional_content = await self._extract_additional_content(soup, url)

            return {
                "content": clean_content,
                "metadata": metadata,
                "images": images,
                "videos": videos,
                "additional_content": additional_content,
                "content_type": ContentType.HTML,
            }

        except Exception as e:
            logger.error(f"HTML processing failed for {url}: {str(e)}")
            raise ContentProcessingError(f"HTML processing failed: {str(e)}")

    async def _extract_main_content(
        self, soup: BeautifulSoup, url: str, custom_selectors: Optional[Dict[str, str]] = None
    ) -> str:
        """Extract main content using readability or custom selectors."""
        try:
            if custom_selectors:
                return await self._extract_with_custom_selectors(soup, custom_selectors)
            else:
                return await self.readability_extractor.extract_main_content(str(soup), url)
        except Exception as e:
            logger.warning(f"Main content extraction failed, using fallback: {str(e)}")
            return await self._extract_with_fallback(soup)

    async def _extract_with_custom_selectors(
        self, soup: BeautifulSoup, custom_selectors: Dict[str, str]
    ) -> str:
        """Extract content using custom CSS selectors."""
        content_parts = []

        for selector_name, selector in custom_selectors.items():
            try:
                elements = soup.select(selector)
                for element in elements:
                    # Remove script and style elements
                    for script in element(["script", "style"]):
                        script.decompose()

                    text = element.get_text(separator=" ", strip=True)
                    if text:
                        content_parts.append(text)
            except Exception as e:
                logger.warning(f"Custom selector '{selector_name}' failed: {str(e)}")
                continue

        return "\n\n".join(content_parts)

    async def _extract_with_fallback(self, soup: BeautifulSoup) -> str:
        """Fallback content extraction method."""
        # Remove unwanted elements
        for element in soup(
            ["script", "style", "nav", "header", "footer", "aside", "advertisement"]
        ):
            element.decompose()

        # Try to find main content areas
        main_selectors = [
            "main",
            "article",
            ".content",
            ".post",
            ".entry",
            ".article",
            "#content",
            "#main",
            "#post",
            "#article",
            ".main-content",
        ]

        for selector in main_selectors:
            elements = soup.select(selector)
            if elements:
                return elements[0].get_text(separator=" ", strip=True)

        # Fallback to body content
        body = soup.find("body")
        if body:
            return body.get_text(separator=" ", strip=True)

        return soup.get_text(separator=" ", strip=True)

    async def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[ProcessedImage]:
        """Extract and process images from HTML."""
        images = []

        try:
            img_tags = soup.find_all("img")

            for img in img_tags:
                try:
                    src = img.get("src")
                    if not src:
                        continue

                    # Resolve relative URLs
                    img_url = urljoin(base_url, src)

                    # Extract image attributes
                    alt_text = img.get("alt", "")
                    title = img.get("title", "")
                    width = self._parse_dimension(img.get("width"))
                    height = self._parse_dimension(img.get("height"))

                    # Create processed image
                    processed_image = ProcessedImage(
                        url=img_url,
                        width=width or 0,
                        height=height or 0,
                        file_size=0,  # Will be updated during processing
                        format=self._detect_image_format(img_url),
                        alt_text=alt_text,
                        caption=title,
                        is_optimized=False,
                        quality_score=0.0,
                    )

                    images.append(processed_image)

                except Exception as e:
                    logger.warning(f"Failed to process image {src}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Image extraction failed: {str(e)}")

        return images

    async def _extract_videos(self, soup: BeautifulSoup, base_url: str) -> List[VideoMetadata]:
        """Extract video metadata from HTML."""
        videos = []

        try:
            # Extract video tags
            video_tags = soup.find_all("video")
            for video in video_tags:
                try:
                    src = video.get("src")
                    if not src:
                        continue

                    video_url = urljoin(base_url, src)

                    video_metadata = VideoMetadata(
                        url=video_url,
                        title=video.get("title", ""),
                        duration=self._parse_duration(video.get("duration")),
                        width=self._parse_dimension(video.get("width")),
                        height=self._parse_dimension(video.get("height")),
                        format=self._detect_video_format(video_url),
                    )

                    videos.append(video_metadata)

                except Exception as e:
                    logger.warning(f"Failed to process video {src}: {str(e)}")
                    continue

            # Extract iframe videos (YouTube, Vimeo, etc.)
            iframe_tags = soup.find_all("iframe")
            for iframe in iframe_tags:
                try:
                    src = iframe.get("src", "")
                    if not src:
                        continue

                    # Check if it's a video iframe
                    if any(
                        domain in src
                        for domain in ["youtube.com", "youtu.be", "vimeo.com", "dailymotion.com"]
                    ):
                        video_metadata = VideoMetadata(
                            url=src,
                            title=iframe.get("title", ""),
                            width=self._parse_dimension(iframe.get("width")),
                            height=self._parse_dimension(iframe.get("height")),
                            format="iframe",
                        )

                        videos.append(video_metadata)

                except Exception as e:
                    logger.warning(f"Failed to process iframe video {src}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Video extraction failed: {str(e)}")

        return videos

    async def _extract_additional_content(
        self, soup: BeautifulSoup, base_url: str
    ) -> Dict[str, Any]:
        """Extract additional content elements like links, tables, etc."""
        additional_content = {
            "links": [],
            "tables": [],
            "lists": [],
            "quotes": [],
            "code_blocks": [],
        }

        try:
            # Extract links
            link_tags = soup.find_all("a", href=True)
            for link in link_tags:
                href = link.get("href")
                if href:
                    link_url = urljoin(base_url, href)
                    additional_content["links"].append(
                        {
                            "url": link_url,
                            "text": link.get_text(strip=True),
                            "title": link.get("title", ""),
                        }
                    )

            # Extract tables
            table_tags = soup.find_all("table")
            for table in table_tags:
                table_data = self._extract_table_data(table)
                if table_data:
                    additional_content["tables"].append(table_data)

            # Extract lists
            list_tags = soup.find_all(["ul", "ol"])
            for list_tag in list_tags:
                list_items = [li.get_text(strip=True) for li in list_tag.find_all("li")]
                if list_items:
                    additional_content["lists"].append({"type": list_tag.name, "items": list_items})

            # Extract quotes
            quote_tags = soup.find_all(["blockquote", "q"])
            for quote in quote_tags:
                quote_text = quote.get_text(strip=True)
                if quote_text:
                    additional_content["quotes"].append(
                        {"text": quote_text, "author": quote.get("cite", ""), "type": quote.name}
                    )

            # Extract code blocks
            code_tags = soup.find_all(["code", "pre"])
            for code in code_tags:
                code_text = code.get_text(strip=True)
                if code_text and len(code_text) > 10:  # Only include substantial code
                    additional_content["code_blocks"].append(
                        {
                            "text": code_text,
                            "language": (
                                code.get("class", [""])[0].replace("language-", "")
                                if code.get("class")
                                else ""
                            ),
                            "type": code.name,
                        }
                    )

        except Exception as e:
            logger.error(f"Additional content extraction failed: {str(e)}")

        return additional_content

    def _extract_table_data(self, table) -> Optional[Dict[str, Any]]:
        """Extract data from HTML table."""
        try:
            rows = table.find_all("tr")
            if not rows:
                return None

            table_data = {"headers": [], "rows": [], "caption": ""}

            # Extract caption
            caption = table.find("caption")
            if caption:
                table_data["caption"] = caption.get_text(strip=True)

            # Extract headers
            header_row = table.find("thead")
            if header_row:
                headers = header_row.find_all(["th", "td"])
                table_data["headers"] = [h.get_text(strip=True) for h in headers]
            else:
                # Use first row as headers
                first_row = rows[0]
                headers = first_row.find_all(["th", "td"])
                table_data["headers"] = [h.get_text(strip=True) for h in headers]
                rows = rows[1:]  # Skip first row

            # Extract data rows
            for row in rows:
                cells = row.find_all(["td", "th"])
                row_data = [cell.get_text(strip=True) for cell in cells]
                if row_data:
                    table_data["rows"].append(row_data)

            return table_data

        except Exception as e:
            logger.warning(f"Table extraction failed: {str(e)}")
            return None

    def _parse_dimension(self, value: Optional[str]) -> Optional[int]:
        """Parse dimension value to integer."""
        if not value:
            return None

        try:
            # Remove 'px' suffix if present
            value = str(value).replace("px", "").strip()
            return int(float(value))
        except (ValueError, TypeError):
            return None

    def _parse_duration(self, value: Optional[str]) -> Optional[int]:
        """Parse duration value to seconds."""
        if not value:
            return None

        try:
            # Handle different duration formats
            if ":" in value:
                # Format: MM:SS or HH:MM:SS
                parts = value.split(":")
                if len(parts) == 2:
                    minutes, seconds = map(int, parts)
                    return minutes * 60 + seconds
                elif len(parts) == 3:
                    hours, minutes, seconds = map(int, parts)
                    return hours * 3600 + minutes * 60 + seconds
            else:
                # Assume seconds
                return int(float(value))
        except (ValueError, TypeError):
            return None

    def _detect_image_format(self, url: str) -> str:
        """Detect image format from URL."""
        url_lower = url.lower()
        if url_lower.endswith(".jpg") or url_lower.endswith(".jpeg"):
            return "JPEG"
        elif url_lower.endswith(".png"):
            return "PNG"
        elif url_lower.endswith(".gif"):
            return "GIF"
        elif url_lower.endswith(".webp"):
            return "WebP"
        elif url_lower.endswith(".svg"):
            return "SVG"
        else:
            return "Unknown"

    def _detect_video_format(self, url: str) -> str:
        """Detect video format from URL."""
        url_lower = url.lower()
        if url_lower.endswith(".mp4"):
            return "MP4"
        elif url_lower.endswith(".webm"):
            return "WebM"
        elif url_lower.endswith(".ogg"):
            return "OGG"
        elif "youtube.com" in url_lower or "youtu.be" in url_lower:
            return "YouTube"
        elif "vimeo.com" in url_lower:
            return "Vimeo"
        else:
            return "Unknown"

    async def clean_html_content(
        self,
        html_content: str,
        remove_scripts: bool = True,
        remove_styles: bool = True,
        remove_comments: bool = True,
        remove_ads: bool = True,
    ) -> str:
        """
        Clean HTML content by removing unwanted elements.

        Args:
            html_content: Raw HTML content
            remove_scripts: Whether to remove script tags
            remove_styles: Whether to remove style tags
            remove_comments: Whether to remove HTML comments
            remove_ads: Whether to remove advertisement elements

        Returns:
            Cleaned HTML content
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove scripts
            if remove_scripts:
                for script in soup(["script", "noscript"]):
                    script.decompose()

            # Remove styles
            if remove_styles:
                for style in soup(["style", "link"]):
                    if style.get("rel") == ["stylesheet"]:
                        style.decompose()

            # Remove comments
            if remove_comments:
                comments = soup.find_all(string=lambda text: isinstance(text, Comment))
                for comment in comments:
                    comment.extract()

            # Remove ads and unwanted elements
            if remove_ads:
                ad_selectors = [
                    '[class*="ad"]',
                    '[class*="advertisement"]',
                    '[class*="banner"]',
                    '[id*="ad"]',
                    '[id*="advertisement"]',
                    '[id*="banner"]',
                    ".ads",
                    ".advertisement",
                    ".banner",
                    ".sponsor",
                    ".promo",
                ]

                for selector in ad_selectors:
                    for element in soup.select(selector):
                        element.decompose()

            # Remove empty elements
            for element in soup.find_all():
                if not element.get_text(strip=True) and not element.find(["img", "video", "audio"]):
                    element.decompose()

            return str(soup)

        except Exception as e:
            logger.error(f"HTML cleaning failed: {str(e)}")
            return html_content
