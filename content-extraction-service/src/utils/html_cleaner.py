"""
HTML cleaning and sanitization utilities.
"""

import re
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup, Comment
from loguru import logger

from ..exceptions import ContentProcessingError


class HTMLCleaner:
    """Advanced HTML cleaning and sanitization system."""

    def __init__(
        self,
        allowed_tags: List[str] = None,
        allowed_attributes: List[str] = None,
        remove_scripts: bool = True,
        remove_styles: bool = True,
        remove_comments: bool = True,
    ):
        """Initialize HTML cleaner."""
        self.allowed_tags = allowed_tags or [
            "a",
            "abbr",
            "acronym",
            "b",
            "blockquote",
            "code",
            "em",
            "i",
            "li",
            "ol",
            "p",
            "strong",
            "ul",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "img",
            "br",
            "hr",
            "pre",
            "span",
            "div",
            "table",
            "tbody",
            "td",
            "th",
            "thead",
            "tr",
            "figcaption",
            "figure",
        ]
        self.allowed_attributes = allowed_attributes or [
            "href",
            "src",
            "alt",
            "title",
            "class",
            "style",
            "width",
            "height",
        ]
        self.remove_scripts = remove_scripts
        self.remove_styles = remove_styles
        self.remove_comments = remove_comments

    async def clean_html(self, html_content: str) -> str:
        """
        Clean HTML content by removing unwanted elements and attributes.

        Args:
            html_content: Raw HTML content

        Returns:
            Cleaned HTML content
        """
        try:
            logger.info("Cleaning HTML content")

            # Parse HTML
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove unwanted elements
            await self._remove_unwanted_elements(soup)

            # Clean attributes
            await self._clean_attributes(soup)

            # Remove empty elements
            await self._remove_empty_elements(soup)

            # Clean text content
            await self._clean_text_content(soup)

            return str(soup)

        except Exception as e:
            logger.error(f"HTML cleaning failed: {str(e)}")
            raise ContentProcessingError(f"HTML cleaning failed: {str(e)}")

    async def sanitize_text(self, text_content: str) -> str:
        """
        Sanitize plain text content.

        Args:
            text_content: Raw text content

        Returns:
            Sanitized text content
        """
        try:
            # Remove excessive whitespace
            text = re.sub(r"\s+", " ", text_content)

            # Remove control characters
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

            # Normalize line endings
            text = text.replace("\r\n", "\n").replace("\r", "\n")

            # Remove excessive newlines
            text = re.sub(r"\n{3,}", "\n\n", text)

            # Strip leading/trailing whitespace
            text = text.strip()

            return text

        except Exception as e:
            logger.error(f"Text sanitization failed: {str(e)}")
            raise ContentProcessingError(f"Text sanitization failed: {str(e)}")

    async def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Remove unwanted HTML elements."""
        try:
            # Remove scripts and noscript
            if self.remove_scripts:
                for element in soup(["script", "noscript", "iframe"]):
                    element.decompose()

            # Remove styles
            if self.remove_styles:
                for element in soup(["style", "link"]):
                    if element.get("rel") == ["stylesheet"]:
                        element.decompose()

            # Remove comments
            if self.remove_comments:
                comments = soup.find_all(
                    string=lambda text: isinstance(
                        text, Comment))
                for comment in comments:
                    comment.extract()

            # Remove advertisement elements
            await self._remove_advertisement_elements(soup)

            # Remove navigation elements
            await self._remove_navigation_elements(soup)

            # Remove social media widgets
            await self._remove_social_widgets(soup)

        except Exception as e:
            logger.warning(f"Element removal failed: {str(e)}")

    async def _remove_advertisement_elements(
            self, soup: BeautifulSoup) -> None:
        """Remove advertisement and promotional elements."""
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
            '[class*="popup"]',
            '[class*="modal"]',
            '[class*="overlay"]',
        ]

        for selector in ad_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    element.decompose()
            except Exception as e:
                logger.warning(
                    f"Failed to remove ads with selector {selector}: {str(e)}")

    async def _remove_navigation_elements(self, soup: BeautifulSoup) -> None:
        """Remove navigation and menu elements."""
        nav_selectors = [
            "nav",
            "header",
            "footer",
            "aside",
            "menu",
            "sidebar",
            '[class*="nav"]',
            '[class*="menu"]',
            '[class*="sidebar"]',
            '[class*="header"]',
            '[class*="footer"]',
        ]

        for selector in nav_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    element.decompose()
            except Exception as e:
                logger.warning(
                    f"Failed to remove nav with selector {selector}: {str(e)}")

    async def _remove_social_widgets(self, soup: BeautifulSoup) -> None:
        """Remove social media widgets and sharing buttons."""
        social_selectors = [
            '[class*="social"]',
            '[class*="share"]',
            '[class*="follow"]',
            '[class*="facebook"]',
            '[class*="twitter"]',
            '[class*="linkedin"]',
            '[class*="instagram"]',
            '[class*="youtube"]',
            '[class*="pinterest"]',
        ]

        for selector in social_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    element.decompose()
            except Exception as e:
                logger.warning(
                    f"Failed to remove social widgets with selector {selector}: {str(e)}"
                )

    async def _clean_attributes(self, soup: BeautifulSoup) -> None:
        """Clean HTML attributes."""
        try:
            for element in soup.find_all():
                # Remove disallowed attributes
                attrs_to_remove = []
                for attr in element.attrs:
                    if attr not in self.allowed_attributes:
                        attrs_to_remove.append(attr)

                for attr in attrs_to_remove:
                    del element.attrs[attr]

                # Clean specific attributes
                await self._clean_specific_attributes(element)

        except Exception as e:
            logger.warning(f"Attribute cleaning failed: {str(e)}")

    async def _clean_specific_attributes(self, element) -> None:
        """Clean specific attributes like style and class."""
        try:
            # Clean style attributes
            if "style" in element.attrs:
                style = element.attrs["style"]
                # Remove dangerous CSS properties
                dangerous_props = [
                    "javascript:", "expression(", "eval(", "url("]
                for prop in dangerous_props:
                    if prop in style.lower():
                        del element.attrs["style"]
                        break

            # Clean class attributes
            if "class" in element.attrs:
                classes = element.attrs["class"]
                if isinstance(classes, list):
                    # Remove suspicious class names
                    suspicious_classes = [
                        "ad", "advertisement", "banner", "popup", "modal"]
                    element.attrs["class"] = [
                        cls
                        for cls in classes
                        if not any(sus in cls.lower() for sus in suspicious_classes)
                    ]

            # Clean href attributes
            if "href" in element.attrs:
                href = element.attrs["href"]
                if href.startswith("javascript:") or href.startswith("data:"):
                    del element.attrs["href"]

            # Clean src attributes
            if "src" in element.attrs:
                src = element.attrs["src"]
                if src.startswith("data:") and not src.startswith(
                        "data:image/"):
                    del element.attrs["src"]

        except Exception as e:
            logger.warning(f"Specific attribute cleaning failed: {str(e)}")

    async def _remove_empty_elements(self, soup: BeautifulSoup) -> None:
        """Remove empty elements."""
        try:
            for element in soup.find_all():
                # Check if element is empty
                if not element.get_text(strip=True) and not element.find(
                    ["img", "video", "audio", "br", "hr"]
                ):
                    element.decompose()

        except Exception as e:
            logger.warning(f"Empty element removal failed: {str(e)}")

    async def _clean_text_content(self, soup: BeautifulSoup) -> None:
        """Clean text content within elements."""
        try:
            for element in soup.find_all():
                if element.string:
                    # Clean text content
                    text = element.string
                    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
                    text = text.strip()
                    element.string.replace_with(text)

        except Exception as e:
            logger.warning(f"Text content cleaning failed: {str(e)}")

    async def extract_main_content(self, html_content: str) -> str:
        """
        Extract main content from HTML using heuristics.

        Args:
            html_content: Raw HTML content

        Returns:
            Extracted main content text
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove unwanted elements first
            await self._remove_unwanted_elements(soup)

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
                ".entry-content",
                ".post-content",
                ".article-content",
            ]

            for selector in main_selectors:
                elements = soup.select(selector)
                if elements:
                    # Get text from the first matching element
                    main_element = elements[0]
                    text = main_element.get_text(separator=" ", strip=True)
                    if len(text) > 100:  # Ensure substantial content
                        return text

            # Fallback to body content
            body = soup.find("body")
            if body:
                return body.get_text(separator=" ", strip=True)

            # Final fallback
            return soup.get_text(separator=" ", strip=True)

        except Exception as e:
            logger.error(f"Main content extraction failed: {str(e)}")
            raise ContentProcessingError(
                f"Main content extraction failed: {str(e)}")

    async def remove_boilerplate(self, html_content: str) -> str:
        """
        Remove boilerplate content from HTML.

        Args:
            html_content: Raw HTML content

        Returns:
            HTML with boilerplate removed
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove common boilerplate elements
            boilerplate_selectors = [
                "script",
                "style",
                "nav",
                "header",
                "footer",
                "aside",
                '[class*="cookie"]',
                '[class*="privacy"]',
                '[class*="terms"]',
                '[class*="disclaimer"]',
                '[class*="copyright"]',
            ]

            for selector in boilerplate_selectors:
                elements = soup.select(selector)
                for element in elements:
                    element.decompose()

            return str(soup)

        except Exception as e:
            logger.error(f"Boilerplate removal failed: {str(e)}")
            raise ContentProcessingError(
                f"Boilerplate removal failed: {str(e)}")

    async def validate_html(self, html_content: str) -> Dict[str, Any]:
        """
        Validate HTML content and return validation results.

        Args:
            html_content: HTML content to validate

        Returns:
            Validation results dictionary
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Check for basic HTML structure
            has_doctype = html_content.strip().startswith("<!DOCTYPE")
            has_html_tag = soup.find("html") is not None
            has_head_tag = soup.find("head") is not None
            has_body_tag = soup.find("body") is not None

            # Count elements
            script_count = len(soup.find_all("script"))
            style_count = len(soup.find_all("style"))
            link_count = len(soup.find_all("link"))
            img_count = len(soup.find_all("img"))

            # Check for potential issues
            issues = []
            if script_count > 10:
                issues.append("High number of script tags")
            if style_count > 5:
                issues.append("High number of style tags")
            if not has_doctype:
                issues.append("Missing DOCTYPE declaration")
            if not has_html_tag:
                issues.append("Missing HTML tag")

            return {
                "valid": True,
                "has_doctype": has_doctype,
                "has_html_tag": has_html_tag,
                "has_head_tag": has_head_tag,
                "has_body_tag": has_body_tag,
                "script_count": script_count,
                "style_count": style_count,
                "link_count": link_count,
                "img_count": img_count,
                "issues": issues,
                "element_count": len(soup.find_all()),
            }

        except Exception as e:
            logger.error(f"HTML validation failed: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "has_doctype": False,
                "has_html_tag": False,
                "has_head_tag": False,
                "has_body_tag": False,
                "script_count": 0,
                "style_count": 0,
                "link_count": 0,
                "img_count": 0,
                "issues": [f"Validation error: {str(e)}"],
                "element_count": 0,
            }
