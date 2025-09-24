"""
Readability-based content extraction using python-readability.
"""

import re
from typing import Any, Dict, Optional

from bs4 import BeautifulSoup
from loguru import logger
from readability import Document

from ..exceptions import ContentProcessingError


class ReadabilityExtractor:
    """Content extraction using readability algorithms."""

    def __init__(
        self, min_text_length: int = 250, min_image_width: int = 100, min_image_height: int = 100
    ):
        """Initialize readability extractor."""
        self.min_text_length = min_text_length
        self.min_image_width = min_image_width
        self.min_image_height = min_image_height

    async def extract_main_content(self, html_content: str, url: Optional[str] = None) -> str:
        """
        Extract main content using readability algorithm.

        Args:
            html_content: Raw HTML content
            url: Source URL (optional)

        Returns:
            Extracted main content text
        """
        try:
            logger.info(f"Extracting main content using readability for {url or 'unknown'}")

            # Create readability document
            doc = Document(html_content)

            # Extract main content
            main_content = doc.summary()

            # Parse and clean the extracted content
            soup = BeautifulSoup(main_content, "html.parser")

            # Remove unwanted elements
            await self._clean_extracted_content(soup)

            # Get clean text
            clean_text = soup.get_text(separator=" ", strip=True)

            # Validate content length
            if len(clean_text) < self.min_text_length:
                logger.warning(
                    f"Extracted content too short ({len(clean_text)} chars), trying fallback"
                )
                return await self._fallback_extraction(html_content)

            return clean_text

        except Exception as e:
            logger.error(f"Readability extraction failed: {str(e)}")
            # Try fallback extraction
            return await self._fallback_extraction(html_content)

    async def _clean_extracted_content(self, soup: BeautifulSoup) -> None:
        """Clean extracted content by removing unwanted elements."""
        try:
            # Remove script and style elements
            for element in soup(["script", "style", "noscript"]):
                element.decompose()

            # Remove comments
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                comment.extract()

            # Remove empty elements
            for element in soup.find_all():
                if not element.get_text(strip=True) and not element.find(["img", "br", "hr"]):
                    element.decompose()

            # Clean up whitespace
            for element in soup.find_all():
                if element.string:
                    text = element.string
                    text = re.sub(r"\s+", " ", text)
                    text = text.strip()
                    element.string.replace_with(text)

        except Exception as e:
            logger.warning(f"Content cleaning failed: {str(e)}")

    async def _fallback_extraction(self, html_content: str) -> str:
        """Fallback content extraction when readability fails."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
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
                    text = elements[0].get_text(separator=" ", strip=True)
                    if len(text) > self.min_text_length:
                        return text

            # Fallback to body content
            body = soup.find("body")
            if body:
                return body.get_text(separator=" ", strip=True)

            return soup.get_text(separator=" ", strip=True)

        except Exception as e:
            logger.error(f"Fallback extraction failed: {str(e)}")
            return ""

    async def extract_with_metadata(
        self, html_content: str, url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract content with additional metadata.

        Args:
            html_content: Raw HTML content
            url: Source URL (optional)

        Returns:
            Dictionary with content and metadata
        """
        try:
            doc = Document(html_content)

            # Extract content
            main_content = doc.summary()
            title = doc.title()

            # Parse content
            soup = BeautifulSoup(main_content, "html.parser")
            await self._clean_extracted_content(soup)

            # Extract images
            images = []
            for img in soup.find_all("img"):
                src = img.get("src", "")
                alt = img.get("alt", "")
                if src:
                    images.append(
                        {
                            "src": src,
                            "alt": alt,
                            "width": img.get("width"),
                            "height": img.get("height"),
                        }
                    )

            # Extract links
            links = []
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                text = link.get_text(strip=True)
                if href and text:
                    links.append({"href": href, "text": text})

            return {
                "content": soup.get_text(separator=" ", strip=True),
                "title": title,
                "images": images,
                "links": links,
                "word_count": len(soup.get_text().split()),
                "char_count": len(soup.get_text()),
            }

        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
            return {
                "content": "",
                "title": "",
                "images": [],
                "links": [],
                "word_count": 0,
                "char_count": 0,
            }

    async def score_content_quality(self, content: str) -> Dict[str, float]:
        """
        Score content quality based on readability metrics.

        Args:
            content: Content text to score

        Returns:
            Dictionary with quality scores
        """
        try:
            # Basic metrics
            word_count = len(content.split())
            char_count = len(content)
            sentence_count = len(re.findall(r"[.!?]+", content))

            # Calculate averages
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            avg_word_length = char_count / word_count if word_count > 0 else 0

            # Readability score (simplified Flesch-Kincaid)
            readability_score = 0
            if sentence_count > 0 and word_count > 0:
                # Count syllables (simplified)
                syllables = sum(self._count_syllables(word) for word in content.split())
                avg_syllables = syllables / word_count
                readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
                readability_score = max(0, min(100, readability_score))

            # Content length score
            length_score = min(100, (word_count / 500) * 100)  # Optimal around 500 words

            # Structure score
            structure_score = 0
            if word_count > 0:
                # Check for proper paragraph structure
                paragraphs = content.split("\n\n")
                if len(paragraphs) > 1:
                    structure_score += 30

                # Check for headings
                if any(
                    word in content.lower() for word in ["introduction", "conclusion", "summary"]
                ):
                    structure_score += 20

                # Check for lists
                if any(char in content for char in ["•", "-", "*", "1.", "2.", "3."]):
                    structure_score += 20

                structure_score = min(100, structure_score)

            return {
                "readability_score": readability_score,
                "length_score": length_score,
                "structure_score": structure_score,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": avg_sentence_length,
                "avg_word_length": avg_word_length,
            }

        except Exception as e:
            logger.error(f"Quality scoring failed: {str(e)}")
            return {
                "readability_score": 0,
                "length_score": 0,
                "structure_score": 0,
                "word_count": 0,
                "sentence_count": 0,
                "avg_sentence_length": 0,
                "avg_word_length": 0,
            }

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        if not word:
            return 0

        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel

        # Handle silent 'e'
        if word.endswith("e") and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    async def extract_structured_content(
        self, html_content: str, url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured content with headings, paragraphs, and lists.

        Args:
            html_content: Raw HTML content
            url: Source URL (optional)

        Returns:
            Dictionary with structured content
        """
        try:
            doc = Document(html_content)
            main_content = doc.summary()

            soup = BeautifulSoup(main_content, "html.parser")
            await self._clean_extracted_content(soup)

            # Extract headings
            headings = []
            for i in range(1, 7):
                for heading in soup.find_all(f"h{i}"):
                    text = heading.get_text(strip=True)
                    if text:
                        headings.append({"level": i, "text": text, "id": heading.get("id", "")})

            # Extract paragraphs
            paragraphs = []
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if text and len(text) > 20:  # Only substantial paragraphs
                    paragraphs.append(text)

            # Extract lists
            lists = []
            for ul in soup.find_all(["ul", "ol"]):
                items = []
                for li in ul.find_all("li"):
                    text = li.get_text(strip=True)
                    if text:
                        items.append(text)
                if items:
                    lists.append({"type": ul.name, "items": items})

            # Extract quotes
            quotes = []
            for blockquote in soup.find_all("blockquote"):
                text = blockquote.get_text(strip=True)
                if text:
                    quotes.append(text)

            return {
                "headings": headings,
                "paragraphs": paragraphs,
                "lists": lists,
                "quotes": quotes,
                "full_text": soup.get_text(separator=" ", strip=True),
            }

        except Exception as e:
            logger.error(f"Structured content extraction failed: {str(e)}")
            return {"headings": [], "paragraphs": [], "lists": [], "quotes": [], "full_text": ""}
