"""Text processing utilities for content enrichment."""

import html
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class TextProcessor:
    """Text processing utilities for content enrichment."""

    def __init__(self):
        """Initialize the text processor."""
        self.html_parser = None
        self._initialize_parser()

    def _initialize_parser(self):
        """Initialize HTML parser."""
        try:
            from bs4 import BeautifulSoup

            self.html_parser = BeautifulSoup
        except ImportError:
            logger.warning("BeautifulSoup not available, using basic HTML cleaning")
            self.html_parser = None

    def clean_text(self, text: str, remove_html: bool = True) -> str:
        """Clean and normalize text."""
        try:
            # Remove HTML tags if requested
            if remove_html:
                text = self._remove_html_tags(text)

            # Decode HTML entities
            text = html.unescape(text)

            # Normalize unicode
            text = unicodedata.normalize("NFKD", text)

            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text)

            # Remove control characters
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

            return text.strip()

        except Exception as e:
            logger.error("Text cleaning failed", error=str(e))
            return text

    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        if self.html_parser:
            try:
                soup = self.html_parser(text, "html.parser")
                return soup.get_text()
            except Exception:
                pass

        # Fallback to regex
        return re.sub(r"<[^>]+>", "", text)

    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        try:
            # Simple sentence splitting
            sentences = re.split(r"[.!?]+", text)

            # Clean and filter sentences
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Filter out very short sentences
                    cleaned_sentences.append(sentence)

            return cleaned_sentences

        except Exception as e:
            logger.error("Sentence extraction failed", error=str(e))
            return [text]

    def extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text."""
        try:
            # Split by double newlines or paragraph tags
            paragraphs = re.split(r"\n\s*\n|<p[^>]*>", text)

            # Clean paragraphs
            cleaned_paragraphs = []
            for paragraph in paragraphs:
                paragraph = self.clean_text(paragraph)
                if len(paragraph) > 20:  # Filter out very short paragraphs
                    cleaned_paragraphs.append(paragraph)

            return cleaned_paragraphs

        except Exception as e:
            logger.error("Paragraph extraction failed", error=str(e))
            return [text]

    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract keywords from text."""
        try:
            # Clean text
            cleaned_text = self.clean_text(text)

            # Convert to lowercase
            text_lower = cleaned_text.lower()

            # Remove common stop words
            stop_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "must",
                "can",
                "this",
                "that",
                "these",
                "those",
                "i",
                "you",
                "he",
                "she",
                "it",
                "we",
                "they",
                "me",
                "him",
                "her",
                "us",
                "them",
                "my",
                "your",
                "his",
                "her",
                "its",
                "our",
                "their",
            }

            # Extract words
            words = re.findall(r"\b[a-zA-Z]+\b", text_lower)

            # Filter words
            keywords = []
            for word in words:
                if len(word) >= min_length and word not in stop_words and not word.isdigit():
                    keywords.append(word)

            return keywords

        except Exception as e:
            logger.error("Keyword extraction failed", error=str(e))
            return []

    def extract_phrases(self, text: str, min_words: int = 2, max_words: int = 4) -> List[str]:
        """Extract phrases from text."""
        try:
            # Clean text
            cleaned_text = self.clean_text(text)

            # Extract sentences
            sentences = self.extract_sentences(cleaned_text)

            phrases = []
            for sentence in sentences:
                words = sentence.split()

                # Generate n-grams
                for i in range(len(words) - min_words + 1):
                    for n in range(min_words, min(max_words + 1, len(words) - i + 1)):
                        phrase = " ".join(words[i : i + n])
                        if len(phrase) > 10:  # Filter out very short phrases
                            phrases.append(phrase)

            return phrases

        except Exception as e:
            logger.error("Phrase extraction failed", error=str(e))
            return []

    def calculate_text_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate various text metrics."""
        try:
            cleaned_text = self.clean_text(text)

            # Basic metrics
            char_count = len(cleaned_text)
            word_count = len(cleaned_text.split())
            sentence_count = len(self.extract_sentences(cleaned_text))
            paragraph_count = len(self.extract_paragraphs(cleaned_text))

            # Calculate averages
            avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
            avg_sentences_per_paragraph = sentence_count / paragraph_count if paragraph_count > 0 else 0

            # Calculate readability metrics
            readability_score = self._calculate_simple_readability(cleaned_text)

            # Calculate complexity metrics
            unique_words = len(set(cleaned_text.lower().split()))
            lexical_diversity = unique_words / word_count if word_count > 0 else 0

            return {
                "character_count": char_count,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
                "avg_words_per_sentence": avg_words_per_sentence,
                "avg_sentences_per_paragraph": avg_sentences_per_paragraph,
                "readability_score": readability_score,
                "unique_words": unique_words,
                "lexical_diversity": lexical_diversity,
            }

        except Exception as e:
            logger.error("Text metrics calculation failed", error=str(e))
            return {}

    def _calculate_simple_readability(self, text: str) -> float:
        """Calculate a simple readability score."""
        try:
            words = text.split()
            sentences = self.extract_sentences(text)

            if not words or not sentences:
                return 0.5

            # Simple readability based on average sentence length
            avg_sentence_length = len(words) / len(sentences)

            # Convert to 0-1 scale (shorter sentences = higher readability)
            if avg_sentence_length <= 10:
                return 1.0
            elif avg_sentence_length <= 15:
                return 0.8
            elif avg_sentence_length <= 20:
                return 0.6
            elif avg_sentence_length <= 25:
                return 0.4
            else:
                return 0.2

        except Exception as e:
            logger.error("Readability calculation failed", error=str(e))
            return 0.5

    def extract_quotes(self, text: str) -> List[str]:
        """Extract quoted text from content."""
        try:
            # Find text within quotes
            quote_patterns = [
                r'"([^"]*)"',  # Double quotes
                r"'([^']*)'",  # Single quotes
                r"«([^»]*)»",  # French quotes
                r'„([^"]*)"',  # German quotes
            ]

            quotes = []
            for pattern in quote_patterns:
                matches = re.findall(pattern, text)
                quotes.extend(matches)

            # Clean quotes
            cleaned_quotes = []
            for quote in quotes:
                cleaned_quote = self.clean_text(quote)
                if len(cleaned_quote) > 5:  # Filter out very short quotes
                    cleaned_quotes.append(cleaned_quote)

            return cleaned_quotes

        except Exception as e:
            logger.error("Quote extraction failed", error=str(e))
            return []

    def extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Extract numbers and their context from text."""
        try:
            # Pattern for various number formats
            number_patterns = [
                (r"\b\d+\.\d+\b", "decimal"),
                (r"\b\d+\b", "integer"),
                (r"\b\d{1,3}(,\d{3})*\b", "formatted_number"),
                (r"\$\d+(?:\.\d{2})?\b", "currency"),
                (r"\b\d+%\b", "percentage"),
                (r"\b\d{4}\b", "year"),
                (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "date"),
            ]

            numbers = []
            for pattern, number_type in number_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    numbers.append(
                        {
                            "value": match.group(),
                            "type": number_type,
                            "start": match.start(),
                            "end": match.end(),
                            "context": self._get_context(text, match.start(), match.end()),
                        }
                    )

            return numbers

        except Exception as e:
            logger.error("Number extraction failed", error=str(e))
            return []

    def _get_context(self, text: str, start: int, end: int, context_length: int = 50) -> str:
        """Get context around a match."""
        try:
            context_start = max(0, start - context_length)
            context_end = min(len(text), end + context_length)
            return text[context_start:context_end].strip()

        except Exception:
            return ""

    def extract_urls(self, text: str) -> List[Dict[str, Any]]:
        """Extract URLs from text."""
        try:
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls = []

            matches = re.finditer(url_pattern, text)
            for match in matches:
                urls.append(
                    {
                        "url": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "context": self._get_context(text, match.start(), match.end()),
                    }
                )

            return urls

        except Exception as e:
            logger.error("URL extraction failed", error=str(e))
            return []

    def extract_emails(self, text: str) -> List[Dict[str, Any]]:
        """Extract email addresses from text."""
        try:
            email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            emails = []

            matches = re.finditer(email_pattern, text)
            for match in matches:
                emails.append(
                    {
                        "email": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "context": self._get_context(text, match.start(), match.end()),
                    }
                )

            return emails

        except Exception as e:
            logger.error("Email extraction failed", error=str(e))
            return []

    def extract_phone_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Extract phone numbers from text."""
        try:
            phone_patterns = [
                # US format
                r"(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
                r"\+?[1-9]\d{1,14}",  # International format
            ]

            phone_numbers = []
            for pattern in phone_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    phone_numbers.append(
                        {
                            "phone": match.group(),
                            "start": match.start(),
                            "end": match.end(),
                            "context": self._get_context(text, match.start(), match.end()),
                        }
                    )

            return phone_numbers

        except Exception as e:
            logger.error("Phone number extraction failed", error=str(e))
            return []

    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        try:
            hashtag_pattern = r"#\w+"
            hashtags = re.findall(hashtag_pattern, text)
            return hashtags

        except Exception as e:
            logger.error("Hashtag extraction failed", error=str(e))
            return []

    def extract_mentions(self, text: str) -> List[str]:
        """Extract mentions (@username) from text."""
        try:
            mention_pattern = r"@\w+"
            mentions = re.findall(mention_pattern, text)
            return mentions

        except Exception as e:
            logger.error("Mention extraction failed", error=str(e))
            return []
