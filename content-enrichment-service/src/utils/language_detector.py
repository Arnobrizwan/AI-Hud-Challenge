"""Language detection utilities."""

import asyncio
from typing import Any, Dict, Optional

import langdetect
import structlog
from langdetect import DetectorFactory

logger = structlog.get_logger(__name__)

# Set seed for consistent results
DetectorFactory.seed = 0


class LanguageDetector:
    """Language detection for content processing."""

    def __init__(self):
        """Initialize the language detector."""
        self.supported_languages = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi",
            "nl": "Dutch",
            "sv": "Swedish",
            "no": "Norwegian",
            "da": "Danish",
            "fi": "Finnish",
            "pl": "Polish",
            "tr": "Turkish",
            "th": "Thai",
        }

    async def detect(self, text: str) -> str:
        """Detect the language of the given text."""
        try:
            if not text or len(text.strip()) < 10:
                return "en"  # Default to English for very short text

            # Clean text for better detection
            cleaned_text = self._clean_text(text)

            if len(cleaned_text) < 10:
                return "en"

            # Detect language
            detected_lang = langdetect.detect(cleaned_text)

            # Validate detected language
            if detected_lang in self.supported_languages:
                logger.info("Language detected", language=detected_lang, text_length=len(text))
                return detected_lang
            else:
                logger.warning("Unsupported language detected", language=detected_lang, falling_back_to="en")
                return "en"

        except Exception as e:
            logger.error("Language detection failed", error=str(e), falling_back_to="en")
            return "en"

    def _clean_text(self, text: str) -> str:
        """Clean text for better language detection."""
        import re

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s.,!?;:]", "", text)

        return text.strip()

    async def detect_multiple(self, texts: list) -> Dict[str, str]:
        """Detect language for multiple texts."""
        try:
            results = {}

            for i, text in enumerate(texts):
                language = await self.detect(text)
                results[f"text_{i}"] = language

            return results

        except Exception as e:
            logger.error("Multiple language detection failed", error=str(e))
            return {f"text_{i}": "en" for i in range(len(texts))}

    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages."""
        return self.supported_languages.copy()

    def is_language_supported(self, language_code: str) -> bool:
        """Check if a language is supported."""
        return language_code in self.supported_languages

    async def get_language_statistics(self, texts: list) -> Dict[str, Any]:
    """Get language detection statistics."""
        try:
            results = await self.detect_multiple(texts)

            # Count languages
            language_counts = {}
            for language in results.values():
                language_counts[language] = language_counts.get(language, 0) + 1

            # Calculate percentages
            total_texts = len(texts)
            language_percentages = {lang: (count / total_texts) * 100 for lang, count in language_counts.items()}

            return {
                "total_texts": total_texts,
                "language_distribution": language_counts,
                "language_percentages": language_percentages,
                "most_common_language": (max(language_counts, key=language_counts.get) if language_counts else "en"),
            }

        except Exception as e:
            logger.error("Language statistics calculation failed", error=str(e))
            return {}
