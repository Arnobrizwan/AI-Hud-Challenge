"""
Quality analysis and scoring for feedback and content
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
import structlog
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = structlog.get_logger(__name__)


class QualityAnalyzer:
    """Analyze quality of feedback and content"""

    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.bias_detector = BiasDetector()
        self.fact_checker = FactChecker()
        self.readability_analyzer = ReadabilityAnalyzer()
        self.spam_detector = SpamDetector()

    async def score_feedback(self, feedback: Any) -> float:
        """Calculate overall quality score for feedback"""

        try:
            scores = []

            # Text quality score
            if hasattr(feedback, "comment") and feedback.comment:
                text_score = await self.analyze_text_quality(feedback.comment)
                scores.append(text_score)

            # Rating quality score
            if hasattr(feedback, "rating") and feedback.rating is not None:
                rating_score = self.analyze_rating_quality(feedback.rating)
                scores.append(rating_score)

            # Metadata quality score
            if hasattr(feedback, "metadata") and feedback.metadata:
                metadata_score = self.analyze_metadata_quality(
                    feedback.metadata)
                scores.append(metadata_score)

            # Calculate weighted average
            if scores:
                return sum(scores) / len(scores)
            else:
                return 0.5  # Default neutral score

        except Exception as e:
            logger.error("Error scoring feedback", error=str(e))
            return 0.5

    async def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""

        try:
            if not text or not text.strip():
                return 0.0

            # Use VADER sentiment analyzer
            scores = self.sentiment_analyzer.polarity_scores(text)

            # Return compound score (-1 to 1)
            return scores["compound"]

        except Exception as e:
            logger.error("Error analyzing sentiment", error=str(e))
            return 0.0

    async def analyze_text_quality(self, text: str) -> float:
        """Analyze quality of text content"""

        try:
            if not text or not text.strip():
                return 0.0

            scores = []

            # Length score (prefer moderate length)
            length_score = self._calculate_length_score(text)
            scores.append(length_score)

            # Readability score
            readability_score = await self.readability_analyzer.analyze(text)
            scores.append(readability_score)

            # Language quality score
            language_score = self._analyze_language_quality(text)
            scores.append(language_score)

            # Spam score (inverted)
            spam_score = await self.spam_detector.detect(text)
            scores.append(1.0 - spam_score)

            return sum(scores) / len(scores)

        except Exception as e:
            logger.error("Error analyzing text quality", error=str(e))
            return 0.5

    def analyze_rating_quality(self, rating: float) -> float:
        """Analyze quality of rating"""

        try:
            # Check if rating is within valid range
            if rating < 0 or rating > 5:
                return 0.0

            # Prefer ratings that are not just the middle value
            if rating == 3.0:
                return 0.6  # Slightly lower for neutral ratings

            # Higher scores for more extreme ratings (more decisive)
            distance_from_center = abs(rating - 2.5)
            return min(1.0, distance_from_center / 2.5)

        except Exception as e:
            logger.error("Error analyzing rating quality", error=str(e))
            return 0.5

    def analyze_metadata_quality(self, metadata: Dict[str, Any]) -> float:
        """Analyze quality of metadata"""

        try:
            if not metadata:
                return 0.5

            scores = []

            # Check for required fields
            required_fields = ["timestamp", "source"]
            for field in required_fields:
                if field in metadata and metadata[field]:
                    scores.append(1.0)
                else:
                    scores.append(0.0)

            # Check for additional useful fields
            useful_fields = ["user_agent", "ip_address", "session_id"]
            for field in useful_fields:
                if field in metadata and metadata[field]:
                    scores.append(0.5)

            return sum(scores) / len(scores) if scores else 0.5

        except Exception as e:
            logger.error("Error analyzing metadata quality", error=str(e))
            return 0.5

    def _calculate_length_score(self, text: str) -> float:
        """Calculate score based on text length"""

        word_count = len(text.split())

        # Prefer text between 10-100 words
        if word_count < 5:
            return 0.2  # Too short
        elif word_count < 10:
            return 0.6  # Short but acceptable
        elif word_count <= 100:
            return 1.0  # Ideal length
        elif word_count <= 200:
            return 0.8  # Long but acceptable
        else:
            return 0.4  # Too long

    def _analyze_language_quality(self, text: str) -> float:
        """Analyze language quality indicators"""

        try:
            scores = []

            # Check for proper capitalization
            if text[0].isupper() and any(c.isupper() for c in text[1:]):
                scores.append(1.0)
            else:
                scores.append(0.5)

            # Check for proper punctuation
            if text.endswith((".", "!", "?")):
                scores.append(1.0)
            else:
                scores.append(0.7)

            # Check for excessive repetition
            words = text.lower().split()
            if len(set(words)) / len(words) > 0.7:  # Good diversity
                scores.append(1.0)
            else:
                scores.append(0.6)

            # Check for excessive caps
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio < 0.3:
                scores.append(1.0)
            else:
                scores.append(0.3)

            return sum(scores) / len(scores)

        except Exception as e:
            logger.error("Error analyzing language quality", error=str(e))
            return 0.5


class BiasDetector:
    """Detect bias in content"""

    def __init__(self):
        self.bias_keywords = {
            "gender": ["he", "she", "man", "woman", "male", "female"],
            "race": ["black", "white", "asian", "hispanic", "latino"],
            "age": ["old", "young", "elderly", "teenager", "millennial"],
            "political": ["liberal", "conservative", "democrat", "republican"],
            "religious": ["christian", "muslim", "jewish", "hindu", "buddhist"],
        }

    async def detect_bias(self, text: str) -> Dict[str, float]:
        """Detect various types of bias in text"""

        try:
            text_lower = text.lower()
            bias_scores = {}

            for bias_type, keywords in self.bias_keywords.items():
                keyword_count = sum(
                    1 for keyword in keywords if keyword in text_lower)
                bias_scores[bias_type] = min(
                    1.0, keyword_count / 5.0)  # Normalize

            return bias_scores

        except Exception as e:
            logger.error("Error detecting bias", error=str(e))
            return {}


class FactChecker:
    """Basic fact checking capabilities"""

    def __init__(self):
        self.fact_patterns = [
            r"\d{4}",  # Years
            r"\$[\d,]+",  # Money amounts
            r"\d+%",  # Percentages
        ]

    async def check_facts(self, text: str) -> Dict[str, Any]:
        """Check factual claims in text"""
        try:
            facts_found = []

            for pattern in self.fact_patterns:
                matches = re.findall(pattern, text)
                facts_found.extend(matches)

            return {
                "facts_found": len(facts_found),
                "factual_claims": facts_found,
                "needs_verification": len(facts_found) > 0,
            }

        except Exception as e:
            logger.error("Error checking facts", error=str(e))
            return {
                "facts_found": 0,
                "factual_claims": [],
                "needs_verification": False}


class ReadabilityAnalyzer:
    """Analyze text readability"""

    async def analyze(self, text: str) -> float:
        """Analyze readability of text"""

        try:
            if not text or not text.strip():
                return 0.0

            # Simple readability metrics
            sentences = text.split(".")
            words = text.split()

            if not sentences or not words:
                return 0.0

            # Average words per sentence
            avg_words_per_sentence = len(words) / len(sentences)

            # Average characters per word
            avg_chars_per_word = sum(len(word) for word in words) / len(words)

            # Simple readability score (higher is more readable)
            readability_score = 1.0 - \
                min(1.0, (avg_words_per_sentence - 10) / 20)
            readability_score *= 1.0 - min(1.0, (avg_chars_per_word - 4) / 6)

            return max(0.0, min(1.0, readability_score))

        except Exception as e:
            logger.error("Error analyzing readability", error=str(e))
            return 0.5


class SpamDetector:
    """Detect spam and low-quality content"""

    def __init__(self):
        self.spam_indicators = [
            r"click here",
            r"buy now",
            r"free money",
            r"act now",
            r"limited time",
            r"guaranteed",
            r"no risk",
            r"work from home",
            r"make money",
            r"get rich",
        ]

    async def detect(self, text: str) -> float:
        """Detect spam likelihood in text"""

        try:
            if not text or not text.strip():
                return 0.0

            text_lower = text.lower()
            spam_score = 0.0

            # Check for spam indicators
            for pattern in self.spam_indicators:
                if re.search(pattern, text_lower):
                    spam_score += 0.2

            # Check for excessive repetition
            words = text_lower.split()
            if len(words) > 0:
                unique_words = len(set(words))
                repetition_ratio = 1.0 - (unique_words / len(words))
                spam_score += repetition_ratio * 0.5

            # Check for excessive caps
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.5:
                spam_score += 0.3

            return min(1.0, spam_score)

        except Exception as e:
            logger.error("Error detecting spam", error=str(e))
            return 0.0


class QualityController:
    """Control quality of annotations and submissions"""

    def __init__(self):
        self.quality_analyzer = QualityAnalyzer()

    async def score_annotation(self, annotation_data: Dict[str, Any]) -> float:
        """Score quality of annotation"""

        try:
            scores = []

            # Check completeness
            if annotation_data:
                scores.append(1.0)
            else:
                scores.append(0.0)

            # Check for required fields based on annotation type
            if "sentiment" in annotation_data:
                sentiment = annotation_data["sentiment"]
                if sentiment in ["positive", "negative", "neutral"]:
                    scores.append(1.0)
                else:
                    scores.append(0.0)

            if "quality_score" in annotation_data:
                score = annotation_data["quality_score"]
                if isinstance(score, (int, float)) and 1 <= score <= 5:
                    scores.append(1.0)
                else:
                    scores.append(0.0)

            return sum(scores) / len(scores) if scores else 0.5

        except Exception as e:
            logger.error("Error scoring annotation", error=str(e))
            return 0.5
