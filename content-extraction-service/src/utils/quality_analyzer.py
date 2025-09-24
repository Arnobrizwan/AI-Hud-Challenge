"""
Content quality analyzer with comprehensive scoring and validation.
"""

import math
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..exceptions import QualityAnalysisError
from ..models.content import QualityMetrics


class QualityAnalyzer:
    """Advanced content quality analyzer with multiple scoring algorithms."""

    def __init__(self):
        """Initialize quality analyzer."""
        self.spam_keywords = self._load_spam_keywords()
        self.quality_indicators = self._load_quality_indicators()
        self.readability_weights = {
            "sentence_length": 0.3,
            "word_length": 0.2,
            "syllable_count": 0.2,
            "complex_words": 0.3,
        }

    async def analyze_content_quality(
        self,
        content: str,
        url: Optional[str] = None,
        title: Optional[str] = None,
        language_hint: Optional[str] = None,
    ) -> QualityMetrics:
        """
        Analyze content quality and return comprehensive metrics.

        Args:
            content: Content text to analyze
            url: Source URL (optional)
            title: Content title (optional)
            language_hint: Language hint for analysis (optional)

        Returns:
            QualityMetrics object with detailed analysis
        """
        try:
            logger.info(f"Analyzing content quality for {url or 'unknown'}")

            # Basic text statistics
            word_count = len(content.split())
            character_count = len(content)
            sentence_count = self._count_sentences(content)
            paragraph_count = self._count_paragraphs(content)

            # Calculate averages
            average_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            average_word_length = character_count / word_count if word_count > 0 else 0

            # Readability analysis
            readability_score = await self._calculate_readability_score(content)

            # Spam detection
            spam_score = await self._calculate_spam_score(content, url, title)

            # Duplicate content detection
            duplicate_score = await self._calculate_duplicate_score(content, url)

            # Content freshness analysis
            content_freshness = await self._calculate_content_freshness(content, url)

            # Image to text ratio (placeholder - would need image data)
            image_to_text_ratio = 0.0

            # Link density analysis
            link_density = await self._calculate_link_density(content)

            # Overall quality score
            overall_quality = await self._calculate_overall_quality(
                readability_score,
                spam_score,
                duplicate_score,
                content_freshness,
                word_count,
                average_sentence_length,
            )

            return QualityMetrics(
                readability_score=readability_score,
                word_count=word_count,
                character_count=character_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                average_sentence_length=average_sentence_length,
                average_word_length=average_word_length,
                image_to_text_ratio=image_to_text_ratio,
                link_density=link_density,
                spam_score=spam_score,
                duplicate_score=duplicate_score,
                content_freshness=content_freshness,
                overall_quality=overall_quality,
            )

        except Exception as e:
            logger.error(f"Quality analysis failed: {str(e)}")
            raise QualityAnalysisError(f"Quality analysis failed: {str(e)}")

    async def _calculate_readability_score(self, content: str) -> float:
        """Calculate Flesch-Kincaid readability score."""
        try:
            sentences = self._split_into_sentences(content)
            words = content.split()

            if not sentences or not words:
                return 0.0

            # Count syllables
            total_syllables = sum(self._count_syllables(word) for word in words)

            # Flesch-Kincaid formula
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = total_syllables / len(words)

            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

            # Normalize to 0-100 range
            return max(0.0, min(100.0, score))

        except Exception as e:
            logger.warning(f"Readability calculation failed: {str(e)}")
            return 50.0  # Default score

    async def _calculate_spam_score(
        self, content: str, url: Optional[str], title: Optional[str]
    ) -> float:
        """Calculate spam detection score (0-100, higher = more spam)."""
        try:
            spam_indicators = 0
            total_checks = 0

            # Check for spam keywords
            content_lower = content.lower()
            spam_keyword_count = sum(
                1 for keyword in self.spam_keywords if keyword in content_lower
            )
            spam_indicators += min(spam_keyword_count * 10, 50)  # Max 50 points
            total_checks += 1

            # Check for excessive capitalization
            caps_ratio = sum(1 for c in content if c.isupper()) / len(content) if content else 0
            if caps_ratio > 0.3:
                spam_indicators += 20
            total_checks += 1

            # Check for excessive punctuation
            punct_ratio = sum(1 for c in content if c in "!?") / len(content) if content else 0
            if punct_ratio > 0.1:
                spam_indicators += 15
            total_checks += 1

            # Check for repeated phrases
            repeated_phrases = self._find_repeated_phrases(content)
            if repeated_phrases:
                spam_indicators += min(len(repeated_phrases) * 5, 25)
            total_checks += 1

            # Check for suspicious URL patterns
            if url and self._is_suspicious_url(url):
                spam_indicators += 30
            total_checks += 1

            # Check for clickbait title patterns
            if title and self._is_clickbait_title(title):
                spam_indicators += 25
            total_checks += 1

            # Calculate final score
            if total_checks == 0:
                return 0.0

            spam_score = (spam_indicators / total_checks) * 2  # Scale to 0-100
            return min(100.0, spam_score)

        except Exception as e:
            logger.warning(f"Spam score calculation failed: {str(e)}")
            return 0.0

    async def _calculate_duplicate_score(self, content: str, url: Optional[str]) -> float:
        """Calculate duplicate content score (0-100, higher = more duplicate)."""
        try:
            # This is a simplified version - in production, you'd compare against a database
            # of known content or use more sophisticated similarity algorithms

            # Check for common boilerplate text
            boilerplate_phrases = [
                "click here",
                "read more",
                "continue reading",
                "see more",
                "advertisement",
                "sponsored content",
                "this site uses cookies",
            ]

            content_lower = content.lower()
            boilerplate_count = sum(1 for phrase in boilerplate_phrases if phrase in content_lower)

            # Check for very short content (likely incomplete)
            if len(content.split()) < 50:
                return 80.0  # High duplicate score for very short content

            # Check for repetitive content
            words = content.split()
            if len(words) > 0:
                word_freq = Counter(words)
                most_common_freq = word_freq.most_common(1)[0][1]
                repetition_ratio = most_common_freq / len(words)

                if repetition_ratio > 0.1:  # More than 10% repetition
                    return min(100.0, repetition_ratio * 1000)

            # Base score on boilerplate content
            duplicate_score = min(100.0, boilerplate_count * 10)
            return duplicate_score

        except Exception as e:
            logger.warning(f"Duplicate score calculation failed: {str(e)}")
            return 0.0

    async def _calculate_content_freshness(self, content: str, url: Optional[str]) -> float:
        """Calculate content freshness score (0-100, higher = fresher)."""
        try:
            # Look for date indicators in content
            date_patterns = [
                r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
                r"\b(?:yesterday|today|tomorrow|this week|this month|this year)\b",
            ]

            current_year = datetime.now().year
            found_dates = []

            for pattern in date_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                found_dates.extend(matches)

            if not found_dates:
                return 50.0  # Neutral score if no dates found

            # Analyze found dates
            freshness_score = 0.0
            for date_str in found_dates:
                try:
                    # Simple year extraction
                    year_match = re.search(r"\b(20\d{2})\b", date_str)
                    if year_match:
                        year = int(year_match.group(1))
                        if year == current_year:
                            freshness_score += 100
                        elif year == current_year - 1:
                            freshness_score += 80
                        elif year == current_year - 2:
                            freshness_score += 60
                        elif year >= current_year - 5:
                            freshness_score += 40
                        else:
                            freshness_score += 20
                except (ValueError, AttributeError):
                    continue

            # Average the scores
            if found_dates:
                freshness_score = freshness_score / len(found_dates)

            return min(100.0, freshness_score)

        except Exception as e:
            logger.warning(f"Content freshness calculation failed: {str(e)}")
            return 50.0

    async def _calculate_link_density(self, content: str) -> float:
        """Calculate link density in content."""
        try:
            # Count links in content (simple regex approach)
            link_pattern = r"https?://[^\s]+"
            links = re.findall(link_pattern, content)

            word_count = len(content.split())
            if word_count == 0:
                return 0.0

            link_density = len(links) / word_count
            return min(1.0, link_density)

        except Exception as e:
            logger.warning(f"Link density calculation failed: {str(e)}")
            return 0.0

    async def _calculate_overall_quality(
        self,
        readability_score: float,
        spam_score: float,
        duplicate_score: float,
        content_freshness: float,
        word_count: int,
        average_sentence_length: float,
    ) -> float:
        """Calculate overall quality score."""
        try:
            # Weighted combination of different quality factors
            weights = {
                "readability": 0.25,
                "spam": 0.25,
                "duplicate": 0.20,
                "freshness": 0.15,
                "length": 0.15,
            }

            # Normalize scores
            readability_norm = readability_score / 100.0
            spam_norm = 1.0 - (spam_score / 100.0)  # Invert spam score
            duplicate_norm = 1.0 - (duplicate_score / 100.0)  # Invert duplicate score
            freshness_norm = content_freshness / 100.0

            # Length score (prefer content between 200-2000 words)
            if word_count < 100:
                length_norm = word_count / 100.0
            elif word_count <= 2000:
                length_norm = 1.0
            else:
                length_norm = max(0.5, 1.0 - (word_count - 2000) / 5000.0)

            # Calculate weighted average
            overall_score = (
                weights["readability"] * readability_norm
                + weights["spam"] * spam_norm
                + weights["duplicate"] * duplicate_norm
                + weights["freshness"] * freshness_norm
                + weights["length"] * length_norm
            )

            # Convert back to 0-100 scale
            return min(100.0, max(0.0, overall_score * 100.0))

        except Exception as e:
            logger.warning(f"Overall quality calculation failed: {str(e)}")
            return 50.0

    def _count_sentences(self, content: str) -> int:
        """Count sentences in content."""
        if not content:
            return 0

        # Simple sentence counting based on punctuation
        sentence_endings = re.findall(r"[.!?]+", content)
        return len(sentence_endings)

    def _count_paragraphs(self, content: str) -> int:
        """Count paragraphs in content."""
        if not content:
            return 0

        # Split by double newlines or single newlines with proper formatting
        paragraphs = re.split(r"\n\s*\n", content)
        return len([p for p in paragraphs if p.strip()])

    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences."""
        if not content:
            return []

        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", content)
        return [s.strip() for s in sentences if s.strip()]

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
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

    def _find_repeated_phrases(self, content: str, min_length: int = 3) -> List[str]:
        """Find repeated phrases in content."""
        words = content.lower().split()
        if len(words) < min_length * 2:
            return []

        phrase_counts = {}
        for i in range(len(words) - min_length + 1):
            phrase = " ".join(words[i : i + min_length])
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        return [phrase for phrase, count in phrase_counts.items() if count > 1]

    def _is_suspicious_url(self, url: str) -> bool:
        """Check if URL has suspicious patterns."""
        suspicious_patterns = [
            r"bit\.ly",
            r"tinyurl\.com",
            r"short\.link",
            r"click\w+\.com",
            r"redirect\w+\.com",
            r"\d+\.\d+\.\d+\.\d+",  # IP addresses
            r"[a-z0-9-]+\.tk$",
            r"[a-z0-9-]+\.ml$",  # Suspicious TLDs
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True

        return False

    def _is_clickbait_title(self, title: str) -> bool:
        """Check if title has clickbait patterns."""
        clickbait_patterns = [
            r"you won\'t believe",
            r"shocking",
            r"amazing",
            r"incredible",
            r"this will blow your mind",
            r"number \d+ will shock you",
            r"what happens next",
            r"doctors hate this",
            r"one weird trick",
            r"click here",
            r"read more",
            r"see what happens",
        ]

        title_lower = title.lower()
        for pattern in clickbait_patterns:
            if re.search(pattern, title_lower):
                return True

        return False

    def _load_spam_keywords(self) -> List[str]:
        """Load spam keywords for detection."""
        return [
            "click here",
            "buy now",
            "free money",
            "make money fast",
            "work from home",
            "no experience",
            "guaranteed",
            "limited time",
            "act now",
            "don't miss out",
            "exclusive offer",
            "special deal",
            "spam",
            "advertisement",
            "promo",
            "discount",
            "sale",
            "viagra",
            "casino",
            "lottery",
            "winner",
            "congratulations",
        ]

    def _load_quality_indicators(self) -> List[str]:
        """Load quality indicators for content analysis."""
        return [
            "research",
            "study",
            "analysis",
            "report",
            "findings",
            "evidence",
            "data",
            "statistics",
            "survey",
            "interview",
            "expert",
            "professional",
            "authority",
            "credible",
            "reliable",
        ]

    async def get_quality_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Get recommendations for improving content quality."""
        recommendations = []

        if metrics.readability_score < 30:
            recommendations.append(
                "Content is difficult to read. Consider shorter sentences and simpler words."
            )
        elif metrics.readability_score > 80:
            recommendations.append("Content may be too simple. Consider adding more complex ideas.")

        if metrics.spam_score > 30:
            recommendations.append(
                "Content contains spam indicators. Review for promotional language."
            )

        if metrics.duplicate_score > 50:
            recommendations.append("Content appears to have duplicate or boilerplate text.")

        if metrics.word_count < 200:
            recommendations.append("Content is too short. Consider adding more detail and context.")
        elif metrics.word_count > 3000:
            recommendations.append(
                "Content is very long. Consider breaking into sections or summaries."
            )

        if metrics.average_sentence_length > 25:
            recommendations.append(
                "Sentences are too long. Consider breaking them into shorter ones."
            )

        if metrics.link_density > 0.1:
            recommendations.append(
                "High link density may affect readability. Consider reducing links."
            )

        return recommendations
