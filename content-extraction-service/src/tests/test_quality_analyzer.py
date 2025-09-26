"""
Unit tests for quality analyzer.
"""

from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from models.content import QualityAnalysis
from utils.quality_analyzer import QualityAnalyzer


class TestQualityAnalyzer:
    """Test QualityAnalyzer class."""

    def setup_method(self):
        """Setup test method."""
        self.analyzer = QualityAnalyzer()

    @pytest.mark.asyncio
    async def test_analyze_content_quality_basic(self) -> Dict[str, Any]:
        """Test basic content quality analysis."""
        content = "This is a test article with multiple sentences. It contains enough words to be considered valid content. The quality should be reasonable."

        result = await self.analyzer.analyze_content_quality(content)

        assert isinstance(result, QualityMetrics)
        assert result.word_count > 0
        assert result.character_count > 0
        assert result.sentence_count > 0
        assert result.paragraph_count > 0
        assert 0 <= result.readability_score <= 100
        assert 0 <= result.spam_score <= 100
        assert 0 <= result.duplicate_score <= 100
        assert 0 <= result.overall_quality <= 100

    @pytest.mark.asyncio
    async def test_analyze_content_quality_with_url(self) -> Dict[str, Any]:
        """Test content quality analysis with URL."""
        content = "This is a test article about technology and innovation."
        url = "https://example.com/article"

        result = await self.analyzer.analyze_content_quality(content, url=url)

        assert isinstance(result, QualityMetrics)
        assert result.word_count > 0

    @pytest.mark.asyncio
    async def test_analyze_content_quality_with_title(self) -> Dict[str, Any]:
        """Test content quality analysis with title."""
        content = "This is a test article about technology and innovation."
        title = "Technology Innovation Article"

        result = await self.analyzer.analyze_content_quality(content, title=title)

        assert isinstance(result, QualityMetrics)
        assert result.word_count > 0

    @pytest.mark.asyncio
    async def test_analyze_content_quality_empty_content(self) -> Dict[str, Any]:
        """Test content quality analysis with empty content."""
        content = ""

        result = await self.analyzer.analyze_content_quality(content)

        assert isinstance(result, QualityMetrics)
        assert result.word_count == 0
        assert result.character_count == 0
        assert result.sentence_count == 0

    @pytest.mark.asyncio
    async def test_analyze_content_quality_short_content(self) -> Dict[str, Any]:
        """Test content quality analysis with short content."""
        content = "Short content."

        result = await self.analyzer.analyze_content_quality(content)

        assert isinstance(result, QualityMetrics)
        assert result.word_count == 2
        assert result.character_count > 0

    @pytest.mark.asyncio
    async def test_calculate_readability_score(self) -> Dict[str, Any]:
        """Test readability score calculation."""
        # Simple content should have high readability
        simple_content = "The cat sat on the mat. The dog ran in the yard. The bird flew in the sky."
        score = await self.analyzer._calculate_readability_score(simple_content)

        assert 0 <= score <= 100
        assert score > 50  # Simple content should be readable

    @pytest.mark.asyncio
    async def test_calculate_spam_score(self) -> Dict[str, Any]:
        """Test spam score calculation."""
        # Normal content should have low spam score
        normal_content = "This is a normal article about technology and innovation."
        score = await self.analyzer._calculate_spam_score(normal_content, None, None)

        assert 0 <= score <= 100
        assert score < 50  # Normal content should have low spam score

    @pytest.mark.asyncio
    async def test_calculate_spam_score_with_spam_keywords(self) -> Dict[str, Any]:
        """Test spam score calculation with spam keywords."""
        # Content with spam keywords should have high spam score
        spam_content = "Click here now! Buy now! Free money! Make money fast! Work from home!"
        score = await self.analyzer._calculate_spam_score(spam_content, None, None)

        assert 0 <= score <= 100
        assert score > 50  # Spam content should have high spam score

    @pytest.mark.asyncio
    async def test_calculate_duplicate_score(self) -> Dict[str, Any]:
        """Test duplicate score calculation."""
        # Normal content should have low duplicate score
        normal_content = "This is a unique article about technology and innovation."
        score = await self.analyzer._calculate_duplicate_score(normal_content, None)

        assert 0 <= score <= 100
        assert score < 50  # Normal content should have low duplicate score

    @pytest.mark.asyncio
    async def test_calculate_content_freshness(self) -> Dict[str, Any]:
        """Test content freshness calculation."""
        # Content with current year should have high freshness
        fresh_content = "This article was published in 2024 and discusses current technology trends."
        score = await self.analyzer._calculate_content_freshness(fresh_content, None)

        assert 0 <= score <= 100
        assert score > 50  # Fresh content should have high freshness score

    @pytest.mark.asyncio
    async def test_calculate_link_density(self) -> Dict[str, Any]:
        """Test link density calculation."""
        # Content with links should have higher link density
        content_with_links = "This article has links to https://example.com and https://test.com websites."
        score = await self.analyzer._calculate_link_density(content_with_links)

        assert 0 <= score <= 1
        assert score > 0  # Content with links should have positive link density

    @pytest.mark.asyncio
    async def test_calculate_overall_quality(self) -> Dict[str, Any]:
        """Test overall quality calculation."""
        # High quality metrics should result in high overall quality
        score = await self.analyzer._calculate_overall_quality(
            readability_score=80.0,
            spam_score=10.0,  # Low spam is good
            duplicate_score=5.0,  # Low duplicate is good
            content_freshness=90.0,
            word_count=500,
            average_sentence_length=15.0,
        )

        assert 0 <= score <= 100
        assert score > 70  # High quality metrics should result in high overall quality

    def test_count_sentences(self):
        """Test sentence counting."""
        content = "This is sentence one. This is sentence two! This is sentence three?"
        count = self.analyzer._count_sentences(content)

        assert count == 3

    def test_count_paragraphs(self):
        """Test paragraph counting."""
        content = "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three."
        count = self.analyzer._count_paragraphs(content)

        assert count == 3

    def test_split_into_sentences(self):
        """Test sentence splitting."""
        content = "This is sentence one. This is sentence two! This is sentence three?"
        sentences = self.analyzer._split_into_sentences(content)

        assert len(sentences) == 3
        assert "This is sentence one" in sentences[0]
        assert "This is sentence two" in sentences[1]
        assert "This is sentence three" in sentences[2]

    def test_count_syllables(self):
        """Test syllable counting."""
        # Simple words
        assert self.analyzer._count_syllables("cat") == 1
        assert self.analyzer._count_syllables("dog") == 1
        assert self.analyzer._count_syllables("hello") == 2
        assert self.analyzer._count_syllables("beautiful") == 3

        # Empty word
        assert self.analyzer._count_syllables("") == 0

    def test_find_repeated_phrases(self):
        """Test repeated phrase detection."""
        content = "This is a test. This is a test. This is another test."
        phrases = self.analyzer._find_repeated_phrases(content, min_length=3)

        assert len(phrases) > 0
        assert "This is a" in phrases

    def test_is_suspicious_url(self):
        """Test suspicious URL detection."""
        # Normal URLs should not be suspicious
        assert self.analyzer._is_suspicious_url("https://example.com/article") is False
        assert self.analyzer._is_suspicious_url("https://news.bbc.co.uk/story") is False

        # Suspicious URLs should be detected
        assert self.analyzer._is_suspicious_url("https://bit.ly/abc123") is True
        assert self.analyzer._is_suspicious_url("https://tinyurl.com/xyz") is True

    def test_is_clickbait_title(self):
        """Test clickbait title detection."""
        # Normal titles should not be clickbait
        assert self.analyzer._is_clickbait_title("Technology Innovation in 2024") is False
        assert self.analyzer._is_clickbait_title("How to Build Better Software") is False

        # Clickbait titles should be detected
        assert self.analyzer._is_clickbait_title("You Won't Believe What Happens Next!") is True
        assert self.analyzer._is_clickbait_title("This Will Blow Your Mind") is True

    @pytest.mark.asyncio
    async def test_get_quality_recommendations(self) -> Dict[str, Any]:
        """Test quality recommendations generation."""
        # Low readability score should generate recommendations
        low_readability_metrics = QualityMetrics(
            readability_score=20.0,  # Low readability
            word_count=500,
            character_count=2500,
            sentence_count=25,
            paragraph_count=5,
            average_sentence_length=20.0,
            average_word_length=5.0,
            image_to_text_ratio=0.1,
            link_density=0.05,
            spam_score=10.0,
            duplicate_score=5.0,
            content_freshness=80.0,
            overall_quality=70.0,
        )

        recommendations = await self.analyzer.get_quality_recommendations(low_readability_metrics)

        assert len(recommendations) > 0
        assert any("readability" in rec.lower() for rec in recommendations)

    @pytest.mark.asyncio
    async def test_analyze_content_quality_error_handling(self) -> Dict[str, Any]:
        """Test error handling in content quality analysis."""
        # Test with None content
        result = await self.analyzer.analyze_content_quality(None)
        assert isinstance(result, QualityMetrics)
        assert result.word_count == 0

    def test_load_spam_keywords(self):
        """Test spam keywords loading."""
        keywords = self.analyzer._load_spam_keywords()

        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "click here" in keywords
        assert "buy now" in keywords

    def test_load_quality_indicators(self):
        """Test quality indicators loading."""
        indicators = self.analyzer._load_quality_indicators()

        assert isinstance(indicators, list)
        assert len(indicators) > 0
        assert "research" in indicators
        assert "analysis" in indicators
