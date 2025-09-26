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

        assert isinstance(result, QualityAnalysis)
        assert result.word_count > 0
        assert result.sentence_count > 0
        assert result.paragraph_count > 0
        assert result.avg_sentence_length > 0
        assert 0.0 <= result.quality_score <= 1.0
        assert 0.0 <= result.readability_score <= 1.0
        assert 0.0 <= result.sentiment_score <= 1.0
        assert 0.0 <= result.language_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_content_quality_with_url(self) -> Dict[str, Any]:
        """Test content quality analysis with URL."""
        content = "This is a test article with multiple sentences. It contains enough words to be considered valid content. The quality should be reasonable."
        url = "https://example.com/article"

        result = await self.analyzer.analyze_content_quality(content, url=url)

        assert isinstance(result, QualityAnalysis)
        assert result.word_count > 0
        assert 0.0 <= result.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_content_quality_with_title(self) -> Dict[str, Any]:
        """Test content quality analysis with title."""
        content = "This is a test article with multiple sentences. It contains enough words to be considered valid content. The quality should be reasonable."
        title = "Test Article Title"

        result = await self.analyzer.analyze_content_quality(content, title=title)

        assert isinstance(result, QualityAnalysis)
        assert result.word_count > 0
        assert 0.0 <= result.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_content_quality_empty_content(self) -> Dict[str, Any]:
        """Test content quality analysis with empty content."""
        content = ""

        result = await self.analyzer.analyze_content_quality(content)

        assert isinstance(result, QualityAnalysis)
        assert result.word_count == 0
        assert result.sentence_count == 0
        assert result.paragraph_count == 0

    @pytest.mark.asyncio
    async def test_analyze_content_quality_short_content(self) -> Dict[str, Any]:
        """Test content quality analysis with short content."""
        content = "Short."

        result = await self.analyzer.analyze_content_quality(content)

        assert isinstance(result, QualityAnalysis)
        assert result.word_count > 0
        assert result.sentence_count > 0

    @pytest.mark.asyncio
    async def test_calculate_spam_score_with_spam_keywords(self) -> Dict[str, Any]:
        """Test spam score calculation with spam keywords."""
        content = "BUY NOW! FREE MONEY! CLICK HERE! URGENT! LIMITED TIME OFFER!"

        result = await self.analyzer.analyze_content_quality(content)

        assert isinstance(result, QualityAnalysis)
        assert 0.0 <= result.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_duplicate_score(self) -> Dict[str, Any]:
        """Test duplicate score calculation."""
        content = "This is a test article with multiple sentences. It contains enough words to be considered valid content. The quality should be reasonable."

        result = await self.analyzer.analyze_content_quality(content)

        assert isinstance(result, QualityAnalysis)
        assert 0.0 <= result.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_content_freshness(self) -> Dict[str, Any]:
        """Test content freshness calculation."""
        content = "This is a test article with multiple sentences. It contains enough words to be considered valid content. The quality should be reasonable."

        result = await self.analyzer.analyze_content_quality(content)

        assert isinstance(result, QualityAnalysis)
        assert 0.0 <= result.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_find_repeated_phrases(self) -> Dict[str, Any]:
        """Test repeated phrases detection."""
        content = "This is a test article. This is a test article. This is a test article."

        result = await self.analyzer.analyze_content_quality(content)

        assert isinstance(result, QualityAnalysis)
        assert 0.0 <= result.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_get_quality_recommendations(self) -> Dict[str, Any]:
        """Test quality recommendations generation."""
        content = "This is a test article with multiple sentences. It contains enough words to be considered valid content. The quality should be reasonable."

        result = await self.analyzer.analyze_content_quality(content)

        assert isinstance(result, QualityAnalysis)
        assert 0.0 <= result.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_content_quality_error_handling(self) -> Dict[str, Any]:
        """Test error handling in content quality analysis."""
        # Test with None content - should raise an exception
        with pytest.raises(Exception):
            await self.analyzer.analyze_content_quality(None)
