"""Tests for sentiment analysis functionality."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from ..models.content import ContentType, EmotionLabel, ExtractedContent, SentimentLabel
from ..sentiment.analyzer import SentimentAnalyzer


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create SentimentAnalyzer instance for testing."""
        return SentimentAnalyzer()

    @pytest.fixture
    def positive_content(self):
        """Create positive content for testing."""
        return ExtractedContent(
            title="Amazing New Technology",
            content="This is absolutely fantastic! I love this new technology. It's incredible how it works and makes everything so much better. I'm so excited about the future!",
            content_type=ContentType.ARTICLE,
            language="en",
        )

    @pytest.fixture
    def negative_content(self):
        """Create negative content for testing."""
        return ExtractedContent(
            title="Terrible Experience",
            content="This is awful and disappointing. I hate this product. It's terrible and doesn't work at all. I'm so frustrated and angry about this waste of money.",
            content_type=ContentType.ARTICLE,
            language="en",
        )

    @pytest.fixture
    def neutral_content(self):
        """Create neutral content for testing."""
        return ExtractedContent(
            title="Product Information",
            content="This product is available in three colors: red, blue, and green. It weighs 2.5 pounds and measures 10x5x3 inches. The price is $99.99.",
            content_type=ContentType.ARTICLE,
            language="en",
        )

    @pytest.mark.asyncio
    async def test_analyze_sentiment_positive(self, analyzer, positive_content) -> Dict[str, Any]:
        """Test sentiment analysis for positive content."""
        sentiment = await analyzer.analyze_sentiment(positive_content)

        assert sentiment.sentiment in [SentimentLabel.POSITIVE, SentimentLabel.NEUTRAL]
        assert sentiment.confidence > 0.0
        assert sentiment.subjectivity > 0.0  # Should be somewhat subjective
        assert sentiment.polarity > 0.0  # Should be positive

    @pytest.mark.asyncio
    async def test_analyze_sentiment_negative(self, analyzer, negative_content) -> Dict[str, Any]:
        """Test sentiment analysis for negative content."""
        sentiment = await analyzer.analyze_sentiment(negative_content)

        assert sentiment.sentiment in [SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL]
        assert sentiment.confidence > 0.0
        assert sentiment.subjectivity > 0.0  # Should be somewhat subjective
        assert sentiment.polarity < 0.0  # Should be negative

    @pytest.mark.asyncio
    async def test_analyze_sentiment_neutral(self, analyzer, neutral_content) -> Dict[str, Any]:
        """Test sentiment analysis for neutral content."""
        sentiment = await analyzer.analyze_sentiment(neutral_content)

        assert sentiment.sentiment in [
            SentimentLabel.NEUTRAL,
            SentimentLabel.POSITIVE,
            SentimentLabel.NEGATIVE,
        ]
        assert sentiment.confidence > 0.0
        assert sentiment.subjectivity < 0.5  # Should be more objective
        assert -0.2 < sentiment.polarity < 0.2  # Should be close to neutral

    @pytest.mark.asyncio
    async def test_analyze_sentiment_with_emotions(self, analyzer, positive_content) -> Dict[str, Any]:
        """Test sentiment analysis with emotion detection."""
        sentiment = await analyzer.analyze_sentiment(positive_content)

        assert isinstance(sentiment.emotions, dict)
        assert len(sentiment.emotions) > 0

        # Check that emotions are valid
        for emotion, intensity in sentiment.emotions.items():
            assert emotion in EmotionLabel
            assert 0.0 <= intensity <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_sentiment_batch(
        self, analyzer, positive_content, negative_content, neutral_content
    ) -> Dict[str, Any]:
        """Test batch sentiment analysis."""
        contents = [positive_content, negative_content, neutral_content]
        sentiments = await analyzer.analyze_sentiment_batch(contents)

        assert len(sentiments) == len(contents)
        assert all(isinstance(s, type(sentiments[0])) for s in sentiments)
        assert all(s.confidence > 0.0 for s in sentiments)

    @pytest.mark.asyncio
    async def test_analyze_sentiment_error_handling(self, analyzer) -> Dict[str, Any]:
        """Test error handling in sentiment analysis."""
        # Test with empty content
        empty_content = ExtractedContent(title="", content="", content_type=ContentType.ARTICLE, language="en")

        sentiment = await analyzer.analyze_sentiment(empty_content)
        assert sentiment.sentiment == SentimentLabel.NEUTRAL
        assert sentiment.confidence == 0.5

    @pytest.mark.asyncio
    async def test_analyze_sentiment_fallback(self, analyzer, positive_content) -> Dict[str, Any]:
        """Test fallback sentiment analysis when models fail."""
        with patch.object(analyzer, "model_loaded", False):
            sentiment = await analyzer.analyze_sentiment(positive_content)

            assert sentiment.sentiment in SentimentLabel
            assert sentiment.confidence > 0.0
            assert isinstance(sentiment.emotions, dict)

    def test_get_sentiment_statistics(self, analyzer):
        """Test sentiment statistics calculation."""
        # Create mock sentiment analyses
        sentiments = [
            Mock(
                sentiment=SentimentLabel.POSITIVE,
                confidence=0.8,
                subjectivity=0.6,
                polarity=0.7,
                emotions={EmotionLabel.JOY: 0.8},
            ),
            Mock(
                sentiment=SentimentLabel.NEGATIVE,
                confidence=0.9,
                subjectivity=0.8,
                polarity=-0.6,
                emotions={EmotionLabel.ANGER: 0.7},
            ),
            Mock(
                sentiment=SentimentLabel.NEUTRAL,
                confidence=0.7,
                subjectivity=0.3,
                polarity=0.1,
                emotions={EmotionLabel.NEUTRAL: 1.0},
            ),
        ]

        stats = analyzer.get_sentiment_statistics(sentiments)

        assert stats["total_analyses"] == 3
        assert stats["sentiment_distribution"]["positive"] == 1
        assert stats["sentiment_distribution"]["negative"] == 1
        assert stats["sentiment_distribution"]["neutral"] == 1
        assert stats["average_confidence"] > 0.7
        assert stats["average_subjectivity"] > 0.0
        assert "emotion_distribution" in stats

    @pytest.mark.asyncio
    async def test_detect_sentiment_shift(self, analyzer) -> Dict[str, Any]:
        """Test sentiment shift detection."""
        old_analysis = Mock(sentiment=SentimentLabel.NEGATIVE, confidence=0.8, polarity=-0.7, subjectivity=0.6)

        new_analysis = Mock(sentiment=SentimentLabel.POSITIVE, confidence=0.9, polarity=0.8, subjectivity=0.7)

        shift = await analyzer.detect_sentiment_shift(old_analysis, new_analysis)

        assert shift["old_sentiment"] == "negative"
        assert shift["new_sentiment"] == "positive"
        assert shift["polarity_change"] > 0
        assert shift["confidence_change"] > 0
        assert "significant_changes" in shift
        assert shift["has_significant_change"]

    @pytest.mark.asyncio
    async def test_analyze_sentiment_multilingual(self, analyzer) -> Dict[str, Any]:
        """Test sentiment analysis with different languages."""
        # Test with Spanish content
        spanish_content = ExtractedContent(
            title="Tecnología Increíble",
            content="Esta tecnología es absolutamente fantástica. Me encanta cómo funciona.",
            content_type=ContentType.ARTICLE,
            language="es",
        )

        sentiment = await analyzer.analyze_sentiment(spanish_content, language="es")

        assert sentiment.sentiment in SentimentLabel
        assert sentiment.confidence > 0.0
        assert isinstance(sentiment.emotions, dict)

    @pytest.mark.asyncio
    async def test_analyze_sentiment_with_model_failure(self, analyzer, positive_content):
        """Test sentiment analysis when transformer models fail."""
        with patch.object(analyzer, "sentiment_pipeline") as mock_pipeline:
            mock_pipeline.side_effect = Exception("Model error")

            sentiment = await analyzer.analyze_sentiment(positive_content)

            # Should fall back to TextBlob/VADER
            assert sentiment.sentiment in SentimentLabel
            assert sentiment.confidence > 0.0

    def test_emotion_keywords_fallback(self, analyzer):
        """Test emotion detection with keyword fallback."""
        text = "I am so happy and joyful about this amazing news!"

        emotions = asyncio.run(analyzer._analyze_emotions_with_fallback(text))

        assert isinstance(emotions, dict)
        assert len(emotions) > 0
        assert all(isinstance(emotion, EmotionLabel) for emotion in emotions.keys())
        assert all(0.0 <= intensity <= 1.0 for intensity in emotions.values())

    def test_sentiment_keywords_fallback(self, analyzer):
        """Test sentiment analysis with keyword fallback."""
        text = "This is absolutely terrible and I hate it!"

        result = asyncio.run(analyzer._analyze_with_fallback(text))

        assert result["sentiment"] in SentimentLabel
        assert result["confidence"] > 0.0
        assert 0.0 <= result["subjectivity"] <= 1.0
        assert -1.0 <= result["polarity"] <= 1.0
