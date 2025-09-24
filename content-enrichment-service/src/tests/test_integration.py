"""Integration tests for the content enrichment service."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ..enrichment.pipeline import ContentEnrichmentPipeline
from ..models.content import ContentType, ExtractedContent, ProcessingMode


class TestContentEnrichmentPipeline:
    """Integration tests for the enrichment pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create ContentEnrichmentPipeline instance for testing."""
        return ContentEnrichmentPipeline()

    @pytest.fixture
    def sample_content(self):
        """Create sample content for testing."""
        return ExtractedContent(
            title="Apple Inc. Announces Revolutionary AI Technology",
            content="Apple Inc., the Cupertino-based technology giant led by CEO Tim Cook, announced today the launch of their revolutionary artificial intelligence platform. The new AI system, developed by a team of world-class engineers and researchers, promises to transform how we interact with technology. The announcement was made at the company's annual Worldwide Developers Conference in San Jose, California. The AI platform leverages advanced machine learning algorithms and natural language processing to provide unprecedented capabilities. Industry experts predict this could be a game-changer for the tech industry, with potential applications in healthcare, education, and business automation. The technology is expected to be available to developers starting next year, with consumer products following shortly after.",
            summary="Apple announces new AI platform with advanced ML capabilities",
            url="https://example.com/apple-ai-announcement",
            source="Tech News",
            author="Sarah Johnson",
            content_type=ContentType.ARTICLE,
            language="en",
        )

    @pytest.mark.asyncio
    async def test_enrich_content_full_pipeline(self, pipeline, sample_content):
        """Test full content enrichment pipeline."""
        enriched = await pipeline.enrich_content(
            content=sample_content,
            processing_mode=ProcessingMode.REALTIME,
            include_entities=True,
            include_topics=True,
            include_sentiment=True,
            include_signals=True,
            include_trust_score=True,
        )

        # Verify enriched content structure
        assert enriched.id == sample_content.id
        assert enriched.original_content == sample_content
        assert enriched.processing_mode == ProcessingMode.REALTIME
        assert enriched.language_detected == "en"
        assert enriched.processing_time_ms > 0

        # Verify entities
        assert isinstance(enriched.entities, list)
        assert len(enriched.entities) > 0

        # Check for expected entities
        entity_texts = [entity.text for entity in enriched.entities]
        assert any("Apple" in text for text in entity_texts)
        assert any("Tim Cook" in text for text in entity_texts)
        assert any("artificial intelligence" in text.lower() for text in entity_texts)

        # Verify topics
        assert isinstance(enriched.topics, list)
        assert len(enriched.topics) > 0

        # Verify sentiment
        assert enriched.sentiment is not None
        assert enriched.sentiment.confidence > 0.0
        assert enriched.sentiment.subjectivity >= 0.0
        assert enriched.sentiment.polarity >= -1.0

        # Verify signals
        assert enriched.signals is not None
        assert enriched.signals.readability_score > 0.0
        assert enriched.signals.authority_score > 0.0

        # Verify trust score
        assert enriched.trust_score is not None
        assert enriched.trust_score.overall_score > 0.0

        # Verify model versions
        assert isinstance(enriched.model_versions, dict)
        assert len(enriched.model_versions) > 0

    @pytest.mark.asyncio
    async def test_enrich_content_selective_components(self, pipeline, sample_content):
        """Test enrichment with selective components."""
        enriched = await pipeline.enrich_content(
            content=sample_content,
            include_entities=True,
            include_topics=False,
            include_sentiment=True,
            include_signals=False,
            include_trust_score=False,
        )

        # Should have entities and sentiment
        assert len(enriched.entities) > 0
        assert enriched.sentiment is not None

        # Should not have topics, signals, or trust score
        assert len(enriched.topics) == 0
        assert enriched.signals is None
        assert enriched.trust_score is None

    @pytest.mark.asyncio
    async def test_enrich_content_batch_processing(self, pipeline):
        """Test batch content enrichment."""
        contents = [
            ExtractedContent(
                title="Tech News 1",
                content="First article about technology and innovation.",
                content_type=ContentType.ARTICLE,
                language="en",
            ),
            ExtractedContent(
                title="Tech News 2",
                content="Second article about artificial intelligence and machine learning.",
                content_type=ContentType.ARTICLE,
                language="en",
            ),
            ExtractedContent(
                title="Tech News 3",
                content="Third article about data science and analytics.",
                content_type=ContentType.ARTICLE,
                language="en",
            ),
        ]

        enriched_contents = await pipeline.enrich_batch(
            contents=contents, processing_mode=ProcessingMode.BATCH
        )

        assert len(enriched_contents) == len(contents)
        assert all(
            enriched.id == content.id for enriched, content in zip(enriched_contents, contents)
        )
        assert all(
            enriched.processing_mode == ProcessingMode.BATCH for enriched in enriched_contents
        )

    @pytest.mark.asyncio
    async def test_enrich_content_error_handling(self, pipeline):
        """Test error handling in content enrichment."""
        # Test with invalid content
        invalid_content = ExtractedContent(
            title="", content="", content_type=ContentType.ARTICLE, language="en"  # Empty content
        )

        enriched = await pipeline.enrich_content(invalid_content)

        # Should still return enriched content with fallback values
        assert enriched is not None
        assert enriched.entities == []
        assert enriched.sentiment is not None
        assert enriched.signals is not None
        assert enriched.trust_score is not None

    @pytest.mark.asyncio
    async def test_enrich_content_different_languages(self, pipeline):
        """Test enrichment with different languages."""
        spanish_content = ExtractedContent(
            title="Noticias de Tecnología",
            content="Apple Inc. anunció hoy su nueva tecnología de inteligencia artificial. La empresa con sede en Cupertino, dirigida por el CEO Tim Cook, presentó una plataforma revolucionaria que promete transformar la industria tecnológica.",
            content_type=ContentType.ARTICLE,
            language="es",
        )

        enriched = await pipeline.enrich_content(spanish_content, language_hint="es")

        assert enriched.language_detected == "es"
        assert len(enriched.entities) > 0
        assert enriched.sentiment is not None

    @pytest.mark.asyncio
    async def test_enrich_content_processing_time(self, pipeline, sample_content):
        """Test that processing time is recorded correctly."""
        enriched = await pipeline.enrich_content(sample_content)

        assert enriched.processing_time_ms > 0
        assert enriched.processing_time_ms < 30000  # Should be less than 30 seconds

    @pytest.mark.asyncio
    async def test_enrich_content_model_versions(self, pipeline, sample_content):
        """Test that model versions are recorded."""
        enriched = await pipeline.enrich_content(sample_content)

        assert isinstance(enriched.model_versions, dict)
        assert "entity_extractor" in enriched.model_versions
        assert "topic_classifier" in enriched.model_versions
        assert "sentiment_analyzer" in enriched.model_versions
        assert "signal_extractor" in enriched.model_versions

        # Check model version structure
        for model_name, version in enriched.model_versions.items():
            assert hasattr(version, "name")
            assert hasattr(version, "version")
            assert hasattr(version, "created_at")
            assert hasattr(version, "performance_metrics")

    @pytest.mark.asyncio
    async def test_enrich_content_parallel_processing(self, pipeline, sample_content):
        """Test that components are processed in parallel."""
        import time

        start_time = time.time()
        enriched = await pipeline.enrich_content(sample_content)
        end_time = time.time()

        # Should complete in reasonable time (parallel processing)
        processing_time = end_time - start_time
        assert processing_time < 10.0  # Should be much faster than sequential

    @pytest.mark.asyncio
    async def test_enrich_content_with_mock_components(self, pipeline, sample_content):
        """Test enrichment with mocked components to verify integration."""
        # Mock the individual components
        with (
            patch.object(pipeline.entity_extractor, "extract_entities") as mock_entities,
            patch.object(pipeline.topic_classifier, "classify_topics") as mock_topics,
            patch.object(pipeline.sentiment_analyzer, "analyze_sentiment") as mock_sentiment,
            patch.object(pipeline.signal_extractor, "extract_signals") as mock_signals,
            patch.object(pipeline.signal_extractor, "compute_trustworthiness") as mock_trust,
        ):

            # Set up mock returns
            mock_entities.return_value = []
            mock_topics.return_value = []
            mock_sentiment.return_value = Mock(
                sentiment="positive", confidence=0.8, subjectivity=0.6, polarity=0.7, emotions={}
            )
            mock_signals.return_value = Mock(readability_score=0.8, authority_score=0.7)
            mock_trust.return_value = Mock(overall_score=0.8)

            enriched = await pipeline.enrich_content(sample_content)

            # Verify mocks were called
            mock_entities.assert_called_once()
            mock_topics.assert_called_once()
            mock_sentiment.assert_called_once()
            mock_signals.assert_called_once()
            mock_trust.assert_called_once()

            # Verify enriched content
            assert enriched is not None
            assert enriched.entities == []
            assert enriched.topics == []
            assert enriched.sentiment is not None
            assert enriched.signals is not None
            assert enriched.trust_score is not None

    @pytest.mark.asyncio
    async def test_enrich_content_large_content(self, pipeline):
        """Test enrichment with large content."""
        large_content = ExtractedContent(
            title="Large Article",
            content="This is a very long article. " * 1000,  # Large content
            content_type=ContentType.ARTICLE,
            language="en",
        )

        enriched = await pipeline.enrich_content(large_content)

        assert enriched is not None
        assert enriched.processing_time_ms > 0
        # Should handle large content without errors

    @pytest.mark.asyncio
    async def test_enrich_content_special_characters(self, pipeline):
        """Test enrichment with special characters and unicode."""
        special_content = ExtractedContent(
            title="Special Characters & Unicode: 中文, العربية, हिन्दी",
            content="This article contains special characters: émojis 🚀, symbols ©®, and unicode text in multiple languages: 中文, العربية, हिन्दी, and more!",
            content_type=ContentType.ARTICLE,
            language="en",
        )

        enriched = await pipeline.enrich_content(special_content)

        assert enriched is not None
        assert enriched.entities is not None
        assert enriched.sentiment is not None
        # Should handle special characters without errors
