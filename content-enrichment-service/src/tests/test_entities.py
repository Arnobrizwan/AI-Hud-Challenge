"""Tests for entity extraction functionality."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from ..entities.extractor import EntityExtractor
from ..models.content import ContentType, EntityType, ExtractedContent


class TestEntityExtractor:
    """Test cases for EntityExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create EntityExtractor instance for testing."""
        return EntityExtractor()

    @pytest.fixture
    def sample_content(self):
        """Create sample content for testing."""
        return ExtractedContent(
            title="Apple Inc. Announces New iPhone",
            content="Apple Inc., the technology company led by CEO Tim Cook, announced the new iPhone 15 today. The device features advanced AI capabilities and will be available in stores starting September 15, 2023.",
            summary="Apple announces iPhone 15 with AI features",
            url="https://example.com/apple-iphone-15",
            source="Tech News",
            author="John Smith",
            content_type=ContentType.ARTICLE,
            language="en",
        )

    @pytest.mark.asyncio
    async def test_extract_entities_basic(self, extractor, sample_content):
        """Test basic entity extraction."""
        entities = await extractor.extract_entities(sample_content.content)

        assert isinstance(entities, list)
        assert len(entities) > 0

        # Check for expected entities
        entity_texts = [entity.text for entity in entities]
        assert "Apple Inc." in entity_texts or "Apple" in entity_texts
        assert "Tim Cook" in entity_texts
        assert "iPhone 15" in entity_texts

    @pytest.mark.asyncio
    async def test_extract_entities_with_confidence(self, extractor, sample_content):
        """Test entity extraction with confidence scores."""
        entities = await extractor.extract_entities(sample_content.content)

        for entity in entities:
            assert 0.0 <= entity.confidence <= 1.0
            assert entity.start >= 0
            assert entity.end > entity.start
            assert entity.text is not None
            assert entity.label in EntityType

    @pytest.mark.asyncio
    async def test_extract_custom_entities(self, extractor):
        """Test custom entity extraction."""
        text = "Contact us at support@example.com or call +1-555-123-4567. Visit https://example.com for more info."

        entities = await extractor._extract_custom_entities(text)

        assert len(entities) > 0

        # Check for email
        email_entities = [e for e in entities if e.text == "support@example.com"]
        assert len(email_entities) == 1

        # Check for phone
        phone_entities = [e for e in entities if "+1-555-123-4567" in e.text]
        assert len(phone_entities) == 1

        # Check for URL
        url_entities = [e for e in entities if e.text == "https://example.com"]
        assert len(url_entities) == 1

    @pytest.mark.asyncio
    async def test_entity_resolution(self, extractor):
        """Test entity resolution and deduplication."""
        # Create mock entities with overlap
        entities = [
            Mock(text="Apple Inc.", start=0, end=10, confidence=0.9, label=EntityType.ORGANIZATION),
            Mock(text="Apple", start=0, end=5, confidence=0.8, label=EntityType.ORGANIZATION),
            Mock(text="iPhone", start=20, end=26, confidence=0.9, label=EntityType.PRODUCT),
        ]

        resolved = extractor._resolve_entities(entities)

        # Should remove overlapping entities (keep higher confidence)
        assert len(resolved) <= len(entities)
        assert all(entity.confidence >= 0.8 for entity in resolved)

    @pytest.mark.asyncio
    async def test_extract_entities_from_content(self, extractor, sample_content):
        """Test entity extraction from ExtractedContent object."""
        entities = await extractor.extract_entities_from_content(sample_content)

        assert isinstance(entities, list)
        assert len(entities) > 0

        # Should include entities from both title and content
        entity_texts = [entity.text for entity in entities]
        assert any("Apple" in text for text in entity_texts)

    def test_calculate_confidence(self, extractor):
        """Test confidence calculation."""
        # Mock entity
        entity = Mock()
        entity.text = "Apple Inc."
        entity.label = EntityType.ORGANIZATION
        entity.sent = Mock()
        entity.sent.text = "Apple Inc. is a technology company."

        confidence = extractor._calculate_confidence(entity, EntityType.ORGANIZATION)

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably confident

    def test_map_spacy_label(self, extractor):
        """Test spaCy label mapping."""
        assert extractor._map_spacy_label("PERSON") == EntityType.PERSON
        assert extractor._map_spacy_label("ORG") == EntityType.ORGANIZATION
        assert extractor._map_spacy_label("GPE") == EntityType.LOCATION
        assert extractor._map_spacy_label("UNKNOWN") == EntityType.CUSTOM

    def test_get_entity_statistics(self, extractor):
        """Test entity statistics calculation."""
        entities = [
            Mock(text="Apple", label=EntityType.ORGANIZATION, confidence=0.9, wikidata_id="Q95"),
            Mock(text="Tim Cook", label=EntityType.PERSON, confidence=0.8, wikidata_id=None),
            Mock(text="iPhone", label=EntityType.PRODUCT, confidence=0.7, wikidata_id="Q78"),
        ]

        stats = extractor.get_entity_statistics(entities)

        assert stats["total_entities"] == 3
        assert stats["entity_types"]["ORGANIZATION"] == 1
        assert stats["entity_types"]["PERSON"] == 1
        assert stats["entity_types"]["PRODUCT"] == 1
        assert stats["linked_entities"] == 2
        assert stats["average_confidence"] > 0.7

    @pytest.mark.asyncio
    async def test_extract_entities_empty_text(self, extractor):
        """Test entity extraction with empty text."""
        entities = await extractor.extract_entities("")
        assert entities == []

    @pytest.mark.asyncio
    async def test_extract_entities_short_text(self, extractor):
        """Test entity extraction with very short text."""
        entities = await extractor.extract_entities("Hi")
        assert isinstance(entities, list)  # Should not crash

    @pytest.mark.asyncio
    async def test_extract_entities_error_handling(self, extractor):
        """Test error handling in entity extraction."""
        with patch.object(extractor, "nlp") as mock_nlp:
            mock_nlp.side_effect = Exception("spaCy error")

            entities = await extractor.extract_entities("Test text")
            assert entities == []  # Should return empty list on error
