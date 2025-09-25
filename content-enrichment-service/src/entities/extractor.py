"""Advanced named entity recognition with linking."""

import asyncio
import re
from typing import Any, Dict, List, Optional, Set

import spacy
import structlog
from spacy import displacy

from ..knowledge_base.entity_kb import EntityKnowledgeBase
from ..models.content import Entity, EntityType, ExtractedContent
from ..utils.text_processor import TextProcessor

logger = structlog.get_logger(__name__)


class EntityExtractor:
    """Advanced named entity recognition with linking."""

    def __init__(self):
        """Initialize the entity extractor."""
        self.nlp = spacy.load("en_core_web_lg")
        self.entity_kb = EntityKnowledgeBase()
        self.text_processor = TextProcessor()

        # Custom entity patterns
        self.custom_patterns = self._load_custom_patterns()

        # Entity confidence thresholds
        self.confidence_thresholds = {
            EntityType.PERSON: 0.7,
            EntityType.ORGANIZATION: 0.6,
            EntityType.LOCATION: 0.6,
            EntityType.DATE: 0.8,
            EntityType.MONEY: 0.9,
            EntityType.PERCENT: 0.9,
            EntityType.CARDINAL: 0.8,
            EntityType.ORDINAL: 0.8,
            EntityType.QUANTITY: 0.7,
            EntityType.EVENT: 0.5,
            EntityType.FACILITY: 0.5,
            EntityType.LANGUAGE: 0.8,
            EntityType.LAW: 0.6,
            EntityType.NATIONALITY: 0.7,
            EntityType.PRODUCT: 0.5,
            EntityType.WORK_OF_ART: 0.6,
            EntityType.CUSTOM: 0.5,
        }

    def _load_custom_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load custom entity patterns."""
        return {
            "email": [
                {
                    "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                    "label": "EMAIL",
                }
            ],
            "phone": [
                {
                    "pattern": r"(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
                    "label": "PHONE",
                }
            ],
            "url": [{"pattern": r'https?://[^\s<>"{}|\\^`\[\]]+', "label": "URL"}],
            "hashtag": [{"pattern": r"#\w+", "label": "HASHTAG"}],
            "mention": [{"pattern": r"@\w+", "label": "MENTION"}],
            "cryptocurrency": [{"pattern": r"\b(BTC|ETH|LTC|XRP|ADA|DOT|LINK|UNI|AAVE|COMP)\b", "label": "CRYPTO"}],
            "stock_ticker": [
                {
                    "pattern": r"\b[A-Z]{1,5}\b",
                    "label": "STOCK",
                }  # Basic pattern, would need validation
            ],
        }

    async def extract_entities(self, text: str, language: str = "en") -> List[Entity]:
        """Extract entities from text with linking and disambiguation."""
        try:
            # Stage 1: spaCy NER for base entities
            doc = self.nlp(text)
            entities = []

            # Stage 2: Custom entity patterns
            custom_entities = await self._extract_custom_entities(text)

            # Stage 3: Process spaCy entities
            for ent in doc.ents:
                entity_type = self._map_spacy_label(ent.label_)
                confidence = self._calculate_confidence(ent, entity_type)

                if confidence >= self.confidence_thresholds.get(entity_type, 0.5):
                    entity = Entity(
                        text=ent.text,
                        label=entity_type,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=confidence,
                    )

                    # Link to knowledge base
                    kb_entity = await self.entity_kb.link_entity(entity)
                    if kb_entity:
                        entity.wikidata_id = kb_entity.get("id")
                        entity.canonical_name = kb_entity.get("name")
                        entity.description = kb_entity.get("description")
                        entity.aliases = kb_entity.get("aliases", [])
                        entity.categories = kb_entity.get("categories", [])
                        entity.properties = kb_entity.get("properties", {})

                    entities.append(entity)

            # Stage 4: Add custom entities
            entities.extend(custom_entities)

            # Stage 5: Entity resolution and deduplication
            resolved_entities = self._resolve_entities(entities)

            # Stage 6: Sort by confidence and position
            resolved_entities.sort(key=lambda x: (-x.confidence, x.start))

            logger.info(
                "Entity extraction completed",
                text_length=len(text),
                entities_found=len(resolved_entities),
                language=language,
            )

            return resolved_entities

        except Exception as e:
            logger.error("Entity extraction failed", error=str(e), exc_info=True)
            return []

    async def _extract_custom_entities(self, text: str) -> List[Entity]:
        """Extract custom entities using regex patterns."""
        entities = []

        for pattern_type, patterns in self.custom_patterns.items():
            for pattern_info in patterns:
                matches = re.finditer(pattern_info["pattern"], text, re.IGNORECASE)

                for match in matches:
                    entity = Entity(
                        text=match.group(),
                        label=EntityType.CUSTOM,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,  # High confidence for regex matches
                        properties={"pattern_type": pattern_type},
                    )
                    entities.append(entity)

        return entities

    def _map_spacy_label(self, spacy_label: str) -> EntityType:
        """Map spaCy labels to our entity types."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "MONEY": EntityType.MONEY,
            "PERCENT": EntityType.PERCENT,
            "CARDINAL": EntityType.CARDINAL,
            "ORDINAL": EntityType.ORDINAL,
            "QUANTITY": EntityType.QUANTITY,
            "EVENT": EntityType.EVENT,
            "FAC": EntityType.FACILITY,
            "LANGUAGE": EntityType.LANGUAGE,
            "LAW": EntityType.LAW,
            "NORP": EntityType.NATIONALITY,
            "PRODUCT": EntityType.PRODUCT,
            "WORK_OF_ART": EntityType.WORK_OF_ART,
        }
        return mapping.get(spacy_label, EntityType.CUSTOM)

    def _calculate_confidence(self, ent: Any, entity_type: EntityType) -> float:
        """Calculate confidence score for an entity."""
        base_confidence = 0.8  # Default confidence

        # Adjust based on entity length
        # Longer entities are more confident
        length_factor = min(len(ent.text) / 20, 1.0)

        # Adjust based on entity type
        type_factor = self.confidence_thresholds.get(entity_type, 0.5)

        # Adjust based on context (simplified)
        context_factor = 1.0
        if hasattr(ent, "sent") and ent.sent:
            # Check if entity appears in proper context
            sent_text = ent.sent.text.lower()
            if entity_type == EntityType.PERSON and any(title in sent_text for title in ["mr", "ms", "dr", "prof"]):
                context_factor = 1.2
            elif entity_type == EntityType.ORGANIZATION and any(
                word in sent_text for word in ["inc", "corp", "ltd", "llc"]
            ):
                context_factor = 1.1

        confidence = base_confidence * length_factor * type_factor * context_factor
        return min(confidence, 1.0)

    def _resolve_entities(self, entities: List[Entity]) -> List[Entity]:
        """Resolve overlapping and duplicate entities."""
        if not entities:
            return []

        # Sort by start position
        entities.sort(key=lambda x: x.start)

        resolved = []
        current_entity = entities[0]

        for next_entity in entities[1:]:
            # Check for overlap
            if next_entity.start < current_entity.end:
                # Overlapping entities - keep the one with higher confidence
                if next_entity.confidence > current_entity.confidence:
                    current_entity = next_entity
            else:
                # No overlap - add current entity and move to next
                resolved.append(current_entity)
                current_entity = next_entity

        # Add the last entity
        resolved.append(current_entity)

        # Remove exact duplicates
        seen = set()
        unique_entities = []
        for entity in resolved:
            key = (entity.text.lower(), entity.start, entity.end)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    async def extract_entities_from_content(self, content: ExtractedContent) -> List[Entity]:
        """Extract entities from ExtractedContent object."""
        # Combine title and content for better entity extraction
        full_text = f"{content.title}\n\n{content.content}"
        if content.summary:
            full_text = f"{content.title}\n\n{content.summary}\n\n{content.content}"

        return await self.extract_entities(full_text, content.language)

    def get_entity_statistics(self, entities: List[Entity]) -> Dict[str, Any]:
        """Get statistics about extracted entities."""
        if not entities:
            return {}

        stats = {
            "total_entities": len(entities),
            "entity_types": {},
            "confidence_distribution": {
                "high": 0,  # > 0.8
                "medium": 0,  # 0.5 - 0.8
                "low": 0,  # < 0.5
            },
            "linked_entities": 0,
            "average_confidence": 0.0,
        }

        total_confidence = 0.0

        for entity in entities:
            # Count by type
            entity_type = entity.label.value
            stats["entity_types"][entity_type] = stats["entity_types"].get(entity_type, 0) + 1

            # Confidence distribution
            if entity.confidence > 0.8:
                stats["confidence_distribution"]["high"] += 1
            elif entity.confidence >= 0.5:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["low"] += 1

            # Linked entities
            if entity.wikidata_id:
                stats["linked_entities"] += 1

            total_confidence += entity.confidence

        stats["average_confidence"] = total_confidence / len(entities)

        return stats
