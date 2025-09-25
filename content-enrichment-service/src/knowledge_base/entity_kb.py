"""Entity knowledge base for linking and disambiguation."""

import asyncio
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import structlog
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from ..config import settings

logger = structlog.get_logger(__name__)


class EntityKnowledgeBase:
    """Entity knowledge base with linking and disambiguation capabilities."""

    def __init__(self):
        """Initialize the knowledge base."""
        self.engine = create_async_engine(settings.database_url)
        self.session_factory = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

        # Initialize embedding model for semantic similarity
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Cache for frequently accessed entities
        self.entity_cache = {}
        self.cache_ttl = 3600  # 1 hour

        # Wikidata API configuration
        self.wikidata_base_url = "https://www.wikidata.org/w/api.php"
        self.wikidata_params = {
            "format": "json",
            "action": "wbsearchentities",
            "language": "en",
            "type": "item",
        }

    async def link_entity(self, entity: "Entity") -> Optional[Dict[str, Any]]:
        """Link an entity to knowledge base entries."""
        try:
            # Check cache first
            cache_key = f"{entity.text}_{entity.label.value}"
            if cache_key in self.entity_cache:
                cached_result = self.entity_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    return cached_result["data"]

            # Search in local database first
            local_result = await self._search_local_entities(entity)
            if local_result:
                self._cache_result(cache_key, local_result)
                return local_result

            # Search in Wikidata
            wikidata_result = await self._search_wikidata(entity)
            if wikidata_result:
                # Store in local database for future use
                await self._store_entity(entity, wikidata_result)
                self._cache_result(cache_key, wikidata_result)
                return wikidata_result

            return None

        except Exception as e:
            logger.error(
                "Entity linking failed",
                entity_text=entity.text,
                entity_type=entity.label.value,
                error=str(e),
            )
            return None

    async def _search_local_entities(self, entity: "Entity") -> Optional[Dict[str, Any]]:
        """Search for entities in local database."""
        try:
            async with self.session_factory() as session:
                # Search by exact text match first
                query = text(
                    """
                    SELECT id, name, description, aliases, categories, properties,
                           embedding, confidence_score, created_at
                    FROM entities
                    WHERE LOWER(name) = LOWER(:text)
                       OR LOWER(aliases::text) LIKE LOWER(:text_pattern)
                    ORDER BY confidence_score DESC
                    LIMIT 1
                """
                )

                result = await session.execute(query, {"text": entity.text, "text_pattern": f"%{entity.text}%"})

                row = result.fetchone()
                if row:
                    return {
                        "id": row.id,
                        "name": row.name,
                        "description": row.description,
                        "aliases": row.aliases or [],
                        "categories": row.categories or [],
                        "properties": row.properties or {},
                        "confidence": row.confidence_score,
                    }

                # If no exact match, try semantic similarity
                entity_embedding = self.embedding_model.encode([entity.text])

                similarity_query = text(
                    """
                    SELECT id, name, description, aliases, categories, properties,
                           confidence_score, 1 - (embedding <=> :embedding) as similarity
                    FROM entities
                    WHERE 1 - (embedding <=> :embedding) > 0.7
                    ORDER BY similarity DESC
                    LIMIT 1
                """
                )

                result = await session.execute(similarity_query, {"embedding": entity_embedding[0].tolist()})

                row = result.fetchone()
                if row and row.similarity > 0.7:
                    return {
                        "id": row.id,
                        "name": row.name,
                        "description": row.description,
                        "aliases": row.aliases or [],
                        "categories": row.categories or [],
                        "properties": row.properties or {},
                        "confidence": row.confidence_score * row.similarity,
                    }

                return None

        except Exception as e:
            logger.error("Local entity search failed", error=str(e))
            return None

    async def _search_wikidata(self, entity: "Entity") -> Optional[Dict[str, Any]]:
        """Search for entities in Wikidata."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                params = self.wikidata_params.copy()
                params["search"] = entity.text
                params["limit"] = 5

                response = await client.get(self.wikidata_base_url, params=params)
                response.raise_for_status()

                data = response.json()

                if "search" in data and data["search"]:
                    # Find best match based on entity type and description
                    best_match = self._find_best_wikidata_match(data["search"], entity)

                    if best_match:
                        # Get detailed information
                        detailed_info = await self._get_wikidata_details(best_match["id"], client)

                        return {
                            "id": best_match["id"],
                            "name": best_match["label"],
                            "description": best_match.get("description", ""),
                            "aliases": detailed_info.get("aliases", []),
                            "categories": detailed_info.get("categories", []),
                            "properties": detailed_info.get("properties", {}),
                            "confidence": 0.8,  # High confidence for Wikidata matches
                        }

                return None

        except Exception as e:
            logger.error("Wikidata search failed", error=str(e))
            return None

    def _find_best_wikidata_match(self, search_results: List[Dict], entity: "Entity") -> Optional[Dict[str, Any]]:
        """Find the best Wikidata match for an entity."""
        if not search_results:
            return None

        # Score matches based on label similarity and description relevance
        scored_matches = []

        for result in search_results:
            score = 0.0

            # Label similarity
            label_similarity = self._calculate_text_similarity(entity.text, result["label"])
            score += label_similarity * 0.6

            # Description relevance
            if "description" in result:
                desc_similarity = self._calculate_text_similarity(entity.text, result["description"])
                score += desc_similarity * 0.4

            # Entity type matching (simplified)
            if self._matches_entity_type(result, entity.label):
                score += 0.2

            scored_matches.append((score, result))

        # Return best match
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        best_score, best_match = scored_matches[0]

        return best_match if best_score > 0.5 else None

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using embeddings."""
        try:
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception:
            # Fallback to simple string similarity
            return 1.0 if text1.lower() == text2.lower() else 0.0

    def _matches_entity_type(self, wikidata_result: Dict, entity_type: "EntityType") -> bool:
        """Check if Wikidata result matches entity type."""
        # This is a simplified implementation
        # In practice, you'd check Wikidata properties and categories
        description = wikidata_result.get("description", "").lower()

        type_keywords = {
            "PERSON": ["person", "human", "actor", "politician", "scientist"],
            "ORGANIZATION": ["organization", "company", "corporation", "institution"],
            "LOCATION": ["place", "city", "country", "state", "region"],
            "EVENT": ["event", "conference", "festival", "meeting"],
            "PRODUCT": ["product", "software", "device", "tool"],
        }

        keywords = type_keywords.get(entity_type.value, [])
        return any(keyword in description for keyword in keywords)

    async def _get_wikidata_details(self, entity_id: str, client: httpx.AsyncClient) -> Dict[str, Any]:
        """Get detailed information from Wikidata."""
        try:
            params = {
                "format": "json",
                "action": "wbgetentities",
                "ids": entity_id,
                "props": "aliases|descriptions|claims",
                "languages": "en",
            }

            response = await client.get(self.wikidata_base_url, params=params)
            response.raise_for_status()

            data = response.json()
            entities = data.get("entities", {})
            entity_data = entities.get(entity_id, {})

            # Extract aliases
            aliases = []
            if "aliases" in entity_data and "en" in entity_data["aliases"]:
                aliases = [alias["value"] for alias in entity_data["aliases"]["en"]]

            # Extract categories (simplified)
            categories = []
            claims = entity_data.get("claims", {})

            # P31 (instance of) property
            if "P31" in claims:
                for claim in claims["P31"]:
                    if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
                        categories.append(claim["mainsnak"]["datavalue"]["value"]["id"])

            return {"aliases": aliases, "categories": categories, "properties": claims}

        except Exception as e:
            logger.error("Wikidata details fetch failed", entity_id=entity_id, error=str(e))
            return {}

    async def _store_entity(self, entity: "Entity", kb_data: Dict[str, Any]) -> None:
        """Store entity in local database."""
        try:
            async with self.session_factory() as session:
                # Create embedding
                embedding = self.embedding_model.encode([entity.text])

                query = text(
                    """
                    INSERT INTO entities (name, description, aliases, categories,
                                       properties, embedding, confidence_score,
                                       entity_type, created_at)
                    VALUES (:name, :description, :aliases, :categories,
                           :properties, :embedding, :confidence, :entity_type, NOW())
                    ON CONFLICT (name, entity_type)
                    DO UPDATE SET
                        description = EXCLUDED.description,
                        aliases = EXCLUDED.aliases,
                        categories = EXCLUDED.categories,
                        properties = EXCLUDED.properties,
                        embedding = EXCLUDED.embedding,
                        confidence_score = EXCLUDED.confidence_score,
                        updated_at = NOW()
                """
                )

                await session.execute(
                    query,
                    {
                        "name": entity.text,
                        "description": kb_data.get("description", ""),
                        "aliases": kb_data.get("aliases", []),
                        "categories": kb_data.get("categories", []),
                        "properties": kb_data.get("properties", {}),
                        "embedding": embedding[0].tolist(),
                        "confidence": kb_data.get("confidence", 0.8),
                        "entity_type": entity.label.value,
                    },
                )

                await session.commit()

        except Exception as e:
            logger.error("Entity storage failed", error=str(e))

    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """Check if cached result is still valid."""
        import time

        return time.time() - cached_result.get("timestamp", 0) < self.cache_ttl

    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache a result."""
        import time

        self.entity_cache[cache_key] = {"data": result, "timestamp": time.time()}

    async def get_entity_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        try:
            async with self.session_factory() as session:
                query = text(
                    """
                    SELECT
                        COUNT(*) as total_entities,
                        COUNT(DISTINCT entity_type) as unique_types,
                        AVG(confidence_score) as avg_confidence
                    FROM entities
                """
                )

                result = await session.execute(query)
                row = result.fetchone()

                return {
                    "total_entities": row.total_entities,
                    "unique_types": row.unique_types,
                    "average_confidence": float(row.avg_confidence) if row.avg_confidence else 0.0,
                    "cache_size": len(self.entity_cache),
                }

        except Exception as e:
            logger.error("Failed to get entity statistics", error=str(e))
            return {}
