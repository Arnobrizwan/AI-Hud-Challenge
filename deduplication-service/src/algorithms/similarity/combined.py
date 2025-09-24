"""Combined similarity calculator with multiple similarity metrics."""

import asyncio
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from .semantic import SemanticSimilarityCalculator, ContentSimilarityCalculator
from ..lsh.minhash import MinHashGenerator
from ...models.schemas import NormalizedArticle, SimilarityScore


class CombinedSimilarityCalculator:
    """Combined similarity calculator using multiple metrics."""
    
    def __init__(
        self,
        semantic_calculator: SemanticSimilarityCalculator,
        content_calculator: ContentSimilarityCalculator,
        title_weight: float = 0.4,
        content_weight: float = 0.4,
        entity_weight: float = 0.2
    ):
        """Initialize combined similarity calculator.
        
        Args:
            semantic_calculator: Semantic similarity calculator
            content_calculator: Content similarity calculator
            title_weight: Weight for title similarity
            content_weight: Weight for content similarity
            entity_weight: Weight for entity similarity
        """
        self.semantic_calc = semantic_calculator
        self.content_calc = content_calculator
        
        # Normalize weights
        total_weight = title_weight + content_weight + entity_weight
        self.title_weight = title_weight / total_weight
        self.content_weight = content_weight / total_weight
        self.entity_weight = entity_weight / total_weight
    
    async def compute_similarity(
        self,
        article1: NormalizedArticle,
        article2: NormalizedArticle
    ) -> SimilarityScore:
        """Compute combined similarity between two articles.
        
        Args:
            article1: First article
            article2: Second article
            
        Returns:
            Combined similarity score
        """
        # Compute individual similarities
        title_sim = await self.content_calc.compute_title_similarity(
            article1.title, article2.title
        )
        
        content_sim = await self.content_calc.compute_content_similarity(
            article1.content, article2.content
        )
        
        semantic_sim = await self.semantic_calc.compute_similarity(
            article1, article2
        )
        
        entity_sim = await self.semantic_calc.compute_entity_similarity(
            article1.entities, article2.entities
        )
        
        topic_sim = await self.semantic_calc.compute_topic_similarity(
            article1.topics, article2.topics
        )
        
        location_sim = await self.semantic_calc.compute_location_similarity(
            article1.locations, article2.locations
        )
        
        # Compute weighted similarity
        weighted_similarity = (
            self.title_weight * title_sim +
            self.content_weight * content_sim +
            self.entity_weight * (entity_sim + topic_sim + location_sim) / 3
        )
        
        # Compute confidence based on individual similarities
        confidence = self._compute_confidence([
            title_sim, content_sim, semantic_sim, entity_sim, topic_sim, location_sim
        ])
        
        return SimilarityScore(
            article_id=article2.id,
            similarity=min(weighted_similarity, 1.0),
            similarity_type="combined",
            confidence=confidence
        )
    
    async def compute_batch_similarity(
        self,
        target_article: NormalizedArticle,
        candidate_articles: List[NormalizedArticle]
    ) -> List[SimilarityScore]:
        """Compute similarity between target article and batch of candidates.
        
        Args:
            target_article: Target article
            candidate_articles: List of candidate articles
            
        Returns:
            List of similarity scores
        """
        if not candidate_articles:
            return []
        
        # Compute similarities in parallel
        tasks = [
            self.compute_similarity(target_article, candidate)
            for candidate in candidate_articles
        ]
        
        similarities = await asyncio.gather(*tasks)
        
        return similarities
    
    async def find_similar_articles(
        self,
        target_article: NormalizedArticle,
        candidate_articles: List[NormalizedArticle],
        threshold: float = 0.85,
        max_results: Optional[int] = None
    ) -> List[SimilarityScore]:
        """Find articles similar to target article.
        
        Args:
            target_article: Target article
            candidate_articles: List of candidate articles
            threshold: Similarity threshold
            max_results: Maximum number of results
            
        Returns:
            List of similar articles with scores
        """
        # Compute similarities
        similarities = await self.compute_batch_similarity(
            target_article, candidate_articles
        )
        
        # Filter by threshold
        filtered_similarities = [
            sim for sim in similarities if sim.similarity >= threshold
        ]
        
        # Sort by similarity (descending)
        filtered_similarities.sort(key=lambda x: x.similarity, reverse=True)
        
        # Limit results
        if max_results:
            filtered_similarities = filtered_similarities[:max_results]
        
        return filtered_similarities
    
    async def compute_detailed_similarity(
        self,
        article1: NormalizedArticle,
        article2: NormalizedArticle
    ) -> Dict[str, float]:
        """Compute detailed similarity breakdown.
        
        Args:
            article1: First article
            article2: Second article
            
        Returns:
            Dictionary with detailed similarity scores
        """
        # Compute all similarity metrics
        title_sim = await self.content_calc.compute_title_similarity(
            article1.title, article2.title
        )
        
        content_sim = await self.content_calc.compute_content_similarity(
            article1.content, article2.content
        )
        
        semantic_sim = await self.semantic_calc.compute_similarity(
            article1, article2
        )
        
        entity_sim = await self.semantic_calc.compute_entity_similarity(
            article1.entities, article2.entities
        )
        
        topic_sim = await self.semantic_calc.compute_topic_similarity(
            article1.topics, article2.topics
        )
        
        location_sim = await self.semantic_calc.compute_location_similarity(
            article1.locations, article2.locations
        )
        
        # Compute combined scores
        weighted_similarity = (
            self.title_weight * title_sim +
            self.content_weight * content_sim +
            self.entity_weight * (entity_sim + topic_sim + location_sim) / 3
        )
        
        return {
            'title_similarity': title_sim,
            'content_similarity': content_sim,
            'semantic_similarity': semantic_sim,
            'entity_similarity': entity_sim,
            'topic_similarity': topic_sim,
            'location_similarity': location_sim,
            'weighted_similarity': weighted_similarity,
            'confidence': self._compute_confidence([
                title_sim, content_sim, semantic_sim, entity_sim, topic_sim, location_sim
            ])
        }
    
    def _compute_confidence(self, similarities: List[float]) -> float:
        """Compute confidence score based on similarity consistency.
        
        Args:
            similarities: List of similarity scores
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not similarities:
            return 0.0
        
        # Compute variance of similarities
        mean_sim = sum(similarities) / len(similarities)
        variance = sum((sim - mean_sim) ** 2 for sim in similarities) / len(similarities)
        
        # Convert variance to confidence (lower variance = higher confidence)
        confidence = max(0.0, 1.0 - variance)
        
        return confidence


class SimilarityThresholdManager:
    """Manager for similarity thresholds and filtering."""
    
    def __init__(
        self,
        default_threshold: float = 0.85,
        lsh_threshold: float = 0.7,
        content_threshold: float = 0.8,
        title_threshold: float = 0.9
    ):
        """Initialize threshold manager.
        
        Args:
            default_threshold: Default similarity threshold
            lsh_threshold: LSH threshold for candidate filtering
            content_threshold: Content similarity threshold
            title_threshold: Title similarity threshold
        """
        self.default_threshold = default_threshold
        self.lsh_threshold = lsh_threshold
        self.content_threshold = content_threshold
        self.title_threshold = title_threshold
    
    def should_consider_candidate(
        self,
        lsh_similarity: float,
        content_similarity: float,
        title_similarity: float
    ) -> bool:
        """Check if candidate should be considered for detailed comparison.
        
        Args:
            lsh_similarity: LSH similarity score
            content_similarity: Content similarity score
            title_similarity: Title similarity score
            
        Returns:
            True if candidate should be considered
        """
        # At least one similarity should be above threshold
        return (
            lsh_similarity >= self.lsh_threshold or
            content_similarity >= self.content_threshold or
            title_similarity >= self.title_threshold
        )
    
    def is_duplicate(
        self,
        combined_similarity: float,
        title_similarity: float,
        content_similarity: float
    ) -> bool:
        """Check if articles are duplicates based on thresholds.
        
        Args:
            combined_similarity: Combined similarity score
            title_similarity: Title similarity score
            content_similarity: Content similarity score
            
        Returns:
            True if articles are duplicates
        """
        # High combined similarity
        if combined_similarity >= self.default_threshold:
            return True
        
        # Very high title similarity
        if title_similarity >= self.title_threshold:
            return True
        
        # High content similarity with decent title similarity
        if (content_similarity >= self.content_threshold and 
            title_similarity >= 0.7):
            return True
        
        return False
    
    def get_confidence_level(self, similarity: float) -> str:
        """Get confidence level for similarity score.
        
        Args:
            similarity: Similarity score
            
        Returns:
            Confidence level string
        """
        if similarity >= 0.95:
            return "very_high"
        elif similarity >= 0.9:
            return "high"
        elif similarity >= 0.8:
            return "medium"
        elif similarity >= 0.7:
            return "low"
        else:
            return "very_low"
