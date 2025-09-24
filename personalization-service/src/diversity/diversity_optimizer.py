"""Diversity and serendipity optimization for recommendations."""

import asyncio
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import structlog
from sklearn.metrics.pairwise import cosine_similarity

from ..config.settings import settings
from ..models.schemas import ContentItem, DiversityParams, Recommendation, UserProfile

logger = structlog.get_logger()


class DiversityOptimizer:
    """Optimize recommendation diversity and serendipity."""

    def __init__(self):
        self.diversity_threshold = settings.diversity_threshold
        self.serendipity_weight = settings.serendipity_weight

    async def optimize(
        self,
        recommendations: List[Recommendation],
        user_profile: UserProfile,
        diversity_params: Optional[DiversityParams] = None,
    ) -> List[Recommendation]:
        """Apply diversity and serendipity optimization."""
        if not diversity_params or not diversity_params.enable_diversity:
            return recommendations

        if not recommendations:
            return recommendations

        # Apply diversity constraints
        diversified = await self._apply_diversity_constraints(recommendations, diversity_params)

        # Apply serendipity optimization
        serendipitous = await self._apply_serendipity_optimization(
            diversified, user_profile, diversity_params
        )

        # Re-rank with diversity and serendipity scores
        final_recommendations = await self._rerank_with_diversity(
            serendipitous, user_profile, diversity_params
        )

        return final_recommendations[: diversity_params.max_results]

    async def _apply_diversity_constraints(
        self, recommendations: List[Recommendation], diversity_params: DiversityParams
    ) -> List[Recommendation]:
        """Apply diversity constraints to recommendations."""
        optimized = []
        selected_topics = set()
        selected_sources = set()

        for rec in recommendations:
            # Check topic diversity
            topic_diversity_ok = await self._check_topic_diversity(
                rec, selected_topics, diversity_params.topic_diversity_threshold
            )

            # Check source diversity
            source_diversity_ok = await self._check_source_diversity(
                rec, selected_sources, diversity_params.source_diversity_threshold
            )

            if topic_diversity_ok and source_diversity_ok:
                optimized.append(rec)
                selected_topics.update(rec.topics)
                selected_sources.add(rec.source)

            if len(optimized) >= diversity_params.max_results:
                break

        return optimized

    async def _check_topic_diversity(
        self, recommendation: Recommendation, selected_topics: Set[str], threshold: float
    ) -> bool:
        """Check if recommendation meets topic diversity requirements."""
        if not recommendation.topics:
            return True

        # Calculate topic overlap
        overlap = len(set(recommendation.topics).intersection(selected_topics))
        total_topics = len(recommendation.topics)

        if total_topics == 0:
            return True

        overlap_ratio = overlap / total_topics
        return overlap_ratio <= threshold

    async def _check_source_diversity(
        self, recommendation: Recommendation, selected_sources: Set[str], threshold: float
    ) -> bool:
        """Check if recommendation meets source diversity requirements."""
        if not recommendation.source:
            return True

        # Check if source is already selected
        if recommendation.source in selected_sources:
            return False

        return True

    async def _apply_serendipity_optimization(
        self,
        recommendations: List[Recommendation],
        user_profile: UserProfile,
        diversity_params: DiversityParams,
    ) -> List[Recommendation]:
        """Apply serendipity optimization to recommendations."""
        if diversity_params.serendipity_weight <= 0:
            return recommendations

        serendipitous = []

        for rec in recommendations:
            # Compute serendipity boost
            serendipity_boost = await self._compute_serendipity_boost(
                rec, user_profile, diversity_params.serendipity_weight
            )

            # Apply boost to score
            rec.score += serendipity_boost

            # Add serendipity features
            rec.features["serendipity_boost"] = serendipity_boost
            rec.features["serendipity_score"] = await self._compute_serendipity_score(
                rec, user_profile
            )

            serendipitous.append(rec)

        return serendipitous

    async def _compute_serendipity_boost(
        self, recommendation: Recommendation, user_profile: UserProfile, serendipity_weight: float
    ) -> float:
        """Compute serendipity boost for a recommendation."""
        # Topic serendipity
        topic_serendipity = await self._compute_topic_serendipity(recommendation, user_profile)

        # Source serendipity
        source_serendipity = await self._compute_source_serendipity(recommendation, user_profile)

        # Content novelty
        content_novelty = await self._compute_content_novelty(recommendation, user_profile)

        # Combine serendipity factors
        serendipity_score = (
            topic_serendipity * 0.4 + source_serendipity * 0.3 + content_novelty * 0.3
        )

        return serendipity_score * serendipity_weight

    async def _compute_topic_serendipity(
        self, recommendation: Recommendation, user_profile: UserProfile
    ) -> float:
        """Compute topic serendipity score."""
        if not recommendation.topics or not user_profile.topic_preferences:
            return 0.0

        # Find topics that are less preferred by user
        serendipity_score = 0.0
        for topic in recommendation.topics:
            preference = user_profile.topic_preferences.get(topic, 0.0)
            # Higher serendipity for less preferred topics
            serendipity_score += (1.0 - preference) / len(recommendation.topics)

        return serendipity_score

    async def _compute_source_serendipity(
        self, recommendation: Recommendation, user_profile: UserProfile
    ) -> float:
        """Compute source serendipity score."""
        if not recommendation.source or not user_profile.source_preferences:
            return 0.0

        preference = user_profile.source_preferences.get(recommendation.source, 0.0)
        # Higher serendipity for less preferred sources
        return 1.0 - preference

    async def _compute_content_novelty(
        self, recommendation: Recommendation, user_profile: UserProfile
    ) -> float:
        """Compute content novelty score."""
        # This would typically use content embeddings or other features
        # For now, use a simple heuristic based on content features

        novelty_score = 0.0

        # Check for unusual content features
        if "embedding_dim" in recommendation.features:
            # Use embedding variance as novelty indicator
            embedding_values = [
                v for k, v in recommendation.features.items() if k.startswith("embedding_dim_")
            ]
            if embedding_values:
                variance = np.var(embedding_values)
                novelty_score = min(1.0, variance)

        return novelty_score

    async def _compute_serendipity_score(
        self, recommendation: Recommendation, user_profile: UserProfile
    ) -> float:
        """Compute overall serendipity score for a recommendation."""
        topic_serendipity = await self._compute_topic_serendipity(recommendation, user_profile)
        source_serendipity = await self._compute_source_serendipity(recommendation, user_profile)
        content_novelty = await self._compute_content_novelty(recommendation, user_profile)

        return (topic_serendipity + source_serendipity + content_novelty) / 3.0

    async def _rerank_with_diversity(
        self,
        recommendations: List[Recommendation],
        user_profile: UserProfile,
        diversity_params: DiversityParams,
    ) -> List[Recommendation]:
        """Re-rank recommendations considering diversity and serendipity."""
        if not recommendations:
            return recommendations

        # Compute diversity scores
        diversity_scores = await self._compute_diversity_scores(recommendations)

        # Combine original scores with diversity scores
        for i, rec in enumerate(recommendations):
            diversity_score = diversity_scores[i]
            serendipity_score = rec.features.get("serendipity_score", 0.0)

            # Weighted combination
            rec.score = rec.score * 0.6 + diversity_score * 0.2 + serendipity_score * 0.2

            # Add diversity features
            rec.features["diversity_score"] = diversity_score

        # Sort by combined score
        return sorted(recommendations, key=lambda x: x.score, reverse=True)

    async def _compute_diversity_scores(self, recommendations: List[Recommendation]) -> List[float]:
        """Compute diversity scores for a list of recommendations."""
        if len(recommendations) <= 1:
            return [1.0] * len(recommendations)

        diversity_scores = []

        for i, rec in enumerate(recommendations):
            # Compute diversity with respect to other recommendations
            diversity_score = 0.0
            count = 0

            for j, other_rec in enumerate(recommendations):
                if i != j:
                    # Topic diversity
                    topic_diversity = await self._compute_topic_diversity_score(rec, other_rec)

                    # Source diversity
                    source_diversity = await self._compute_source_diversity_score(rec, other_rec)

                    # Content diversity (using features)
                    content_diversity = await self._compute_content_diversity_score(rec, other_rec)

                    diversity_score += (
                        topic_diversity * 0.4 + source_diversity * 0.3 + content_diversity * 0.3
                    )
                    count += 1

            if count > 0:
                diversity_score /= count

            diversity_scores.append(diversity_score)

        return diversity_scores

    async def _compute_topic_diversity_score(
        self, rec1: Recommendation, rec2: Recommendation
    ) -> float:
        """Compute topic diversity between two recommendations."""
        if not rec1.topics or not rec2.topics:
            return 1.0

        topics1 = set(rec1.topics)
        topics2 = set(rec2.topics)

        # Jaccard distance
        intersection = len(topics1.intersection(topics2))
        union = len(topics1.union(topics2))

        if union == 0:
            return 1.0

        return 1.0 - (intersection / union)

    async def _compute_source_diversity_score(
        self, rec1: Recommendation, rec2: Recommendation
    ) -> float:
        """Compute source diversity between two recommendations."""
        if not rec1.source or not rec2.source:
            return 1.0

        return 1.0 if rec1.source != rec2.source else 0.0

    async def _compute_content_diversity_score(
        self, rec1: Recommendation, rec2: Recommendation
    ) -> float:
        """Compute content diversity between two recommendations."""
        # Extract feature vectors
        features1 = self._extract_feature_vector(rec1)
        features2 = self._extract_feature_vector(rec2)

        if not features1 or not features2:
            return 1.0

        # Compute cosine similarity
        similarity = cosine_similarity([features1], [features2])[0][0]

        # Return diversity (1 - similarity)
        return 1.0 - similarity

    def _extract_feature_vector(self, recommendation: Recommendation) -> Optional[List[float]]:
        """Extract feature vector from recommendation."""
        features = []

        # Add embedding features
        for k, v in recommendation.features.items():
            if k.startswith("embedding_dim_") and isinstance(v, (int, float)):
                features.append(float(v))

        # Add other numerical features
        for k, v in recommendation.features.items():
            if k in ["tfidf_score", "cf_score", "topic_similarity"] and isinstance(v, (int, float)):
                features.append(float(v))

        return features if features else None

    async def compute_diversity_metrics(
        self, recommendations: List[Recommendation]
    ) -> Dict[str, float]:
        """Compute diversity metrics for a set of recommendations."""
        if not recommendations:
            return {}

        # Topic diversity
        all_topics = set()
        for rec in recommendations:
            all_topics.update(rec.topics)

        topic_diversity = len(all_topics) / len(recommendations) if recommendations else 0.0

        # Source diversity
        sources = set(rec.source for rec in recommendations if rec.source)
        source_diversity = len(sources) / len(recommendations) if recommendations else 0.0

        # Content diversity
        diversity_scores = await self._compute_diversity_scores(recommendations)
        avg_content_diversity = np.mean(diversity_scores) if diversity_scores else 0.0

        return {
            "topic_diversity": topic_diversity,
            "source_diversity": source_diversity,
            "content_diversity": avg_content_diversity,
            "overall_diversity": (topic_diversity + source_diversity + avg_content_diversity) / 3.0,
        }
