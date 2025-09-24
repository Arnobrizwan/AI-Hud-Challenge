"""Advanced content ranking engine with ML and heuristics."""

import asyncio
import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import structlog
from sklearn.preprocessing import StandardScaler

from ..features.extractor import RankingFeatureExtractor
from ..monitoring.metrics import RankingMetricsCollector
from ..optimization.cache import CacheManager
from ..personalization.engine import PersonalizationEngine
from ..schemas import (
    Article,
    FeatureVector,
    PersonalizedScore,
    RankedArticle,
    RankedResults,
    RankingRequest,
)
from ..testing.ab_framework import ABTestingFramework

logger = structlog.get_logger(__name__)


class ContentRankingEngine:
    """Advanced content ranking with ML and heuristics."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.feature_extractor = RankingFeatureExtractor(cache_manager)
        self.personalization_engine = PersonalizationEngine(cache_manager)
        self.ab_tester = ABTestingFramework()
        self.metrics_collector = RankingMetricsCollector()

        # ML Models
        self.ranker_model = None
        self.scaler = StandardScaler()
        self.model_loaded = False

        # Configuration
        self.default_weights = {
            "relevance": 0.3,
            "freshness": 0.25,
            "authority": 0.2,
            "personalization": 0.15,
            "diversity": 0.1,
        }

        # Load models asynchronously
        asyncio.create_task(self._load_models())

    async def _load_models(self) -> Dict[str, Any]:
    """Load ML models asynchronously."""
        try:
            # In production, load from model registry
            # For now, create a dummy model for demonstration
            self.ranker_model = self._create_dummy_model()
            self.model_loaded = True
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.error("Failed to load ML models", error=str(e))
            self.model_loaded = False

    def _create_dummy_model(self):
        """Create a dummy LightGBM model for demonstration."""
        # In production, this would load from a trained model file
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        return lgb.LGBMRanker(**params)

    async def rank_content(self, request: RankingRequest) -> RankedResults:
        """Main ranking pipeline with personalization."""
        start_time = time.time()

        try:
            # Get ranking algorithm variant for A/B testing
            algorithm_variant = await self.ab_tester.get_variant(
                user_id=request.user_id, experiment="ranking_algorithm_v4"
            )

            # Retrieve candidate articles
            candidates = await self._get_candidates(request)
            if not candidates:
                return RankedResults(
                    articles=[],
                    total_count=0,
                    algorithm_variant=algorithm_variant,
                    processing_time_ms=0,
                    features_computed=0,
                    cache_hit_rate=0.0,
                )

            # Compute features for all candidates
            features_start = time.time()
            features = await self.compute_ranking_features(candidates, request)
            features_time = (time.time() - features_start) * 1000

            # Apply ranking algorithm
            ranking_start = time.time()
            if algorithm_variant == "ml_ranker" and self.model_loaded:
                ranked_results = await self.ml_ranking(candidates, features, request)
            elif algorithm_variant == "hybrid":
                ranked_results = await self.hybrid_ranking(candidates, features, request)
            else:
                ranked_results = await self.heuristic_ranking(candidates, features, request)
            ranking_time = (time.time() - ranking_start) * 1000

            # Apply diversity and freshness constraints
            final_results = await self._apply_ranking_constraints(ranked_results, request)

            # Log ranking decision for model updates
            await self._log_ranking_decision(request, final_results, algorithm_variant)

            total_time = (time.time() - start_time) * 1000

            # Update metrics
            await self.metrics_collector.record_ranking(
                response_time_ms=total_time,
                feature_time_ms=features_time,
                ranking_time_ms=ranking_time,
                cache_hit_rate=self.cache_manager.get_hit_rate(),
                article_count=len(candidates),
            )

            return RankedResults(
                articles=final_results,
                total_count=len(candidates),
                algorithm_variant=algorithm_variant,
                processing_time_ms=total_time,
                features_computed=len(features),
                cache_hit_rate=self.cache_manager.get_hit_rate(),
            )

        except Exception as e:
            logger.error(
                "Ranking failed",
                error=str(e),
                user_id=request.user_id)
            await self.metrics_collector.record_error()
            raise

    async def _get_candidates(self, request: RankingRequest) -> List[Article]:
        """Retrieve candidate articles for ranking."""
        # In production, this would query a content database
        # For now, return dummy articles
        # Get more for filtering
        return await self._get_dummy_articles(request.limit * 2)

    async def _get_dummy_articles(self, count: int) -> List[Article]:
        """Generate dummy articles for testing."""
        from ..schemas import Article, Author, ContentType, Source

        articles = []
        for i in range(count):
            article = Article(
                id=f"article_{i}",
                title=f"Sample Article {i}",
                content=f"This is sample content for article {i}",
                url=f"https://example.com/article/{i}",
                published_at=datetime.utcnow() - timedelta(hours=i),
                source=Source(
                    id=f"source_{i % 5}", name=f"Source {i % 5}", domain=f"source{i % 5}.com"
                ),
                author=Author(id=f"author_{i}", name=f"Author {i}"),
                word_count=500 + i * 10,
                reading_time=2 + i // 10,
                quality_score=0.5 + (i % 10) * 0.05,
            )
            articles.append(article)

        return articles

    async def compute_ranking_features(
        self, candidates: List[Article], request: RankingRequest
    ) -> np.ndarray:
        """Comprehensive feature computation for ranking."""
        features_matrix = []

        for article in candidates:
            features = {}

            # Content features
            features.update(await self.feature_extractor.compute_content_features(article))

            # Freshness features
            features.update(await self.feature_extractor.compute_freshness_features(article))

            # Authority features
            features.update(await self.feature_extractor.compute_authority_features(article))

            # Personalization features
            if request.enable_personalization:
                personalization_features = (
                    await self.feature_extractor.compute_personalization_features(
                        article, request.user_id
                    )
                )
                features.update(personalization_features)

            # Contextual features
            features.update(
                await self.feature_extractor.compute_contextual_features(article, request)
            )

            # Interaction features
            features.update(
                await self.feature_extractor.compute_interaction_features(article, request.user_id)
            )

            features_matrix.append(list(features.values()))

        return np.array(features_matrix)

    async def ml_ranking(
            self,
            candidates: List[Article],
            features: np.ndarray,
            request: RankingRequest) -> List[RankedArticle]:
        """ML-based ranking using LightGBM."""
        if not self.model_loaded or self.ranker_model is None:
            logger.warning(
                "ML model not loaded, falling back to heuristic ranking")
            return await self.heuristic_ranking(candidates, features, request)

        try:
            # Normalize features
            features_normalized = self.scaler.fit_transform(features)

            # Get predictions
            scores = self.ranker_model.predict(features_normalized)

            # Create ranked articles
            ranked_articles = []
            for i, (article, score) in enumerate(zip(candidates, scores)):
                ranked_article = RankedArticle(
                    article=article,
                    rank=i + 1,
                    score=float(score),
                    explanation="ML-based ranking")
                ranked_articles.append(ranked_article)

            # Sort by score (descending)
            ranked_articles.sort(key=lambda x: x.score, reverse=True)

            # Update ranks
            for i, article in enumerate(ranked_articles):
                article.rank = i + 1

            return ranked_articles[: request.limit]

        except Exception as e:
            logger.error("ML ranking failed", error=str(e))
            return await self.heuristic_ranking(candidates, features, request)

    async def hybrid_ranking(
            self,
            candidates: List[Article],
            features: np.ndarray,
            request: RankingRequest) -> List[RankedArticle]:
        """Hybrid ranking combining ML and heuristics."""
        # Get ML scores
        ml_articles = await self.ml_ranking(candidates, features, request)

        # Get heuristic scores
        heuristic_articles = await self.heuristic_ranking(candidates, features, request)

        # Combine scores with weights
        combined_articles = []
        for ml_article, heuristic_article in zip(
                ml_articles, heuristic_articles):
            combined_score = 0.7 * ml_article.score + 0.3 * heuristic_article.score

            combined_article = RankedArticle(
                article=ml_article.article,
                rank=ml_article.rank,
                score=combined_score,
                explanation="Hybrid ML + Heuristic ranking",
            )
            combined_articles.append(combined_article)

        # Sort by combined score
        combined_articles.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, article in enumerate(combined_articles):
            article.rank = i + 1

        return combined_articles[: request.limit]

    async def heuristic_ranking(
            self,
            candidates: List[Article],
            features: np.ndarray,
            request: RankingRequest) -> List[RankedArticle]:
        """Heuristic-based ranking using multiple signals."""
        ranked_articles = []

        for i, article in enumerate(candidates):
            # Extract features for this article
            article_features = features[i] if len(
                features) > i else np.zeros(10)

            # Compute heuristic score
            score = await self._compute_heuristic_score(article, article_features, request)

            ranked_article = RankedArticle(
                article=article,
                rank=i + 1,
                score=float(score),
                explanation="Heuristic-based ranking",
            )
            ranked_articles.append(ranked_article)

        # Sort by score (descending)
        ranked_articles.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, article in enumerate(ranked_articles):
            article.rank = i + 1

        return ranked_articles[: request.limit]

    async def _compute_heuristic_score(
        self, article: Article, features: np.ndarray, request: RankingRequest
    ) -> float:
        """Compute heuristic ranking score."""
        score = 0.0

        # Freshness score (higher for newer content)
        age_hours = (datetime.utcnow() -
                     article.published_at).total_seconds() / 3600
        freshness_score = math.exp(-age_hours / 24)  # 24-hour half-life
        score += freshness_score * self.default_weights["freshness"]

        # Quality score
        quality_score = article.quality_score
        score += quality_score * self.default_weights["relevance"]

        # Engagement score (normalized)
        total_engagement = article.view_count + article.like_count + article.share_count
        engagement_score = min(total_engagement / 1000, 1.0)  # Cap at 1.0
        score += engagement_score * 0.1

        # Source authority (if available)
        if article.source.authority_score:
            score += article.source.authority_score * \
                self.default_weights["authority"]

        # Personalization (if enabled)
        if request.enable_personalization:
            personalization_score = await self._get_personalization_score(article, request.user_id)
            score += personalization_score * \
                self.default_weights["personalization"]

        return min(score, 1.0)  # Cap at 1.0

    async def _get_personalization_score(
            self, article: Article, user_id: str) -> float:
        """Get personalization score for article."""
        try:
            personalized_scores = await self.personalization_engine.personalize_ranking(
                [article], user_id
            )
            return personalized_scores[0].score if personalized_scores else 0.5
        except Exception as e:
            logger.warning("Personalization failed", error=str(e))
            return 0.5

    async def _apply_ranking_constraints(
        self, ranked_articles: List[RankedArticle], request: RankingRequest
    ) -> List[RankedArticle]:
        """Apply diversity and freshness constraints."""
        # Apply diversity constraint (limit articles from same source)
        source_counts = {}
        filtered_articles = []

        for article in ranked_articles:
            source_id = article.article.source.id
            if source_counts.get(
                    source_id, 0) < 3:  # Max 3 articles per source
                filtered_articles.append(article)
                source_counts[source_id] = source_counts.get(source_id, 0) + 1

        # Apply freshness constraint (boost recent content)
        now = datetime.utcnow()
        for article in filtered_articles:
            age_hours = (
                now - article.article.published_at).total_seconds() / 3600
            if age_hours < 1:  # Breaking news boost
                article.score *= 1.2
            elif age_hours < 6:  # Recent news boost
                article.score *= 1.1

        # Re-sort after applying constraints
        filtered_articles.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, article in enumerate(filtered_articles):
            article.rank = i + 1

        return filtered_articles[: request.limit]

    async def _log_ranking_decision(
            self,
            request: RankingRequest,
            results: List[RankedArticle],
            algorithm_variant: str):
         -> Dict[str, Any]:"""Log ranking decision for model updates and analysis."""
        try:
            # In production, this would log to a data warehouse
            logger.info(
                "Ranking decision logged",
                user_id=request.user_id,
                algorithm_variant=algorithm_variant,
                article_count=len(results),
                avg_score=sum(
                    r.score for r in results) /
                len(results) if results else 0,
            )
        except Exception as e:
            logger.warning("Failed to log ranking decision", error=str(e))
