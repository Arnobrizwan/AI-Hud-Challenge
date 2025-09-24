"""Advanced deduplication pipeline with LSH and semantic similarity."""

import asyncio
import time
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

from ..algorithms.lsh.lsh_index import LSHIndex, LSHIndexManager
from ..algorithms.lsh.minhash import ContentFingerprinter, MinHashGenerator
from ..algorithms.similarity.combined import (
    CombinedSimilarityCalculator,
    SimilarityThresholdManager,
)
from ..algorithms.similarity.semantic import (
    ContentSimilarityCalculator,
    SemanticSimilarityCalculator,
)
from ..config.settings import settings
from ..models.schemas import DuplicateResult, NormalizedArticle, SimilarityScore


class DeduplicationPipeline:
    """Advanced deduplication with LSH and semantic similarity."""

    def __init__(
        self,
        redis_client,
        lsh_index_manager: LSHIndexManager,
        similarity_calculator: CombinedSimilarityCalculator,
        threshold_manager: SimilarityThresholdManager,
    ):
        """Initialize deduplication pipeline.

        Args:
            redis_client: Redis client for caching
            lsh_index_manager: LSH index manager
            similarity_calculator: Combined similarity calculator
            threshold_manager: Similarity threshold manager
        """
        self.redis = redis_client
        self.lsh_manager = lsh_index_manager
        self.similarity_calc = similarity_calculator
        self.threshold_manager = threshold_manager

        # Initialize components
        self.minhash_gen = MinHashGenerator(num_perm=settings.lsh_num_perm, seed=42)
        self.fingerprinter = ContentFingerprinter(self.minhash_gen)

        # Get LSH index
        self.lsh_index = None

    async def initialize(self) -> None:
        """Initialize the pipeline."""
        self.lsh_index = await self.lsh_manager.get_index(
            index_name="main_lsh",
            threshold=settings.lsh_threshold,
            num_perm=settings.lsh_num_perm,
            num_bands=settings.lsh_num_bands,
            band_size=settings.lsh_band_size,
        )

    async def process_article(self, article: NormalizedArticle) -> DuplicateResult:
        """Main deduplication pipeline with multi-stage detection.

        Args:
            article: Article to process

        Returns:
            Duplicate detection result
        """
        start_time = time.time()

        try:
            # Stage 1: Content fingerprinting
            content_fingerprint = self.fingerprinter.compute_fingerprint(
                article.content, article.title
            )

            # Stage 2: LSH-based candidate retrieval
            lsh_candidates = await self._find_lsh_candidates(article)

            # Stage 3: Semantic similarity filtering
            semantic_matches = await self._filter_semantic_matches(
                article, lsh_candidates, settings.similarity_threshold
            )

            # Stage 4: Final duplicate classification
            duplicates = await self._classify_duplicates(article, semantic_matches)

            # Stage 5: Update indices
            await self._update_indices(article, content_fingerprint)

            # Create result
            result = DuplicateResult(
                article_id=article.id,
                is_duplicate=len(duplicates) > 0,
                duplicate_of=duplicates[0].article_id if duplicates else None,
                similarity_scores=[(d.article_id, d.similarity) for d in duplicates],
                cluster_id=await self._get_or_create_cluster(article, duplicates),
                confidence=max([d.confidence for d in duplicates], default=0.0),
                detection_method="combined",
            )

            # Update processing time
            processing_time = time.time() - start_time
            await self._update_metrics("processing_time", processing_time)

            return result

        except Exception as e:
            # Log error and return non-duplicate result
            print(f"Error processing article {article.id}: {e}")
            return DuplicateResult(
                article_id=article.id, is_duplicate=False, confidence=0.0, detection_method="error"
            )

    async def process_batch(self, articles: List[NormalizedArticle]) -> List[DuplicateResult]:
        """Process a batch of articles.

        Args:
            articles: List of articles to process

        Returns:
            List of duplicate detection results
        """
        # Process articles in parallel
        tasks = [self.process_article(article) for article in articles]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Error in batch processing: {result}")
                continue
            valid_results.append(result)

        # Update batch metrics
        await self._update_metrics("articles_processed", len(articles))
        await self._update_metrics(
            "duplicates_found", sum(1 for r in valid_results if r.is_duplicate)
        )

        return valid_results

    async def _find_lsh_candidates(self, article: NormalizedArticle) -> List[NormalizedArticle]:
        """Find LSH candidates for article.

        Args:
            article: Article to find candidates for

        Returns:
            List of candidate articles
        """
        # Generate MinHash for article
        minhash = self.minhash_gen.generate_minhash(article.content)

        # Query LSH index
        candidate_ids = await self.lsh_index.query_candidates(minhash)

        # Get candidate articles from database (placeholder)
        # In real implementation, this would query the database
        candidates = []
        for candidate_id in candidate_ids:
            # This would be replaced with actual database query
            candidate = await self._get_article_by_id(candidate_id)
            if candidate:
                candidates.append(candidate)

        return candidates

    async def _filter_semantic_matches(
        self, article: NormalizedArticle, candidates: List[NormalizedArticle], threshold: float
    ) -> List[SimilarityScore]:
        """Filter candidates using semantic similarity.

        Args:
            article: Target article
            candidates: List of candidate articles
            threshold: Similarity threshold

        Returns:
            List of similar articles with scores
        """
        if not candidates:
            return []

        # Compute similarities
        similarities = await self.similarity_calc.compute_batch_similarity(article, candidates)

        # Filter by threshold
        filtered_similarities = [sim for sim in similarities if sim.similarity >= threshold]

        return filtered_similarities

    async def _classify_duplicates(
        self, article: NormalizedArticle, semantic_matches: List[SimilarityScore]
    ) -> List[SimilarityScore]:
        """Classify duplicates from semantic matches.

        Args:
            article: Target article
            semantic_matches: List of semantic matches

        Returns:
            List of confirmed duplicates
        """
        duplicates = []

        for match in semantic_matches:
            # Get candidate article
            candidate = await self._get_article_by_id(match.article_id)
            if not candidate:
                continue

            # Compute detailed similarity
            detailed_sim = await self.similarity_calc.compute_detailed_similarity(
                article, candidate
            )

            # Check if it's a duplicate
            is_duplicate = self.threshold_manager.is_duplicate(
                detailed_sim["weighted_similarity"],
                detailed_sim["title_similarity"],
                detailed_sim["content_similarity"],
            )

            if is_duplicate:
                # Update similarity score with detailed calculation
                match.similarity = detailed_sim["weighted_similarity"]
                match.confidence = detailed_sim["confidence"]
                duplicates.append(match)

        # Sort by similarity (descending)
        duplicates.sort(key=lambda x: x.similarity, reverse=True)

        return duplicates

    async def _update_indices(self, article: NormalizedArticle, content_fingerprint: Dict) -> None:
        """Update LSH and other indices.

        Args:
            article: Article to add to indices
            content_fingerprint: Content fingerprint
        """
        # Generate MinHash
        minhash = self.minhash_gen.generate_minhash(article.content)

        # Add to LSH index
        await self.lsh_index.add_article(article.id, minhash)

        # Cache fingerprint in Redis
        await self._cache_fingerprint(article.id, content_fingerprint)

    async def _get_or_create_cluster(
        self, article: NormalizedArticle, duplicates: List[SimilarityScore]
    ) -> Optional[UUID]:
        """Get or create cluster for article.

        Args:
            article: Article to cluster
            duplicates: List of duplicates

        Returns:
            Cluster ID if created or found
        """
        if not duplicates:
            return None

        # For now, return the first duplicate's cluster
        # In a full implementation, this would handle cluster management
        return duplicates[0].article_id

    async def _get_article_by_id(self, article_id: UUID) -> Optional[NormalizedArticle]:
        """Get article by ID from database.

        Args:
            article_id: Article UUID

        Returns:
            Article if found, None otherwise
        """
        # This would be replaced with actual database query
        # For now, return None as placeholder
        return None

    async def _cache_fingerprint(self, article_id: UUID, fingerprint: Dict) -> None:
        """Cache article fingerprint in Redis.

        Args:
            article_id: Article UUID
            fingerprint: Content fingerprint
        """
        cache_key = f"fingerprint:{article_id}"
        await self.redis.setex(cache_key, settings.cache_ttl, str(fingerprint))

    async def _update_metrics(self, metric_name: str, value: float) -> None:
        """Update processing metrics.

        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        await self.redis.hincrbyfloat("metrics", metric_name, value)

    async def get_pipeline_stats(self) -> Dict[str, float]:
        """Get pipeline statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        stats = await self.redis.hgetall("metrics")
        return {k.decode(): float(v) for k, v in stats.items()}

    async def clear_pipeline_cache(self) -> None:
        """Clear pipeline cache."""
        # Clear fingerprint cache
        pattern = "fingerprint:*"
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)

        # Clear metrics
        await self.redis.delete("metrics")


class DeduplicationService:
    """High-level deduplication service."""

    def __init__(self, pipeline: DeduplicationPipeline):
        """Initialize deduplication service.

        Args:
            pipeline: Deduplication pipeline
        """
        self.pipeline = pipeline

    async def deduplicate_article(self, article: NormalizedArticle) -> DuplicateResult:
        """Deduplicate a single article.

        Args:
            article: Article to deduplicate

        Returns:
            Duplicate detection result
        """
        return await self.pipeline.process_article(article)

    async def deduplicate_batch(self, articles: List[NormalizedArticle]) -> List[DuplicateResult]:
        """Deduplicate a batch of articles.

        Args:
            articles: List of articles to deduplicate

        Returns:
            List of duplicate detection results
        """
        return await self.pipeline.process_batch(articles)

    async def find_duplicates(
        self, article: NormalizedArticle, threshold: Optional[float] = None
    ) -> List[SimilarityScore]:
        """Find duplicates for an article.

        Args:
            article: Article to find duplicates for
            threshold: Similarity threshold

        Returns:
            List of similar articles with scores
        """
        if threshold is None:
            threshold = settings.similarity_threshold

        # Get LSH candidates
        candidates = await self.pipeline._find_lsh_candidates(article)

        # Find similar articles
        similar_articles = await self.pipeline.similarity_calc.find_similar_articles(
            article, candidates, threshold
        )

        return similar_articles

    async def get_service_stats(self) -> Dict[str, float]:
        """Get service statistics.

        Returns:
            Dictionary with service statistics
        """
        return await self.pipeline.get_pipeline_stats()

    async def health_check(self) -> Dict[str, any]:
        """Perform health check.

        Returns:
            Health check result
        """
        try:
            # Check LSH index
            lsh_stats = await self.pipeline.lsh_index.get_index_stats()

            # Check Redis connection
            await self.pipeline.redis.ping()

            return {
                "status": "healthy",
                "lsh_index_size": lsh_stats.get("index_size", 0),
                "redis_connected": True,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "redis_connected": False}
