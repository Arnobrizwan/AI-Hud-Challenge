"""Tests for deduplication pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.deduplication.pipeline import DeduplicationPipeline, DeduplicationService
from src.models.schemas import DuplicateResult, SimilarityScore


class TestDeduplicationPipeline:
    """Test deduplication pipeline."""
    
    @pytest.mark.asyncio
    async def test_process_article_no_duplicates(
        self,
        deduplication_pipeline,
        sample_article
    ):
        """Test processing article with no duplicates."""
        result = await deduplication_pipeline.process_article(sample_article)
        
        assert isinstance(result, DuplicateResult)
        assert result.article_id == sample_article.id
        assert result.is_duplicate is False
        assert result.duplicate_of is None
        assert result.similarity_scores == []
        assert result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_process_batch(
        self,
        deduplication_pipeline,
        sample_articles
    ):
        """Test batch processing."""
        results = await deduplication_pipeline.process_batch(sample_articles)
        
        assert len(results) == len(sample_articles)
        assert all(isinstance(result, DuplicateResult) for result in results)
        assert all(result.article_id == article.id for result, article in zip(results, sample_articles))
    
    @pytest.mark.asyncio
    async def test_find_lsh_candidates(
        self,
        deduplication_pipeline,
        sample_article
    ):
        """Test LSH candidate finding."""
        # Mock the LSH index query
        deduplication_pipeline.lsh_index.query_candidates = AsyncMock(return_value=[])
        
        candidates = await deduplication_pipeline._find_lsh_candidates(sample_article)
        
        assert isinstance(candidates, list)
        deduplication_pipeline.lsh_index.query_candidates.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_filter_semantic_matches(
        self,
        deduplication_pipeline,
        sample_article,
        sample_articles
    ):
        """Test semantic similarity filtering."""
        candidates = sample_articles[:3]  # Use first 3 articles as candidates
        
        matches = await deduplication_pipeline._filter_semantic_matches(
            sample_article, candidates, threshold=0.85
        )
        
        assert isinstance(matches, list)
        # All matches should be SimilarityScore objects
        assert all(isinstance(match, SimilarityScore) for match in matches)
    
    @pytest.mark.asyncio
    async def test_classify_duplicates(
        self,
        deduplication_pipeline,
        sample_article
    ):
        """Test duplicate classification."""
        # Create mock similarity scores
        similarity_scores = [
            SimilarityScore(
                article_id=sample_article.id,
                similarity=0.9,
                similarity_type="combined",
                confidence=0.8
            )
        ]
        
        # Mock the get_article_by_id method
        deduplication_pipeline._get_article_by_id = AsyncMock(return_value=sample_article)
        
        duplicates = await deduplication_pipeline._classify_duplicates(
            sample_article, similarity_scores
        )
        
        assert isinstance(duplicates, list)
        # Should return duplicates based on threshold
        assert all(isinstance(dup, SimilarityScore) for dup in duplicates)
    
    @pytest.mark.asyncio
    async def test_update_indices(
        self,
        deduplication_pipeline,
        sample_article
    ):
        """Test index updates."""
        content_fingerprint = {
            'content_hash': 'test_hash',
            'content_minhash': MagicMock(),
            'content_length': 100,
            'word_count': 20
        }
        
        # Mock the LSH index add_article method
        deduplication_pipeline.lsh_index.add_article = AsyncMock()
        
        await deduplication_pipeline._update_indices(sample_article, content_fingerprint)
        
        # Verify LSH index was updated
        deduplication_pipeline.lsh_index.add_article.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_pipeline_stats(
        self,
        deduplication_pipeline
    ):
        """Test pipeline statistics."""
        # Mock Redis hgetall
        deduplication_pipeline.redis.hgetall = AsyncMock(return_value={
            b'articles_processed': b'100',
            b'duplicates_found': b'25'
        })
        
        stats = await deduplication_pipeline.get_pipeline_stats()
        
        assert isinstance(stats, dict)
        assert 'articles_processed' in stats
        assert 'duplicates_found' in stats


class TestDeduplicationService:
    """Test deduplication service."""
    
    @pytest.mark.asyncio
    async def test_deduplicate_article(
        self,
        deduplication_service,
        sample_article
    ):
        """Test single article deduplication."""
        result = await deduplication_service.deduplicate_article(sample_article)
        
        assert isinstance(result, DuplicateResult)
        assert result.article_id == sample_article.id
    
    @pytest.mark.asyncio
    async def test_deduplicate_batch(
        self,
        deduplication_service,
        sample_articles
    ):
        """Test batch deduplication."""
        results = await deduplication_service.deduplicate_batch(sample_articles)
        
        assert len(results) == len(sample_articles)
        assert all(isinstance(result, DuplicateResult) for result in results)
    
    @pytest.mark.asyncio
    async def test_find_duplicates(
        self,
        deduplication_service,
        sample_article
    ):
        """Test finding duplicates."""
        # Mock the pipeline's _find_lsh_candidates method
        deduplication_service.pipeline._find_lsh_candidates = AsyncMock(return_value=[])
        
        similar_articles = await deduplication_service.find_duplicates(
            sample_article, threshold=0.85
        )
        
        assert isinstance(similar_articles, list)
        assert all(isinstance(sim, SimilarityScore) for sim in similar_articles)
    
    @pytest.mark.asyncio
    async def test_get_service_stats(
        self,
        deduplication_service
    ):
        """Test service statistics."""
        # Mock the pipeline's get_pipeline_stats method
        deduplication_service.pipeline.get_pipeline_stats = AsyncMock(return_value={
            'articles_processed': 100,
            'duplicates_found': 25
        })
        
        stats = await deduplication_service.get_service_stats()
        
        assert isinstance(stats, dict)
        assert 'articles_processed' in stats
        assert 'duplicates_found' in stats
    
    @pytest.mark.asyncio
    async def test_health_check(
        self,
        deduplication_service
    ):
        """Test health check."""
        # Mock the pipeline's health_check method
        deduplication_service.pipeline.health_check = AsyncMock(return_value={
            'status': 'healthy',
            'lsh_index_size': 1000,
            'redis_connected': True
        })
        
        health = await deduplication_service.health_check()
        
        assert isinstance(health, dict)
        assert 'status' in health
        assert health['status'] == 'healthy'
        assert 'lsh_index_size' in health
        assert 'redis_connected' in health
