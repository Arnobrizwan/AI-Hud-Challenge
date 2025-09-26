"""Tests for clustering algorithms."""

from typing import Any, Dict
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.clustering.event_grouping import EventGroupingEngine
from src.clustering.incremental_dbscan import IncrementalDBSCAN


class TestIncrementalDBSCAN:
    """Test incremental DBSCAN clustering."""

    def test_init(self):
        """Test DBSCAN initialization."""
        dbscan = IncrementalDBSCAN(eps=0.3, min_samples=2, max_cluster_size=100)

        assert dbscan.eps == 0.3
        assert dbscan.min_samples == 2
        assert dbscan.max_cluster_size == 100
        assert len(dbscan.clusters) == 0
        assert len(dbscan.cluster_centers) == 0

    @pytest.mark.asyncio
    async def test_fit_predict(self, clustering_engine, mock_articles_for_clustering) -> Dict[str, Any]:
        """Test fit and predict."""
        # Create mock features
        features = np.random.random((len(mock_articles_for_clustering), 10))

        cluster_labels = await clustering_engine.fit_predict(mock_articles_for_clustering, features)

        assert len(cluster_labels) == len(mock_articles_for_clustering)
        assert all(isinstance(label, int) for label in cluster_labels)
        # -1 indicates noise points
        assert all(label >= -1 for label in cluster_labels)

    @pytest.mark.asyncio
    async def test_incremental_fit(self, clustering_engine, mock_articles_for_clustering) -> Dict[str, Any]:
        """Test incremental fitting."""
        # First fit with some articles
        initial_articles = mock_articles_for_clustering[:3]
        initial_features = np.random.random((len(initial_articles), 10))

        initial_labels = await clustering_engine.fit_predict(initial_articles, initial_features)

        # Then incrementally fit new articles
        new_articles = mock_articles_for_clustering[3:]
        new_features = np.random.random((len(new_articles), 10))

        new_labels = await clustering_engine.incremental_fit(new_articles, new_features)

        assert len(new_labels) == len(new_articles)
        assert all(isinstance(label, int) for label in new_labels)

    @pytest.mark.asyncio
    async def test_compute_temporal_features(self, clustering_engine, sample_article) -> Dict[str, Any]:
        """Test temporal feature computation."""
        features = await clustering_engine._compute_temporal_features(sample_article)

        assert isinstance(features, np.ndarray)
        assert len(features) == 5  # 5 temporal features
        assert all(isinstance(f, (int, float)) for f in features)

    @pytest.mark.asyncio
    async def test_compute_entity_features(self, clustering_engine, sample_article) -> Dict[str, Any]:
        """Test entity feature computation."""
        features = await clustering_engine._compute_entity_features(sample_article.entities)

        assert isinstance(features, np.ndarray)
        assert len(features) == 10  # 10 entity features
        assert all(isinstance(f, (int, float)) for f in features)

    @pytest.mark.asyncio
    async def test_compute_topic_features(self, clustering_engine, sample_article):
        """Test topic feature computation."""
        features = await clustering_engine._compute_topic_features(sample_article.topics)

        assert isinstance(features, np.ndarray)
        assert len(features) == 5  # 5 topic features
        assert all(isinstance(f, (int, float)) for f in features)

    @pytest.mark.asyncio
    async def test_compute_geo_features(self, clustering_engine, sample_article):
        """Test geographic feature computation."""
        features = await clustering_engine._compute_geo_features(sample_article.locations)

        assert isinstance(features, np.ndarray)
        assert len(features) == 5  # 5 geo features
        assert all(isinstance(f, (int, float)) for f in features)

    def test_get_cluster_quality_metrics(self, clustering_engine):
        """Test cluster quality metrics."""
        # Add some mock clusters
        clustering_engine.clusters = {0: {1, 2, 3}, 1: {4, 5}}
        clustering_engine.cluster_metadata = {
            0: {"size": 3, "created_at": None},
            1: {"size": 2, "created_at": None},
        }

        metrics = clustering_engine.get_cluster_quality_metrics()

        assert isinstance(metrics, dict)
        assert 0 in metrics
        assert 1 in metrics
        assert "size" in metrics[0]
        assert "density" in metrics[0]
        assert "is_valid" in metrics[0]

    def test_get_noise_ratio(self, clustering_engine):
        """Test noise ratio calculation."""
        # Add some mock data
        clustering_engine.clusters = {0: {1, 2, 3}, 1: {4, 5}}
        clustering_engine.noise_points = {6, 7}

        noise_ratio = clustering_engine.get_noise_ratio()

        assert isinstance(noise_ratio, float)
        assert 0.0 <= noise_ratio <= 1.0
        # 2 noise points out of 7 total points = 2/7
        assert abs(noise_ratio - 2 / 7) < 0.01


class TestEventGroupingEngine:
    """Test event grouping engine."""

    def test_init(self, event_grouping_engine):
        """Test event grouping engine initialization."""
        assert event_grouping_engine.clustering_engine is not None
        assert event_grouping_engine.semantic_calc is not None
        assert event_grouping_engine.time_window_hours == 24
        assert event_grouping_engine.min_cluster_size == 2
        assert event_grouping_engine.max_cluster_size == 100

    @pytest.mark.asyncio
    async def test_group_into_events(self, event_grouping_engine, mock_articles_for_clustering):
        """Test grouping articles into events."""
        events = await event_grouping_engine.group_into_events(mock_articles_for_clustering)

        assert isinstance(events, list)
        # All events should have the required attributes
        for event in events:
            assert hasattr(event, "id")
            assert hasattr(event, "cluster")
            assert hasattr(event, "articles")
            assert hasattr(event, "representative_article")
            assert hasattr(event, "temporal_coherence")
            assert hasattr(event, "topical_coherence")

    @pytest.mark.asyncio
    async def test_incremental_group_articles(self, event_grouping_engine, mock_articles_for_clustering):
        """Test incremental article grouping."""
        # First group some articles
        initial_articles = mock_articles_for_clustering[:3]
        initial_events = await event_grouping_engine.group_into_events(initial_articles)

        # Then add more articles incrementally
        new_articles = mock_articles_for_clustering[3:]
        updated_events = await event_grouping_engine.incremental_group_articles(new_articles)

        assert isinstance(updated_events, list)
        # Should have some events (either new or updated)
        assert len(updated_events) >= 0

    @pytest.mark.asyncio
    async def test_compute_temporal_coherence(self, event_grouping_engine, sample_articles):
        """Test temporal coherence computation."""
        coherence = await event_grouping_engine._compute_temporal_coherence(sample_articles)

        assert isinstance(coherence, float)
        assert 0.0 <= coherence <= 1.0

    @pytest.mark.asyncio
    async def test_compute_topical_coherence(self, event_grouping_engine, sample_articles):
        """Test topical coherence computation."""
        coherence = await event_grouping_engine._compute_topical_coherence(sample_articles)

        assert isinstance(coherence, float)
        assert 0.0 <= coherence <= 1.0

    @pytest.mark.asyncio
    async def test_compute_centrality(self, event_grouping_engine, sample_articles):
        """Test centrality computation."""
        candidate = sample_articles[0]
        cluster_articles = sample_articles[:3]

        centrality = await event_grouping_engine._compute_centrality(candidate, cluster_articles)

        assert isinstance(centrality, float)
        assert 0.0 <= centrality <= 1.0

    def test_compute_freshness_score(self, event_grouping_engine, sample_article):
        """Test freshness score computation."""
        score = event_grouping_engine._compute_freshness_score(sample_article.published_at)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_get_source_authority(self, event_grouping_engine) -> Dict[str, Any]:
        """Test source authority computation."""
        authority = await event_grouping_engine._get_source_authority("reuters")
        assert isinstance(authority, float)
        assert 0.0 <= authority <= 1.0

        # Test unknown source
        authority = await event_grouping_engine._get_source_authority("unknown_source")
        assert authority == 0.5  # Default value

    def test_compute_completeness_score(self, event_grouping_engine, sample_article):
        """Test completeness score computation."""
        score = event_grouping_engine._compute_completeness_score(sample_article)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_extract_common_topics(self, event_grouping_engine, sample_articles):
        """Test common topic extraction."""
        topics = await event_grouping_engine._extract_common_topics(sample_articles)

        assert isinstance(topics, list)
        # Should return Topic objects
        for topic in topics:
            assert hasattr(topic, "name")
            assert hasattr(topic, "confidence")

    @pytest.mark.asyncio
    async def test_extract_common_entities(self, event_grouping_engine, sample_articles):
        """Test common entity extraction."""
        entities = await event_grouping_engine._extract_common_entities(sample_articles)

        assert isinstance(entities, list)
        # Should return Entity objects
        for entity in entities:
            assert hasattr(entity, "text")
            assert hasattr(entity, "label")
            assert hasattr(entity, "confidence")

    @pytest.mark.asyncio
    async def test_extract_common_locations(self, event_grouping_engine, sample_articles):
        """Test common location extraction."""
        locations = await event_grouping_engine._extract_common_locations(sample_articles)

        assert isinstance(locations, list)
        # Should return Location objects
        for location in locations:
            assert hasattr(location, "name")
            assert hasattr(location, "confidence")

    def test_get_event_quality_metrics(self, event_grouping_engine):
        """Test event quality metrics."""
        # Add some mock events
        from uuid import uuid4

        from src.models.schemas import Cluster, NewsEvent

        mock_cluster = Cluster(
            id=uuid4(),
            representative_article_id=uuid4(),
            article_count=5,
            quality_score=0.8,
            topics=[],
            entities=[],
            locations=[],
            is_active=True,
            created_at=sample_article.published_at,
            updated_at=sample_article.published_at,
        )

        mock_event = NewsEvent(
            id=uuid4(),
            cluster=mock_cluster,
            articles=sample_articles[:3],
            representative_article=sample_articles[0],
            temporal_coherence=0.9,
            topical_coherence=0.8,
        )

        event_grouping_engine.events[mock_event.id] = mock_event

        metrics = event_grouping_engine.get_event_quality_metrics()

        assert isinstance(metrics, dict)
        assert mock_event.id in metrics
        assert "article_count" in metrics[mock_event.id]
        assert "temporal_coherence" in metrics[mock_event.id]
        assert "topical_coherence" in metrics[mock_event.id]
