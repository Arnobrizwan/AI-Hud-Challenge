"""Event grouping engine with temporal and topical coherence."""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

import numpy as np
from sklearn.metrics import silhouette_score

from ..algorithms.similarity.semantic import SemanticSimilarityCalculator
from ..models.schemas import Cluster, Entity, Location, NewsEvent, NormalizedArticle, Topic
from .incremental_dbscan import IncrementalDBSCAN


class EventGroupingEngine:
    """Intelligent event grouping with temporal and topical coherence."""

    def __init__(
        self,
        clustering_engine: IncrementalDBSCAN,
        semantic_calculator: SemanticSimilarityCalculator,
        time_window_hours: int = 24,
        min_cluster_size: int = 2,
        max_cluster_size: int = 100,
    ):
        """Initialize event grouping engine.

        Args:
            clustering_engine: Incremental DBSCAN clustering engine
            semantic_calculator: Semantic similarity calculator
            time_window_hours: Time window for event grouping
            min_cluster_size: Minimum cluster size
            max_cluster_size: Maximum cluster size
        """
        self.clustering_engine = clustering_engine
        self.semantic_calc = semantic_calculator
        self.time_window_hours = time_window_hours
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

        # Event state
        self.events: Dict[UUID, NewsEvent] = {}
        self.cluster_to_event: Dict[int, UUID] = {}
        self.article_to_event: Dict[UUID, UUID] = {}

    async def group_into_events(
        self, articles: List[NormalizedArticle], features: Optional[np.ndarray] = None
    ) -> List[NewsEvent]:
        """Group articles into coherent news events.

        Args:
            articles: List of articles to group
            features: Precomputed features (optional)

        Returns:
            List of news events
        """
        if not articles:
            return []

        # Apply clustering
        cluster_labels = await self.clustering_engine.fit_predict(articles, features)

        # Create events from clusters
        events = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue

            # Get articles in cluster
            cluster_articles = [
                article for article, label in zip(articles, cluster_labels) if label == cluster_id
            ]

            # Create event
            event = await self._create_news_event(cluster_articles, cluster_id)
            if event:
                events.append(event)

                # Update state
                self.events[event.id] = event
                self.cluster_to_event[cluster_id] = event.id
                for article in cluster_articles:
                    self.article_to_event[article.id] = event.id

        return events

    async def incremental_group_articles(
        self, new_articles: List[NormalizedArticle], features: Optional[np.ndarray] = None
    ) -> List[NewsEvent]:
        """Incrementally group new articles into existing or new events.

        Args:
            new_articles: New articles to group
            features: Precomputed features (optional)

        Returns:
            List of updated or new events
        """
        if not new_articles:
            return []

        # Apply incremental clustering
        cluster_labels = await self.clustering_engine.incremental_fit(new_articles, features)

        # Process each article
        updated_events = []
        for article, cluster_id in zip(new_articles, cluster_labels):
            if cluster_id == -1:  # Noise point
                continue

            if cluster_id in self.cluster_to_event:
                # Add to existing event
                event_id = self.cluster_to_event[cluster_id]
                event = self.events[event_id]
                await self._add_article_to_event(event, article)
                updated_events.append(event)
            else:
                # Create new event
                event = await self._create_news_event([article], cluster_id)
                if event:
                    self.events[event.id] = event
                    self.cluster_to_event[cluster_id] = event.id
                    self.article_to_event[article.id] = event.id
                    updated_events.append(event)

        return updated_events

    async def _create_news_event(
        self, articles: List[NormalizedArticle], cluster_id: int
    ) -> Optional[NewsEvent]:
        """Create a news event from cluster articles.

        Args:
            articles: Articles in the cluster
            cluster_id: Cluster ID

        Returns:
            News event or None if invalid
        """
        if len(articles) < self.min_cluster_size:
            return None

        # Select representative article
        representative = await self._select_representative_article(articles)

        # Create cluster
        cluster = await self._create_cluster(articles, representative, cluster_id)

        # Compute event coherence
        temporal_coherence = await self._compute_temporal_coherence(articles)
        topical_coherence = await self._compute_topical_coherence(articles)

        # Generate event summary
        event_summary = await self._generate_event_summary(articles, representative)

        # Extract event keywords
        event_keywords = await self._extract_event_keywords(articles)

        # Create news event
        event = NewsEvent(
            id=UUID(int=cluster_id),  # Use cluster_id as event ID
            cluster=cluster,
            articles=articles,
            representative_article=representative,
            event_summary=event_summary,
            event_keywords=event_keywords,
            temporal_coherence=temporal_coherence,
            topical_coherence=topical_coherence,
        )

        return event

    async def _create_cluster(
        self, articles: List[NormalizedArticle], representative: NormalizedArticle, cluster_id: int
    ) -> Cluster:
        """Create cluster from articles.

        Args:
            articles: Articles in cluster
            representative: Representative article
            cluster_id: Cluster ID

        Returns:
            Cluster object
        """
        # Compute cluster quality score
        quality_score = await self._compute_cluster_quality(articles)

        # Extract common topics
        topics = await self._extract_common_topics(articles)

        # Extract common entities
        entities = await self._extract_common_entities(articles)

        # Extract common locations
        locations = await self._extract_common_locations(articles)

        # Compute time span
        time_span = await self._compute_time_span(articles)

        return Cluster(
            id=UUID(int=cluster_id),
            representative_article_id=representative.id,
            article_count=len(articles),
            quality_score=quality_score,
            topics=topics,
            entities=entities,
            locations=locations,
            time_span=time_span,
            is_active=True,
            created_at=min(article.published_at for article in articles),
            updated_at=max(article.published_at for article in articles),
        )

    async def _select_representative_article(
        self, articles: List[NormalizedArticle]
    ) -> NormalizedArticle:
        """Select representative article for cluster.

        Args:
            articles: Articles in cluster

        Returns:
            Representative article
        """
        if len(articles) == 1:
            return articles[0]

        # Compute representativeness scores
        scores = {}
        for article in articles:
            score = await self._compute_representative_score(article, articles)
            scores[article.id] = score

        # Select article with highest score
        best_article_id = max(scores, key=scores.get)
        return next(a for a in articles if a.id == best_article_id)

    async def _compute_representative_score(
        self, candidate: NormalizedArticle, cluster_articles: List[NormalizedArticle]
    ) -> float:
        """Compute representativeness score for article.

        Args:
            candidate: Candidate article
            cluster_articles: All articles in cluster

        Returns:
            Representativeness score
        """
        # Centrality score
        centrality = await self._compute_centrality(candidate, cluster_articles)

        # Quality score
        quality = candidate.quality_score

        # Freshness score
        freshness = self._compute_freshness_score(candidate.published_at)

        # Source authority score
        source_authority = await self._get_source_authority(candidate.source)

        # Content completeness score
        completeness = self._compute_completeness_score(candidate)

        # Weighted combination
        weights = {
            "centrality": 0.3,
            "quality": 0.25,
            "freshness": 0.2,
            "source_authority": 0.15,
            "completeness": 0.1,
        }

        score = (
            weights["centrality"] * centrality
            + weights["quality"] * quality
            + weights["freshness"] * freshness
            + weights["source_authority"] * source_authority
            + weights["completeness"] * completeness
        )

        return score

    async def _compute_centrality(
        self, candidate: NormalizedArticle, cluster_articles: List[NormalizedArticle]
    ) -> float:
        """Compute centrality score for article.

        Args:
            candidate: Candidate article
            cluster_articles: All articles in cluster

        Returns:
            Centrality score
        """
        if len(cluster_articles) <= 1:
            return 1.0

        # Compute average similarity to other articles
        similarities = []
        for other_article in cluster_articles:
            if other_article.id != candidate.id:
                similarity = await self.semantic_calc.compute_similarity(candidate, other_article)
                similarities.append(similarity)

        if not similarities:
            return 0.0

        return np.mean(similarities)

    def _compute_freshness_score(self, published_at: datetime) -> float:
        """Compute freshness score for article.

        Args:
            published_at: Article publication time

        Returns:
            Freshness score
        """
        now = datetime.now(published_at.tzinfo)
        age_hours = (now - published_at).total_seconds() / 3600

        # Exponential decay with 24-hour half-life
        return np.exp(-age_hours * np.log(2) / 24)

    async def _get_source_authority(self, source: str) -> float:
        """Get source authority score.

        Args:
            source: Source name

        Returns:
            Authority score
        """
        # Placeholder - would use actual source authority data
        authority_scores = {
            "reuters": 0.9,
            "ap": 0.9,
            "bbc": 0.8,
            "cnn": 0.8,
            "nytimes": 0.8,
            "washingtonpost": 0.8,
            "guardian": 0.7,
            "independent": 0.6,
        }

        return authority_scores.get(source.lower(), 0.5)

    def _compute_completeness_score(self, article: NormalizedArticle) -> float:
        """Compute content completeness score.

        Args:
            article: Article to score

        Returns:
            Completeness score
        """
        score = 0.0

        # Title completeness
        if article.title and len(article.title) > 10:
            score += 0.2

        # Content completeness
        if article.content and len(article.content) > 100:
            score += 0.3

        # Summary completeness
        if article.summary and len(article.summary) > 20:
            score += 0.2

        # Entity completeness
        if article.entities:
            score += 0.1

        # Topic completeness
        if article.topics:
            score += 0.1

        # Location completeness
        if article.locations:
            score += 0.1

        return min(score, 1.0)

    async def _compute_temporal_coherence(self, articles: List[NormalizedArticle]) -> float:
        """Compute temporal coherence of articles.

        Args:
            articles: Articles in cluster

        Returns:
            Temporal coherence score
        """
        if len(articles) <= 1:
            return 1.0

        # Compute time differences
        times = [article.published_at for article in articles]
        times.sort()

        # Check if all articles are within time window
        time_span = (times[-1] - times[0]).total_seconds() / 3600  # hours
        if time_span <= self.time_window_hours:
            return 1.0

        # Compute coherence based on time distribution
        coherence = max(0.0, 1.0 - (time_span - self.time_window_hours) / self.time_window_hours)
        return coherence

    async def _compute_topical_coherence(self, articles: List[NormalizedArticle]) -> float:
        """Compute topical coherence of articles.

        Args:
            articles: Articles in cluster

        Returns:
            Topical coherence score
        """
        if len(articles) <= 1:
            return 1.0

        # Compute pairwise similarities
        similarities = []
        for i, article1 in enumerate(articles):
            for article2 in articles[i + 1 :]:
                similarity = await self.semantic_calc.compute_similarity(article1, article2)
                similarities.append(similarity)

        if not similarities:
            return 0.0

        return np.mean(similarities)

    async def _compute_cluster_quality(self, articles: List[NormalizedArticle]) -> float:
        """Compute cluster quality score.

        Args:
            articles: Articles in cluster

        Returns:
            Quality score
        """
        if not articles:
            return 0.0

        # Average article quality
        avg_quality = np.mean([article.quality_score for article in articles])

        # Cluster size factor
        size_factor = min(len(articles) / self.min_cluster_size, 1.0)

        # Temporal coherence
        temporal_coherence = await self._compute_temporal_coherence(articles)

        # Topical coherence
        topical_coherence = await self._compute_topical_coherence(articles)

        # Weighted combination
        quality = (
            0.4 * avg_quality
            + 0.2 * size_factor
            + 0.2 * temporal_coherence
            + 0.2 * topical_coherence
        )

        return min(quality, 1.0)

    async def _extract_common_topics(self, articles: List[NormalizedArticle]) -> List[Topic]:
        """Extract common topics from articles.

        Args:
            articles: Articles in cluster

        Returns:
            List of common topics
        """
        # Count topic occurrences
        topic_counts = {}
        for article in articles:
            for topic in article.topics:
                key = topic.name.lower()
                if key not in topic_counts:
                    topic_counts[key] = {"topic": topic, "count": 0}
                topic_counts[key]["count"] += 1

        # Filter by frequency threshold
        threshold = max(1, len(articles) // 2)
        common_topics = []

        for topic_data in topic_counts.values():
            if topic_data["count"] >= threshold:
                # Update confidence based on frequency
                topic = topic_data["topic"]
                topic.confidence = min(topic.confidence * topic_data["count"] / len(articles), 1.0)
                common_topics.append(topic)

        # Sort by confidence
        common_topics.sort(key=lambda x: x.confidence, reverse=True)

        return common_topics[:10]  # Return top 10 topics

    async def _extract_common_entities(self, articles: List[NormalizedArticle]) -> List[Entity]:
        """Extract common entities from articles.

        Args:
            articles: Articles in cluster

        Returns:
            List of common entities
        """
        # Count entity occurrences
        entity_counts = {}
        for article in articles:
            for entity in article.entities:
                key = entity.text.lower()
                if key not in entity_counts:
                    entity_counts[key] = {"entity": entity, "count": 0}
                entity_counts[key]["count"] += 1

        # Filter by frequency threshold
        threshold = max(1, len(articles) // 2)
        common_entities = []

        for entity_data in entity_counts.values():
            if entity_data["count"] >= threshold:
                # Update confidence based on frequency
                entity = entity_data["entity"]
                entity.confidence = min(
                    entity.confidence * entity_data["count"] / len(articles), 1.0
                )
                common_entities.append(entity)

        # Sort by confidence
        common_entities.sort(key=lambda x: x.confidence, reverse=True)

        return common_entities[:20]  # Return top 20 entities

    async def _extract_common_locations(self, articles: List[NormalizedArticle]) -> List[Location]:
        """Extract common locations from articles.

        Args:
            articles: Articles in cluster

        Returns:
            List of common locations
        """
        # Count location occurrences
        location_counts = {}
        for article in articles:
            for location in article.locations:
                key = location.name.lower()
                if key not in location_counts:
                    location_counts[key] = {"location": location, "count": 0}
                location_counts[key]["count"] += 1

        # Filter by frequency threshold
        threshold = max(1, len(articles) // 2)
        common_locations = []

        for location_data in location_counts.values():
            if location_data["count"] >= threshold:
                # Update confidence based on frequency
                location = location_data["location"]
                location.confidence = min(
                    location.confidence * location_data["count"] / len(articles), 1.0
                )
                common_locations.append(location)

        # Sort by confidence
        common_locations.sort(key=lambda x: x.confidence, reverse=True)

        return common_locations[:10]  # Return top 10 locations

    async def _compute_time_span(self, articles: List[NormalizedArticle]) -> int:
        """Compute time span of articles in seconds.

        Args:
            articles: Articles in cluster

        Returns:
            Time span in seconds
        """
        if len(articles) <= 1:
            return 0

        times = [article.published_at for article in articles]
        return int((max(times) - min(times)).total_seconds())

    async def _generate_event_summary(
        self, articles: List[NormalizedArticle], representative: NormalizedArticle
    ) -> str:
        """Generate event summary.

        Args:
            articles: Articles in cluster
            representative: Representative article

        Returns:
            Event summary
        """
        # Use representative article summary as base
        if representative.summary:
            return representative.summary

        # Fallback to title
        return representative.title

    async def _extract_event_keywords(self, articles: List[NormalizedArticle]) -> List[str]:
        """Extract event keywords.

        Args:
            articles: Articles in cluster

        Returns:
            List of keywords
        """
        # Extract keywords from titles
        keywords = set()
        for article in articles:
            # Simple keyword extraction (would use NLP in production)
            words = article.title.lower().split()
            keywords.update(word for word in words if len(word) > 3)

        # Return top keywords
        return list(keywords)[:10]

    async def _add_article_to_event(self, event: NewsEvent, article: NormalizedArticle) -> None:
        """Add article to existing event.

        Args:
            event: Existing event
            article: Article to add
        """
        # Add article to event
        event.articles.append(article)

        # Update cluster
        event.cluster.article_count += 1
        event.cluster.updated_at = max(event.cluster.updated_at, article.published_at)

        # Update state
        self.article_to_event[article.id] = event.id

    def get_event_quality_metrics(self) -> Dict[UUID, Dict[str, float]]:
        """Get quality metrics for all events.

        Returns:
            Dictionary with event quality metrics
        """
        metrics = {}

        for event_id, event in self.events.items():
            event_metrics = {
                "article_count": len(event.articles),
                "temporal_coherence": event.temporal_coherence,
                "topical_coherence": event.topical_coherence,
                "quality_score": event.cluster.quality_score,
                "time_span_hours": event.cluster.time_span / 3600 if event.cluster.time_span else 0,
                "keyword_count": len(event.event_keywords),
            }

            metrics[event_id] = event_metrics

        return metrics
