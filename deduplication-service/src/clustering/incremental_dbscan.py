"""Incremental DBSCAN clustering implementation."""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..models.schemas import NormalizedArticle, Cluster, NewsEvent, Entity, Topic, Location


class IncrementalDBSCAN:
    """Incremental DBSCAN clustering for streaming data."""
    
    def __init__(
        self,
        eps: float = 0.3,
        min_samples: int = 2,
        max_cluster_size: int = 100,
        temporal_decay_half_life_hours: int = 24
    ):
        """Initialize incremental DBSCAN.
        
        Args:
            eps: Maximum distance between samples in the same neighborhood
            min_samples: Minimum number of samples in a neighborhood
            max_cluster_size: Maximum size of a cluster
            temporal_decay_half_life_hours: Half-life for temporal decay in hours
        """
        self.eps = eps
        self.min_samples = min_samples
        self.max_cluster_size = max_cluster_size
        self.temporal_decay_half_life = temporal_decay_half_lours * 3600  # Convert to seconds
        
        # Clustering state
        self.clusters: Dict[int, Set[UUID]] = {}
        self.cluster_centers: Dict[int, np.ndarray] = {}
        self.cluster_metadata: Dict[int, Dict] = {}
        self.noise_points: Set[UUID] = set()
        
        # Feature scaler
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    async def fit_predict(
        self,
        articles: List[NormalizedArticle],
        features: Optional[np.ndarray] = None
    ) -> List[int]:
        """Fit clustering model and predict cluster labels.
        
        Args:
            articles: List of articles to cluster
            features: Precomputed features (optional)
            
        Returns:
            List of cluster labels (-1 for noise)
        """
        if features is None:
            features = await self._compute_features(articles)
        
        # Scale features
        if not self.is_fitted:
            features_scaled = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            features_scaled = self.scaler.transform(features)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(features_scaled)
        
        # Update clustering state
        await self._update_clustering_state(articles, cluster_labels, features_scaled)
        
        return cluster_labels.tolist()
    
    async def incremental_fit(
        self,
        new_articles: List[NormalizedArticle],
        features: Optional[np.ndarray] = None
    ) -> List[int]:
        """Incrementally fit new articles to existing clusters.
        
        Args:
            new_articles: New articles to cluster
            features: Precomputed features (optional)
            
        Returns:
            List of cluster labels (-1 for noise)
        """
        if features is None:
            features = await self._compute_features(new_articles)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Find nearest clusters for each new article
        cluster_labels = []
        
        for i, article in enumerate(new_articles):
            article_features = features_scaled[i].reshape(1, -1)
            
            # Find nearest existing cluster
            nearest_cluster = await self._find_nearest_cluster(article_features)
            
            if nearest_cluster is not None:
                # Add to existing cluster
                self.clusters[nearest_cluster].add(article.id)
                cluster_labels.append(nearest_cluster)
                
                # Update cluster center
                await self._update_cluster_center(nearest_cluster, article_features[0])
            else:
                # Check if it can form a new cluster with nearby articles
                new_cluster_id = await self._try_create_new_cluster(
                    article, article_features, new_articles
                )
                
                if new_cluster_id is not None:
                    cluster_labels.append(new_cluster_id)
                else:
                    # Add to noise
                    self.noise_points.add(article.id)
                    cluster_labels.append(-1)
        
        return cluster_labels
    
    async def _compute_features(self, articles: List[NormalizedArticle]) -> np.ndarray:
        """Compute feature vectors for articles.
        
        Args:
            articles: List of articles
            
        Returns:
            Feature matrix
        """
        features = []
        
        for article in articles:
            # Semantic features (placeholder - would use actual embeddings)
            semantic_features = np.random.random(384)  # 384-dim embedding
            
            # Temporal features
            temporal_features = await self._compute_temporal_features(article)
            
            # Entity features
            entity_features = await self._compute_entity_features(article.entities)
            
            # Topic features
            topic_features = await self._compute_topic_features(article.topics)
            
            # Geographic features
            geo_features = await self._compute_geo_features(article.locations)
            
            # Combine all features
            combined_features = np.concatenate([
                semantic_features,
                temporal_features,
                entity_features,
                topic_features,
                geo_features
            ])
            
            features.append(combined_features)
        
        return np.array(features)
    
    async def _compute_temporal_features(self, article: NormalizedArticle) -> np.ndarray:
        """Compute temporal features for article.
        
        Args:
            article: Article to compute features for
            
        Returns:
            Temporal feature vector
        """
        import time
        from datetime import datetime
        
        # Current time
        current_time = time.time()
        article_time = article.published_at.timestamp()
        
        # Time difference in hours
        time_diff_hours = (current_time - article_time) / 3600
        
        # Temporal decay
        decay_factor = np.exp(-time_diff_hours * np.log(2) / self.temporal_decay_half_life)
        
        # Day of week (0-6)
        day_of_week = article.published_at.weekday()
        
        # Hour of day (0-23)
        hour_of_day = article.published_at.hour
        
        return np.array([
            decay_factor,
            np.sin(2 * np.pi * day_of_week / 7),  # Cyclical encoding
            np.cos(2 * np.pi * day_of_week / 7),
            np.sin(2 * np.pi * hour_of_day / 24),
            np.cos(2 * np.pi * hour_of_day / 24)
        ])
    
    async def _compute_entity_features(self, entities: List[Entity]) -> np.ndarray:
        """Compute entity-based features.
        
        Args:
            entities: List of entities
            
        Returns:
            Entity feature vector
        """
        if not entities:
            return np.zeros(10)
        
        # Entity type distribution
        entity_types = {}
        for entity in entities:
            entity_types[entity.label] = entity_types.get(entity.label, 0) + 1
        
        # Create feature vector
        features = np.zeros(10)
        common_types = ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART']
        
        for i, entity_type in enumerate(common_types):
            if i < len(features):
                features[i] = entity_types.get(entity_type, 0)
        
        # Entity count
        features[5] = len(entities)
        
        # Average confidence
        features[6] = np.mean([e.confidence for e in entities]) if entities else 0
        
        # Entity diversity (unique types)
        features[7] = len(set(e.label for e in entities))
        
        # Longest entity length
        features[8] = max(len(e.text) for e in entities) if entities else 0
        
        # Entity density
        features[9] = len(entities) / 100  # Normalize by 100
        
        return features
    
    async def _compute_topic_features(self, topics: List[Topic]) -> np.ndarray:
        """Compute topic-based features.
        
        Args:
            topics: List of topics
            
        Returns:
            Topic feature vector
        """
        if not topics:
            return np.zeros(5)
        
        # Topic count
        topic_count = len(topics)
        
        # Average confidence
        avg_confidence = np.mean([t.confidence for t in topics])
        
        # Topic diversity
        unique_topics = len(set(t.name for t in topics))
        
        # High confidence topics
        high_conf_topics = sum(1 for t in topics if t.confidence > 0.8)
        
        # Topic entropy
        topic_probs = [t.confidence for t in topics]
        topic_probs = np.array(topic_probs) / sum(topic_probs)
        entropy = -np.sum(topic_probs * np.log(topic_probs + 1e-10))
        
        return np.array([
            topic_count,
            avg_confidence,
            unique_topics,
            high_conf_topics,
            entropy
        ])
    
    async def _compute_geo_features(self, locations: List[Location]) -> np.ndarray:
        """Compute geographic features.
        
        Args:
            locations: List of locations
            
        Returns:
            Geographic feature vector
        """
        if not locations:
            return np.zeros(5)
        
        # Location count
        location_count = len(locations)
        
        # Countries represented
        countries = set(loc.country for loc in locations if loc.country)
        country_count = len(countries)
        
        # Regions represented
        regions = set(loc.region for loc in locations if loc.region)
        region_count = len(regions)
        
        # Average confidence
        avg_confidence = np.mean([loc.confidence for loc in locations])
        
        # Geographic spread (placeholder)
        geo_spread = 1.0 if location_count > 1 else 0.0
        
        return np.array([
            location_count,
            country_count,
            region_count,
            avg_confidence,
            geo_spread
        ])
    
    async def _update_clustering_state(
        self,
        articles: List[NormalizedArticle],
        cluster_labels: np.ndarray,
        features: np.ndarray
    ) -> None:
        """Update clustering state with new results.
        
        Args:
            articles: List of articles
            cluster_labels: Cluster labels
            features: Feature matrix
        """
        # Clear existing state
        self.clusters.clear()
        self.cluster_centers.clear()
        self.cluster_metadata.clear()
        self.noise_points.clear()
        
        # Update clusters
        for i, (article, label) in enumerate(zip(articles, cluster_labels)):
            if label == -1:
                self.noise_points.add(article.id)
            else:
                if label not in self.clusters:
                    self.clusters[label] = set()
                self.clusters[label].add(article.id)
        
        # Compute cluster centers
        for cluster_id, article_ids in self.clusters.items():
            cluster_indices = [i for i, article in enumerate(articles) 
                             if article.id in article_ids]
            cluster_features = features[cluster_indices]
            self.cluster_centers[cluster_id] = np.mean(cluster_features, axis=0)
            
            # Store metadata
            self.cluster_metadata[cluster_id] = {
                'size': len(article_ids),
                'created_at': articles[0].published_at if articles else None
            }
    
    async def _find_nearest_cluster(self, article_features: np.ndarray) -> Optional[int]:
        """Find nearest cluster for article.
        
        Args:
            article_features: Article feature vector
            
        Returns:
            Nearest cluster ID or None
        """
        if not self.cluster_centers:
            return None
        
        min_distance = float('inf')
        nearest_cluster = None
        
        for cluster_id, center in self.cluster_centers.items():
            distance = np.linalg.norm(article_features - center)
            if distance < min_distance and distance < self.eps:
                min_distance = distance
                nearest_cluster = cluster_id
        
        return nearest_cluster
    
    async def _try_create_new_cluster(
        self,
        article: NormalizedArticle,
        article_features: np.ndarray,
        all_articles: List[NormalizedArticle]
    ) -> Optional[int]:
        """Try to create new cluster with nearby articles.
        
        Args:
            article: Article to cluster
            article_features: Article feature vector
            all_articles: All articles in current batch
            
        Returns:
            New cluster ID or None
        """
        # Find nearby articles
        nearby_articles = []
        for other_article in all_articles:
            if other_article.id == article.id:
                continue
            
            # Compute distance (placeholder)
            distance = np.random.random()  # Would compute actual distance
            
            if distance < self.eps:
                nearby_articles.append(other_article)
        
        # Check if we have enough samples for a new cluster
        if len(nearby_articles) >= self.min_samples - 1:
            # Create new cluster
            new_cluster_id = max(self.clusters.keys(), default=-1) + 1
            self.clusters[new_cluster_id] = {article.id}
            
            # Add nearby articles
            for nearby_article in nearby_articles:
                self.clusters[new_cluster_id].add(nearby_article.id)
            
            # Compute cluster center
            cluster_articles = [article] + nearby_articles
            cluster_features = np.array([article_features[0]] + 
                                      [np.random.random(article_features.shape[1]) 
                                       for _ in nearby_articles])
            self.cluster_centers[new_cluster_id] = np.mean(cluster_features, axis=0)
            
            # Store metadata
            self.cluster_metadata[new_cluster_id] = {
                'size': len(self.clusters[new_cluster_id]),
                'created_at': article.published_at
            }
            
            return new_cluster_id
        
        return None
    
    async def _update_cluster_center(
        self,
        cluster_id: int,
        new_features: np.ndarray
    ) -> None:
        """Update cluster center with new article.
        
        Args:
            cluster_id: Cluster ID
            new_features: New article features
        """
        if cluster_id not in self.cluster_centers:
            self.cluster_centers[cluster_id] = new_features
        else:
            # Update center incrementally
            current_center = self.cluster_centers[cluster_id]
            cluster_size = len(self.clusters[cluster_id])
            
            # Weighted average
            new_center = (current_center * (cluster_size - 1) + new_features) / cluster_size
            self.cluster_centers[cluster_id] = new_center
    
    def get_cluster_quality_metrics(self) -> Dict[int, Dict[str, float]]:
        """Get quality metrics for all clusters.
        
        Returns:
            Dictionary with cluster quality metrics
        """
        metrics = {}
        
        for cluster_id, article_ids in self.clusters.items():
            cluster_size = len(article_ids)
            
            # Basic metrics
            cluster_metrics = {
                'size': cluster_size,
                'density': cluster_size / self.max_cluster_size,
                'is_valid': cluster_size >= self.min_samples,
                'is_oversized': cluster_size > self.max_cluster_size
            }
            
            metrics[cluster_id] = cluster_metrics
        
        return metrics
    
    def get_noise_ratio(self) -> float:
        """Get ratio of noise points to total points.
        
        Returns:
            Noise ratio
        """
        total_points = len(self.noise_points) + sum(len(ids) for ids in self.clusters.values())
        if total_points == 0:
            return 0.0
        
        return len(self.noise_points) / total_points
