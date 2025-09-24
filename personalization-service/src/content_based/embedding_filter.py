"""Content-based filtering using embeddings."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import structlog
import asyncio
from collections import Counter

from ..models.schemas import ContentItem, UserProfile, Recommendation
from ..config.settings import settings

logger = structlog.get_logger()


class EmbeddingContentFilter:
    """Content-based filtering using sentence embeddings."""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.content_embeddings: Optional[np.ndarray] = None
        self.content_ids: List[str] = []
        self.is_fitted = False
        
    async def fit(self, content_items: List[ContentItem]) -> None:
        """Fit the embedding model on content items."""
        if not content_items:
            logger.warning("No content items provided for embedding fitting")
            return
            
        # Prepare text content
        texts = []
        content_ids = []
        
        for item in content_items:
            # Combine title and content
            text = f"{item.title} {item.content or ''}"
            texts.append(text)
            content_ids.append(item.id)
        
        # Generate embeddings
        self.content_embeddings = self.model.encode(texts)
        self.content_ids = content_ids
        self.is_fitted = True
        
        logger.info(f"Embedding model fitted on {len(content_items)} content items")
    
    async def compute_similarities(self, user_profile: UserProfile, 
                                 candidates: List[ContentItem]) -> List[float]:
        """Compute content-based similarity scores for candidates."""
        if not self.is_fitted:
            logger.warning("Embedding model not fitted")
            return [0.5] * len(candidates)
        
        # Build user profile embedding
        user_embedding = await self._build_user_profile_embedding(user_profile)
        
        if user_embedding is None:
            return [0.5] * len(candidates)
        
        # Compute similarities
        similarities = []
        
        for candidate in candidates:
            if candidate.id in self.content_ids:
                # Get content embedding
                content_idx = self.content_ids.index(candidate.id)
                content_embedding = self.content_embeddings[content_idx]
                
                # Compute cosine similarity
                similarity = cosine_similarity([user_embedding], [content_embedding])[0][0]
                similarities.append(float(similarity))
            else:
                # New content - generate embedding on the fly
                candidate_embedding = await self._generate_content_embedding(candidate)
                similarity = cosine_similarity([user_embedding], [candidate_embedding])[0][0]
                similarities.append(float(similarity))
        
        return similarities
    
    async def _build_user_profile_embedding(self, user_profile: UserProfile) -> Optional[np.ndarray]:
        """Build user profile embedding from preferences."""
        if not user_profile.topic_preferences:
            return None
        
        # Create a text representation of user preferences
        preference_texts = []
        weights = []
        
        # Add topic preferences
        for topic, weight in user_profile.topic_preferences.items():
            preference_texts.append(topic)
            weights.append(weight)
        
        # Add source preferences
        for source, weight in user_profile.source_preferences.items():
            preference_texts.append(source)
            weights.append(weight * 0.5)  # Lower weight for sources
        
        if not preference_texts:
            return None
        
        # Generate embeddings for each preference
        preference_embeddings = self.model.encode(preference_texts)
        
        # Weighted average of embeddings
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        user_embedding = np.average(preference_embeddings, axis=0, weights=weights)
        
        return user_embedding
    
    async def _generate_content_embedding(self, content_item: ContentItem) -> np.ndarray:
        """Generate embedding for a content item."""
        text = f"{content_item.title} {content_item.content or ''}"
        return self.model.encode([text])[0]
    
    async def get_similar_content(self, content_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """Get similar content based on embedding similarity."""
        if not self.is_fitted or content_id not in self.content_ids:
            return []
        
        content_idx = self.content_ids.index(content_id)
        content_embedding = self.content_embeddings[content_idx]
        
        # Compute similarities with all content
        similarities = cosine_similarity([content_embedding], self.content_embeddings)[0]
        
        # Get top similar content
        similar_indices = np.argsort(similarities)[-n_similar-1:-1][::-1]
        
        similar_content = []
        for idx in similar_indices:
            if idx != content_idx:  # Exclude self
                similar_content.append((
                    self.content_ids[idx],
                    float(similarities[idx])
                ))
        
        return similar_content
    
    async def get_content_features(self, content_item: ContentItem) -> Dict[str, float]:
        """Extract content features for a given item."""
        # Generate embedding
        embedding = await self._generate_content_embedding(content_item)
        
        # Create feature dictionary with top dimensions
        features = {}
        for i, value in enumerate(embedding):
            if abs(value) > 0.1:  # Only include significant values
                features[f"embedding_dim_{i}"] = float(value)
        
        return features
    
    async def compute_topic_similarity(self, user_profile: UserProfile, 
                                     candidate: ContentItem) -> float:
        """Compute topic-based similarity using embeddings."""
        if not user_profile.topic_preferences or not candidate.topics:
            return 0.5
        
        # Generate embeddings for user topics and candidate topics
        user_topics = list(user_profile.topic_preferences.keys())
        candidate_topics = candidate.topics
        
        if not user_topics or not candidate_topics:
            return 0.5
        
        # Generate embeddings
        user_topic_embeddings = self.model.encode(user_topics)
        candidate_topic_embeddings = self.model.encode(candidate_topics)
        
        # Compute similarities
        similarities = cosine_similarity(user_topic_embeddings, candidate_topic_embeddings)
        
        # Get maximum similarity for each candidate topic
        max_similarities = np.max(similarities, axis=0)
        
        # Weight by user preferences
        weights = []
        for topic in candidate_topics:
            weight = user_profile.topic_preferences.get(topic, 0.0)
            weights.append(weight)
        
        if not weights or sum(weights) == 0:
            return float(np.mean(max_similarities))
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return float(np.average(max_similarities, weights=weights))
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the embedding model."""
        return {
            "is_fitted": self.is_fitted,
            "n_content_items": len(self.content_ids),
            "embedding_dim": settings.content_embedding_dim,
            "model_name": "all-MiniLM-L6-v2"
        }
