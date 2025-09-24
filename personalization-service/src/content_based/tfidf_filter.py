"""TF-IDF based content filtering."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config.settings import settings
from ..models.schemas import ContentItem, UserProfile

logger = structlog.get_logger()


class TFIDFContentFilter:
    """TF-IDF based content filtering."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=settings.tfidf_max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        self.content_vectors: Optional[np.ndarray] = None
        self.content_ids: List[str] = []
        self.is_fitted = False

    async def fit(self, content_items: List[ContentItem]) -> None:
        """Fit the TF-IDF vectorizer on content items."""
        if not content_items:
            logger.warning("No content items provided for TF-IDF fitting")
            return

        # Prepare text content
        texts = []
        content_ids = []

        for item in content_items:
            # Combine title and content
            text = f"{item.title} {item.content or ''}"
            texts.append(text)
            content_ids.append(item.id)

        # Fit vectorizer
        self.content_vectors = self.vectorizer.fit_transform(texts)
        self.content_ids = content_ids
        self.is_fitted = True

        logger.info(f"TF-IDF vectorizer fitted on {len(content_items)} content items")

    async def compute_similarities(self, user_profile: UserProfile, candidates: List[ContentItem]) -> List[float]:
        """Compute content-based similarity scores for candidates."""
        if not self.is_fitted:
            logger.warning("TF-IDF vectorizer not fitted")
            return [0.5] * len(candidates)

        # Build user profile vector
        user_vector = await self._build_user_profile_vector(user_profile)

        if user_vector is None:
            return [0.5] * len(candidates)

        # Compute similarities
        similarities = []

        for candidate in candidates:
            if candidate.id in self.content_ids:
                # Get content vector
                content_idx = self.content_ids.index(candidate.id)
                content_vector = self.content_vectors[content_idx]

                # Compute cosine similarity
                similarity = cosine_similarity([user_vector], [content_vector])[0][0]
                similarities.append(float(similarity))
            else:
                # New content - use topic-based similarity
                topic_similarity = await self._compute_topic_similarity(user_profile, candidate)
                similarities.append(topic_similarity)

        return similarities

    async def _build_user_profile_vector(self, user_profile: UserProfile) -> Optional[np.ndarray]:
        """Build user profile vector from preferences."""
        if not user_profile.topic_preferences:
            return None

        # Create a text representation of user preferences
        preference_text = []

        # Add topic preferences
        for topic, weight in user_profile.topic_preferences.items():
            # Repeat topic based on weight
            count = max(1, int(weight * 10))
            preference_text.extend([topic] * count)

        # Add source preferences
        for source, weight in user_profile.source_preferences.items():
            count = max(1, int(weight * 5))
            preference_text.extend([source] * count)

        if not preference_text:
            return None

        # Transform to vector
        preference_text_str = " ".join(preference_text)
        user_vector = self.vectorizer.transform([preference_text_str])

        return user_vector.toarray()[0]

    async def _compute_topic_similarity(self, user_profile: UserProfile, candidate: ContentItem) -> float:
        """Compute topic-based similarity for new content."""
        if not user_profile.topic_preferences or not candidate.topics:
            return 0.5

        # Compute weighted topic similarity
        total_similarity = 0.0
        total_weight = 0.0

        for candidate_topic in candidate.topics:
            best_match = 0.0
            for user_topic, weight in user_profile.topic_preferences.items():
                # Simple string similarity (in production, use semantic
                # similarity)
                similarity = self._string_similarity(candidate_topic, user_topic)
                weighted_similarity = similarity * weight
                best_match = max(best_match, weighted_similarity)

            total_similarity += best_match
            total_weight += 1.0

        if total_weight == 0:
            return 0.5

        return total_similarity / total_weight

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute string similarity using Jaccard similarity."""
        s1_words = set(s1.lower().split())
        s2_words = set(s2.lower().split())

        if not s1_words and not s2_words:
            return 1.0
        if not s1_words or not s2_words:
            return 0.0

        intersection = len(s1_words.intersection(s2_words))
        union = len(s1_words.union(s2_words))

        return intersection / union if union > 0 else 0.0

    async def get_similar_content(self, content_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """Get similar content based on TF-IDF similarity."""
        if not self.is_fitted or content_id not in self.content_ids:
            return []

        content_idx = self.content_ids.index(content_id)
        content_vector = self.content_vectors[content_idx]

        # Compute similarities with all content
        similarities = cosine_similarity([content_vector], self.content_vectors)[0]

        # Get top similar content
        similar_indices = np.argsort(similarities)[-n_similar - 1 : -1][::-1]

        similar_content = []
        for idx in similar_indices:
            if idx != content_idx:  # Exclude self
                similar_content.append((self.content_ids[idx], float(similarities[idx])))

        return similar_content

    async def get_content_features(self, content_item: ContentItem) -> Dict[str, float]:
        """Extract content features for a given item."""
        if not self.is_fitted:
            return {}

        # Combine title and content
        text = f"{content_item.title} {content_item.content or ''}"

        # Transform to vector
        content_vector = self.vectorizer.transform([text])

        # Get feature names and values
        feature_names = self.vectorizer.get_feature_names_out()
        feature_values = content_vector.toarray()[0]

        # Create feature dictionary
        features = {}
        for name, value in zip(feature_names, feature_values):
            if value > 0:
                features[f"tfidf_{name}"] = float(value)

        return features

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the TF-IDF model."""
        return {
            "is_fitted": self.is_fitted,
            "n_content_items": len(self.content_ids),
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.is_fitted else 0,
            "max_features": settings.tfidf_max_features,
        }
