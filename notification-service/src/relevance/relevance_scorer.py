"""
Content relevance scoring for notification decisioning.
"""

import asyncio
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ..exceptions import RelevanceScoringError
from ..models.schemas import NotificationCandidate, NotificationPreferences, UserProfile

logger = logging.getLogger(__name__)


class PersonalizationEngine:
    """Personalization engine for content relevance."""

    def __init__(self):
        self.engagement_model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    async def initialize(self) -> None:
        """Initialize personalization engine."""
        logger.info("Initializing personalization engine")

        # Try to load existing model
        await self._load_engagement_model()

        # If no model exists, create and train a new one
        if not self.is_trained:
    await self._create_and_train_engagement_model()

        logger.info("Personalization engine initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup personalization engine."""
        logger.info("Cleaning up personalization engine")
        # Model cleanup if needed
        pass

    async def predict_engagement(
            self,
            user_profile: UserProfile,
            content) -> float:
        """Predict user engagement with content."""

        try:
            # Extract features for engagement prediction
            features = await self._extract_engagement_features(user_profile, content)

            # Predict engagement score
            if self.engagement_model and self.is_trained:
                feature_vector = np.array([list(features.values())])
                feature_vector_scaled = self.scaler.transform(feature_vector)
                engagement_score = self.engagement_model.predict(
                    feature_vector_scaled)[0]
            else:
                # Fallback to heuristic-based prediction
                engagement_score = self._heuristic_engagement_prediction(
                    features)

            return max(0.0, min(1.0, engagement_score))

        except Exception as e:
            logger.error(f"Error predicting engagement: {str(e)}")
            return 0.5  # Default neutral score

    async def _extract_engagement_features(
        self, user_profile: UserProfile, content
    ) -> Dict[str, Any]:
    """Extract features for engagement prediction."""
        features = {}

        # User profile features
        features["user_engagement_history"] = len(
            user_profile.engagement_history)
        features["user_topic_diversity"] = len(user_profile.topic_preferences)
        features["user_source_diversity"] = len(
            user_profile.source_preferences)

        # Content features
        features["content_topics_count"] = len(content.topics)
        features["content_locations_count"] = len(content.locations)
        features["content_urgency"] = content.urgency_score
        features["content_is_breaking"] = 1 if content.is_breaking else 0

        # Topic alignment
        topic_alignment = self._calculate_topic_alignment(
            content.topics, user_profile.topic_preferences
        )
        features["topic_alignment"] = topic_alignment

        # Source alignment
        source_alignment = self._calculate_source_alignment(
            content.source, user_profile.source_preferences
        )
        features["source_alignment"] = source_alignment

        # Location alignment
        location_alignment = self._calculate_location_alignment(
            content.locations, user_profile.location_preferences
        )
        features["location_alignment"] = location_alignment

        # Time-based features
        features["content_age_hours"] = self._calculate_content_age(
            content.published_at)
        features["is_recent_content"] = 1 if features["content_age_hours"] < 24 else 0

        return features

    def _calculate_topic_alignment(
        self, content_topics: List[str], user_topics: List[str]
    ) -> float:
        """Calculate topic alignment between content and user preferences."""

        if not content_topics or not user_topics:
            return 0.0

        # Calculate Jaccard similarity
        content_set = set(content_topics)
        user_set = set(user_topics)

        intersection = len(content_set.intersection(user_set))
        union = len(content_set.union(user_set))

        return intersection / union if union > 0 else 0.0

    def _calculate_source_alignment(
            self,
            content_source: str,
            user_sources: List[str]) -> float:
        """Calculate source alignment between content and user preferences."""

        if not user_sources:
            return 0.5  # Neutral if no preferences

        return 1.0 if content_source in user_sources else 0.0

    def _calculate_location_alignment(
        self, content_locations: List[str], user_locations: List[str]
    ) -> float:
        """Calculate location alignment between content and user preferences."""

        if not content_locations or not user_locations:
            return 0.5  # Neutral if no location data

        # Calculate overlap
        content_set = set(content_locations)
        user_set = set(user_locations)

        intersection = len(content_set.intersection(user_set))
        return min(intersection / len(content_locations), 1.0)

    def _calculate_content_age(self, published_at: datetime) -> float:
        """Calculate content age in hours."""
        now = datetime.utcnow()
        return (now - published_at).total_seconds() / 3600

    def _heuristic_engagement_prediction(
            self, features: Dict[str, Any]) -> float:
        """Heuristic-based engagement prediction when ML model is not available."""

        # Base engagement score
        base_score = 0.5

        # Topic alignment bonus
        topic_alignment = features.get("topic_alignment", 0)
        base_score += topic_alignment * 0.3

        # Source alignment bonus
        source_alignment = features.get("source_alignment", 0)
        base_score += source_alignment * 0.2

        # Breaking news bonus
        if features.get("content_is_breaking", 0):
            base_score += 0.2

        # Recent content bonus
        if features.get("is_recent_content", 0):
            base_score += 0.1

        # User engagement history bonus
        engagement_history = features.get("user_engagement_history", 0)
        if engagement_history > 10:
            base_score += 0.1

        return max(0.0, min(1.0, base_score))

    async def _load_engagement_model(self) -> None:
        """Load existing engagement model from file."""
        try:
            from ..config import get_settings

            settings = get_settings()

            model_path = settings.ENGAGEMENT_MODEL_PATH
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
                    self.engagement_model = model_data["model"]
                    self.scaler = model_data["scaler"]
                    self.is_trained = True
                    logger.info("Loaded existing engagement model")
        except Exception as e:
            logger.warning(f"Could not load existing engagement model: {e}")

    async def _create_and_train_engagement_model(self) -> None:
        """Create and train a new engagement model."""
        logger.info("Creating and training new engagement model")

        # Generate synthetic training data
        X, y = self._generate_synthetic_engagement_data()

        # Train Random Forest model
        self.engagement_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )

        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.engagement_model.fit(X_scaled, y)

        self.is_trained = True

        # Save model
        await self._save_engagement_model()

        logger.info("Engagement model trained and saved successfully")

    def _generate_synthetic_engagement_data(self) -> tuple:
        """Generate synthetic training data for engagement model."""
        np.random.seed(42)

        n_samples = 10000
        n_features = 10

        # Generate random features
        X = np.random.rand(n_samples, n_features)

        # Generate synthetic engagement scores based on feature patterns
        y = np.zeros(n_samples)

        for i in range(n_samples):
            # Higher engagement for better topic alignment
            topic_alignment = X[i, 0]
            source_alignment = X[i, 1]
            is_breaking = X[i, 2] > 0.7
            is_recent = X[i, 3] > 0.8

            # Calculate engagement score
            engagement = (
                topic_alignment * 0.4
                + source_alignment * 0.3
                + (0.2 if is_breaking else 0)
                + (0.1 if is_recent else 0)
                + np.random.normal(0, 0.1)
            )

            y[i] = max(0.0, min(1.0, engagement))

        return X, y

    async def _save_engagement_model(self) -> None:
        """Save trained engagement model to file."""
        try:
            from ..config import get_settings

            settings = get_settings()

            model_path = settings.ENGAGEMENT_MODEL_PATH
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            model_data = {
                "model": self.engagement_model,
                "scaler": self.scaler}

            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Engagement model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save engagement model: {e}")


class TrendingDetector:
    """Detect trending content for relevance scoring."""

    def __init__(self):
        self.trending_keywords = set()
        self.trending_sources = set()
        self.trending_topics = set()

    async def initialize(self) -> None:
        """Initialize trending detector."""
        logger.info("Initializing trending detector")

        # Load trending data (in real implementation, this would come from
        # analytics)
        await self._load_trending_data()

        logger.info("Trending detector initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup trending detector."""
        logger.info("Cleaning up trending detector")
        # No specific cleanup needed

    async def get_trending_score(self, content) -> float:
        """Get trending score for content."""

        try:
            score = 0.0

            # Check trending keywords in title and content
            title_words = set(content.title.lower().split())
            content_words = set(content.content.lower().split())

            trending_keyword_matches = len(
                title_words.intersection(
                    self.trending_keywords))
            trending_content_matches = len(
                content_words.intersection(
                    self.trending_keywords))

            score += (trending_keyword_matches * 0.3) + \
                (trending_content_matches * 0.1)

            # Check trending topics
            topic_matches = len(
                set(content.topics).intersection(self.trending_topics))
            score += topic_matches * 0.2

            # Check trending sources
            if content.source in self.trending_sources:
                score += 0.2

            # Normalize score
            return min(score, 1.0)

        except Exception as e:
            logger.error(f"Error calculating trending score: {e}")
            return 0.0

    async def _load_trending_data(self) -> None:
        """Load trending data from external sources."""
        # Mock implementation - in real system, this would fetch from analytics
        self.trending_keywords = {
            "breaking",
            "urgent",
            "important",
            "latest",
            "update",
            "news",
            "crisis",
            "emergency",
            "alert",
            "warning",
            "developing",
        }

        self.trending_sources = {
            "reuters",
            "ap",
            "bbc",
            "cnn",
            "nytimes",
            "washingtonpost"}

        self.trending_topics = {
            "politics",
            "technology",
            "health",
            "economy",
            "climate",
            "sports",
            "entertainment",
            "science",
        }


class UserProfileManager:
    """Manage user profiles for personalization."""

    def __init__(self):
        self.profiles = {}  # In-memory cache

    async def initialize(self) -> None:
        """Initialize user profile manager."""
        logger.info("Initializing user profile manager")
        # No specific initialization needed
        logger.info("User profile manager initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup user profile manager."""
        logger.info("Cleaning up user profile manager")
        self.profiles.clear()

    async def get_profile(self, user_id: str) -> UserProfile:
        """Get user profile."""

        try:
            # Check cache first
            if user_id in self.profiles:
                return self.profiles[user_id]

            # In real implementation, this would fetch from database
            profile = await self._fetch_profile_from_db(user_id)

            # Cache profile
            self.profiles[user_id] = profile

            return profile

        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            # Return default profile
            return self._create_default_profile(user_id)

    async def _fetch_profile_from_db(self, user_id: str) -> UserProfile:
        """Fetch user profile from database."""
        # Mock implementation - would fetch from database
        return UserProfile(
            user_id=user_id,
            topic_preferences=["technology", "politics", "science"],
            source_preferences=["reuters", "bbc", "nytimes"],
            location_preferences=["US", "UK", "EU"],
            engagement_history=[],
            device_info={},
            timezone="UTC",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def _create_default_profile(self, user_id: str) -> UserProfile:
        """Create default user profile."""
        return UserProfile(
            user_id=user_id,
            topic_preferences=[],
            source_preferences=[],
            location_preferences=[],
            engagement_history=[],
            device_info={},
            timezone="UTC",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )


class RelevanceScorer:
    """Score notification content relevance for users."""

    def __init__(self):
        self.personalization_engine = PersonalizationEngine()
        self.trending_detector = TrendingDetector()
        self.user_profiler = UserProfileManager()

    async def initialize(self) -> None:
        """Initialize relevance scorer."""
        logger.info("Initializing relevance scorer")

        # Initialize components
        await self.personalization_engine.initialize()
        await self.trending_detector.initialize()
        await self.user_profiler.initialize()

        logger.info("Relevance scorer initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup relevance scorer."""
        logger.info("Cleaning up relevance scorer")

        # Cleanup components
        await self.personalization_engine.cleanup()
        await self.trending_detector.cleanup()
        await self.user_profiler.cleanup()

    async def score_relevance(
            self,
            candidate: NotificationCandidate,
            user_prefs: NotificationPreferences) -> float:
        """Compute relevance score for notification candidate."""

        try:
            logger.debug(
                "Scoring relevance for candidate",
                user_id=candidate.user_id,
                notification_type=candidate.notification_type.value,
            )

            # Get user profile
            user_profile = await self.user_profiler.get_profile(candidate.user_id)

            # Topic relevance
            topic_score = await self._score_topic_relevance(
                candidate.content.topics, user_profile.topic_preferences
            )

            # Source preference
            source_score = await self._score_source_preference(
                candidate.content.source, user_profile.source_preferences
            )

            # Trending factor
            trending_score = await self.trending_detector.get_trending_score(candidate.content)

            # Geographic relevance
            geo_score = await self._score_geographic_relevance(
                candidate.content.locations, user_profile.location_preferences
            )

            # Time sensitivity
            time_score = self._score_time_sensitivity(
                candidate.content.published_at)

            # Engagement prediction
            engagement_score = await self.personalization_engine.predict_engagement(
                user_profile, candidate.content
            )

            # Weighted combination
            relevance_score = (
                topic_score * 0.25
                + source_score * 0.15
                + trending_score * 0.15
                + geo_score * 0.1
                + time_score * 0.1
                + engagement_score * 0.25
            )

            # Apply user preference adjustments
            relevance_score = self._apply_preference_adjustments(
                relevance_score, candidate, user_prefs
            )

            final_score = min(relevance_score, 1.0)

            logger.debug(
                "Relevance scoring completed",
                user_id=candidate.user_id,
                final_score=final_score,
                components={
                    "topic_score": topic_score,
                    "source_score": source_score,
                    "trending_score": trending_score,
                    "geo_score": geo_score,
                    "time_score": time_score,
                    "engagement_score": engagement_score,
                },
            )

            return final_score

        except Exception as e:
            logger.error(
                "Error scoring relevance",
                user_id=candidate.user_id,
                error=str(e),
                exc_info=True)
            raise RelevanceScoringError(f"Failed to score relevance: {str(e)}")

    async def _score_topic_relevance(
        self, content_topics: List[str], user_topics: List[str]
    ) -> float:
        """Score topic relevance between content and user preferences."""

        if not content_topics or not user_topics:
            return 0.5  # Neutral if no topic data

        # Calculate Jaccard similarity
        content_set = set(content_topics)
        user_set = set(user_topics)

        intersection = len(content_set.intersection(user_set))
        union = len(content_set.union(user_set))

        return intersection / union if union > 0 else 0.0

    async def _score_source_preference(
            self,
            content_source: str,
            user_sources: List[str]) -> float:
        """Score source preference alignment."""

        if not user_sources:
            return 0.5  # Neutral if no source preferences

        return 1.0 if content_source in user_sources else 0.0

    async def _score_geographic_relevance(
        self, content_locations: List[str], user_locations: List[str]
    ) -> float:
        """Score geographic relevance."""

        if not content_locations or not user_locations:
            return 0.5  # Neutral if no location data

        # Calculate overlap
        content_set = set(content_locations)
        user_set = set(user_locations)

        intersection = len(content_set.intersection(user_set))
        return min(intersection / len(content_locations), 1.0)

    def _score_time_sensitivity(self, published_at: datetime) -> float:
        """Score time sensitivity of content."""

        now = datetime.utcnow()
        age_hours = (now - published_at).total_seconds() / 3600

        # More sensitive if very recent
        if age_hours < 1:
            return 1.0
        elif age_hours < 6:
            return 0.8
        elif age_hours < 24:
            return 0.6
        elif age_hours < 72:
            return 0.4
        else:
            return 0.2

    def _apply_preference_adjustments(
        self,
        base_score: float,
        candidate: NotificationCandidate,
        user_prefs: NotificationPreferences,
    ) -> float:
        """Apply user preference adjustments to relevance score."""

        adjusted_score = base_score

        # Boost for preferred notification types
        if candidate.notification_type in user_prefs.enabled_types:
            adjusted_score *= 1.1

        # Reduce for non-preferred types
        else:
            adjusted_score *= 0.8

        # Apply time-based adjustments (quiet hours)
        if self._is_in_quiet_hours(user_prefs):
            adjusted_score *= 0.5

        return min(adjusted_score, 1.0)

    def _is_in_quiet_hours(self, user_prefs: NotificationPreferences) -> bool:
        """Check if current time is in user's quiet hours."""

        if not user_prefs.quiet_hours_start or not user_prefs.quiet_hours_end:
            return False

        current_hour = datetime.utcnow().hour

        if user_prefs.quiet_hours_start <= user_prefs.quiet_hours_end:
            # Same day quiet hours
            return user_prefs.quiet_hours_start <= current_hour < user_prefs.quiet_hours_end
        else:
            # Overnight quiet hours
            return (
                current_hour >= user_prefs.quiet_hours_start
                or current_hour < user_prefs.quiet_hours_end
            )
