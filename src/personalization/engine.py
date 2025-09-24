"""User-specific content personalization engine."""

import asyncio
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import structlog

from ..schemas import Article, UserProfile, PersonalizedScore
from ..optimization.cache import CacheManager

logger = structlog.get_logger(__name__)


class PersonalizationEngine:
    """User-specific content personalization."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.user_profiles = UserProfileManager(cache_manager)
        self.collaborative_filter = CollaborativeFilter(cache_manager)
        self.content_filter = ContentBasedFilter(cache_manager)
        self.topic_analyzer = TopicAnalyzer()
        
        # Vectorizers for content analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        
        # Initialize models
        asyncio.create_task(self._initialize_models())
    
    async def _initialize_models(self):
        """Initialize ML models for personalization."""
        try:
            # Load pre-trained models or train on historical data
            await self._load_user_profiles()
            await self._train_collaborative_filter()
            logger.info("Personalization models initialized")
        except Exception as e:
            logger.error("Failed to initialize personalization models", error=str(e))
    
    async def _load_user_profiles(self):
        """Load user profiles from cache or database."""
        # In production, this would load from a database
        pass
    
    async def _train_collaborative_filter(self):
        """Train collaborative filtering model."""
        # In production, this would train on historical interaction data
        pass
    
    async def personalize_ranking(self, articles: List[Article], 
                                user_id: str) -> List[PersonalizedScore]:
        """Apply personalization to content ranking."""
        try:
            user_profile = await self.user_profiles.get_profile(user_id)
            if not user_profile:
                # Return neutral scores for new users
                return [PersonalizedScore(article_id=a.id, score=0.5) for a in articles]
            
            scores = []
            
            for article in articles:
                # Topic affinity scoring
                topic_score = await self._compute_topic_affinity(
                    article.topics, user_profile.topic_preferences
                )
                
                # Source preference scoring
                source_score = await self._compute_source_preference(
                    article.source.id, user_profile.source_preferences
                )
                
                # Collaborative filtering score
                cf_score = await self.collaborative_filter.predict(
                    user_id, article.id
                )
                
                # Content-based similarity score
                cb_score = await self.content_filter.compute_similarity(
                    article, user_profile.content_preferences
                )
                
                # Time-based preference scoring
                time_score = await self._compute_time_preference(
                    article.published_at, user_profile.reading_patterns
                )
                
                # Geographic preference (if available)
                geo_score = await self._compute_geographic_preference(
                    article, user_profile
                )
                
                # Combine scores with weights
                final_score = (
                    topic_score * 0.25 +
                    source_score * 0.20 +
                    cf_score * 0.20 +
                    cb_score * 0.20 +
                    time_score * 0.10 +
                    geo_score * 0.05
                )
                
                # Generate explanation
                explanation = self._generate_explanation(
                    topic_score, source_score, cf_score, cb_score, time_score, geo_score
                )
                
                scores.append(PersonalizedScore(
                    article_id=article.id,
                    score=min(final_score, 1.0),  # Cap at 1.0
                    explanation=explanation,
                    feature_breakdown={
                        'topic_affinity': topic_score,
                        'source_preference': source_score,
                        'collaborative_filtering': cf_score,
                        'content_similarity': cb_score,
                        'time_preference': time_score,
                        'geographic_preference': geo_score
                    }
                ))
            
            return scores
            
        except Exception as e:
            logger.error("Personalization failed", error=str(e), user_id=user_id)
            # Return neutral scores on error
            return [PersonalizedScore(article_id=a.id, score=0.5) for a in articles]
    
    async def _compute_topic_affinity(self, article_topics: List, 
                                    user_preferences: Dict[str, float]) -> float:
        """Compute topic affinity score."""
        if not article_topics or not user_preferences:
            return 0.5
        
        total_score = 0.0
        total_weight = 0.0
        
        for topic in article_topics:
            topic_name = topic.name.lower()
            confidence = topic.confidence
            
            # Find best matching user preference
            best_match_score = 0.0
            for pref_topic, pref_score in user_preferences.items():
                if topic_name in pref_topic.lower() or pref_topic.lower() in topic_name:
                    best_match_score = max(best_match_score, pref_score)
            
            total_score += best_match_score * confidence
            total_weight += confidence
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    async def _compute_source_preference(self, source_id: str, 
                                       source_preferences: Dict[str, float]) -> float:
        """Compute source preference score."""
        return source_preferences.get(source_id, 0.5)
    
    async def _compute_time_preference(self, published_at: datetime, 
                                     reading_patterns: Dict[str, Any]) -> float:
        """Compute time-based preference score."""
        if not reading_patterns:
            return 0.5
        
        # Get preferred reading hours
        preferred_hours = reading_patterns.get('preferred_hours', [])
        if not preferred_hours:
            return 0.5
        
        current_hour = published_at.hour
        
        # Check if published during preferred hours
        if current_hour in preferred_hours:
            return 0.8
        
        # Check if published during adjacent hours
        for preferred_hour in preferred_hours:
            if abs(current_hour - preferred_hour) <= 2:
                return 0.6
        
        return 0.4
    
    async def _compute_geographic_preference(self, article: Article, 
                                           user_profile: UserProfile) -> float:
        """Compute geographic preference score."""
        # This would use location data if available
        return 0.5  # Neutral score for now
    
    def _generate_explanation(self, topic_score: float, source_score: float,
                            cf_score: float, cb_score: float, 
                            time_score: float, geo_score: float) -> str:
        """Generate explanation for personalization score."""
        explanations = []
        
        if topic_score > 0.7:
            explanations.append("matches your topic interests")
        elif topic_score < 0.3:
            explanations.append("different from your usual topics")
        
        if source_score > 0.7:
            explanations.append("from a source you like")
        elif source_score < 0.3:
            explanations.append("from a new source")
        
        if cf_score > 0.7:
            explanations.append("similar to what you've enjoyed")
        
        if cb_score > 0.7:
            explanations.append("similar content style")
        
        if time_score > 0.7:
            explanations.append("published when you're active")
        
        return ", ".join(explanations) if explanations else "personalized for you"


class UserProfileManager:
    """Manages user profiles and preferences."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.profiles: Dict[str, UserProfile] = {}
    
    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile from cache or create default."""
        # Check cache first
        cached_profile = await self.cache_manager.get(f"user_profile:{user_id}")
        if cached_profile:
            return UserProfile(**cached_profile)
        
        # Create default profile for new users
        default_profile = UserProfile(
            user_id=user_id,
            topic_preferences={},
            source_preferences={},
            reading_patterns={},
            content_preferences={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Cache the profile
        await self.cache_manager.set(
            f"user_profile:{user_id}",
            default_profile.dict(),
            ttl=3600  # 1 hour
        )
        
        return default_profile
    
    async def update_profile(self, user_id: str, 
                           interaction_data: Dict[str, Any]) -> None:
        """Update user profile based on interactions."""
        profile = await self.get_profile(user_id)
        if not profile:
            return
        
        # Update topic preferences based on article topics
        if 'article_topics' in interaction_data:
            await self._update_topic_preferences(
                profile, interaction_data['article_topics']
            )
        
        # Update source preferences
        if 'source_id' in interaction_data:
            await self._update_source_preferences(
                profile, interaction_data['source_id']
            )
        
        # Update reading patterns
        if 'reading_time' in interaction_data:
            await self._update_reading_patterns(
                profile, interaction_data
            )
        
        profile.updated_at = datetime.utcnow()
        
        # Cache updated profile
        await self.cache_manager.set(
            f"user_profile:{user_id}",
            profile.dict(),
            ttl=3600
        )
    
    async def _update_topic_preferences(self, profile: UserProfile, 
                                      topics: List) -> None:
        """Update topic preferences based on user interactions."""
        for topic in topics:
            topic_name = topic.name.lower()
            current_score = profile.topic_preferences.get(topic_name, 0.5)
            
            # Update with exponential moving average
            new_score = 0.7 * current_score + 0.3 * topic.confidence
            profile.topic_preferences[topic_name] = min(new_score, 1.0)
    
    async def _update_source_preferences(self, profile: UserProfile, 
                                       source_id: str) -> None:
        """Update source preferences based on user interactions."""
        current_score = profile.source_preferences.get(source_id, 0.5)
        new_score = 0.8 * current_score + 0.2  # Boost for interaction
        profile.source_preferences[source_id] = min(new_score, 1.0)
    
    async def _update_reading_patterns(self, profile: UserProfile, 
                                     interaction_data: Dict[str, Any]) -> None:
        """Update reading patterns based on user behavior."""
        reading_time = interaction_data.get('reading_time', 0)
        hour = datetime.utcnow().hour
        
        # Update preferred reading hours
        if 'preferred_hours' not in profile.reading_patterns:
            profile.reading_patterns['preferred_hours'] = []
        
        if reading_time > 60:  # If user spent more than 1 minute reading
            if hour not in profile.reading_patterns['preferred_hours']:
                profile.reading_patterns['preferred_hours'].append(hour)
                # Keep only last 10 preferred hours
                profile.reading_patterns['preferred_hours'] = \
                    profile.reading_patterns['preferred_hours'][-10:]


class CollaborativeFilter:
    """Collaborative filtering for user-item recommendations."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_similarity_matrix = None
    
    async def predict(self, user_id: str, article_id: str) -> float:
        """Predict user preference for article using collaborative filtering."""
        try:
            # In production, this would use a trained model
            # For now, return a random score based on user and article IDs
            hash_value = hash(f"{user_id}_{article_id}")
            return (hash_value % 100) / 100.0
        except Exception as e:
            logger.warning("Collaborative filtering prediction failed", error=str(e))
            return 0.5
    
    async def train(self, interactions: List[Dict[str, Any]]) -> None:
        """Train collaborative filtering model on interaction data."""
        # In production, this would implement matrix factorization
        # or other collaborative filtering algorithms
        pass


class ContentBasedFilter:
    """Content-based filtering using article features."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.article_vectors = {}
        self.user_content_preferences = {}
    
    async def compute_similarity(self, article: Article, 
                               user_preferences: Dict[str, Any]) -> float:
        """Compute content similarity between article and user preferences."""
        try:
            # In production, this would use TF-IDF vectors and cosine similarity
            # For now, return a score based on article features
            if not user_preferences:
                return 0.5
            
            # Simple similarity based on content length and quality
            content_score = 0.0
            if 'preferred_length' in user_preferences:
                preferred_length = user_preferences['preferred_length']
                actual_length = article.word_count
                length_similarity = 1.0 - abs(preferred_length - actual_length) / max(preferred_length, actual_length)
                content_score += length_similarity * 0.5
            
            if 'preferred_quality' in user_preferences:
                preferred_quality = user_preferences['preferred_quality']
                quality_similarity = 1.0 - abs(preferred_quality - article.quality_score)
                content_score += quality_similarity * 0.5
            
            return min(content_score, 1.0)
            
        except Exception as e:
            logger.warning("Content-based filtering failed", error=str(e))
            return 0.5


class TopicAnalyzer:
    """Analyzes and processes topic information."""
    
    def __init__(self):
        self.topic_embeddings = {}
    
    async def analyze_topics(self, content: str) -> List[Dict[str, Any]]:
        """Analyze topics in content."""
        # In production, this would use NLP models for topic extraction
        # For now, return dummy topics
        return [
            {"name": "technology", "confidence": 0.8},
            {"name": "artificial intelligence", "confidence": 0.6}
        ]
    
    async def compute_topic_similarity(self, topics1: List, topics2: List) -> float:
        """Compute similarity between two sets of topics."""
        if not topics1 or not topics2:
            return 0.0
        
        # Simple Jaccard similarity
        set1 = {t.name.lower() for t in topics1}
        set2 = {t.name.lower() for t in topics2}
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
