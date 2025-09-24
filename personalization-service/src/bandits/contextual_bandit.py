"""Contextual bandit implementation."""

import asyncio
from typing import List, Dict, Optional, Tuple
import structlog
import numpy as np

from .thompson_sampling import ThompsonSamplingBandit
from .ucb import UpperConfidenceBoundBandit
from .epsilon_greedy import EpsilonGreedyBandit
from ..models.schemas import BanditRecommendation, UserContext, ContentItem
from ..database.redis_client import RedisClient
from ..database.postgres_client import PostgreSQLClient

logger = structlog.get_logger()


class ContextualBandit:
    """Contextual multi-armed bandit for exploration/exploitation."""
    
    def __init__(self, redis_client: RedisClient, postgres_client: PostgreSQLClient):
        self.redis = redis_client
        self.postgres = postgres_client
        self.algorithms = {
            'thompson_sampling': ThompsonSamplingBandit(),
            'ucb': UpperConfidenceBoundBandit(),
            'epsilon_greedy': EpsilonGreedyBandit()
        }
        self.current_algorithm = 'thompson_sampling'
        self.cache_ttl = 3600  # 1 hour
        
    async def initialize(self) -> None:
        """Initialize the contextual bandit models."""
        logger.info("Initializing contextual bandit models")
        # Models are initialized on first use
        logger.info("Contextual bandit models ready")
    
    async def select_content(self, user_context: UserContext, 
                           candidates: List[ContentItem]) -> List[BanditRecommendation]:
        """Select content using contextual bandit algorithms."""
        if not candidates:
            return []
        
        # Extract contextual features
        context_features = await self._extract_context_features(user_context)
        
        # Get bandit algorithm
        algorithm = self.algorithms[self.current_algorithm]
        
        recommendations = []
        
        for candidate in candidates:
            # Extract content features
            content_features = await self._extract_content_features(candidate)
            
            # Combine context and content features
            combined_features = np.concatenate([context_features, content_features])
            
            # Get bandit recommendation
            action_value, uncertainty = await algorithm.predict(
                features=combined_features,
                candidate_id=candidate.id
            )
            
            # Compute confidence
            confidence = max(0.0, min(1.0, 1.0 - uncertainty))
            
            recommendations.append(BanditRecommendation(
                item_id=candidate.id,
                expected_reward=action_value,
                uncertainty=uncertainty,
                features=combined_features.tolist(),
                confidence=confidence
            ))
        
        return sorted(recommendations, key=lambda x: x.expected_reward, reverse=True)
    
    async def _extract_context_features(self, user_context: UserContext) -> np.ndarray:
        """Extract contextual features from user context."""
        features = []
        
        # Device type features
        device_features = [0.0] * 4  # mobile, desktop, tablet, other
        if user_context.device_type:
            device_mapping = {
                'mobile': 0,
                'desktop': 1,
                'tablet': 2
            }
            device_idx = device_mapping.get(user_context.device_type.lower(), 3)
            device_features[device_idx] = 1.0
        features.extend(device_features)
        
        # Time of day features
        time_features = [0.0] * 4  # morning, afternoon, evening, night
        if user_context.time_of_day:
            time_mapping = {
                'morning': 0,
                'afternoon': 1,
                'evening': 2,
                'night': 3
            }
            time_idx = time_mapping.get(user_context.time_of_day.lower(), 0)
            time_features[time_idx] = 1.0
        features.extend(time_features)
        
        # Day of week features
        day_features = [0.0] * 7  # Monday to Sunday
        if user_context.day_of_week:
            day_mapping = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            day_idx = day_mapping.get(user_context.day_of_week.lower(), 0)
            day_features[day_idx] = 1.0
        features.extend(day_features)
        
        # Location features (simplified)
        location_features = [0.0] * 3  # local, national, international
        if user_context.location:
            # Simple heuristic based on location string
            if 'local' in user_context.location.lower():
                location_features[0] = 1.0
            elif 'national' in user_context.location.lower():
                location_features[1] = 1.0
            else:
                location_features[2] = 1.0
        features.extend(location_features)
        
        # Custom context features
        if user_context.custom_context:
            # Add custom features (normalized)
            custom_values = list(user_context.custom_context.values())
            if custom_values:
                # Normalize to [0, 1]
                max_val = max(custom_values) if custom_values else 1.0
                normalized_values = [v / max_val for v in custom_values]
                features.extend(normalized_values[:10])  # Limit to 10 features
        
        # Pad or truncate to fixed size
        target_size = 20
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        elif len(features) > target_size:
            features = features[:target_size]
        
        return np.array(features)
    
    async def _extract_content_features(self, content_item: ContentItem) -> np.ndarray:
        """Extract content features from content item."""
        features = []
        
        # Topic features (one-hot encoding for top topics)
        topic_features = [0.0] * 10  # Top 10 topics
        if content_item.topics:
            # Simple topic mapping (in production, use proper topic modeling)
            topic_mapping = {
                'technology': 0, 'business': 1, 'sports': 2, 'entertainment': 3,
                'politics': 4, 'science': 5, 'health': 6, 'education': 7,
                'lifestyle': 8, 'news': 9
            }
            for topic in content_item.topics[:3]:  # Top 3 topics
                topic_idx = topic_mapping.get(topic.lower(), 0)
                topic_features[topic_idx] = 1.0
        features.extend(topic_features)
        
        # Source features (one-hot encoding for top sources)
        source_features = [0.0] * 5  # Top 5 sources
        if content_item.source:
            source_mapping = {
                'cnn': 0, 'bbc': 1, 'reuters': 2, 'ap': 3, 'guardian': 4
            }
            source_idx = source_mapping.get(content_item.source.lower(), 0)
            source_features[source_idx] = 1.0
        features.extend(source_features)
        
        # Content length features
        content_length = len(content_item.content or '')
        length_features = [
            min(1.0, content_length / 1000),  # Normalized length
            1.0 if content_length > 500 else 0.0,  # Long content flag
            1.0 if content_length < 200 else 0.0   # Short content flag
        ]
        features.extend(length_features)
        
        # Author features (simplified)
        author_features = [
            1.0 if content_item.author else 0.0,  # Has author
            1.0 if len(content_item.author or '') > 10 else 0.0  # Long author name
        ]
        features.extend(author_features)
        
        # Pad or truncate to fixed size
        target_size = 20
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        elif len(features) > target_size:
            features = features[:target_size]
        
        return np.array(features)
    
    async def update_from_feedback(self, recommendation: BanditRecommendation,
                                 reward: float) -> None:
        """Update bandit model from user feedback."""
        algorithm = self.algorithms[self.current_algorithm]
        
        await algorithm.update(
            features=np.array(recommendation.features),
            action_id=recommendation.item_id,
            reward=reward
        )
        
        # Log the update
        logger.info(
            f"Updated bandit model: item={recommendation.item_id}, "
            f"reward={reward}, algorithm={self.current_algorithm}"
        )
    
    async def set_algorithm(self, algorithm_name: str) -> None:
        """Set the active bandit algorithm."""
        if algorithm_name in self.algorithms:
            self.current_algorithm = algorithm_name
            logger.info(f"Switched to bandit algorithm: {algorithm_name}")
        else:
            logger.warning(f"Unknown bandit algorithm: {algorithm_name}")
    
    async def get_algorithm_statistics(self) -> Dict[str, any]:
        """Get statistics for all bandit algorithms."""
        stats = {}
        for name, algorithm in self.algorithms.items():
            stats[name] = algorithm.get_model_info()
        return stats
    
    async def get_arm_statistics(self, algorithm_name: Optional[str] = None) -> Dict[str, any]:
        """Get arm statistics for the specified algorithm."""
        if algorithm_name is None:
            algorithm_name = self.current_algorithm
        
        if algorithm_name not in self.algorithms:
            return {}
        
        algorithm = self.algorithms[algorithm_name]
        return await algorithm.get_all_arm_statistics()
    
    async def retrain_models(self) -> None:
        """Retrain the bandit models (reset statistics)."""
        logger.info("Retraining bandit models")
        
        # Reset all algorithms
        for algorithm in self.algorithms.values():
            algorithm.arm_counts.clear()
            algorithm.arm_rewards.clear()
            algorithm.context_weights.clear()
            algorithm.context_dim = 0
            if hasattr(algorithm, 'total_pulls'):
                algorithm.total_pulls = 0
        
        logger.info("Bandit models retrained (reset)")
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the bandit models."""
        return {
            "current_algorithm": self.current_algorithm,
            "available_algorithms": list(self.algorithms.keys()),
            "algorithms": await self.get_algorithm_statistics()
        }
