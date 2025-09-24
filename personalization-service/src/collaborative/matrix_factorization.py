"""Matrix factorization for collaborative filtering."""

import numpy as np
import asyncio
from typing import List, Dict, Optional, Tuple
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import structlog
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender

from ..models.schemas import UserInteraction, SimilarUser
from ..config.settings import settings

logger = structlog.get_logger()


class ImplicitMatrixFactorization:
    """Implicit matrix factorization for collaborative filtering."""
    
    def __init__(self):
        self.model = AlternatingLeastSquares(
            factors=settings.collaborative_factors,
            regularization=settings.collaborative_regularization,
            iterations=settings.collaborative_iterations,
            random_state=42
        )
        self.user_mapping: Dict[str, int] = {}
        self.item_mapping: Dict[str, int] = {}
        self.reverse_user_mapping: Dict[int, str] = {}
        self.reverse_item_mapping: Dict[int, str] = {}
        self.interaction_matrix: Optional[csr_matrix] = None
        self.is_trained = False
        
    async def train_model(self, interactions: List[UserInteraction]) -> None:
        """Train the collaborative filtering model."""
        logger.info(f"Training CF model with {len(interactions)} interactions")
        
        # Build interaction matrix
        self.interaction_matrix = await self._build_interaction_matrix(interactions)
        
        if self.interaction_matrix is None or self.interaction_matrix.nnz == 0:
            logger.warning("No interactions found for training")
            return
            
        # Train the model
        self.model.fit(self.interaction_matrix)
        self.is_trained = True
        
        logger.info(
            f"Trained CF model with {len(self.user_mapping)} users and {len(self.item_mapping)} items"
        )
    
    async def _build_interaction_matrix(self, interactions: List[UserInteraction]) -> Optional[csr_matrix]:
        """Build user-item interaction matrix."""
        if not interactions:
            return None
            
        # Create mappings
        user_ids = sorted(set(interaction.user_id for interaction in interactions))
        item_ids = sorted(set(interaction.item_id for interaction in interactions))
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        
        # Build matrix
        rows = []
        cols = []
        data = []
        
        for interaction in interactions:
            user_idx = self.user_mapping[interaction.user_id]
            item_idx = self.item_mapping[interaction.item_id]
            
            # Weight interactions based on type
            weight = self._get_interaction_weight(interaction)
            
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(weight)
        
        # Create sparse matrix
        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(item_ids))
        )
        
        return matrix
    
    def _get_interaction_weight(self, interaction: UserInteraction) -> float:
        """Get weight for interaction based on type."""
        weights = {
            'click': 1.0,
            'view': 0.5,
            'share': 2.0,
            'save': 2.5,
            'like': 1.5,
            'dislike': -1.0,
            'read': 3.0,
            'skip': -0.5
        }
        return weights.get(interaction.interaction_type.value, 1.0)
    
    async def predict_batch(self, user_id: str, item_ids: List[str]) -> List[float]:
        """Batch prediction for multiple items."""
        if not self.is_trained or user_id not in self.user_mapping:
            # Cold start - return neutral scores
            return [0.5] * len(item_ids)
        
        user_idx = self.user_mapping[user_id]
        scores = []
        
        for item_id in item_ids:
            if item_id in self.item_mapping:
                item_idx = self.item_mapping[item_id]
                score = self.model.predict(user_idx, item_idx)
                # Normalize score to [0, 1]
                normalized_score = max(0.0, min(1.0, (score + 1) / 2))
                scores.append(normalized_score)
            else:
                scores.append(0.5)  # Cold start for new items
        
        return scores
    
    async def get_similar_users(self, user_id: str, n_users: int = 10) -> List[SimilarUser]:
        """Find similar users for collaborative filtering."""
        if not self.is_trained or user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        
        # Get user factors
        user_factors = self.model.user_factors[user_idx]
        
        # Compute cosine similarity with all users
        similarities = cosine_similarity([user_factors], self.model.user_factors)[0]
        
        # Get top similar users
        similar_indices = np.argsort(similarities)[-n_users-1:-1][::-1]
        
        similar_users = []
        for idx in similar_indices:
            if idx != user_idx:  # Exclude self
                reverse_user_id = self.reverse_user_mapping.get(idx)
                if reverse_user_id:
                    # Count common interactions
                    common_interactions = self._count_common_interactions(user_idx, idx)
                    
                    similar_users.append(SimilarUser(
                        user_id=reverse_user_id,
                        similarity=float(similarities[idx]),
                        common_interactions=common_interactions
                    ))
        
        return similar_users
    
    def _count_common_interactions(self, user1_idx: int, user2_idx: int) -> int:
        """Count common interactions between two users."""
        if self.interaction_matrix is None:
            return 0
            
        user1_items = set(self.interaction_matrix[user1_idx].indices)
        user2_items = set(self.interaction_matrix[user2_idx].indices)
        
        return len(user1_items.intersection(user2_items))
    
    async def get_user_factors(self, user_id: str) -> Optional[np.ndarray]:
        """Get user factor vector."""
        if not self.is_trained or user_id not in self.user_mapping:
            return None
        
        user_idx = self.user_mapping[user_id]
        return self.model.user_factors[user_idx]
    
    async def get_item_factors(self, item_id: str) -> Optional[np.ndarray]:
        """Get item factor vector."""
        if not self.is_trained or item_id not in self.item_mapping:
            return None
        
        item_idx = self.item_mapping[item_id]
        return self.model.item_factors[item_idx]
    
    async def update_user_factors(self, user_id: str, item_id: str, rating: float) -> None:
        """Update user factors with new interaction (online learning)."""
        if not self.is_trained:
            return
            
        # This is a simplified online update
        # In production, you might want to use more sophisticated online learning
        if user_id in self.user_mapping and item_id in self.item_mapping:
            user_idx = self.user_mapping[user_id]
            item_idx = self.item_mapping[item_id]
            
            # Update the interaction matrix
            if self.interaction_matrix is not None:
                self.interaction_matrix[user_idx, item_idx] = rating
            
            # Retrain model periodically (in production, use incremental updates)
            # For now, we'll just log the update
            logger.info(f"Updated interaction: user={user_id}, item={item_id}, rating={rating}")
    
    async def get_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Get top recommendations for a user."""
        if not self.is_trained or user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        
        # Get recommendations
        recommendations = self.model.recommend(
            user_idx, 
            self.interaction_matrix[user_idx], 
            N=n_recommendations
        )
        
        # Convert to item IDs and scores
        result = []
        for item_idx, score in recommendations:
            item_id = self.reverse_item_mapping.get(item_idx)
            if item_id:
                normalized_score = max(0.0, min(1.0, (score + 1) / 2))
                result.append((item_id, normalized_score))
        
        return result
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the trained model."""
        return {
            "is_trained": self.is_trained,
            "n_users": len(self.user_mapping),
            "n_items": len(self.item_mapping),
            "n_interactions": self.interaction_matrix.nnz if self.interaction_matrix is not None else 0,
            "factors": settings.collaborative_factors,
            "regularization": settings.collaborative_regularization
        }
