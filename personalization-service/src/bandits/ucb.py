"""Upper Confidence Bound (UCB) contextual bandit algorithm."""

import asyncio
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from ..config.settings import settings
from ..models.schemas import BanditRecommendation, ContentItem, UserContext

logger = structlog.get_logger()


class UpperConfidenceBoundBandit:
    """Upper Confidence Bound contextual bandit algorithm."""

    def __init__(self):
        self.arm_counts: Dict[str, int] = {}
        self.arm_rewards: Dict[str, List[float]] = {}
        self.context_weights: Dict[str, np.ndarray] = {}
        self.context_dim = 0
        self.total_pulls = 0
        self.confidence_level = 0.95

    async def predict(self, features: np.ndarray, candidate_id: str) -> Tuple[float, float]:
        """Predict expected reward and uncertainty for a candidate."""
        if candidate_id not in self.arm_counts:
            # New arm - initialize
            self.arm_counts[candidate_id] = 0
            self.arm_rewards[candidate_id] = []
            self.context_weights[candidate_id] = np.zeros(len(features))
            self.context_dim = len(features)

        # Get arm statistics
        count = self.arm_counts[candidate_id]
        rewards = self.arm_rewards[candidate_id]

        if count == 0:
            # New arm - return high uncertainty to encourage exploration
            return 0.5, 1.0

        # Compute average reward
        average_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # Compute confidence bound
        confidence_bound = self._compute_confidence_bound(count)

        # UCB score
        ucb_score = average_reward + confidence_bound

        # Uncertainty is the confidence bound
        uncertainty = confidence_bound

        return float(ucb_score), float(uncertainty)

    def _compute_confidence_bound(self, count: int) -> float:
        """Compute confidence bound for UCB."""
        if count == 0:
            return 1.0

        # UCB1 confidence bound
        confidence_bound = math.sqrt((2 * math.log(self.total_pulls)) / count)

        return confidence_bound

    async def update(self, features: np.ndarray, action_id: str, reward: float) -> None:
        """Update the bandit model with new feedback."""
        if action_id not in self.arm_counts:
            # Initialize new arm
            self.arm_counts[action_id] = 0
            self.arm_rewards[action_id] = []
            self.context_weights[action_id] = np.zeros(len(features))
            self.context_dim = len(features)

        # Update arm statistics
        self.arm_counts[action_id] += 1
        self.arm_rewards[action_id].append(reward)
        self.total_pulls += 1

        # Update context weights using online gradient descent
        learning_rate = 0.01
        prediction = np.dot(self.context_weights[action_id], features)
        error = reward - prediction

        # Update weights
        self.context_weights[action_id] += learning_rate * error * features

        # Clip weights to prevent explosion
        self.context_weights[action_id] = np.clip(self.context_weights[action_id], -10, 10)

    async def get_arm_statistics(self, action_id: str) -> Dict[str, float]:
        """Get statistics for a specific arm."""
        if action_id not in self.arm_counts:
            return {"count": 0, "total_reward": 0.0, "average_reward": 0.0, "confidence_bound": 1.0}

        count = self.arm_counts[action_id]
        rewards = self.arm_rewards[action_id]
        total_reward = sum(rewards)
        average_reward = total_reward / len(rewards) if rewards else 0.0
        confidence_bound = self._compute_confidence_bound(count)

        return {
            "count": count,
            "total_reward": total_reward,
            "average_reward": average_reward,
            "confidence_bound": confidence_bound,
        }

    async def get_all_arm_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all arms."""
        stats = {}
        for action_id in self.arm_counts:
            stats[action_id] = await self.get_arm_statistics(action_id)
        return stats

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the bandit model."""
        return {
            "n_arms": len(self.arm_counts),
            "context_dim": self.context_dim,
            "total_pulls": self.total_pulls,
            "confidence_level": self.confidence_level,
        }
