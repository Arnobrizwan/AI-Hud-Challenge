"""Epsilon-Greedy contextual bandit algorithm."""

import asyncio
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from ..config.settings import settings
from ..models.schemas import BanditRecommendation, ContentItem, UserContext

logger = structlog.get_logger()


class EpsilonGreedyBandit:
    """Epsilon-Greedy contextual bandit algorithm."""

    def __init__(self):
        self.epsilon = settings.epsilon
        self.arm_counts: Dict[str, int] = {}
        self.arm_rewards: Dict[str, List[float]] = {}
        self.context_weights: Dict[str, np.ndarray] = {}
        self.context_dim = 0
        self.total_pulls = 0

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
            # New arm - return neutral score with high uncertainty
            return 0.5, 1.0

        # Compute average reward
        average_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # Compute uncertainty (standard error)
        if count > 1:
            variance = np.var(rewards)
            uncertainty = np.sqrt(variance / count)
        else:
            uncertainty = 1.0

        return float(average_reward), float(uncertainty)

    async def select_action(self, features: np.ndarray, candidate_ids: List[str]) -> str:
        """Select action using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            # Explore - select random action
            return random.choice(candidate_ids)
        else:
            # Exploit - select best action
            best_action = None
            best_score = -float("inf")

            for candidate_id in candidate_ids:
                score, _ = await self.predict(features, candidate_id)
                if score > best_score:
                    best_score = score
                    best_action = candidate_id

            return best_action or candidate_ids[0]

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
            return {"count": 0, "total_reward": 0.0, "average_reward": 0.0, "uncertainty": 1.0}

        count = self.arm_counts[action_id]
        rewards = self.arm_rewards[action_id]
        total_reward = sum(rewards)
        average_reward = total_reward / len(rewards) if rewards else 0.0

        # Compute uncertainty
        if count > 1:
            variance = np.var(rewards)
            uncertainty = np.sqrt(variance / count)
        else:
            uncertainty = 1.0

        return {
            "count": count,
            "total_reward": total_reward,
            "average_reward": average_reward,
            "uncertainty": uncertainty,
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
            "epsilon": self.epsilon,
        }
