"""
Multi-Armed Bandit Testing Framework
"""

import asyncio
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


class BanditTestingFramework:
    """Multi-armed bandit testing framework"""

    def __init__(self):
        self.active_bandits: Dict[str, Dict[str, Any]] = {}
        self.bandit_algorithms = {
            "epsilon_greedy": self._epsilon_greedy,
            "ucb1": self._ucb1,
            "thompson_sampling": self._thompson_sampling,
            "softmax": self._softmax,
        }

    async def initialize(self) -> Dict[str, Any]:
    """Initialize the bandit testing framework"""
        logger.info("Initializing bandit testing framework...")
        # No specific initialization needed
        logger.info("Bandit testing framework initialized successfully")

    async def cleanup(self) -> Dict[str, Any]:
    """Cleanup bandit testing framework resources"""
        logger.info("Cleaning up bandit testing framework...")
        self.active_bandits.clear()
        logger.info("Bandit testing framework cleanup completed")

    async def create_bandit(self, bandit_config: Dict[str, Any]) -> str:
        """Create a new multi-armed bandit"""

        bandit_id = str(uuid4())
        algorithm = bandit_config.get("algorithm", "epsilon_greedy")

        if algorithm not in self.bandit_algorithms:
            raise ValueError(f"Unsupported bandit algorithm: {algorithm}")

        bandit = {
            "id": bandit_id,
            "algorithm": algorithm,
            "arms": bandit_config.get("arms", []),
            "parameters": bandit_config.get("parameters", {}),
            "rewards": {arm: [] for arm in bandit_config.get("arms", [])},
            "pulls": {arm: 0 for arm in bandit_config.get("arms", [])},
            "total_pulls": 0,
            "created_at": datetime.utcnow(),
            "status": "active",
        }

        self.active_bandits[bandit_id] = bandit

        logger.info(f"Created bandit {bandit_id} with algorithm {algorithm}")
        return bandit_id

    async def select_arm(self, bandit_id: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Select an arm to pull based on bandit algorithm"""

        if bandit_id not in self.active_bandits:
            raise ValueError(f"Bandit {bandit_id} not found")

        bandit = self.active_bandits[bandit_id]
        algorithm = bandit["algorithm"]

        if algorithm not in self.bandit_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Select arm using the specified algorithm
        arm = await self.bandit_algorithms[algorithm](bandit, context)

        # Update pull count
        bandit["pulls"][arm] += 1
        bandit["total_pulls"] += 1

        return arm

    async def update_reward(
        self, bandit_id: str, arm: str, reward: float, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update reward for a pulled arm"""

        if bandit_id not in self.active_bandits:
            return False

        bandit = self.active_bandits[bandit_id]

        if arm not in bandit["rewards"]:
            return False

        bandit["rewards"][arm].append({"reward": reward, "context": context, "timestamp": datetime.utcnow()})

        return True

    async def analyze_bandit(self, bandit_id: str) -> Dict[str, Any]:
    """Analyze bandit performance"""
        if bandit_id not in self.active_bandits:
            raise ValueError(f"Bandit {bandit_id} not found")

        bandit = self.active_bandits[bandit_id]

        # Calculate arm statistics
        arm_stats = {}
        for arm in bandit["arms"]:
            rewards = [r["reward"] for r in bandit["rewards"][arm]]
            pulls = bandit["pulls"][arm]

            if pulls > 0:
                arm_stats[arm] = {
                    "pulls": pulls,
                    "total_reward": sum(rewards),
                    "average_reward": np.mean(rewards) if rewards else 0,
                    "reward_std": np.std(rewards) if len(rewards) > 1 else 0,
                    "confidence_interval": self._calculate_confidence_interval(rewards),
                }
            else:
                arm_stats[arm] = {
                    "pulls": 0,
                    "total_reward": 0,
                    "average_reward": 0,
                    "reward_std": 0,
                    "confidence_interval": {"lower": 0, "upper": 0},
                }

        # Find best arm
        best_arm = max(arm_stats.keys(), key=lambda x: arm_stats[x]["average_reward"])

        # Calculate regret (difference from optimal)
        optimal_reward = max(arm_stats[arm]["average_reward"] for arm in bandit["arms"])
        regret = sum(
            (optimal_reward - arm_stats[arm]["average_reward"]) * arm_stats[arm]["pulls"] for arm in bandit["arms"]
        )

        return {
            "bandit_id": bandit_id,
            "algorithm": bandit["algorithm"],
            "total_pulls": bandit["total_pulls"],
            "arm_statistics": arm_stats,
            "best_arm": best_arm,
            "total_regret": regret,
            "analysis_timestamp": datetime.utcnow(),
        }

    async def _epsilon_greedy(self, bandit: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """Epsilon-greedy algorithm"""

        epsilon = bandit["parameters"].get("epsilon", 0.1)

        if np.random.random() < epsilon:
            # Explore: choose random arm
            return np.random.choice(bandit["arms"])
        else:
            # Exploit: choose arm with highest average reward
            arm_rewards = {}
            for arm in bandit["arms"]:
                rewards = [r["reward"] for r in bandit["rewards"][arm]]
                arm_rewards[arm] = np.mean(rewards) if rewards else 0

            return max(arm_rewards.keys(), key=lambda x: arm_rewards[x])

    async def _ucb1(self, bandit: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """Upper Confidence Bound (UCB1) algorithm"""

        total_pulls = bandit["total_pulls"]

        if total_pulls < len(bandit["arms"]):
            # Initial exploration phase
            return bandit["arms"][total_pulls % len(bandit["arms"])]

        # Calculate UCB for each arm
        ucb_values = {}
        for arm in bandit["arms"]:
            rewards = [r["reward"] for r in bandit["rewards"][arm]]
            pulls = bandit["pulls"][arm]

            if pulls == 0:
                ucb_values[arm] = float("inf")
            else:
                average_reward = np.mean(rewards)
                confidence_bound = math.sqrt(2 * math.log(total_pulls) / pulls)
                ucb_values[arm] = average_reward + confidence_bound

        return max(ucb_values.keys(), key=lambda x: ucb_values[x])

    async def _thompson_sampling(self, bandit: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """Thompson Sampling algorithm (Beta-Bernoulli)"""

        # Sample from Beta distribution for each arm
        sampled_rewards = {}
        for arm in bandit["arms"]:
            rewards = [r["reward"] for r in bandit["rewards"][arm]]
            pulls = bandit["pulls"][arm]

            # Beta distribution parameters
            alpha = sum(rewards) + 1  # successes + 1
            beta = pulls - sum(rewards) + 1  # failures + 1

            # Sample from Beta distribution
            sampled_rewards[arm] = np.random.beta(alpha, beta)

        return max(sampled_rewards.keys(), key=lambda x: sampled_rewards[x])

    async def _softmax(self, bandit: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """Softmax algorithm"""

        temperature = bandit["parameters"].get("temperature", 1.0)

        # Calculate average rewards
        arm_rewards = {}
        for arm in bandit["arms"]:
            rewards = [r["reward"] for r in bandit["rewards"][arm]]
            arm_rewards[arm] = np.mean(rewards) if rewards else 0

        # Calculate softmax probabilities
        exp_values = {arm: math.exp(reward / temperature) for arm, reward in arm_rewards.items()}
        total_exp = sum(exp_values.values())

        if total_exp == 0:
            # Equal probability if no rewards
            probabilities = {arm: 1.0 / len(bandit["arms"]) for arm in bandit["arms"]}
        else:
            probabilities = {arm: exp_values[arm] / total_exp for arm in bandit["arms"]}

        # Sample arm based on probabilities
        arms = list(probabilities.keys())
        probs = list(probabilities.values())
        return np.random.choice(arms, p=probs)

    def _calculate_confidence_interval(self, rewards: List[float], confidence: float = 0.95) -> Dict[str, float]:
        """Calculate confidence interval for rewards"""

        if len(rewards) < 2:
            return {"lower": 0, "upper": 0}

        mean = np.mean(rewards)
        std = np.std(rewards, ddof=1)
        n = len(rewards)

        # t-distribution critical value
        alpha = 1 - confidence
        t_critical = 1.96  # Approximate for large n

        margin_error = t_critical * (std / math.sqrt(n))

        return {"lower": mean - margin_error, "upper": mean + margin_error}

    async def get_bandit(self, bandit_id: str) -> Optional[Dict[str, Any]]:
        """Get bandit by ID"""
        return self.active_bandits.get(bandit_id)

    async def list_bandits(self) -> List[Dict[str, Any]]:
        """List all active bandits"""
        return list(self.active_bandits.values())

    async def stop_bandit(self, bandit_id: str) -> bool:
        """Stop a bandit"""
        if bandit_id not in self.active_bandits:
            return False

        self.active_bandits[bandit_id]["status"] = "stopped"
        return True
