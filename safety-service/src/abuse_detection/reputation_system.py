"""
Reputation System
User reputation scoring and management
"""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from safety_engine.config import get_abuse_config

logger = logging.getLogger(__name__)


class ReputationSystem:
    """User reputation scoring and management system"""

    def __init__(self):
        self.config = get_abuse_config()
        self.is_initialized = False

        # Reputation storage
        self.user_reputations: Dict[str, float] = {}
        self.reputation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.reputation_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Reputation calculation parameters
        self.base_reputation = 0.5  # Starting reputation for new users
        self.max_reputation = 1.0
        self.min_reputation = 0.0
        self.decay_rate = self.config.reputation_decay

        # Event weights
        self.event_weights = {
            "positive_activity": 0.1,
            "negative_activity": -0.2,
            "abuse_detection": -0.3,
            "content_violation": -0.4,
            "successful_verification": 0.2,
            "community_contribution": 0.15,
            "long_term_activity": 0.05,
        }

    async def initialize(self):
        """Initialize the reputation system"""
        try:
            # Load existing reputations from storage
            await self.load_reputations()

            # Start reputation decay task
            asyncio.create_task(self.reputation_decay_task())

            self.is_initialized = True
            logger.info("Reputation system initialized")

        except Exception as e:
            logger.error(f"Failed to initialize reputation system: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Save current reputations
            await self.save_reputations()

            self.user_reputations.clear()
            self.reputation_history.clear()
            self.reputation_events.clear()

            self.is_initialized = False
            logger.info("Reputation system cleanup completed")

        except Exception as e:
            logger.error(f"Error during reputation system cleanup: {str(e)}")

    async def get_user_reputation(self, user_id: str) -> float:
        """Get current reputation score for a user"""
        try:
            if user_id not in self.user_reputations:
                # Initialize new user with base reputation
                self.user_reputations[user_id] = self.base_reputation
                await self.save_reputations()

            return self.user_reputations[user_id]

        except Exception as e:
            logger.error(f"Failed to get reputation for user {user_id}: {str(e)}")
            return self.base_reputation

    async def update_reputation(
        self, user_id: str, event_type: str, event_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Update user reputation based on an event"""

        if not self.is_initialized:
            raise RuntimeError("Reputation system not initialized")

        try:
            # Get current reputation
            current_reputation = await self.get_user_reputation(user_id)

            # Calculate reputation change
            reputation_change = self.calculate_reputation_change(event_type, event_data)

            # Apply change with bounds
            new_reputation = max(
                self.min_reputation,
                min(self.max_reputation, current_reputation + reputation_change),
            )

            # Update reputation
            self.user_reputations[user_id] = new_reputation

            # Record event
            event_record = {
                "timestamp": datetime.utcnow(),
                "event_type": event_type,
                "reputation_change": reputation_change,
                "old_reputation": current_reputation,
                "new_reputation": new_reputation,
                "event_data": event_data or {},
            }

            self.reputation_events[user_id].append(event_record)
            self.reputation_history[user_id].append(new_reputation)

            # Cleanup old events (keep last 1000)
            if len(self.reputation_events[user_id]) > 1000:
                self.reputation_events[user_id] = self.reputation_events[user_id][-1000:]

            logger.info(
                f"Updated reputation for user {user_id}: {current_reputation:.3f} -> {new_reputation:.3f} ({event_type})"
            )

            return new_reputation

        except Exception as e:
            logger.error(f"Failed to update reputation for user {user_id}: {str(e)}")
            return current_reputation

    def calculate_reputation_change(
        self, event_type: str, event_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate reputation change based on event type and data"""
        try:
            # Get base weight for event type
            base_weight = self.event_weights.get(event_type, 0.0)

            if base_weight == 0.0:
                return 0.0

            # Apply multipliers based on event data
            multiplier = 1.0

            if event_data:
                # Severity multiplier
                severity = event_data.get("severity", "medium")
                severity_multipliers = {"low": 0.5, "medium": 1.0, "high": 1.5, "critical": 2.0}
                multiplier *= severity_multipliers.get(severity, 1.0)

                # Frequency multiplier (recent similar events)
                frequency = event_data.get("frequency", 1)
                if frequency > 1:
                    multiplier *= min(1.0 + (frequency - 1) * 0.1, 2.0)  # Cap at 2x

                # Context multiplier
                context = event_data.get("context", {})
                if context.get("first_offense", False):
                    multiplier *= 0.5  # Reduce penalty for first offense
                if context.get("repeat_offense", False):
                    multiplier *= 1.5  # Increase penalty for repeat offenses

            # Calculate final change
            reputation_change = base_weight * multiplier

            return reputation_change

        except Exception as e:
            logger.error(f"Reputation change calculation failed: {str(e)}")
            return 0.0

    async def batch_update_reputations(self, updates: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update multiple user reputations in batch"""
        try:
            results = {}

            for update in updates:
                user_id = update.get("user_id")
                event_type = update.get("event_type")
                event_data = update.get("event_data")

                if user_id and event_type:
                    new_reputation = await self.update_reputation(user_id, event_type, event_data)
                    results[user_id] = new_reputation

            return results

        except Exception as e:
            logger.error(f"Batch reputation update failed: {str(e)}")
            return {}

    async def get_reputation_history(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get reputation history for a user"""
        try:
            events = self.reputation_events.get(user_id, [])
            return events[-limit:] if limit > 0 else events

        except Exception as e:
            logger.error(f"Failed to get reputation history for user {user_id}: {str(e)}")
            return []

    async def get_reputation_trend(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get reputation trend analysis for a user"""
        try:
            history = self.reputation_history.get(user_id, deque())

            if not history:
                return {"trend": "stable", "change": 0.0, "volatility": 0.0}

            # Convert to list and filter by time
            history_list = list(history)
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            recent_events = [
                event
                for event in self.reputation_events.get(user_id, [])
                if event["timestamp"] >= cutoff_date
            ]

            if len(recent_events) < 2:
                return {"trend": "stable", "change": 0.0, "volatility": 0.0}

            # Calculate trend
            recent_reputations = [event["new_reputation"] for event in recent_events]
            trend_change = recent_reputations[-1] - recent_reputations[0]

            # Calculate volatility (standard deviation of changes)
            changes = [
                recent_reputations[i + 1] - recent_reputations[i]
                for i in range(len(recent_reputations) - 1)
            ]
            volatility = np.std(changes) if changes else 0.0

            # Determine trend direction
            if trend_change > 0.1:
                trend = "improving"
            elif trend_change < -0.1:
                trend = "declining"
            else:
                trend = "stable"

            return {
                "trend": trend,
                "change": trend_change,
                "volatility": volatility,
                "current_reputation": recent_reputations[-1],
                "events_count": len(recent_events),
            }

        except Exception as e:
            logger.error(f"Reputation trend calculation failed for user {user_id}: {str(e)}")
            return {"trend": "stable", "change": 0.0, "volatility": 0.0}

    async def get_top_users(
        self, limit: int = 10, min_reputation: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Get top users by reputation"""
        try:
            # Filter users by minimum reputation
            filtered_users = {
                user_id: reputation
                for user_id, reputation in self.user_reputations.items()
                if reputation >= min_reputation
            }

            # Sort by reputation (descending)
            sorted_users = sorted(filtered_users.items(), key=lambda x: x[1], reverse=True)

            # Format results
            top_users = []
            for user_id, reputation in sorted_users[:limit]:
                trend = await self.get_reputation_trend(user_id)
                top_users.append(
                    {
                        "user_id": user_id,
                        "reputation": reputation,
                        "trend": trend["trend"],
                        "change": trend["change"],
                    }
                )

            return top_users

        except Exception as e:
            logger.error(f"Failed to get top users: {str(e)}")
            return []

    async def get_reputation_distribution(self) -> Dict[str, int]:
        """Get distribution of reputation scores"""
        try:
            if not self.user_reputations:
                return {}

            # Define reputation ranges
            ranges = {
                "excellent": (0.8, 1.0),
                "good": (0.6, 0.8),
                "average": (0.4, 0.6),
                "poor": (0.2, 0.4),
                "very_poor": (0.0, 0.2),
            }

            distribution = {range_name: 0 for range_name in ranges.keys()}

            for reputation in self.user_reputations.values():
                for range_name, (min_val, max_val) in ranges.items():
                    if min_val <= reputation < max_val:
                        distribution[range_name] += 1
                        break

            return distribution

        except Exception as e:
            logger.error(f"Reputation distribution calculation failed: {str(e)}")
            return {}

    async def reputation_decay_task(self):
        """Background task to apply reputation decay"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                if not self.is_initialized:
                    break

                # Apply decay to all users
                for user_id in list(self.user_reputations.keys()):
                    current_reputation = self.user_reputations[user_id]

                    # Apply decay
                    decayed_reputation = current_reputation * (1 - self.decay_rate)
                    decayed_reputation = max(self.min_reputation, decayed_reputation)

                    if (
                        abs(decayed_reputation - current_reputation) > 0.001
                    ):  # Only update if significant change
                        self.user_reputations[user_id] = decayed_reputation

                        # Record decay event
                        await self.update_reputation(
                            user_id,
                            "reputation_decay",
                            {"decay_rate": self.decay_rate, "old_reputation": current_reputation},
                        )

                logger.debug("Applied reputation decay to all users")

            except Exception as e:
                logger.error(f"Reputation decay task failed: {str(e)}")
                await asyncio.sleep(3600)  # Wait before retrying

    async def load_reputations(self):
        """Load reputations from persistent storage"""
        try:
            # In a real implementation, this would load from a database
            # For now, we'll start with empty reputations
            logger.info("Loaded reputations from storage")

        except Exception as e:
            logger.error(f"Failed to load reputations: {str(e)}")

    async def save_reputations(self):
        """Save reputations to persistent storage"""
        try:
            # In a real implementation, this would save to a database
            logger.info(f"Saved {len(self.user_reputations)} reputations to storage")

        except Exception as e:
            logger.error(f"Failed to save reputations: {str(e)}")

    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get overall reputation system statistics"""
        try:
            total_users = len(self.user_reputations)

            if total_users == 0:
                return {"message": "No reputation data available"}

            # Calculate statistics
            reputations = list(self.user_reputations.values())

            stats = {
                "total_users": total_users,
                "average_reputation": np.mean(reputations),
                "median_reputation": np.median(reputations),
                "min_reputation": np.min(reputations),
                "max_reputation": np.max(reputations),
                "reputation_std": np.std(reputations),
                "distribution": await self.get_reputation_distribution(),
                "decay_rate": self.decay_rate,
            }

            return stats

        except Exception as e:
            logger.error(f"Reputation system statistics calculation failed: {str(e)}")
            return {"error": str(e)}
