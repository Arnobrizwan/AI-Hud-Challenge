"""
Notification fatigue detection and prevention system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import structlog
from redis.asyncio import Redis

from ..exceptions import FatigueDetectionError
from ..models.schemas import FatigueCheck, NotificationType

logger = structlog.get_logger()


class FatigueDetector:
    """Detect and prevent notification fatigue."""

    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client
        self.fatigue_thresholds = {
            NotificationType.BREAKING_NEWS: {"daily": 5, "hourly": 2},
            NotificationType.PERSONALIZED: {"daily": 10, "hourly": 3},
            NotificationType.TRENDING: {"daily": 8, "hourly": 2},
            NotificationType.DIGEST: {"daily": 2, "hourly": 1},
            NotificationType.URGENT: {"daily": 15, "hourly": 5},
            NotificationType.MARKETING: {"daily": 3, "hourly": 1},
        }

        # Advanced fatigue detection parameters
        self.engagement_decay_factor = 0.1  # How much engagement drops with each notification
        self.fatigue_recovery_time = 2  # Hours to wait before fatigue starts recovering
        self.max_fatigue_score = 1.0

    async def initialize(self) -> None:
        """Initialize fatigue detector."""
        logger.info("Initializing fatigue detector")

        # Test Redis connection
        try:
            await self.redis_client.ping()
            logger.info("Fatigue detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize fatigue detector: {e}")
            raise FatigueDetectionError(f"Redis connection failed: {e}")

    async def cleanup(self) -> None:
        """Cleanup fatigue detector resources."""
        logger.info("Cleaning up fatigue detector")
        # No specific cleanup needed for Redis client

    async def check_fatigue(
        self, user_id: str, notification_type: NotificationType
    ) -> FatigueCheck:
        """Check if user is experiencing notification fatigue."""

        try:
            logger.debug(
                "Checking fatigue for user",
                user_id=user_id,
                notification_type=notification_type.value,
            )

            # Get notification counts for different time windows
            hourly_count = await self._get_notification_count(user_id, notification_type, hours=1)
            daily_count = await self._get_notification_count(user_id, notification_type, hours=24)

            # Get user's fatigue score
            fatigue_score = await self._get_user_fatigue_score(user_id)

            # Check against thresholds
            thresholds = self.fatigue_thresholds.get(notification_type, {"daily": 10, "hourly": 3})
            hourly_threshold = thresholds.get("hourly", 3)
            daily_threshold = thresholds.get("daily", 10)

            # Determine if user is fatigued
            is_fatigued = (
                hourly_count >= hourly_threshold
                or daily_count >= daily_threshold
                or fatigue_score >= 0.8
            )

            # Calculate next eligible time
            next_eligible_time = None
            if is_fatigued:
                next_eligible_time = await self._calculate_next_eligible_time(
                    user_id, notification_type, hourly_count, daily_count, thresholds
                )

            # Calculate overall fatigue score
            overall_fatigue_score = self._compute_fatigue_score(
                hourly_count, daily_count, fatigue_score, thresholds
            )

            result = FatigueCheck(
                is_fatigued=is_fatigued,
                hourly_count=hourly_count,
                daily_count=daily_count,
                next_eligible_time=next_eligible_time,
                fatigue_score=overall_fatigue_score,
            )

            logger.debug(
                "Fatigue check completed",
                user_id=user_id,
                is_fatigued=is_fatigued,
                fatigue_score=overall_fatigue_score,
            )

            return result

        except Exception as e:
            logger.error("Error checking fatigue", user_id=user_id, error=str(e), exc_info=True)
            raise FatigueDetectionError(f"Failed to check fatigue: {str(e)}")

    async def record_notification_sent(
        self, user_id: str, notification_type: NotificationType
    ) -> None:
        """Record notification for fatigue tracking."""

        try:
            current_time = datetime.utcnow()

            # Record in Redis with TTL
            hourly_key = f"notif_count:hourly:{user_id}:{notification_type.value}:{current_time.strftime('%Y-%m-%d-%H')}"
            daily_key = f"notif_count:daily:{user_id}:{notification_type.value}:{current_time.strftime('%Y-%m-%d')}"

            # Increment counters
            await self.redis_client.incr(hourly_key)
            await self.redis_client.expire(hourly_key, 3600)  # 1 hour TTL

            await self.redis_client.incr(daily_key)
            await self.redis_client.expire(daily_key, 86400)  # 24 hour TTL

            # Update user fatigue score
            await self._update_user_fatigue_score(user_id, notification_type)

            # Record notification timestamp for pattern analysis
            await self._record_notification_timestamp(user_id, notification_type, current_time)

            logger.debug(
                "Recorded notification sent",
                user_id=user_id,
                notification_type=notification_type.value,
            )

        except Exception as e:
            logger.error(
                "Error recording notification sent", user_id=user_id, error=str(e), exc_info=True
            )
            raise FatigueDetectionError(f"Failed to record notification: {str(e)}")

    async def get_user_fatigue_analytics(self, user_id: str) -> Dict:
        """Get detailed fatigue analytics for a user."""

        try:
            analytics = {
                "user_id": user_id,
                "overall_fatigue_score": await self._get_user_fatigue_score(user_id),
                "notification_counts": {},
                "fatigue_patterns": {},
                "recommendations": [],
            }

            # Get counts for each notification type
            for notification_type in NotificationType:
                hourly_count = await self._get_notification_count(
                    user_id, notification_type, hours=1
                )
                daily_count = await self._get_notification_count(
                    user_id, notification_type, hours=24
                )

                analytics["notification_counts"][notification_type.value] = {
                    "hourly": hourly_count,
                    "daily": daily_count,
                }

            # Analyze fatigue patterns
            analytics["fatigue_patterns"] = await self._analyze_fatigue_patterns(user_id)

            # Generate recommendations
            analytics["recommendations"] = await self._generate_fatigue_recommendations(
                user_id, analytics
            )

            return analytics

        except Exception as e:
            logger.error(
                "Error getting fatigue analytics", user_id=user_id, error=str(e), exc_info=True
            )
            raise FatigueDetectionError(f"Failed to get fatigue analytics: {str(e)}")

    async def _get_notification_count(
        self, user_id: str, notification_type: NotificationType, hours: int
    ) -> int:
        """Get notification count for user and type within time window."""

        try:
            current_time = datetime.utcnow()

            if hours == 1:
                # Hourly count
                key_pattern = f"notif_count:hourly:{user_id}:{notification_type.value}:*"
                keys = await self.redis_client.keys(key_pattern)

                total_count = 0
                for key in keys:
                    count = await self.redis_client.get(key)
                    if count:
                        total_count += int(count)

                return total_count

            elif hours == 24:
                # Daily count
                key_pattern = f"notif_count:daily:{user_id}:{notification_type.value}:*"
                keys = await self.redis_client.keys(key_pattern)

                total_count = 0
                for key in keys:
                    count = await self.redis_client.get(key)
                    if count:
                        total_count += int(count)

                return total_count

            else:
                # Custom time window - count from timestamp records
                cutoff_time = current_time - timedelta(hours=hours)
                timestamp_key = f"notif_timestamps:{user_id}:{notification_type.value}"

                # Get all timestamps
                timestamps = await self.redis_client.lrange(timestamp_key, 0, -1)

                count = 0
                for timestamp_str in timestamps:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp >= cutoff_time:
                        count += 1

                return count

        except Exception as e:
            logger.error(f"Error getting notification count: {e}")
            return 0

    async def _get_user_fatigue_score(self, user_id: str) -> float:
        """Get user's current fatigue score."""

        try:
            fatigue_key = f"fatigue_score:{user_id}"
            score = await self.redis_client.get(fatigue_key)

            if score:
                return float(score)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error getting fatigue score: {e}")
            return 0.0

    async def _update_user_fatigue_score(
        self, user_id: str, notification_type: NotificationType
    ) -> None:
        """Update user's fatigue score based on notification sent."""

        try:
            fatigue_key = f"fatigue_score:{user_id}"
            current_score = await self._get_user_fatigue_score(user_id)

            # Increase fatigue score based on notification type
            fatigue_increase = self._get_fatigue_increase(notification_type)
            new_score = min(current_score + fatigue_increase, self.max_fatigue_score)

            # Set with expiration (fatigue decays over time)
            await self.redis_client.setex(fatigue_key, 86400, new_score)  # 24 hour TTL

            logger.debug(
                "Updated fatigue score",
                user_id=user_id,
                old_score=current_score,
                new_score=new_score,
            )

        except Exception as e:
            logger.error(f"Error updating fatigue score: {e}")

    def _get_fatigue_increase(self, notification_type: NotificationType) -> float:
        """Get fatigue increase amount for notification type."""

        fatigue_increases = {
            NotificationType.BREAKING_NEWS: 0.1,
            NotificationType.PERSONALIZED: 0.05,
            NotificationType.TRENDING: 0.08,
            NotificationType.DIGEST: 0.15,
            NotificationType.URGENT: 0.12,
            NotificationType.MARKETING: 0.2,
        }

        return fatigue_increases.get(notification_type, 0.1)

    async def _record_notification_timestamp(
        self, user_id: str, notification_type: NotificationType, timestamp: datetime
    ) -> None:
        """Record notification timestamp for pattern analysis."""

        try:
            timestamp_key = f"notif_timestamps:{user_id}:{notification_type.value}"

            # Add timestamp to list
            await self.redis_client.lpush(timestamp_key, timestamp.isoformat())

            # Keep only last 100 timestamps
            await self.redis_client.ltrim(timestamp_key, 0, 99)

            # Set expiration
            await self.redis_client.expire(timestamp_key, 86400 * 7)  # 7 days TTL

        except Exception as e:
            logger.error(f"Error recording timestamp: {e}")

    async def _calculate_next_eligible_time(
        self,
        user_id: str,
        notification_type: NotificationType,
        hourly_count: int,
        daily_count: int,
        thresholds: Dict,
    ) -> datetime:
        """Calculate next eligible time for notifications."""

        current_time = datetime.utcnow()

        # If hourly threshold exceeded, wait 1 hour
        if hourly_count >= thresholds.get("hourly", 3):
            return current_time + timedelta(hours=1)

        # If daily threshold exceeded, wait until next day
        if daily_count >= thresholds.get("daily", 10):
            next_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            next_day += timedelta(days=1)
            return next_day

        # If fatigue score is high, wait for recovery
        fatigue_score = await self._get_user_fatigue_score(user_id)
        if fatigue_score >= 0.8:
            recovery_hours = int((fatigue_score - 0.5) * 4)  # 0-2 hours based on score
            return current_time + timedelta(hours=recovery_hours)

        # Default: wait 30 minutes
        return current_time + timedelta(minutes=30)

    def _compute_fatigue_score(
        self, hourly_count: int, daily_count: int, user_fatigue_score: float, thresholds: Dict
    ) -> float:
        """Compute overall fatigue score."""

        hourly_threshold = thresholds.get("hourly", 3)
        daily_threshold = thresholds.get("daily", 10)

        # Calculate fatigue based on counts
        hourly_fatigue = min(hourly_count / hourly_threshold, 1.0)
        daily_fatigue = min(daily_count / daily_threshold, 1.0)

        # Weighted combination
        count_fatigue = (hourly_fatigue * 0.6) + (daily_fatigue * 0.4)

        # Combine with user fatigue score
        overall_fatigue = (count_fatigue * 0.7) + (user_fatigue_score * 0.3)

        return min(overall_fatigue, 1.0)

    async def _analyze_fatigue_patterns(self, user_id: str) -> Dict:
        """Analyze user's fatigue patterns."""

        try:
            patterns = {"peak_hours": [], "fatigue_trends": {}, "notification_frequency": {}}

            # Analyze notification timestamps for each type
            for notification_type in NotificationType:
                timestamp_key = f"notif_timestamps:{user_id}:{notification_type.value}"
                timestamps = await self.redis_client.lrange(timestamp_key, 0, -1)

                if timestamps:
                    # Convert to datetime objects
                    dt_timestamps = [datetime.fromisoformat(ts) for ts in timestamps]

                    # Analyze peak hours
                    hour_counts = {}
                    for dt in dt_timestamps:
                        hour = dt.hour
                        hour_counts[hour] = hour_counts.get(hour, 0) + 1

                    # Find peak hours
                    if hour_counts:
                        max_count = max(hour_counts.values())
                        peak_hours = [h for h, c in hour_counts.items() if c >= max_count * 0.8]
                        patterns["peak_hours"].extend(peak_hours)

                    # Calculate frequency
                    if len(dt_timestamps) > 1:
                        time_span = (max(dt_timestamps) - min(dt_timestamps)).total_seconds() / 3600
                        frequency = len(dt_timestamps) / max(time_span, 1)
                        patterns["notification_frequency"][notification_type.value] = frequency

            # Remove duplicates from peak hours
            patterns["peak_hours"] = list(set(patterns["peak_hours"]))

            return patterns

        except Exception as e:
            logger.error(f"Error analyzing fatigue patterns: {e}")
            return {}

    async def _generate_fatigue_recommendations(self, user_id: str, analytics: Dict) -> list:
        """Generate fatigue management recommendations."""

        recommendations = []

        # Check for high fatigue score
        if analytics["overall_fatigue_score"] > 0.7:
            recommendations.append(
                {
                    "type": "fatigue_high",
                    "message": "Consider reducing notification frequency",
                    "priority": "high",
                }
            )

        # Check for excessive hourly notifications
        for notif_type, counts in analytics["notification_counts"].items():
            if counts["hourly"] > 5:
                recommendations.append(
                    {
                        "type": "hourly_excessive",
                        "message": f"Too many {notif_type} notifications per hour",
                        "priority": "medium",
                    }
                )

        # Check for excessive daily notifications
        for notif_type, counts in analytics["notification_counts"].items():
            if counts["daily"] > 20:
                recommendations.append(
                    {
                        "type": "daily_excessive",
                        "message": f"Too many {notif_type} notifications per day",
                        "priority": "medium",
                    }
                )

        return recommendations
