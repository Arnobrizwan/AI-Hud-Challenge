"""
User notification preference management system.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import UserPreferences, get_async_session
from ..exceptions import PreferenceManagementError
from ..models.schemas import DeliveryChannel, NotificationPreferences, NotificationType

logger = structlog.get_logger()


class NotificationPreferenceManager:
    """Manage user notification preferences."""

    def __init__(self):
        self.preferences_cache = {}  # In-memory cache
        self.cache_ttl = 300  # 5 minutes

    async def initialize(self) -> None:
        """Initialize preference manager."""
        logger.info("Initializing notification preference manager")
        # No specific initialization needed
        logger.info("Notification preference manager initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup preference manager."""
        logger.info("Cleaning up notification preference manager")
        self.preferences_cache.clear()

    async def get_preferences(self, user_id: str) -> NotificationPreferences:
        """Get user notification preferences."""

        try:
            # Check cache first
            if user_id in self.preferences_cache:
                cached_prefs, timestamp = self.preferences_cache[user_id]
                if (datetime.utcnow() - timestamp).seconds < self.cache_ttl:
                    return cached_prefs

            # Fetch from database
            prefs = await self._fetch_preferences_from_db(user_id)

            # Cache preferences
            self.preferences_cache[user_id] = (prefs, datetime.utcnow())

            logger.debug(
                "Retrieved user preferences",
                user_id=user_id,
                enabled_types=[t.value for t in prefs.enabled_types],
                channels=[c.value for c in prefs.delivery_channels],
            )

            return prefs

        except Exception as e:
            logger.error(
                "Error getting user preferences",
                user_id=user_id,
                error=str(e),
                exc_info=True)
            # Return default preferences
            return self._create_default_preferences(user_id)

    async def update_preferences(
        self, user_id: str, preferences: NotificationPreferences
    ) -> NotificationPreferences:
        """Update user notification preferences."""

        try:
            # Update in database
            await self._update_preferences_in_db(user_id, preferences)

            # Update cache
            self.preferences_cache[user_id] = (preferences, datetime.utcnow())

            logger.info(
                "Updated user preferences",
                user_id=user_id,
                enabled_types=[t.value for t in preferences.enabled_types],
            )

            return preferences

        except Exception as e:
            logger.error(
                "Error updating user preferences",
                user_id=user_id,
                error=str(e),
                exc_info=True)
            raise PreferenceManagementError(
                f"Failed to update preferences: {str(e)}")

    async def update_notification_type_preference(
        self, user_id: str, notification_type: NotificationType, enabled: bool
    ) -> NotificationPreferences:
        """Update preference for specific notification type."""

        try:
            # Get current preferences
            prefs = await self.get_preferences(user_id)

            # Update notification type
            if enabled and notification_type not in prefs.enabled_types:
                prefs.enabled_types.append(notification_type)
            elif not enabled and notification_type in prefs.enabled_types:
                prefs.enabled_types.remove(notification_type)

            # Update preferences
            updated_prefs = await self.update_preferences(user_id, prefs)

            logger.info(
                "Updated notification type preference",
                user_id=user_id,
                notification_type=notification_type.value,
                enabled=enabled,
            )

            return updated_prefs

        except Exception as e:
            logger.error(
                "Error updating notification type preference",
                user_id=user_id,
                notification_type=notification_type.value,
                error=str(e),
            )
            raise PreferenceManagementError(
                f"Failed to update notification type preference: {str(e)}"
            )

    async def update_delivery_channel_preference(
        self, user_id: str, channel: DeliveryChannel, enabled: bool
    ) -> NotificationPreferences:
        """Update preference for specific delivery channel."""

        try:
            # Get current preferences
            prefs = await self.get_preferences(user_id)

            # Update delivery channel
            if enabled and channel not in prefs.delivery_channels:
                prefs.delivery_channels.append(channel)
            elif not enabled and channel in prefs.delivery_channels:
                prefs.delivery_channels.remove(channel)

            # Update preferences
            updated_prefs = await self.update_preferences(user_id, prefs)

            logger.info(
                "Updated delivery channel preference",
                user_id=user_id,
                channel=channel.value,
                enabled=enabled,
            )

            return updated_prefs

        except Exception as e:
            logger.error(
                "Error updating delivery channel preference",
                user_id=user_id,
                channel=channel.value,
                error=str(e),
            )
            raise PreferenceManagementError(
                f"Failed to update delivery channel preference: {str(e)}"
            )

    async def update_quiet_hours(
        self, user_id: str, start_hour: int, end_hour: int
    ) -> NotificationPreferences:
        """Update user's quiet hours."""

        try:
            # Validate hours
            if not (0 <= start_hour <= 23) or not (0 <= end_hour <= 23):
                raise PreferenceManagementError(
                    "Invalid quiet hours: must be between 0-23")

            # Get current preferences
            prefs = await self.get_preferences(user_id)

            # Update quiet hours
            prefs.quiet_hours_start = start_hour
            prefs.quiet_hours_end = end_hour

            # Update preferences
            updated_prefs = await self.update_preferences(user_id, prefs)

            logger.info(
                "Updated quiet hours",
                user_id=user_id,
                start_hour=start_hour,
                end_hour=end_hour)

            return updated_prefs

        except Exception as e:
            logger.error(
                "Error updating quiet hours",
                user_id=user_id,
                error=str(e))
            raise PreferenceManagementError(
                f"Failed to update quiet hours: {str(e)}")

    async def update_relevance_threshold(
            self,
            user_id: str,
            notification_type: NotificationType,
            threshold: float) -> NotificationPreferences:
        """Update relevance threshold for notification type."""

        try:
            # Validate threshold
            if not (0.0 <= threshold <= 1.0):
                raise PreferenceManagementError(
                    "Invalid threshold: must be between 0.0 and 1.0")

            # Get current preferences
            prefs = await self.get_preferences(user_id)

            # Update threshold
            prefs.relevance_thresholds[notification_type] = threshold

            # Update preferences
            updated_prefs = await self.update_preferences(user_id, prefs)

            logger.info(
                "Updated relevance threshold",
                user_id=user_id,
                notification_type=notification_type.value,
                threshold=threshold,
            )

            return updated_prefs

        except Exception as e:
            logger.error(
                "Error updating relevance threshold",
                user_id=user_id,
                error=str(e))
            raise PreferenceManagementError(
                f"Failed to update relevance threshold: {str(e)}")

    async def get_preference_analytics(self, user_id: str) -> Dict[str, Any]:
    """Get preference analytics for user."""
        try:
            prefs = await self.get_preferences(user_id)

            analytics = {
                "user_id": user_id,
                "enabled_types": [
                    t.value for t in prefs.enabled_types],
                "delivery_channels": [
                    c.value for c in prefs.delivery_channels],
                "quiet_hours": {
                    "start": prefs.quiet_hours_start,
                    "end": prefs.quiet_hours_end},
                "timezone": prefs.timezone,
                "allow_emojis": prefs.allow_emojis,
                "max_daily_notifications": prefs.max_daily_notifications,
                "max_hourly_notifications": prefs.max_hourly_notifications,
                "relevance_thresholds": {
                    k.value: v for k,
                    v in prefs.relevance_thresholds.items()},
                "preference_diversity": self._calculate_preference_diversity(prefs),
            }

            return analytics

        except Exception as e:
            logger.error(
                "Error getting preference analytics",
                user_id=user_id,
                error=str(e))
            return {}

    async def _fetch_preferences_from_db(
            self, user_id: str) -> NotificationPreferences:
        """Fetch preferences from database."""

        try:
            async with get_async_session() as session:
                # Query user preferences
                result = await session.execute(
                    "SELECT * FROM user_preferences WHERE user_id = :user_id", {"user_id": user_id}
                )
                row = result.fetchone()

                if row:
                    # Convert database row to NotificationPreferences
                    return NotificationPreferences(
                        user_id=row.user_id,
                        enabled_types=[NotificationType(t) for t in (row.enabled_types or [])],
                        delivery_channels=[
                            DeliveryChannel(c) for c in (row.delivery_channels or [])
                        ],
                        quiet_hours_start=row.quiet_hours_start,
                        quiet_hours_end=row.quiet_hours_end,
                        timezone=row.timezone or "UTC",
                        allow_emojis=row.allow_emojis,
                        max_daily_notifications=row.max_daily_notifications or 50,
                        max_hourly_notifications=row.max_hourly_notifications or 10,
                        relevance_thresholds={
                            NotificationType(k): v
                            for k, v in (row.relevance_thresholds or {}).items()
                        },
                        metadata=row.metadata or {},
                    )
                else:
                    # Create default preferences if not found
                    return self._create_default_preferences(user_id)

        except Exception as e:
            logger.error(f"Error fetching preferences from database: {e}")
            return self._create_default_preferences(user_id)

    async def _update_preferences_in_db(
        self, user_id: str, preferences: NotificationPreferences
    ) -> None:
        """Update preferences in database."""

        try:
            async with get_async_session() as session:
                # Check if preferences exist
                result = await session.execute(
                    "SELECT user_id FROM user_preferences WHERE user_id = :user_id",
                    {"user_id": user_id},
                )
                exists = result.fetchone() is not None

                if exists:
                    # Update existing preferences
                    await session.execute(
                        """
                        UPDATE user_preferences
                        SET enabled_types = :enabled_types,
                            delivery_channels = :delivery_channels,
                            quiet_hours_start = :quiet_hours_start,
                            quiet_hours_end = :quiet_hours_end,
                            timezone = :timezone,
                            allow_emojis = :allow_emojis,
                            max_daily_notifications = :max_daily_notifications,
                            max_hourly_notifications = :max_hourly_notifications,
                            relevance_thresholds = :relevance_thresholds,
                            metadata = :metadata,
                            updated_at = :updated_at
                        WHERE user_id = :user_id
                        """,
                        {
                            "user_id": user_id,
                            "enabled_types": [t.value for t in preferences.enabled_types],
                            "delivery_channels": [c.value for c in preferences.delivery_channels],
                            "quiet_hours_start": preferences.quiet_hours_start,
                            "quiet_hours_end": preferences.quiet_hours_end,
                            "timezone": preferences.timezone,
                            "allow_emojis": preferences.allow_emojis,
                            "max_daily_notifications": preferences.max_daily_notifications,
                            "max_hourly_notifications": preferences.max_hourly_notifications,
                            "relevance_thresholds": {
                                k.value: v for k, v in preferences.relevance_thresholds.items()
                            },
                            "metadata": preferences.metadata,
                            "updated_at": datetime.utcnow(),
                        },
                    )
                else:
                    # Insert new preferences
                    await session.execute(
                        """
                        INSERT INTO user_preferences
                        (user_id, enabled_types, delivery_channels, quiet_hours_start,
                         quiet_hours_end, timezone, allow_emojis, max_daily_notifications,
                         max_hourly_notifications, relevance_thresholds, metadata,
                         created_at, updated_at)
                        VALUES
                        (:user_id, :enabled_types, :delivery_channels, :quiet_hours_start,
                         :quiet_hours_end, :timezone, :allow_emojis, :max_daily_notifications,
                         :max_hourly_notifications, :relevance_thresholds, :metadata,
                         :created_at, :updated_at)
                        """,
                        {
                            "user_id": user_id,
                            "enabled_types": [t.value for t in preferences.enabled_types],
                            "delivery_channels": [c.value for c in preferences.delivery_channels],
                            "quiet_hours_start": preferences.quiet_hours_start,
                            "quiet_hours_end": preferences.quiet_hours_end,
                            "timezone": preferences.timezone,
                            "allow_emojis": preferences.allow_emojis,
                            "max_daily_notifications": preferences.max_daily_notifications,
                            "max_hourly_notifications": preferences.max_hourly_notifications,
                            "relevance_thresholds": {
                                k.value: v for k, v in preferences.relevance_thresholds.items()
                            },
                            "metadata": preferences.metadata,
                            "created_at": datetime.utcnow(),
                            "updated_at": datetime.utcnow(),
                        },
                    )

                await session.commit()

        except Exception as e:
            logger.error(f"Error updating preferences in database: {e}")
            raise PreferenceManagementError(
                f"Database update failed: {str(e)}")

    def _create_default_preferences(
            self, user_id: str) -> NotificationPreferences:
        """Create default notification preferences for user."""

        return NotificationPreferences(
            user_id=user_id,
            enabled_types=[
                NotificationType.BREAKING_NEWS,
                NotificationType.PERSONALIZED,
                NotificationType.TRENDING,
            ],
            delivery_channels=[DeliveryChannel.PUSH, DeliveryChannel.EMAIL],
            quiet_hours_start=22,  # 10 PM
            quiet_hours_end=8,  # 8 AM
            timezone="UTC",
            allow_emojis=True,
            max_daily_notifications=50,
            max_hourly_notifications=10,
            relevance_thresholds={
                NotificationType.BREAKING_NEWS: 0.2,
                NotificationType.PERSONALIZED: 0.3,
                NotificationType.TRENDING: 0.4,
                NotificationType.DIGEST: 0.5,
                NotificationType.URGENT: 0.1,
                NotificationType.MARKETING: 0.6,
            },
            metadata={},
        )

    def _calculate_preference_diversity(
            self, prefs: NotificationPreferences) -> Dict[str, float]:
        """Calculate preference diversity metrics."""

        return {
            "notification_type_diversity": len(prefs.enabled_types) / len(NotificationType),
            "channel_diversity": len(prefs.delivery_channels) / len(DeliveryChannel),
            "threshold_diversity": len(set(prefs.relevance_thresholds.values()))
            / len(NotificationType),
            "has_quiet_hours": prefs.quiet_hours_start is not None
            and prefs.quiet_hours_end is not None,
        }
