"""
Content optimization for notification engagement.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import structlog

from ..exceptions import ContentOptimizationError
from ..models.schemas import (
    NotificationCandidate,
    NotificationContent,
    NotificationPreferences,
    Priority,
)

logger = structlog.get_logger()


class HeadlineOptimizer:
    """Optimize notification headlines for engagement."""

    def __init__(self):
        self.engagement_keywords = {
            "urgent": [
                "breaking", "urgent", "important", "alert", "crisis"], "curiosity": [
                "mystery", "secret", "revealed", "discovered", "shocking"], "action": [
                "action", "urgent", "now", "immediately", "critical"], "personal": [
                    "you", "your", "personal", "customized", "tailored"], }

        self.emoji_mapping = {
            "breaking_news": "🚨",
            "urgent": "⚡",
            "important": "❗",
            "trending": "📈",
            "personalized": "👤",
            "technology": "💻",
            "politics": "🏛️",
            "sports": "⚽",
            "entertainment": "🎬",
            "science": "🔬",
            "health": "🏥",
            "economy": "💰",
            "climate": "🌍",
        }

    async def make_engaging(self, title: str, max_length: int = 60) -> str:
        """Make title more engaging."""

        try:
            # Start with original title
            optimized = title.strip()

            # Add urgency words if not present
            if not any(word in optimized.lower()
                       for word in self.engagement_keywords["urgent"]):
                optimized = f"Breaking: {optimized}"

            # Add curiosity words
            if not any(word in optimized.lower()
                       for word in self.engagement_keywords["curiosity"]):
                # Add a curiosity hook
                if len(optimized) < max_length - 10:
                    optimized = f"Shocking: {optimized}"

            # Truncate if too long
            if len(optimized) > max_length:
                optimized = optimized[: max_length - 3] + "..."

            return optimized

        except Exception as e:
            logger.error(f"Error making title engaging: {e}")
            return title[:max_length]

    async def add_urgency(self, title: str, max_length: int = 60) -> str:
        """Add urgency to title."""

        try:
            # Start with original title
            optimized = title.strip()

            # Add urgent prefix if not present
            urgent_prefixes = ["URGENT:", "BREAKING:", "ALERT:", "CRITICAL:"]
            if not any(prefix in optimized.upper()
                       for prefix in urgent_prefixes):
                optimized = f"URGENT: {optimized}"

            # Add action words
            if not any(word in optimized.lower()
                       for word in self.engagement_keywords["action"]):
                if len(optimized) < max_length - 8:
                    optimized = f"{optimized} - Action Required"

            # Truncate if too long
            if len(optimized) > max_length:
                optimized = optimized[: max_length - 3] + "..."

            return optimized

        except Exception as e:
            logger.error(f"Error adding urgency to title: {e}")
            return title[:max_length]

    async def personalize(
            self,
            title: str,
            user_prefs: NotificationPreferences,
            max_length: int = 60) -> str:
        """Personalize title based on user preferences."""

        try:
            # Start with original title
            optimized = title.strip()

            # Add personal touch
            if not any(word in optimized.lower()
                       for word in self.engagement_keywords["personal"]):
                if len(optimized) < max_length - 8:
                    optimized = f"Your {optimized}"

            # Add timezone-aware personalization
            if user_prefs.timezone != "UTC":
                if len(optimized) < max_length - 15:
                    optimized = f"{optimized} (Local Update)"

            # Truncate if too long
            if len(optimized) > max_length:
                optimized = optimized[: max_length - 3] + "..."

            return optimized

        except Exception as e:
            logger.error(f"Error personalizing title: {e}")
            return title[:max_length]

    def optimize_for_ab_test(
            self,
            title: str,
            variant: str,
            max_length: int = 60) -> str:
        """Optimize title for A/B test variant."""

        if variant == "engaging":
            return asyncio.run(self.make_engaging(title, max_length))
        elif variant == "urgent":
            return asyncio.run(self.add_urgency(title, max_length))
        elif variant == "personalized":
            # Use default preferences for personalization
            default_prefs = NotificationPreferences(user_id="default")
            return asyncio.run(
                self.personalize(
                    title,
                    default_prefs,
                    max_length))
        else:
            return title[:max_length]


class EmojiManager:
    """Manage emoji usage in notifications."""

    def __init__(self):
        self.category_emojis = {
            "breaking_news": "🚨",
            "urgent": "⚡",
            "important": "❗",
            "trending": "📈",
            "personalized": "👤",
            "technology": "💻",
            "politics": "🏛️",
            "sports": "⚽",
            "entertainment": "🎬",
            "science": "🔬",
            "health": "🏥",
            "economy": "💰",
            "climate": "🌍",
            "default": "📢",
        }

        self.urgency_emojis = {
            "urgent": "🚨",
            "high": "⚡",
            "medium": "📢",
            "low": "💬"}

    async def add_appropriate_emoji(
        self, title: str, category: str, priority: str = "medium"
    ) -> str:
        """Add appropriate emoji to title."""

        try:
            # Don't add emoji if title already has one
            if any(ord(char) > 127 for char in title):
                return title

            # Get emoji for category
            category_emoji = self.category_emojis.get(
                category, self.category_emojis["default"])

            # Get emoji for priority
            priority_emoji = self.urgency_emojis.get(
                priority, self.urgency_emojis["medium"])

            # Choose emoji based on priority
            emoji = priority_emoji if priority in [
                "urgent", "high"] else category_emoji

            # Add emoji to beginning of title
            return f"{emoji} {title}"

        except Exception as e:
            logger.error(f"Error adding emoji: {e}")
            return title

    def get_emoji_for_category(self, category: str) -> str:
        """Get emoji for specific category."""
        return self.category_emojis.get(
            category, self.category_emojis["default"])

    def get_emoji_for_priority(self, priority: str) -> str:
        """Get emoji for specific priority."""
        return self.urgency_emojis.get(priority, self.urgency_emojis["medium"])


class BodyGenerator:
    """Generate notification body content."""

    def __init__(self):
        self.body_templates = {
            "breaking_news": "🚨 {title}\n\n{summary}\n\nTap to read more",
            "urgent": "⚡ {title}\n\n{summary}\n\nAction required",
            "personalized": "👤 {title}\n\n{summary}\n\nPersonalized for you",
            "trending": "📈 {title}\n\n{summary}\n\nTrending now",
            "default": "{title}\n\n{summary}\n\nTap to read more",
        }

        self.summary_generators = {
            "short": lambda content: content[:100] + "..." if len(content) > 100 else content,
            "medium": lambda content: content[:200] + "..." if len(content) > 200 else content,
            "long": lambda content: content[:300] + "..." if len(content) > 300 else content,
        }

    async def generate_notification_body(
        self, content, strategy_variant: str, max_length: int = 120
    ) -> str:
        """Generate notification body based on strategy."""

        try:
            # Get template based on strategy
            template = self.body_templates.get(
                strategy_variant, self.body_templates["default"])

            # Generate summary
            summary_length = "short" if max_length < 100 else "medium"
            summary = self.summary_generators[summary_length](content.content)

            # Format template
            body = template.format(title=content.title, summary=summary)

            # Truncate if too long
            if len(body) > max_length:
                body = body[: max_length - 3] + "..."

            return body

        except Exception as e:
            logger.error(f"Error generating notification body: {e}")
            return content.title[:max_length]


class NotificationContentOptimizer:
    """Optimize notification content for engagement."""

    def __init__(self):
        self.headline_optimizer = HeadlineOptimizer()
        self.emoji_manager = EmojiManager()
        self.body_generator = BodyGenerator()

    async def initialize(self) -> None:
        """Initialize content optimizer."""
        logger.info("Initializing content optimizer")
        # No specific initialization needed
        logger.info("Content optimizer initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup content optimizer."""
        logger.info("Cleaning up content optimizer")
        # No specific cleanup needed

    async def optimize_notification_content(
        self,
        candidate: NotificationCandidate,
        strategy_variant: str,
        user_prefs: NotificationPreferences,
    ) -> NotificationContent:
        """Optimize notification title and body for maximum engagement."""

        try:
            logger.debug(
                "Optimizing notification content",
                user_id=candidate.user_id,
                strategy_variant=strategy_variant,
            )

            original_content = candidate.content

            # Optimize headline based on strategy
            optimized_title = await self._optimize_title(
                original_content.title, strategy_variant, user_prefs, max_length=60
            )

            # Generate notification body
            notification_body = await self.body_generator.generate_notification_body(
                original_content, strategy_variant, max_length=120
            )

            # Add appropriate emoji if user preference allows
            if user_prefs.allow_emojis:
                optimized_title = await self.emoji_manager.add_appropriate_emoji(
                    optimized_title, original_content.category, candidate.priority.value
                )

            # Determine priority
            priority = self._determine_priority(candidate.urgency_score)

            # Create optimized content
            optimized_content = NotificationContent(
                title=optimized_title,
                body=notification_body,
                action_url=original_content.url,
                image_url=original_content.image_url,
                category=original_content.category,
                priority=priority,
            )

            logger.debug(
                "Content optimization completed",
                user_id=candidate.user_id,
                original_title=original_content.title,
                optimized_title=optimized_title,
            )

            return optimized_content

        except Exception as e:
            logger.error(
                "Error optimizing notification content",
                user_id=candidate.user_id,
                error=str(e),
                exc_info=True,
            )
            raise ContentOptimizationError(
                f"Failed to optimize content: {str(e)}")

    async def _optimize_title(
        self,
        title: str,
        strategy_variant: str,
        user_prefs: NotificationPreferences,
        max_length: int,
    ) -> str:
        """Optimize title based on strategy variant."""

        try:
            if strategy_variant == "engaging":
                return await self.headline_optimizer.make_engaging(title, max_length)
            elif strategy_variant == "urgent":
                return await self.headline_optimizer.add_urgency(title, max_length)
            elif strategy_variant == "personalized":
                return await self.headline_optimizer.personalize(title, user_prefs, max_length)
        else:
                # Default optimization
                return title[:max_length]

        except Exception as e:
            logger.error(f"Error optimizing title: {e}")
            return title[:max_length]

    def _determine_priority(self, urgency_score: float) -> Priority:
        """Determine notification priority based on urgency score."""

        if urgency_score >= 0.9:
            return Priority.URGENT
        elif urgency_score >= 0.7:
            return Priority.HIGH
        elif urgency_score >= 0.5:
            return Priority.MEDIUM
        else:
            return Priority.LOW

    async def get_optimization_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get content optimization analytics for user."""
        try:
            analytics = {
                "user_id": user_id,
                "optimization_strategies": {
                    "engaging": "Makes content more engaging with curiosity hooks",
                    "urgent": "Adds urgency and action words",
                    "personalized": "Personalizes content for user preferences",
                },
                "emoji_usage": {
                    "enabled": True,  # Would check user preferences
                    "categories": list(self.emoji_manager.category_emojis.keys()),
                    "priorities": list(self.emoji_manager.urgency_emojis.keys()),
                },
                "body_templates": list(self.body_generator.body_templates.keys()),
                "optimization_metrics": {
                    "title_max_length": 60,
                    "body_max_length": 120,
                    "summary_lengths": list(self.body_generator.summary_generators.keys()),
                },
            }

            return analytics

        except Exception as e:
            logger.error(f"Error getting optimization analytics: {e}")
            return {}

    async def test_optimization_strategies(
        self, title: str, content: str, user_prefs: NotificationPreferences
    ) -> Dict[str, str]:
        """Test different optimization strategies on content."""

        try:
            strategies = ["engaging", "urgent", "personalized", "default"]
            results = {}

            for strategy in strategies:
                # Create mock candidate
                mock_candidate = NotificationCandidate(
                    user_id="test",
                    content=type(
                        "MockContent",
                        (),
                        {
                            "title": title,
                            "content": content,
                            "category": "test",
                            "url": "https://example.com",
                            "image_url": None,
                        },
                    )(),
                    notification_type=candidate.notification_type,
                    urgency_score=0.5,
                )

                # Optimize content
                optimized = await self.optimize_notification_content(
                    mock_candidate, strategy, user_prefs
                )

                results[strategy] = {
                    "title": optimized.title,
                    "body": optimized.body,
                    "priority": optimized.priority.value,
                }

            return results

        except Exception as e:
            logger.error(f"Error testing optimization strategies: {e}")
            return {}
