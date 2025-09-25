"""
Breaking news detection and thresholding system.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import structlog
from redis.asyncio import Redis

from ..models.schemas import NotificationCandidate, BreakingNewsEvent

logger = structlog.get_logger()


class BreakingNewsDetector:
    """Detects and scores breaking news events."""

    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client
        self.velocity_threshold = 0.7
        self.source_weight_threshold = 0.8
        self.user_interest_threshold = 0.6
        self.time_window_hours = 1

    async def initialize(self) -> None:
        """Initialize the breaking news detector."""
        logger.info("Initializing breaking news detector")
        # Initialize any required models or caches
        logger.info("Breaking news detector initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up breaking news detector")

    async def detect_breaking_news(
        self, 
        candidate: NotificationCandidate,
        cluster_velocity: float,
        source_weight: float,
        user_interest: float
    ) -> Tuple[bool, float, BreakingNewsEvent]:
        """
        Detect if content qualifies as breaking news.
        
        Args:
            candidate: Notification candidate
            cluster_velocity: Velocity of articles in the cluster
            source_weight: Authority weight of the source
            user_interest: User's interest in the topic
            
        Returns:
            Tuple of (is_breaking, breaking_score, breaking_event)
        """
        try:
            # Calculate breaking news score
            breaking_score = await self._calculate_breaking_score(
                candidate, cluster_velocity, source_weight, user_interest
            )

            # Determine if it's breaking news
            is_breaking = breaking_score >= self.velocity_threshold

            # Create breaking news event
            breaking_event = BreakingNewsEvent(
                event_id=f"breaking_{candidate.content.id}",
                title=candidate.content.title,
                cluster_id=candidate.content.cluster_id,
                breaking_score=breaking_score,
                velocity=cluster_velocity,
                source_weight=source_weight,
                user_interest=user_interest,
                detected_at=datetime.utcnow(),
                metadata={
                    "content_id": candidate.content.id,
                    "source": candidate.content.source,
                    "topics": candidate.content.topics,
                    "entities": candidate.content.entities,
                }
            )

            if is_breaking:
                # Store breaking news event
                await self._store_breaking_event(breaking_event)
                
                # Update cluster velocity
                await self._update_cluster_velocity(candidate.content.cluster_id, cluster_velocity)

                logger.info(
                    "Breaking news detected",
                    content_id=candidate.content.id,
                    breaking_score=breaking_score,
                    velocity=cluster_velocity,
                    source_weight=source_weight
                )

            return is_breaking, breaking_score, breaking_event

        except Exception as e:
            logger.error(
                "Error detecting breaking news",
                content_id=candidate.content.id,
                error=str(e),
                exc_info=True
            )
            return False, 0.0, None

    async def _calculate_breaking_score(
        self,
        candidate: NotificationCandidate,
        cluster_velocity: float,
        source_weight: float,
        user_interest: float
    ) -> float:
        """Calculate breaking news score."""
        
        # Velocity component (40% weight)
        velocity_score = min(cluster_velocity, 1.0)
        
        # Source weight component (30% weight)
        source_score = min(source_weight, 1.0)
        
        # User interest component (20% weight)
        interest_score = min(user_interest, 1.0)
        
        # Content freshness component (10% weight)
        freshness_score = self._calculate_freshness_score(candidate.content.published_at)
        
        # Calculate weighted score
        breaking_score = (
            velocity_score * 0.4 +
            source_score * 0.3 +
            interest_score * 0.2 +
            freshness_score * 0.1
        )
        
        return min(breaking_score, 1.0)

    def _calculate_freshness_score(self, published_at: datetime) -> float:
        """Calculate content freshness score."""
        now = datetime.utcnow()
        age_minutes = (now - published_at).total_seconds() / 60
        
        # More fresh = higher score
        if age_minutes < 15:
            return 1.0
        elif age_minutes < 60:
            return 0.8
        elif age_minutes < 240:  # 4 hours
            return 0.6
        elif age_minutes < 1440:  # 24 hours
            return 0.4
        else:
            return 0.2

    async def _store_breaking_event(self, breaking_event: BreakingNewsEvent) -> None:
        """Store breaking news event in Redis."""
        try:
            event_key = f"breaking_news:{breaking_event.event_id}"
            event_data = {
                "event_id": breaking_event.event_id,
                "title": breaking_event.title,
                "cluster_id": breaking_event.cluster_id,
                "breaking_score": breaking_event.breaking_score,
                "velocity": breaking_event.velocity,
                "source_weight": breaking_event.source_weight,
                "user_interest": breaking_event.user_interest,
                "detected_at": breaking_event.detected_at.isoformat(),
                "metadata": breaking_event.metadata,
            }
            
            # Store with TTL of 24 hours
            await self.redis_client.hset(event_key, mapping=event_data)
            await self.redis_client.expire(event_key, 86400)
            
            # Add to breaking news list
            await self.redis_client.lpush("breaking_news_events", breaking_event.event_id)
            await self.redis_client.ltrim("breaking_news_events", 0, 999)  # Keep last 1000 events
            
        except Exception as e:
            logger.error("Error storing breaking news event", error=str(e))

    async def _update_cluster_velocity(self, cluster_id: str, velocity: float) -> None:
        """Update cluster velocity in Redis."""
        try:
            velocity_key = f"cluster_velocity:{cluster_id}"
            await self.redis_client.hset(velocity_key, "velocity", velocity)
            await self.redis_client.hset(velocity_key, "updated_at", datetime.utcnow().isoformat())
            await self.redis_client.expire(velocity_key, 3600)  # 1 hour TTL
            
        except Exception as e:
            logger.error("Error updating cluster velocity", cluster_id=cluster_id, error=str(e))

    async def get_recent_breaking_events(self, limit: int = 10) -> List[BreakingNewsEvent]:
        """Get recent breaking news events."""
        try:
            event_ids = await self.redis_client.lrange("breaking_news_events", 0, limit - 1)
            events = []
            
            for event_id in event_ids:
                event_key = f"breaking_news:{event_id.decode()}"
                event_data = await self.redis_client.hgetall(event_key)
                
                if event_data:
                    event = BreakingNewsEvent(
                        event_id=event_data[b"event_id"].decode(),
                        title=event_data[b"title"].decode(),
                        cluster_id=event_data[b"cluster_id"].decode(),
                        breaking_score=float(event_data[b"breaking_score"]),
                        velocity=float(event_data[b"velocity"]),
                        source_weight=float(event_data[b"source_weight"]),
                        user_interest=float(event_data[b"user_interest"]),
                        detected_at=datetime.fromisoformat(event_data[b"detected_at"].decode()),
                        metadata=eval(event_data[b"metadata"].decode()) if event_data.get(b"metadata") else {},
                    )
                    events.append(event)
            
            return events
            
        except Exception as e:
            logger.error("Error getting recent breaking events", error=str(e))
            return []

    async def get_cluster_velocity(self, cluster_id: str) -> float:
        """Get current cluster velocity."""
        try:
            velocity_key = f"cluster_velocity:{cluster_id}"
            velocity_data = await self.redis_client.hget(velocity_key, "velocity")
            
            if velocity_data:
                return float(velocity_data)
            else:
                return 0.0
                
        except Exception as e:
            logger.error("Error getting cluster velocity", cluster_id=cluster_id, error=str(e))
            return 0.0

    async def cleanup_old_events(self) -> None:
        """Clean up old breaking news events."""
        try:
            # Remove events older than 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            # Get all breaking news events
            event_ids = await self.redis_client.lrange("breaking_news_events", 0, -1)
            
            for event_id in event_ids:
                event_key = f"breaking_news:{event_id.decode()}"
                event_data = await self.redis_client.hget(event_key, "detected_at")
                
                if event_data:
                    detected_at = datetime.fromisoformat(event_data.decode())
                    if detected_at < cutoff_time:
                        # Remove old event
                        await self.redis_client.delete(event_key)
                        await self.redis_client.lrem("breaking_news_events", 1, event_id)
            
            logger.info("Cleaned up old breaking news events")
            
        except Exception as e:
            logger.error("Error cleaning up old events", error=str(e))
