"""
Core notification decision engine with ML optimization.
"""

import asyncio
from datetime import datetime
from typing import List

import structlog
from redis.asyncio import Redis

from ..ab_testing.ab_tester import ABTestingFramework
from ..breaking_news.breaking_news_detector import BreakingNewsDetector
from ..delivery.delivery_manager import MultiChannelDelivery
from ..exceptions import NotificationDeliveryError, NotificationError
from ..fatigue.fatigue_detector import FatigueDetector
from ..models.schemas import (
    DeliveryChannel,
    DeliveryResult,
    NotificationCandidate,
    NotificationDecision,
    NotificationType,
    Priority,
)
from ..optimization.content_optimizer import NotificationContentOptimizer
from ..policy.policy_engine import PolicyEngine
from ..preferences.preference_manager import NotificationPreferenceManager
from ..relevance.relevance_scorer import RelevanceScorer
from ..timing.timing_predictor import NotificationTimingModel

logger = structlog.get_logger()


class NotificationDecisionEngine:
    """Intelligent notification decisioning with ML optimization."""

    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client
        self.fatigue_detector = FatigueDetector(redis_client)
        self.preference_manager = NotificationPreferenceManager()
        self.timing_predictor = NotificationTimingModel()
        self.relevance_scorer = RelevanceScorer()
        self.delivery_manager = MultiChannelDelivery()
        self.content_optimizer = NotificationContentOptimizer()
        self.ab_tester = ABTestingFramework(redis_client)
        self.breaking_news_detector = BreakingNewsDetector(redis_client)
        self.policy_engine = PolicyEngine(redis_client)

        # Performance tracking
        self._decision_count = 0
        self._max_concurrent = 1000
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

    async def initialize(self) -> None:
        """Initialize the decision engine."""
        logger.info("Initializing notification decision engine")

        # Initialize components
        await self.fatigue_detector.initialize()
        await self.preference_manager.initialize()
        await self.timing_predictor.initialize()
        await self.relevance_scorer.initialize()
        await self.delivery_manager.initialize()
        await self.content_optimizer.initialize()
        await self.ab_tester.initialize()
        await self.breaking_news_detector.initialize()
        await self.policy_engine.initialize()

        logger.info("Notification decision engine initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up notification decision engine")

        # Cleanup components
        await self.fatigue_detector.cleanup()
        await self.preference_manager.cleanup()
        await self.timing_predictor.cleanup()
        await self.relevance_scorer.cleanup()
        await self.delivery_manager.cleanup()
        await self.content_optimizer.cleanup()
        await self.ab_tester.cleanup()
        await self.breaking_news_detector.cleanup()
        await self.policy_engine.cleanup()

        logger.info("Notification decision engine cleanup completed")

    async def process_notification_candidate(self, candidate: NotificationCandidate) -> NotificationDecision:
        """Main notification decisioning pipeline."""

        async with self._semaphore:
            start_time = datetime.utcnow()

            try:
                logger.info(
                    "Processing notification candidate",
                    user_id=candidate.user_id,
                    notification_type=candidate.notification_type.value,
                    urgency_score=candidate.urgency_score,
                )

                # Get user notification preferences
                user_prefs = await self.preference_manager.get_preferences(candidate.user_id)

                # Check if notifications are enabled for this type
                if not self._is_notification_type_enabled(candidate.notification_type, user_prefs):
                    decision = NotificationDecision(
                        should_send=False,
                        reason="notification_type_disabled",
                        user_id=candidate.user_id,
                    )
                    await self._log_decision(decision, start_time)
                    return decision

                # Check for notification fatigue (unless bypassed)
                if not candidate.bypass_fatigue:
                    fatigue_check = await self.fatigue_detector.check_fatigue(
                        candidate.user_id, candidate.notification_type
                    )

                    if fatigue_check.is_fatigued:
                        decision = NotificationDecision(
                            should_send=False,
                            reason="user_fatigue_detected",
                            next_eligible_time=fatigue_check.next_eligible_time,
                            user_id=candidate.user_id,
                        )
                        await self._log_decision(decision, start_time)
                        return decision

                # Score content relevance and urgency
                relevance_score = await self.relevance_scorer.score_relevance(candidate, user_prefs)

                urgency_score = await self._compute_urgency_score(candidate)

                # Detect breaking news
                cluster_velocity = await self.breaking_news_detector.get_cluster_velocity(
                    candidate.content.cluster_id if hasattr(candidate.content, 'cluster_id') else "unknown"
                )
                source_weight = await self._get_source_weight(candidate.content.source)
                user_interest = await self._get_user_interest(candidate.user_id, candidate.content.topics)
                
                is_breaking, breaking_score, breaking_event = await self.breaking_news_detector.detect_breaking_news(
                    candidate, cluster_velocity, source_weight, user_interest
                )

                # Check if content meets notification threshold
                notification_threshold = user_prefs.get_threshold(candidate.notification_type)
                
                # Adjust threshold for breaking news
                if is_breaking:
                    notification_threshold *= 0.7  # Lower threshold for breaking news
                
                combined_score = (relevance_score * 0.7) + (urgency_score * 0.3)
                
                # Boost score for breaking news
                if is_breaking:
                    combined_score = max(combined_score, breaking_score)

                if combined_score < notification_threshold:
                    decision = NotificationDecision(
                        should_send=False,
                        reason="below_relevance_threshold",
                        score=combined_score,
                        threshold=notification_threshold,
                        user_id=candidate.user_id,
                    )
                    await self._log_decision(decision, start_time)
                    return decision

                # Predict optimal delivery time
                optimal_timing = await self.timing_predictor.predict_optimal_time(
                    candidate.user_id, candidate.notification_type
                )

                # Determine delivery channel
                delivery_channel = await self._select_delivery_channel(candidate, user_prefs, optimal_timing)

                # Get A/B test variant for notification strategy
                strategy_variant = await self.ab_tester.get_variant(
                    user_id=candidate.user_id, experiment="notification_strategy_v3"
                )

                # Optimize notification content
                optimized_content = await self.content_optimizer.optimize_notification_content(
                    candidate, strategy_variant, user_prefs
                )

                # Create preliminary decision
                preliminary_decision = NotificationDecision(
                    should_send=True,
                    delivery_time=optimal_timing.scheduled_time,
                    delivery_channel=delivery_channel,
                    content=optimized_content,
                    priority=self._compute_priority(urgency_score, relevance_score),
                    strategy_variant=strategy_variant,
                    user_id=candidate.user_id,
                    score=combined_score,
                    is_breaking=is_breaking,
                    breaking_score=breaking_score,
                    metadata={
                        "relevance_score": relevance_score,
                        "urgency_score": urgency_score,
                        "predicted_engagement": optimal_timing.predicted_engagement,
                        "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                        "breaking_event": breaking_event.dict() if breaking_event else None,
                    },
                )

                # Evaluate notification policies
                policy_allowed, policy_reason, policy_metadata = await self.policy_engine.evaluate_notification_policy(
                    preliminary_decision, candidate.user_id, candidate.notification_type.value
                )

                if not policy_allowed:
                    decision = NotificationDecision(
                        should_send=False,
                        reason=f"policy_violation: {policy_reason}",
                        user_id=candidate.user_id,
                        metadata=policy_metadata,
                    )
                    await self._log_decision(decision, start_time)
                    return decision

                # Final decision
                decision = preliminary_decision

                await self._log_decision(decision, start_time)
                return decision

            except Exception as e:
                logger.error(
                    "Error processing notification candidate",
                    user_id=candidate.user_id,
                    error=str(e),
                    exc_info=True,
                )

                decision = NotificationDecision(
                    should_send=False,
                    reason="processing_error",
                    user_id=candidate.user_id,
                    metadata={"error": str(e)},
                )
                await self._log_decision(decision, start_time)
                return decision

    async def execute_notification_delivery(self, decision: NotificationDecision) -> DeliveryResult:
        """Execute notification delivery with retry logic."""

        if not decision.should_send:
            raise NotificationError("Cannot deliver notification that should not be sent")

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                logger.info(
                    "Executing notification delivery",
                    user_id=decision.user_id,
                    channel=decision.delivery_channel.value,
                    attempt=retry_count + 1,
                )

                result = await self.delivery_manager.deliver_notification(decision)

                # Record notification sent for fatigue tracking
                await self.fatigue_detector.record_notification_sent(decision.user_id, decision.content.category)

                # Log successful delivery
                await self._log_delivery_success(decision, result)

                return result

            except Exception as e:
                retry_count += 1
                logger.error(
                    "Notification delivery failed",
                    user_id=decision.user_id,
                    attempt=retry_count,
                    error=str(e),
                    exc_info=True,
                )

                if retry_count < max_retries:
                    # Exponential backoff
                    await asyncio.sleep(2**retry_count)
                else:
                    # Log failed delivery
                    await self._log_delivery_failure(decision, str(e))
                    raise NotificationDeliveryError(f"Failed to deliver notification: {str(e)}")

    async def process_batch_notifications(self, candidates: List[NotificationCandidate]) -> List[NotificationDecision]:
        """Process multiple notification candidates in parallel."""

        logger.info("Processing batch notifications", batch_size=len(candidates))

        # Process candidates in parallel with concurrency limit
        tasks = [self.process_notification_candidate(candidate) for candidate in candidates]

        decisions = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_decisions = []
        for i, decision in enumerate(decisions):
            if isinstance(decision, Exception):
                logger.error("Error processing candidate in batch", candidate_index=i, error=str(decision))
                processed_decisions.append(
                    NotificationDecision(
                        should_send=False,
                        reason="batch_processing_error",
                        user_id=candidates[i].user_id,
                        metadata={"error": str(decision)},
                    )
                )
            else:
                processed_decisions.append(decision)

        logger.info(
            "Batch processing completed",
            total_candidates=len(candidates),
            successful_decisions=len([d for d in processed_decisions if not isinstance(d, Exception)]),
        )

        return processed_decisions

    def _is_notification_type_enabled(self, notification_type: NotificationType, user_prefs) -> bool:
        """Check if notification type is enabled for user."""
        return user_prefs.is_notification_type_enabled(notification_type)

    async def _compute_urgency_score(self, candidate: NotificationCandidate) -> float:
        """Compute urgency score for notification candidate."""
        # Base urgency from candidate
        base_urgency = candidate.urgency_score

        # Time-based urgency (more urgent if older)
        time_urgency = self._compute_time_urgency(candidate.content.published_at)

        # Breaking news urgency
        breaking_urgency = 0.9 if candidate.content.is_breaking else 0.0

        # Combine urgency factors
        urgency_score = max(base_urgency, time_urgency, breaking_urgency)

        return min(urgency_score, 1.0)

    def _compute_time_urgency(self, published_at: datetime) -> float:
        """Compute urgency based on content age."""
        now = datetime.utcnow()
        age_hours = (now - published_at).total_seconds() / 3600

        # More urgent if very recent, less urgent if older
        if age_hours < 1:
            return 0.8
        elif age_hours < 6:
            return 0.6
        elif age_hours < 24:
            return 0.4
        else:
            return 0.2

    async def _select_delivery_channel(
        self, candidate: NotificationCandidate, user_prefs, optimal_timing
    ) -> DeliveryChannel:
        """Select optimal delivery channel."""
        return await self.delivery_manager.select_optimal_channel(
            candidate.user_id, candidate.notification_type, candidate.urgency_score
        )

    def _compute_priority(self, urgency_score: float, relevance_score: float) -> Priority:
        """Compute notification priority."""
        combined_score = (urgency_score * 0.6) + (relevance_score * 0.4)

        if combined_score >= 0.8:
            return Priority.URGENT
        elif combined_score >= 0.6:
            return Priority.HIGH
        elif combined_score >= 0.4:
            return Priority.MEDIUM
        else:
            return Priority.LOW

    async def _log_decision(self, decision: NotificationDecision, start_time: datetime) -> None:
        """Log notification decision for analytics."""
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Store decision in Redis for analytics
        decision_data = {
            "user_id": decision.user_id,
            "should_send": decision.should_send,
            "reason": decision.reason,
            "processing_time_ms": processing_time * 1000,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": decision.metadata or {},
        }

        await self.redis_client.lpush("notification_decisions", str(decision_data))

        # Keep only last 10000 decisions
        await self.redis_client.ltrim("notification_decisions", 0, 9999)

    async def _log_delivery_success(self, decision: NotificationDecision, result: DeliveryResult) -> None:
        """Log successful delivery."""
        logger.info(
            "Notification delivered successfully",
            user_id=decision.user_id,
            delivery_id=result.delivery_id,
            channel=result.channel.value,
        )

    async def _log_delivery_failure(self, decision: NotificationDecision, error: str) -> None:
        """Log delivery failure."""
        logger.error("Notification delivery failed", user_id=decision.user_id, error=error)

    async def _get_source_weight(self, source: str) -> float:
        """Get source authority weight."""
        try:
            source_weight_key = f"source_weight:{source}"
            weight = await self.redis_client.get(source_weight_key)
            
            if weight:
                return float(weight)
            else:
                # Default weight based on source type
                default_weights = {
                    "reuters": 0.9,
                    "ap": 0.9,
                    "bbc": 0.8,
                    "cnn": 0.8,
                    "nytimes": 0.8,
                    "washingtonpost": 0.8,
                    "guardian": 0.7,
                    "independent": 0.7,
                }
                return default_weights.get(source.lower(), 0.5)
                
        except Exception as e:
            logger.error("Error getting source weight", source=source, error=str(e))
            return 0.5

    async def _get_user_interest(self, user_id: str, topics: List[str]) -> float:
        """Get user interest score for topics."""
        try:
            if not topics:
                return 0.5
                
            # Get user topic preferences
            user_prefs_key = f"user_topic_prefs:{user_id}"
            user_prefs = await self.redis_client.hgetall(user_prefs_key)
            
            if not user_prefs:
                return 0.5  # Default interest
            
            # Calculate average interest across topics
            total_interest = 0.0
            topic_count = 0
            
            for topic in topics:
                topic_key = topic.lower().replace(" ", "_")
                interest = user_prefs.get(topic_key.encode())
                
                if interest:
                    total_interest += float(interest)
                    topic_count += 1
            
            if topic_count > 0:
                return total_interest / topic_count
            else:
                return 0.5
                
        except Exception as e:
            logger.error("Error getting user interest", user_id=user_id, topics=topics, error=str(e))
            return 0.5
