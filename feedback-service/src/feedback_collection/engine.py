"""
Comprehensive feedback collection and processing system
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from ..models.schemas import (
    Feedback, FeedbackCreate, ProcessingResult, FeedbackType, 
    SignalType, UserFeedback, RealtimeFeedback
)
from ..models.database import Feedback as FeedbackDB, FeedbackProcessingResult
from ..quality_assurance.analyzer import QualityAnalyzer
from ..annotation.manager import AnnotationManager
from ..editorial_workflow.engine import EditorialWorkflowEngine
from ..active_learning.manager import ModelUpdateManager
from ..realtime.websocket_manager import WebSocketManager
from ..analytics.aggregator import DataAggregator

logger = structlog.get_logger(__name__)

class FeedbackCollectionEngine:
    """Comprehensive feedback collection and processing system"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.quality_analyzer = QualityAnalyzer()
        self.annotation_manager = AnnotationManager(db_session)
        self.workflow_engine = EditorialWorkflowEngine(db_session)
        self.model_updater = ModelUpdateManager(db_session)
        self.websocket_manager = WebSocketManager()
        self.data_aggregator = DataAggregator(db_session)
        
    async def process_user_feedback(self, feedback: UserFeedback) -> ProcessingResult:
        """Main feedback processing pipeline"""
        
        try:
            # Validate and normalize feedback
            normalized_feedback = await self.normalize_feedback(feedback)
            
            # Store feedback with timestamp and metadata
            feedback_id = await self.store_feedback(normalized_feedback)
            
            # Process based on feedback type
            if normalized_feedback.feedback_type == FeedbackType.IMPLICIT:
                await self.process_implicit_feedback(normalized_feedback)
            elif normalized_feedback.feedback_type == FeedbackType.EXPLICIT:
                await self.process_explicit_feedback(normalized_feedback)
            
            # Update user profile based on feedback
            await self.update_user_profile(normalized_feedback)
            
            # Trigger model updates if needed
            if await self.should_trigger_model_update(normalized_feedback):
                await self.model_updater.queue_model_update(normalized_feedback)
            
            # Send real-time notifications to relevant stakeholders
            await self.notify_stakeholders(normalized_feedback)
            
            # Calculate quality score
            quality_score = await self.quality_analyzer.score_feedback(normalized_feedback)
            
            # Store processing result
            processing_result = await self.store_processing_result(
                feedback_id, quality_score, normalized_feedback
            )
            
            return ProcessingResult(
                feedback_id=feedback_id,
                processed_at=datetime.utcnow(),
                quality_score=quality_score,
                actions_taken=self.get_processing_actions(normalized_feedback),
                requires_immediate_attention=quality_score < 0.3
            )
            
        except Exception as e:
            logger.error("Error processing feedback", error=str(e), feedback_id=str(feedback.id))
            raise
    
    async def normalize_feedback(self, feedback: UserFeedback) -> UserFeedback:
        """Validate and normalize feedback data"""
        
        # Basic validation
        if not feedback.content_id:
            raise ValueError("Content ID is required")
        
        if feedback.rating is not None and (feedback.rating < 0 or feedback.rating > 5):
            raise ValueError("Rating must be between 0 and 5")
        
        # Normalize text content
        if feedback.comment:
            feedback.comment = feedback.comment.strip()
        
        # Set default values
        if not feedback.feedback_type:
            feedback.feedback_type = FeedbackType.EXPLICIT
        
        if not feedback.metadata:
            feedback.metadata = {}
        
        # Add processing metadata
        feedback.metadata.update({
            "normalized_at": datetime.utcnow().isoformat(),
            "processing_version": "1.0.0"
        })
        
        return feedback
    
    async def store_feedback(self, feedback: UserFeedback) -> UUID:
        """Store feedback in database"""
        
        feedback_db = FeedbackDB(
            id=uuid4(),
            user_id=feedback.user_id,
            content_id=feedback.content_id,
            feedback_type=feedback.feedback_type,
            signal_type=feedback.signal_type,
            rating=feedback.rating,
            comment=feedback.comment,
            metadata=feedback.metadata,
            processed_at=datetime.utcnow()
        )
        
        self.db.add(feedback_db)
        await self.db.commit()
        await self.db.refresh(feedback_db)
        
        logger.info("Feedback stored", feedback_id=str(feedback_db.id))
        return feedback_db.id
    
    async def process_implicit_feedback(self, feedback: UserFeedback) -> None:
        """Process implicit user behavior signals"""
        
        try:
            # Update engagement metrics
            await self.update_engagement_metrics(feedback)
            
            # Update personalization models
            await self.update_personalization_signals(feedback)
            
            # Update content quality signals
            if feedback.signal_type in [SignalType.CLICK, SignalType.DWELL_TIME, SignalType.SHARE]:
                await self.update_content_quality_signals(feedback)
            
            # Detect and flag potential issues
            if (feedback.signal_type == SignalType.COMPLAINT or 
                (feedback.rating is not None and feedback.rating < 2.0)):
                await self.flag_for_review(feedback)
                
        except Exception as e:
            logger.error("Error processing implicit feedback", error=str(e))
    
    async def process_explicit_feedback(self, feedback: UserFeedback) -> None:
        """Process direct user feedback and ratings"""
        
        try:
            # Analyze sentiment and intent
            feedback_analysis = await self.analyze_feedback_content(feedback)
            
            # Route to appropriate workflow
            if feedback_analysis.get("requires_human_review", False):
                await self.workflow_engine.create_review_task(
                    content_id=feedback.content_id,
                    task_type="feedback_review",
                    priority="high" if feedback_analysis.get("severity") == "high" else "normal"
                )
            
            # Update content ratings
            await self.update_content_ratings(feedback)
            
            # Flag high-priority issues
            if feedback_analysis.get("severity") == "high":
                await self.create_priority_alert(feedback, feedback_analysis)
                
        except Exception as e:
            logger.error("Error processing explicit feedback", error=str(e))
    
    async def update_engagement_metrics(self, feedback: UserFeedback) -> None:
        """Update user engagement metrics"""
        
        # This would update user engagement tracking
        # Implementation depends on your metrics system
        logger.info("Updating engagement metrics", user_id=str(feedback.user_id))
    
    async def update_personalization_signals(self, feedback: UserFeedback) -> None:
        """Update personalization model signals"""
        
        # This would update personalization models
        # Implementation depends on your ML pipeline
        logger.info("Updating personalization signals", user_id=str(feedback.user_id))
    
    async def update_content_quality_signals(self, feedback: UserFeedback) -> None:
        """Update content quality signals based on user behavior"""
        
        # This would update content quality metrics
        logger.info("Updating content quality signals", content_id=str(feedback.content_id))
    
    async def flag_for_review(self, feedback: UserFeedback) -> None:
        """Flag content for human review"""
        
        await self.workflow_engine.create_review_task(
            content_id=feedback.content_id,
            task_type="quality_review",
            priority="high"
        )
        
        logger.info("Content flagged for review", content_id=str(feedback.content_id))
    
    async def analyze_feedback_content(self, feedback: UserFeedback) -> Dict[str, Any]:
        """Analyze feedback content for sentiment and intent"""
        
        if not feedback.comment:
            return {"requires_human_review": False, "severity": "low"}
        
        # Basic sentiment analysis (would use ML models in production)
        sentiment_score = await self.quality_analyzer.analyze_sentiment(feedback.comment)
        
        # Determine if human review is needed
        requires_review = (
            sentiment_score < 0.2 or  # Very negative
            any(word in feedback.comment.lower() for word in ["inappropriate", "offensive", "wrong"])
        )
        
        severity = "high" if sentiment_score < 0.1 else "medium" if sentiment_score < 0.3 else "low"
        
        return {
            "sentiment_score": sentiment_score,
            "requires_human_review": requires_review,
            "severity": severity
        }
    
    async def update_content_ratings(self, feedback: UserFeedback) -> None:
        """Update content ratings based on feedback"""
        
        if feedback.rating is not None:
            # This would update content rating aggregations
            logger.info("Updating content ratings", 
                       content_id=str(feedback.content_id), 
                       rating=feedback.rating)
    
    async def create_priority_alert(self, feedback: UserFeedback, analysis: Dict[str, Any]) -> None:
        """Create priority alert for high-severity issues"""
        
        # This would create alerts for stakeholders
        logger.warning("Priority alert created", 
                      content_id=str(feedback.content_id),
                      severity=analysis.get("severity"))
    
    async def update_user_profile(self, feedback: UserFeedback) -> None:
        """Update user profile based on feedback patterns"""
        
        if feedback.user_id:
            # This would update user preference models
            logger.info("Updating user profile", user_id=str(feedback.user_id))
    
    async def should_trigger_model_update(self, feedback: UserFeedback) -> bool:
        """Determine if feedback should trigger model update"""
        
        # Simple heuristic - in production this would be more sophisticated
        return (
            feedback.feedback_type == FeedbackType.EXPLICIT and
            feedback.rating is not None and
            feedback.comment is not None
        )
    
    async def notify_stakeholders(self, feedback: UserFeedback) -> None:
        """Send real-time notifications to relevant stakeholders"""
        
        # Send WebSocket notification
        await self.websocket_manager.broadcast_feedback_update(feedback)
        
        # Send other notifications as needed
        logger.info("Stakeholders notified", feedback_id=str(feedback.id))
    
    async def store_processing_result(self, feedback_id: UUID, quality_score: float, 
                                    feedback: UserFeedback) -> None:
        """Store feedback processing results"""
        
        processing_result = FeedbackProcessingResult(
            id=uuid4(),
            feedback_id=feedback_id,
            quality_score=quality_score,
            sentiment_score=await self.quality_analyzer.analyze_sentiment(feedback.comment or ""),
            confidence_score=0.8,  # Would be calculated by ML models
            actions_taken=self.get_processing_actions(feedback),
            processing_metadata={
                "processing_time": datetime.utcnow().isoformat(),
                "feedback_type": feedback.feedback_type.value
            }
        )
        
        self.db.add(processing_result)
        await self.db.commit()
    
    def get_processing_actions(self, feedback: UserFeedback) -> List[str]:
        """Get list of actions taken during processing"""
        
        actions = ["stored", "analyzed"]
        
        if feedback.feedback_type == FeedbackType.IMPLICIT:
            actions.append("engagement_updated")
        elif feedback.feedback_type == FeedbackType.EXPLICIT:
            actions.append("sentiment_analyzed")
        
        if feedback.rating is not None:
            actions.append("rating_processed")
        
        return actions
    
    async def get_feedback_stats(self, time_window: timedelta) -> Dict[str, Any]:
        """Get feedback processing statistics"""
        
        start_time = datetime.utcnow() - time_window
        
        # Get feedback counts by type
        result = await self.db.execute(
            select(FeedbackDB.feedback_type, FeedbackDB.rating)
            .where(FeedbackDB.created_at >= start_time)
        )
        
        feedbacks = result.fetchall()
        
        stats = {
            "total_feedback": len(feedbacks),
            "by_type": {},
            "average_rating": 0.0,
            "time_window": time_window.total_seconds()
        }
        
        # Count by type
        for feedback in feedbacks:
            feedback_type = feedback.feedback_type.value
            stats["by_type"][feedback_type] = stats["by_type"].get(feedback_type, 0) + 1
        
        # Calculate average rating
        ratings = [f.rating for f in feedbacks if f.rating is not None]
        if ratings:
            stats["average_rating"] = sum(ratings) / len(ratings)
        
        return stats
