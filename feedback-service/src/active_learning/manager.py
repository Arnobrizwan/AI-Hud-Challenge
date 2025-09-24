"""
Active learning manager for continuous model improvement
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
import structlog
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.database import Feedback as FeedbackDB
from ..models.database import (
    FeedbackProcessingResult,
)
from ..models.database import ModelUpdate as ModelUpdateDB
from ..models.database import TrainingBatch as TrainingBatchDB
from ..models.schemas import (
    Feedback,
    FeedbackType,
    ModelUpdate,
    TrainingBatch,
    UncertainPrediction,
    UserFeedback,
)

logger = structlog.get_logger(__name__)


class ActiveLearningManager:
    """Manage active learning for continuous model improvement"""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.uncertainty_estimator = UncertaintyEstimator()
        self.sample_selector = SampleSelector()
        self.model_trainer = ModelTrainer()
        self.performance_tracker = PerformanceTracker(db_session)

    async def identify_uncertain_predictions(
        self, model_name: str, threshold: float = 0.3, hours: int = 24
    ) -> List[UncertainPrediction]:
        """Identify predictions that need human review"""

        try:
            # Get recent predictions with uncertainty scores
            start_time = datetime.utcnow() - timedelta(hours=hours)

            result = await self.db.execute(
                select(FeedbackProcessingResult).where(
                    and_(
                        FeedbackProcessingResult.created_at >= start_time,
                        FeedbackProcessingResult.confidence_score.isnot(None),
                    )
                )
            )

            recent_predictions = result.scalars().all()

            uncertain_predictions = []

            for prediction in recent_predictions:
                # Calculate uncertainty (entropy, confidence intervals, etc.)
                uncertainty_score = await self.uncertainty_estimator.calculate_uncertainty(
                    prediction
                )

                if uncertainty_score > threshold:
                    uncertain_predictions.append(
                        UncertainPrediction(
                            prediction_id=str(prediction.id),
                            model_name=model_name,
                            content_id=str(prediction.feedback.content_id),
                            uncertainty_score=uncertainty_score,
                            prediction_confidence=prediction.confidence_score or 0.0,
                            requires_review=True,
                        )
                    )

            # Prioritize by uncertainty and business impact
            prioritized_predictions = await self.prioritize_for_review(uncertain_predictions)

            logger.info(
                "Uncertain predictions identified",
                count=len(prioritized_predictions),
                model_name=model_name,
            )

            return prioritized_predictions

        except Exception as e:
            logger.error("Error identifying uncertain predictions", error=str(e))
            return []

    async def create_training_data_from_feedback(
        self, feedback_batch: List[UserFeedback]
    ) -> TrainingBatch:
        """Convert user feedback into training data"""

        try:
            training_examples = []

            for feedback in feedback_batch:
                # Extract features from original content
                features = await self.extract_features(feedback.content_id)

                # Convert feedback to label
                label = await self.feedback_to_label(feedback)

                # Add to training set with metadata
                training_examples.append(
                    {
                        "features": features,
                        "label": label,
                        "weight": self.calculate_example_weight(feedback),
                        "metadata": {
                            "feedback_id": str(feedback.id),
                            "user_id": str(feedback.user_id) if feedback.user_id else None,
                            "timestamp": (
                                feedback.created_at.isoformat()
                                if hasattr(feedback, "created_at")
                                else None
                            ),
                            "feedback_type": feedback.feedback_type.value,
                        },
                    }
                )

            # Create training batch
            batch = TrainingBatchDB(
                id=uuid4(),
                model_name="feedback_classifier",
                batch_data={
                    "examples": training_examples,
                    "feature_count": (
                        len(training_examples[0]["features"]) if training_examples else 0
                    ),
                    "label_distribution": self.calculate_label_distribution(training_examples),
                },
                example_count=len(training_examples),
                status="pending",
                created_at=datetime.utcnow(),
            )

            self.db.add(batch)
            await self.db.commit()
            await self.db.refresh(batch)

            logger.info(
                "Training data created",
                batch_id=str(batch.id),
                example_count=len(training_examples),
            )

            return self._db_to_schema(batch)

        except Exception as e:
            logger.error("Error creating training data", error=str(e))
            raise

    async def trigger_model_retraining(
        self, model_name: str, training_batch: TrainingBatch
    ) -> Dict[str, Any]:
        """Trigger model retraining with new data"""

        try:
            # Validate training batch quality
            quality_check = await self.validate_training_batch(training_batch)
            if not quality_check["is_valid"]:
                raise ValueError(f"Invalid training batch: {quality_check['error']}")

            # Submit retraining job
            job_id = await self.model_trainer.submit_retraining_job(
                model_name=model_name,
                training_batch=training_batch,
                validation_split=0.2,
                early_stopping=True,
            )

            # Track retraining progress
            await self.performance_tracker.track_retraining_job(job_id)

            # Create model update record
            model_update = ModelUpdateDB(
                id=uuid4(),
                model_name=model_name,
                training_batch_id=UUID(training_batch.id),
                performance_metrics={},
                status="pending",
                started_at=datetime.utcnow(),
                created_at=datetime.utcnow(),
            )

            self.db.add(model_update)
            await self.db.commit()

            logger.info("Model retraining triggered", job_id=job_id, model_name=model_name)

            return {
                "job_id": job_id,
                "model_name": model_name,
                "training_examples_count": training_batch.example_count,
                "estimated_completion": (datetime.utcnow() + timedelta(hours=2)).isoformat(),
            }

        except Exception as e:
            logger.error("Error triggering model retraining", error=str(e))
            raise

    async def schedule_model_updates(self) -> List[Dict[str, Any]]:
        """Schedule regular model updates based on feedback"""

        try:
            # Check if enough new feedback has accumulated
            models_to_update = await self.identify_models_needing_updates()

            update_results = []

            for model_info in models_to_update:
                # Create training data from recent feedback
                training_data = await self.create_training_data_from_feedback(
                    model_info["feedback_batch"]
                )

                # Submit retraining job
                result = await self.trigger_model_retraining(model_info["name"], training_data)

                update_results.append(result)

                logger.info("Scheduled update for model", model_name=model_info["name"])

            return update_results

        except Exception as e:
            logger.error("Error scheduling model updates", error=str(e))
            return []

    async def prioritize_for_review(
        self, uncertain_predictions: List[UncertainPrediction]
    ) -> List[UncertainPrediction]:
        """Prioritize uncertain predictions for human review"""

        # Sort by uncertainty score (highest first)
        # In production, this would consider business impact, user importance, etc.
        return sorted(uncertain_predictions, key=lambda x: x.uncertainty_score, reverse=True)

    async def extract_features(self, content_id: UUID) -> List[float]:
        """Extract features from content for training"""

        # This would extract relevant features from content
        # For now, return dummy features
        return [0.1, 0.2, 0.3, 0.4, 0.5]  # Dummy features

    async def feedback_to_label(self, feedback: UserFeedback) -> Any:
        """Convert feedback to training label"""

        if feedback.rating is not None:
            # Convert rating to binary classification
            return 1 if feedback.rating >= 3.0 else 0

        if feedback.comment:
            # Use sentiment analysis result
            return await self.analyze_sentiment_label(feedback.comment)

        # Default label
        return 0

    async def analyze_sentiment_label(self, comment: str) -> int:
        """Analyze sentiment and return binary label"""

        # Simple keyword-based sentiment analysis
        positive_words = ["good", "great", "excellent", "amazing", "love", "like"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "poor"]

        comment_lower = comment.lower()
        positive_count = sum(1 for word in positive_words if word in comment_lower)
        negative_count = sum(1 for word in negative_words if word in comment_lower)

        return 1 if positive_count > negative_count else 0

    def calculate_example_weight(self, feedback: UserFeedback) -> float:
        """Calculate weight for training example"""

        # Weight based on feedback type and user importance
        base_weight = 1.0

        if feedback.feedback_type == FeedbackType.EXPLICIT:
            base_weight *= 2.0  # Explicit feedback is more valuable

        if feedback.rating is not None:
            # Higher confidence for extreme ratings
            if feedback.rating <= 1.0 or feedback.rating >= 4.0:
                base_weight *= 1.5

        return base_weight

    def calculate_label_distribution(
        self, training_examples: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Calculate label distribution in training set"""

        labels = [example["label"] for example in training_examples]
        distribution = {}

        for label in labels:
            distribution[str(label)] = distribution.get(str(label), 0) + 1

        return distribution

    async def validate_training_batch(self, training_batch: TrainingBatch) -> Dict[str, Any]:
        """Validate training batch quality"""

        try:
            # Check minimum examples
            if training_batch.example_count < 10:
                return {"is_valid": False, "error": "Insufficient training examples"}

            # Check label distribution
            label_dist = training_batch.batch_data.get("label_distribution", {})
            if len(label_dist) < 2:
                return {"is_valid": False, "error": "Insufficient label diversity"}

            # Check for class imbalance
            total_examples = sum(label_dist.values())
            max_class_count = max(label_dist.values())
            if max_class_count / total_examples > 0.9:
                return {"is_valid": False, "error": "Severe class imbalance"}

            return {"is_valid": True, "error": None}

        except Exception as e:
            return {"is_valid": False, "error": str(e)}

    async def identify_models_needing_updates(self) -> List[Dict[str, Any]]:
        """Identify models that need updates based on new feedback"""

        # Get recent feedback
        start_time = datetime.utcnow() - timedelta(hours=24)

        result = await self.db.execute(
            select(FeedbackDB).where(
                and_(
                    FeedbackDB.created_at >= start_time,
                    FeedbackDB.feedback_type == FeedbackType.EXPLICIT,
                )
            )
        )

        recent_feedback = result.scalars().all()

        if len(recent_feedback) >= 100:  # Threshold for retraining
            return [
                {
                    "name": "feedback_classifier",
                    "feedback_batch": recent_feedback[:100],  # Limit batch size
                }
            ]

        return []

    def _db_to_schema(self, batch: TrainingBatchDB) -> TrainingBatch:
        """Convert database model to schema"""

        return TrainingBatch(
            id=str(batch.id),
            model_name=batch.model_name,
            batch_data=batch.batch_data,
            example_count=batch.example_count,
            status=batch.status,
            created_at=batch.created_at,
        )


class UncertaintyEstimator:
    """Estimate prediction uncertainty"""

    def calculate_uncertainty(self, prediction: Any) -> float:
        """Calculate uncertainty score for a prediction"""

        # Simple uncertainty based on confidence score
        confidence = getattr(prediction, "confidence_score", 0.5)
        return 1.0 - confidence


class SampleSelector:
    """Select samples for active learning"""

    def select_diverse_samples(self, candidates: List[Any], count: int) -> List[Any]:
        """Select diverse samples from candidates"""

        # Simple random selection
        # In production, this would use diversity metrics
        import random

        return random.sample(candidates, min(count, len(candidates)))


class ModelTrainer:
    """Handle model training and retraining"""

    async def submit_retraining_job(
        self,
        model_name: str,
        training_batch: TrainingBatch,
        validation_split: float = 0.2,
        early_stopping: bool = True,
    ) -> str:
        """Submit model retraining job"""

        # This would submit to a job queue or ML platform
        job_id = f"retrain_{model_name}_{uuid4().hex[:8]}"

        # Simulate job submission
        logger.info("Retraining job submitted", job_id=job_id, model_name=model_name)

        return job_id


class PerformanceTracker:
    """Track model performance metrics"""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def track_retraining_job(self, job_id: str) -> None:
        """Track retraining job progress"""

        # This would track job progress and update metrics
        logger.info("Tracking retraining job", job_id=job_id)
