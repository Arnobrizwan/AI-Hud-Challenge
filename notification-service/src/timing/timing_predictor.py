"""
ML model for predicting optimal notification timing.
"""

import asyncio
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ..exceptions import TimingPredictionError
from ..models.schemas import NotificationType, OptimalTiming, TimingPrediction

logger = logging.getLogger(__name__)


class TimingFeatureExtractor:
    """Extract features for timing prediction."""

    def __init__(self):
        self.scaler = StandardScaler()

    async def extract_timing_features(
        self, user_id: str, notification_type: NotificationType, current_time: datetime
    ) -> Dict[str, Any]:
        """Extract features for timing prediction."""

        features = {}

        # Time-based features
        features["hour_of_day"] = current_time.hour
        features["day_of_week"] = current_time.weekday()
        features["is_weekend"] = 1 if current_time.weekday() >= 5 else 0
        features["month"] = current_time.month
        features["day_of_month"] = current_time.day

        # User behavior features (would be fetched from database in real implementation)
        features["user_engagement_score"] = await self._get_user_engagement_score(user_id)
        features["user_timezone_offset"] = await self._get_user_timezone_offset(user_id)

        # Notification type features
        features["notification_type_breaking"] = (
            1 if notification_type == NotificationType.BREAKING_NEWS else 0
        )
        features["notification_type_personalized"] = (
            1 if notification_type == NotificationType.PERSONALIZED else 0
        )
        features["notification_type_trending"] = (
            1 if notification_type == NotificationType.TRENDING else 0
        )

        # Historical engagement features
        features["avg_engagement_hour"] = await self._get_avg_engagement_hour(user_id)
        features["peak_engagement_hour"] = await self._get_peak_engagement_hour(user_id)

        return features

    def extract_time_window_features(self, window_start: datetime) -> Dict[str, Any]:
        """Extract features for a specific time window."""
        return {
            "window_hour": window_start.hour,
            "window_day_of_week": window_start.weekday(),
            "window_is_weekend": 1 if window_start.weekday() >= 5 else 0,
            "window_is_business_hours": 1 if 9 <= window_start.hour <= 17 else 0,
            "window_is_evening": 1 if 18 <= window_start.hour <= 22 else 0,
            "window_is_night": 1 if window_start.hour >= 23 or window_start.hour <= 6 else 0,
        }

    async def _get_user_engagement_score(self, user_id: str) -> float:
        """Get user's overall engagement score."""
        # Mock implementation - would fetch from database
        return np.random.uniform(0.3, 0.9)

    async def _get_user_timezone_offset(self, user_id: str) -> int:
        """Get user's timezone offset in hours."""
        # Mock implementation - would fetch from user preferences
        return np.random.randint(-12, 15)

    async def _get_avg_engagement_hour(self, user_id: str) -> int:
        """Get user's average engagement hour."""
        # Mock implementation - would calculate from historical data
        return np.random.randint(8, 20)

    async def _get_peak_engagement_hour(self, user_id: str) -> int:
        """Get user's peak engagement hour."""
        # Mock implementation - would calculate from historical data
        return np.random.randint(10, 18)


class TimezoneHandler:
    """Handle timezone operations."""

    async def get_user_timezone(self, user_id: str) -> pytz.timezone:
        """Get user's timezone."""
        # Mock implementation - would fetch from user preferences
        timezone_name = "UTC"  # Default timezone
        return pytz.timezone(timezone_name)

    def apply_timezone_constraints(
        self, scheduled_time: datetime, user_id: str, notification_type: NotificationType
    ) -> datetime:
        """Apply timezone and business rule constraints."""

        # Apply quiet hours (e.g., no notifications between 10 PM and 8 AM)
        if 22 <= scheduled_time.hour or scheduled_time.hour < 8:
            # Move to next available time
            if scheduled_time.hour >= 22:
                scheduled_time = scheduled_time.replace(hour=8, minute=0, second=0, microsecond=0)
                scheduled_time += timedelta(days=1)
            else:
                scheduled_time = scheduled_time.replace(hour=8, minute=0, second=0, microsecond=0)

        return scheduled_time


class NotificationTimingModel:
    """ML model for predicting optimal notification timing."""

    def __init__(self):
        self.model = None
        self.feature_extractor = TimingFeatureExtractor()
        self.timezone_handler = TimezoneHandler()
        self.is_trained = False

    async def initialize(self) -> None:
        """Initialize the timing model."""
        logger.info("Initializing notification timing model")

        # Try to load existing model
        await self._load_model()

        # If no model exists, create and train a new one
        if not self.is_trained:
            await self._create_and_train_model()

        logger.info("Notification timing model initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup model resources."""
        logger.info("Cleaning up notification timing model")
        # Model cleanup if needed
        pass

    async def predict_optimal_time(
        self, user_id: str, notification_type: NotificationType
    ) -> OptimalTiming:
        """Predict when user is most likely to engage with notification."""

        try:
            # Get user's timezone and current time
            user_timezone = await self.timezone_handler.get_user_timezone(user_id)
            current_time = datetime.now(user_timezone)

            # Extract timing features
            features = await self.feature_extractor.extract_timing_features(
                user_id, notification_type, current_time
            )

            # Predict engagement probability for different time windows
            time_windows = self._generate_time_windows(current_time, hours_ahead=24)
            predictions = []

            for window_start in time_windows:
                window_features = features.copy()
                window_features.update(
                    self.feature_extractor.extract_time_window_features(window_start)
                )

                # Prepare features for prediction
                feature_vector = np.array([list(window_features.values())])

                # Predict engagement probability
                if self.model and self.is_trained:
                    engagement_prob = self.model.predict_proba(feature_vector)[0][1]
                else:
                    # Fallback to heuristic-based prediction
                    engagement_prob = self._heuristic_engagement_prediction(window_features)

                predictions.append(
                    TimingPrediction(
                        scheduled_time=window_start,
                        engagement_probability=engagement_prob,
                        features=window_features,
                    )
                )

            # Select optimal timing based on highest engagement probability
            optimal_prediction = max(predictions, key=lambda x: x.engagement_probability)

            # Apply business rules and constraints
            scheduled_time = self.timezone_handler.apply_timezone_constraints(
                optimal_prediction.scheduled_time, user_id, notification_type
            )

            return OptimalTiming(
                scheduled_time=scheduled_time,
                predicted_engagement=optimal_prediction.engagement_probability,
                alternative_times=[
                    p.scheduled_time for p in predictions if p != optimal_prediction
                ][:3],
                confidence=optimal_prediction.engagement_probability,
            )

        except Exception as e:
            logger.error(f"Error predicting optimal timing: {str(e)}")
            raise TimingPredictionError(f"Failed to predict optimal timing: {str(e)}")

    async def update_model_from_feedback(self, delivery_result) -> None:
        """Update timing model based on user engagement feedback."""
        try:
            # Extract features from original delivery
            features = delivery_result.original_features

            # Determine engagement label
            engagement_label = 1 if delivery_result.was_engaged else 0

            # Add to training data
            await self._add_training_example(features, engagement_label)

            # Trigger model retraining if enough new examples
            await self._check_and_trigger_retraining()

        except Exception as e:
            logger.error(f"Error updating model from feedback: {str(e)}")
            raise TimingPredictionError(f"Failed to update model: {str(e)}")

    def _generate_time_windows(self, current_time: datetime, hours_ahead: int) -> List[datetime]:
        """Generate time windows for prediction."""
        windows = []

        for hour_offset in range(0, hours_ahead, 1):
            window_time = current_time + timedelta(hours=hour_offset)
            windows.append(window_time)

        return windows

    def _heuristic_engagement_prediction(self, features: Dict[str, Any]) -> float:
        """Heuristic-based engagement prediction when ML model is not available."""

        # Base engagement probability
        base_prob = 0.5

        # Adjust based on time of day
        hour = features.get("window_hour", 12)
        if 9 <= hour <= 11:  # Morning
            base_prob += 0.2
        elif 14 <= hour <= 16:  # Afternoon
            base_prob += 0.15
        elif 19 <= hour <= 21:  # Evening
            base_prob += 0.1
        elif hour >= 22 or hour <= 6:  # Night
            base_prob -= 0.3

        # Adjust based on day of week
        if features.get("window_is_weekend", 0):
            base_prob += 0.1

        # Adjust based on business hours
        if features.get("window_is_business_hours", 0):
            base_prob += 0.05

        return max(0.0, min(1.0, base_prob))

    async def _load_model(self) -> None:
        """Load existing model from file."""
        try:
            from ..config import get_settings

            settings = get_settings()

            model_path = settings.TIMING_MODEL_PATH
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
                    self.model = model_data["model"]
                    self.feature_extractor.scaler = model_data["scaler"]
                    self.is_trained = True
                    logger.info("Loaded existing timing model")
        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")

    async def _create_and_train_model(self) -> None:
        """Create and train a new timing model."""
        logger.info("Creating and training new timing model")

        # Generate synthetic training data
        X, y = self._generate_synthetic_training_data()

        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

        # Fit scaler and model
        X_scaled = self.feature_extractor.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        self.is_trained = True

        # Save model
        await self._save_model()

        logger.info("Timing model trained and saved successfully")

    def _generate_synthetic_training_data(self) -> tuple:
        """Generate synthetic training data for model training."""
        np.random.seed(42)

        n_samples = 10000
        n_features = 10

        # Generate random features
        X = np.random.randn(n_samples, n_features)

        # Generate synthetic labels based on time patterns
        y = np.zeros(n_samples)

        for i in range(n_samples):
            hour = X[i, 0] * 12 + 12  # Scale to 0-24 range
            is_weekend = X[i, 1] > 0.5
            is_business_hours = 9 <= hour <= 17

            # Higher engagement probability for certain times
            if 9 <= hour <= 11 or 14 <= hour <= 16:
                y[i] = 1 if np.random.random() > 0.3 else 0
            elif is_weekend and 10 <= hour <= 20:
                y[i] = 1 if np.random.random() > 0.4 else 0
            else:
                y[i] = 1 if np.random.random() > 0.7 else 0

        return X, y

    async def _save_model(self) -> None:
        """Save trained model to file."""
        try:
            from ..config import get_settings

            settings = get_settings()

            model_path = settings.TIMING_MODEL_PATH
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            model_data = {"model": self.model, "scaler": self.feature_extractor.scaler}

            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    async def _add_training_example(self, features: Dict[str, Any], label: int) -> None:
        """Add new training example to the model."""
        # In a real implementation, this would store the example in a database
        # and trigger retraining when enough examples are collected
        logger.info(f"Added training example with label {label}")

    async def _check_and_trigger_retraining(self) -> None:
        """Check if model should be retrained and trigger if needed."""
        # In a real implementation, this would check if enough new examples
        # have been collected and trigger model retraining
        logger.info("Checking for model retraining")
