"""
ML-based Abuse Classification
Machine learning models for abuse detection
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from safety_engine.config import get_abuse_config
from safety_engine.models import MLPrediction
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class AbuseClassificationModel:
    """ML-based abuse classification system"""

    def __init__(self):
        self.config = get_abuse_config()
        self.is_initialized = False

        # Models
        self.primary_model = None
        self.ensemble_models = []
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Model metadata
        self.model_version = "1.0.0"
        self.feature_names = []
        self.model_performance = {}

        # Model paths
        self.model_dir = "models/abuse_classifiers"
        os.makedirs(self.model_dir, exist_ok=True)

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the ML classifier"""
        try:
            # Try to load existing models
            await self.load_models()

            # If no models exist, create new ones
            if self.primary_model is None:
    await self.create_models()

            self.is_initialized = True
            logger.info("ML abuse classifier initialized")

        except Exception as e:
            logger.error(f"Failed to initialize ML classifier: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup resources"""
        try:
            # Save models before cleanup
            await self.save_models()

            self.primary_model = None
            self.ensemble_models.clear()
            self.scaler = None
            self.label_encoder = None

            self.is_initialized = False
            logger.info("ML classifier cleanup completed")

        except Exception as e:
            logger.error(f"Error during ML classifier cleanup: {str(e)}")

    async def predict_abuse_probability(
        self, user_features: Dict[str, Any], activity_features: Dict[str, Any]
    ) -> MLPrediction:
        """Predict abuse probability using ML models"""

        if not self.is_initialized:
            raise RuntimeError("ML classifier not initialized")

        try:
            # Combine features
            combined_features = {**user_features, **activity_features}

            # Extract feature vector
            feature_vector = self.extract_features(combined_features)

            if len(feature_vector) == 0:
                return MLPrediction(
                    abuse_probability=0.0,
                    confidence=0.0,
                    feature_importance={},
                    model_version=self.model_version,
                )

            # Normalize features
            feature_array = np.array(feature_vector).reshape(1, -1)
            normalized_features = self.scaler.transform(feature_array)

            # Get predictions from primary model
            primary_prob = self.primary_model.predict_proba(normalized_features)[
                0][1]

            # Get ensemble predictions
            ensemble_probs = []
            for model in self.ensemble_models:
                try:
                    prob = model.predict_proba(normalized_features)[0][1]
                    ensemble_probs.append(prob)
                except Exception as e:
                    logger.warning(
                        f"Ensemble model prediction failed: {str(e)}")
                    continue

            # Calculate final probability
            if ensemble_probs:
                final_prob = np.mean([primary_prob] + ensemble_probs)
            else:
                final_prob = primary_prob

            # Calculate confidence
            confidence = self.calculate_confidence(
                primary_prob, ensemble_probs)

            # Get feature importance
            feature_importance = self.get_feature_importance(combined_features)

            return MLPrediction(
                abuse_probability=final_prob,
                confidence=confidence,
                feature_importance=feature_importance,
                model_version=self.model_version,
            )

        except Exception as e:
            logger.error(f"ML prediction failed: {str(e)}")
            return MLPrediction(
                abuse_probability=0.0,
                confidence=0.0,
                feature_importance={},
                model_version=self.model_version,
            )

    def extract_features(self, features: Dict[str, Any]) -> List[float]:
        """Extract numerical features from input data"""
        try:
            feature_vector = []

            # User features
            feature_vector.append(
                features.get(
                    "account_age_days",
                    0) / 365.0)  # Normalized age
            feature_vector.append(
                features.get(
                    "reputation_score",
                    0.5))  # Reputation
            feature_vector.append(
                1.0 if features.get("is_verified", False) else 0.0
            )  # Verification
            feature_vector.append(
                features.get(
                    "activity_score",
                    0.5))  # Activity level
            feature_vector.append(
                features.get("connection_count", 0) / 100.0
            )  # Normalized connections

            # Activity features
            feature_vector.append(
                features.get("request_frequency", 0) / 100.0
            )  # Normalized frequency
            feature_vector.append(
                features.get(
                    "avg_request_size",
                    0) / 1000.0)  # Normalized size
            feature_vector.append(
                features.get(
                    "error_rate",
                    0.0))  # Error rate
            feature_vector.append(
                features.get("response_time_avg", 0) / 1000.0
            )  # Normalized response time
            feature_vector.append(
                features.get("unique_endpoints", 0) / 50.0
            )  # Normalized endpoints

            # Behavioral features
            feature_vector.append(
                features.get("session_duration_avg", 0) / 3600.0
            )  # Normalized duration
            feature_vector.append(
                features.get("time_variance", 0) / 3600.0
            )  # Normalized time variance
            feature_vector.append(
                features.get(
                    "geographic_diversity",
                    0.0))  # Geographic diversity
            feature_vector.append(
                features.get(
                    "device_diversity",
                    0.0))  # Device diversity
            feature_vector.append(
                features.get(
                    "ip_diversity",
                    0.0))  # IP diversity

            # Risk indicators
            feature_vector.append(
                len(features.get("suspicious_patterns", []))
            )  # Suspicious patterns count
            feature_vector.append(
                features.get("failed_login_attempts", 0) / 10.0
            )  # Normalized failed logins
            feature_vector.append(
                features.get("unusual_hours_activity", 0.0)
            )  # Unusual hours activity
            feature_vector.append(
                features.get(
                    "automated_behavior_score",
                    0.0))  # Automation score
            feature_vector.append(
                features.get("content_violation_count", 0) / 10.0
            )  # Normalized violations

            return feature_vector

        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return []

    def calculate_confidence(
            self,
            primary_prob: float,
            ensemble_probs: List[float]) -> float:
        """Calculate prediction confidence"""
        try:
            if not ensemble_probs:
                # Use primary model uncertainty
                return (
                    1.0 - abs(primary_prob - 0.5) * 2
                )  # Higher confidence for extreme predictions

            # Calculate variance in ensemble predictions
            all_probs = [primary_prob] + ensemble_probs
            variance = np.var(all_probs)

            # Lower variance = higher confidence
            # Scale variance to [0, 1]
            confidence = max(0.0, 1.0 - variance * 4)

            return min(confidence, 1.0)

        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.5

    def get_feature_importance(
            self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get feature importance from the model"""
        try:
            if not hasattr(self.primary_model, "feature_importances_"):
                return {}

            # Map feature indices to names
            feature_names = [
                "account_age",
                "reputation_score",
                "is_verified",
                "activity_score",
                "connection_count",
                "request_frequency",
                "avg_request_size",
                "error_rate",
                "response_time_avg",
                "unique_endpoints",
                "session_duration_avg",
                "time_variance",
                "geographic_diversity",
                "device_diversity",
                "ip_diversity",
                "suspicious_patterns_count",
                "failed_login_attempts",
                "unusual_hours_activity",
                "automated_behavior_score",
                "content_violation_count",
            ]

            importances = self.primary_model.feature_importances_

            # Create importance dictionary
            importance_dict = {}
            for i, importance in enumerate(importances):
                if i < len(feature_names):
                    importance_dict[feature_names[i]] = float(importance)

            return importance_dict

        except Exception as e:
            logger.error(f"Feature importance calculation failed: {str(e)}")
            return {}

    async def create_models(self) -> Dict[str, Any]:
        """Create new ML models"""
        try:
            # Create primary model (Random Forest)
            self.primary_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )

            # Create ensemble models
            self.ensemble_models = [
                GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, random_state=42), LogisticRegression(
                    random_state=42, max_iter=1000), SVC(
                    probability=True, random_state=42), ]

            # Initialize scaler
            self.scaler = StandardScaler()

            # Train models with synthetic data (in production, use real
            # training data)
            await self.train_models_with_synthetic_data()

            logger.info("ML models created and trained")

        except Exception as e:
            logger.error(f"Model creation failed: {str(e)}")
            raise

    async def train_models_with_synthetic_data(self) -> Dict[str, Any]:
        """Train models with synthetic data for demonstration"""
        try:
            # Generate synthetic training data
            n_samples = 1000
            n_features = 20

            # Generate features
            X = np.random.randn(n_samples, n_features)

            # Generate labels (0 = normal, 1 = abuse)
            # Create some patterns for abuse
            abuse_indices = []
            for i in range(n_samples):
                # Abuse if high activity + low reputation + high error rate
                if X[i, 3] > 0.8 and X[i, 1] < 0.3 and X[i, 7] > 0.5:
                    abuse_indices.append(i)

            y = np.zeros(n_samples)
            y[abuse_indices] = 1

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Fit scaler
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train primary model
            self.primary_model.fit(X_train_scaled, y_train)

            # Train ensemble models
            for model in self.ensemble_models:
                model.fit(X_train_scaled, y_train)

            # Evaluate models
            primary_score = self.primary_model.score(X_test_scaled, y_test)
            self.model_performance["primary_model"] = primary_score

            # Calculate ensemble performance
            ensemble_scores = []
            for model in self.ensemble_models:
                score = model.score(X_test_scaled, y_test)
                ensemble_scores.append(score)

            self.model_performance["ensemble_models"] = ensemble_scores
            self.model_performance["average_ensemble"] = np.mean(
                ensemble_scores)

            logger.info(
                f"Models trained - Primary: {primary_score:.3f}, Ensemble: {np.mean(ensemble_scores):.3f}"
            )

        except Exception as e:
            logger.error(f"Synthetic data training failed: {str(e)}")
            raise

    async def save_models(self) -> Dict[str, Any]:
        """Save trained models to disk"""
        try:
            if self.primary_model is not None:
                joblib.dump(
                    self.primary_model,
                    os.path.join(
                        self.model_dir,
                        "primary_model.pkl"))

            for i, model in enumerate(self.ensemble_models):
                joblib.dump(
                    model,
                    os.path.join(
                        self.model_dir,
                        f"ensemble_model_{i}.pkl"))

            joblib.dump(
                self.scaler,
                os.path.join(
                    self.model_dir,
                    "scaler.pkl"))
            joblib.dump(
                self.label_encoder,
                os.path.join(
                    self.model_dir,
                    "label_encoder.pkl"))

            # Save metadata
            metadata = {
                "model_version": self.model_version,
                "feature_names": self.feature_names,
                "model_performance": self.model_performance,
            }

            joblib.dump(metadata, os.path.join(self.model_dir, "metadata.pkl"))

            logger.info("Models saved successfully")

        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")

    async def load_models(self) -> Dict[str, Any]:
        """Load trained models from disk"""
        try:
            # Load primary model
            primary_model_path = os.path.join(
                self.model_dir, "primary_model.pkl")
            if os.path.exists(primary_model_path):
                self.primary_model = joblib.load(primary_model_path)

            # Load ensemble models
            for i in range(3):  # Assuming 3 ensemble models
                ensemble_model_path = os.path.join(
                    self.model_dir, f"ensemble_model_{i}.pkl")
                if os.path.exists(ensemble_model_path):
                    model = joblib.load(ensemble_model_path)
                    self.ensemble_models.append(model)

            # Load scaler and encoder
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            encoder_path = os.path.join(self.model_dir, "label_encoder.pkl")
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)

            # Load metadata
            metadata_path = os.path.join(self.model_dir, "metadata.pkl")
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.model_version = metadata.get("model_version", "1.0.0")
                self.feature_names = metadata.get("feature_names", [])
                self.model_performance = metadata.get("model_performance", {})

            if self.primary_model is not None:
                logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")

    async def retrain_models(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Retrain models with new data"""
        try:
            if not training_data:
                logger.warning("No training data provided for retraining")
                return

            # Extract features and labels
            X = []
            y = []

            for data_point in training_data:
                features = data_point.get("features", {})
                label = data_point.get("label", 0)

                feature_vector = self.extract_features(features)
                if feature_vector:
                    X.append(feature_vector)
                    y.append(label)

            if not X:
                logger.warning("No valid training data extracted")
                return

            X = np.array(X)
            y = np.array(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Fit scaler
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Retrain primary model
            self.primary_model.fit(X_train_scaled, y_train)

            # Retrain ensemble models
            for model in self.ensemble_models:
                model.fit(X_train_scaled, y_train)

            # Update model version
            self.model_version = f"1.{int(datetime.now().timestamp())}"

            # Save updated models
            await self.save_models()

            logger.info("Models retrained successfully")

        except Exception as e:
            logger.error(f"Model retraining failed: {str(e)}")

    async def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            return {
                "model_version": self.model_version,
                "performance": self.model_performance,
                "feature_count": len(self.feature_names),
                "ensemble_model_count": len(self.ensemble_models),
                "is_initialized": self.is_initialized,
            }

        except Exception as e:
            logger.error(f"Performance metrics retrieval failed: {str(e)}")
            return {}
