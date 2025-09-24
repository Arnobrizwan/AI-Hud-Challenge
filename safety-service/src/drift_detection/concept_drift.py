"""
Concept Drift Detection
Detect changes in the relationship between features and labels
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from safety_engine.config import get_drift_config
from safety_engine.models import ConceptDriftResult
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class ConceptDriftDetector:
    """Detect concept drift in classification problems"""

    def __init__(self):
        self.config = get_drift_config()
        self.is_initialized = False

        # Models for drift detection
        self.reference_model = None
        self.drift_detector_model = None

    async def initialize(self):
        """Initialize the concept drift detector"""
        try:
            # Initialize models
            self.reference_model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )

            self.drift_detector_model = LogisticRegression(random_state=42, max_iter=1000)

            self.is_initialized = True
            logger.info("Concept drift detector initialized")

        except Exception as e:
            logger.error(f"Failed to initialize concept drift detector: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.reference_model = None
            self.drift_detector_model = None
            logger.info("Concept drift detector cleanup completed")

        except Exception as e:
            logger.error(f"Error during concept drift detector cleanup: {str(e)}")

    async def detect_concept_drift(
        self, reference_labels: pd.Series, current_labels: pd.Series, feature_data: pd.DataFrame
    ) -> ConceptDriftResult:
        """Detect concept drift between reference and current data"""

        if not self.is_initialized:
            raise RuntimeError("Concept drift detector not initialized")

        try:
            # Prepare data
            ref_data, curr_data = self.prepare_data(reference_labels, current_labels, feature_data)

            if ref_data is None or curr_data is None:
                return ConceptDriftResult(
                    drift_detected=False, drift_score=0.0, affected_features=[], confidence=1.0
                )

            # Method 1: Performance-based drift detection
            performance_drift = await self.detect_performance_drift(ref_data, curr_data)

            # Method 2: Distribution-based drift detection
            distribution_drift = await self.detect_distribution_drift(ref_data, curr_data)

            # Method 3: Feature importance drift detection
            importance_drift = await self.detect_importance_drift(ref_data, curr_data)

            # Combine results
            drift_scores = [performance_drift, distribution_drift, importance_drift]
            drift_detected = any(score > self.config.confidence_threshold for score in drift_scores)
            overall_drift_score = max(drift_scores)

            # Determine affected features
            affected_features = self.identify_affected_features(ref_data, curr_data)

            # Calculate confidence
            confidence = self.calculate_confidence(drift_scores)

            return ConceptDriftResult(
                drift_detected=drift_detected,
                drift_score=overall_drift_score,
                affected_features=affected_features,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Concept drift detection failed: {str(e)}")
            return ConceptDriftResult(
                drift_detected=False, drift_score=0.0, affected_features=[], confidence=0.0
            )

    def prepare_data(
        self, reference_labels: pd.Series, current_labels: pd.Series, feature_data: pd.DataFrame
    ) -> tuple:
        """Prepare data for concept drift detection"""
        try:
            # Ensure we have enough data
            if (
                len(reference_labels) < self.config.min_samples
                or len(current_labels) < self.config.min_samples
            ):
                logger.warning("Insufficient data for concept drift detection")
                return None, None

            # Align data
            ref_indices = reference_labels.index
            curr_indices = current_labels.index

            # Get common features
            ref_features = (
                feature_data.loc[ref_indices]
                if ref_indices.isin(feature_data.index).any()
                else None
            )
            curr_features = (
                feature_data.loc[curr_indices]
                if curr_indices.isin(feature_data.index).any()
                else None
            )

            if ref_features is None or curr_features is None:
                logger.warning("Feature data not available for concept drift detection")
                return None, None

            # Create data structures
            ref_data = {"features": ref_features, "labels": reference_labels}

            curr_data = {"features": curr_features, "labels": current_labels}

            return ref_data, curr_data

        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            return None, None

    async def detect_performance_drift(self, ref_data: Dict, curr_data: Dict) -> float:
        """Detect drift based on model performance degradation"""
        try:
            # Train model on reference data
            self.reference_model.fit(ref_data["features"], ref_data["labels"])

            # Evaluate on reference data (baseline)
            ref_pred = self.reference_model.predict(ref_data["features"])
            ref_accuracy = accuracy_score(ref_data["labels"], ref_pred)

            # Evaluate on current data
            curr_pred = self.reference_model.predict(curr_data["features"])
            curr_accuracy = accuracy_score(curr_data["labels"], curr_pred)

            # Calculate performance degradation
            performance_degradation = ref_accuracy - curr_accuracy

            # Normalize to drift score
            drift_score = min(max(performance_degradation, 0.0), 1.0)

            return drift_score

        except Exception as e:
            logger.error(f"Performance drift detection failed: {str(e)}")
            return 0.0

    async def detect_distribution_drift(self, ref_data: Dict, curr_data: Dict) -> float:
        """Detect drift based on label distribution changes"""
        try:
            # Calculate label distributions
            ref_dist = ref_data["labels"].value_counts(normalize=True)
            curr_dist = curr_data["labels"].value_counts(normalize=True)

            # Align distributions
            all_labels = set(ref_dist.index) | set(curr_dist.index)
            ref_dist = ref_dist.reindex(all_labels, fill_value=0.0)
            curr_dist = curr_dist.reindex(all_labels, fill_value=0.0)

            # Calculate total variation distance
            tv_distance = 0.5 * np.sum(np.abs(ref_dist - curr_dist))

            # Convert to drift score
            drift_score = min(tv_distance * 2, 1.0)  # Scale to [0, 1]

            return drift_score

        except Exception as e:
            logger.error(f"Distribution drift detection failed: {str(e)}")
            return 0.0

    async def detect_importance_drift(self, ref_data: Dict, curr_data: Dict) -> float:
        """Detect drift based on feature importance changes"""
        try:
            # Train models on both datasets
            ref_model = RandomForestClassifier(n_estimators=50, random_state=42)
            curr_model = RandomForestClassifier(n_estimators=50, random_state=42)

            ref_model.fit(ref_data["features"], ref_data["labels"])
            curr_model.fit(curr_data["features"], curr_data["labels"])

            # Get feature importances
            ref_importance = ref_model.feature_importances_
            curr_importance = curr_model.feature_importances_

            # Calculate importance drift
            importance_drift = np.mean(np.abs(ref_importance - curr_importance))

            # Convert to drift score
            drift_score = min(importance_drift * 2, 1.0)  # Scale to [0, 1]

            return drift_score

        except Exception as e:
            logger.error(f"Importance drift detection failed: {str(e)}")
            return 0.0

    def identify_affected_features(self, ref_data: Dict, curr_data: Dict) -> List[str]:
        """Identify features most affected by concept drift"""
        try:
            # Train models on both datasets
            ref_model = RandomForestClassifier(n_estimators=50, random_state=42)
            curr_model = RandomForestClassifier(n_estimators=50, random_state=42)

            ref_model.fit(ref_data["features"], ref_data["labels"])
            curr_model.fit(curr_data["features"], curr_data["labels"])

            # Get feature importances
            ref_importance = ref_model.feature_importances_
            curr_importance = curr_model.feature_importances_

            # Calculate importance changes
            importance_changes = np.abs(ref_importance - curr_importance)

            # Get feature names
            feature_names = ref_data["features"].columns.tolist()

            # Sort features by importance change
            feature_importance_pairs = list(zip(feature_names, importance_changes))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

            # Return top affected features
            threshold = np.percentile(importance_changes, 75)  # Top 25%
            affected_features = [
                feature for feature, change in feature_importance_pairs if change > threshold
            ]

            return affected_features[:10]  # Limit to top 10 features

        except Exception as e:
            logger.error(f"Feature identification failed: {str(e)}")
            return []

    def calculate_confidence(self, drift_scores: List[float]) -> float:
        """Calculate confidence in drift detection"""
        try:
            if not drift_scores:
                return 0.0

            # Use the maximum drift score as confidence
            max_score = max(drift_scores)

            # Add consistency bonus if multiple methods agree
            if len(drift_scores) > 1:
                consistency = 1.0 - np.std(drift_scores)
                max_score = max_score * (0.7 + 0.3 * consistency)

            return min(max_score, 1.0)

        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.0
