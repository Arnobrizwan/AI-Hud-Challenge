"""
Feature Importance Monitoring
Monitor changes in feature importance over time
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from safety_engine.config import get_drift_config
from safety_engine.models import ImportanceDriftResult
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression

logger = logging.getLogger(__name__)


class FeatureImportanceMonitor:
    """Monitor changes in feature importance over time"""

    def __init__(self):
        self.config = get_drift_config()
        self.is_initialized = False

    async def initialize(self):
        """Initialize the feature importance monitor"""
        try:
            self.is_initialized = True
            logger.info("Feature importance monitor initialized")

        except Exception as e:
            logger.error(f"Failed to initialize feature importance monitor: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.is_initialized = False
            logger.info("Feature importance monitor cleanup completed")

        except Exception as e:
            logger.error(f"Error during feature importance monitor cleanup: {str(e)}")

    async def detect_importance_drift(
        self, reference_model: Any, current_model: Any
    ) -> ImportanceDriftResult:
        """Detect drift in feature importance between models"""

        if not self.is_initialized:
            raise RuntimeError("Feature importance monitor not initialized")

        try:
            # Extract feature importances
            ref_importance = self.extract_feature_importance(reference_model)
            curr_importance = self.extract_feature_importance(current_model)

            if ref_importance is None or curr_importance is None:
                return ImportanceDriftResult(
                    drift_detected=False,
                    drift_score=0.0,
                    feature_importance_changes={},
                    confidence=0.0,
                )

            # Calculate importance drift
            drift_score = self.calculate_importance_drift(ref_importance, curr_importance)

            # Determine if drift is detected
            drift_detected = drift_score > self.config.importance_drift_threshold

            # Calculate feature importance changes
            importance_changes = self.calculate_feature_changes(ref_importance, curr_importance)

            # Calculate confidence
            confidence = self.calculate_confidence(ref_importance, curr_importance)

            return ImportanceDriftResult(
                drift_detected=drift_detected,
                drift_score=drift_score,
                feature_importance_changes=importance_changes,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Feature importance drift detection failed: {str(e)}")
            return ImportanceDriftResult(
                drift_detected=False, drift_score=0.0, feature_importance_changes={}, confidence=0.0
            )

    def extract_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """Extract feature importance from a model"""
        try:
            if hasattr(model, "feature_importances_"):
                # Tree-based models (RandomForest, XGBoost, etc.)
                importances = model.feature_importances_
                feature_names = getattr(model, "feature_names_in_", None)

                if feature_names is not None:
                    return dict(zip(feature_names, importances))
                else:
                    return {f"feature_{i}": imp for i, imp in enumerate(importances)}

            elif hasattr(model, "coef_"):
                # Linear models (LogisticRegression, LinearRegression, etc.)
                coef = model.coef_
                feature_names = getattr(model, "feature_names_in_", None)

                if coef.ndim > 1:
                    # Multi-class case
                    coef = np.mean(np.abs(coef), axis=0)

                if feature_names is not None:
                    return dict(zip(feature_names, coef))
                else:
                    return {f"feature_{i}": imp for i, imp in enumerate(coef)}

            else:
                logger.warning(
                    f"Model type {type(model)} does not support feature importance extraction"
                )
                return None

        except Exception as e:
            logger.error(f"Feature importance extraction failed: {str(e)}")
            return None

    def calculate_importance_drift(
        self, ref_importance: Dict[str, float], curr_importance: Dict[str, float]
    ) -> float:
        """Calculate drift in feature importance"""
        try:
            # Get common features
            common_features = set(ref_importance.keys()) & set(curr_importance.keys())

            if not common_features:
                return 0.0

            # Calculate drift for each feature
            feature_drifts = []

            for feature in common_features:
                ref_imp = ref_importance[feature]
                curr_imp = curr_importance[feature]

                # Calculate relative change
                if ref_imp > 0:
                    relative_change = abs(curr_imp - ref_imp) / ref_imp
                else:
                    relative_change = abs(curr_imp - ref_imp)

                feature_drifts.append(relative_change)

            # Calculate overall drift score
            if feature_drifts:
                # Use mean of relative changes
                drift_score = np.mean(feature_drifts)

                # Normalize to [0, 1]
                drift_score = min(drift_score, 1.0)
            else:
                drift_score = 0.0

            return drift_score

        except Exception as e:
            logger.error(f"Importance drift calculation failed: {str(e)}")
            return 0.0

    def calculate_feature_changes(
        self, ref_importance: Dict[str, float], curr_importance: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate individual feature importance changes"""
        try:
            changes = {}

            # Get all features
            all_features = set(ref_importance.keys()) | set(curr_importance.keys())

            for feature in all_features:
                ref_imp = ref_importance.get(feature, 0.0)
                curr_imp = curr_importance.get(feature, 0.0)

                # Calculate absolute change
                absolute_change = curr_imp - ref_imp

                # Calculate relative change
                if ref_imp > 0:
                    relative_change = absolute_change / ref_imp
                else:
                    relative_change = absolute_change

                changes[feature] = {
                    "absolute_change": absolute_change,
                    "relative_change": relative_change,
                    "reference_importance": ref_imp,
                    "current_importance": curr_imp,
                }

            return changes

        except Exception as e:
            logger.error(f"Feature changes calculation failed: {str(e)}")
            return {}

    def calculate_confidence(
        self, ref_importance: Dict[str, float], curr_importance: Dict[str, float]
    ) -> float:
        """Calculate confidence in importance drift detection"""
        try:
            # Get common features
            common_features = set(ref_importance.keys()) & set(curr_importance.keys())

            if not common_features:
                return 0.0

            # Calculate stability of importance rankings
            ref_ranking = self.calculate_feature_ranking(ref_importance)
            curr_ranking = self.calculate_feature_ranking(curr_importance)

            # Calculate ranking correlation
            ranking_correlation = self.calculate_ranking_correlation(ref_ranking, curr_ranking)

            # Calculate importance magnitude consistency
            ref_magnitude = np.mean(list(ref_importance.values()))
            curr_magnitude = np.mean(list(curr_importance.values()))
            magnitude_consistency = 1.0 - abs(ref_magnitude - curr_magnitude) / max(
                ref_magnitude, curr_magnitude, 1e-10
            )

            # Combine metrics
            confidence = (ranking_correlation + magnitude_consistency) / 2.0

            return min(confidence, 1.0)

        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.0

    def calculate_feature_ranking(self, importance: Dict[str, float]) -> List[str]:
        """Calculate feature ranking by importance"""
        try:
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            return [feature for feature, _ in sorted_features]

        except Exception as e:
            logger.error(f"Feature ranking calculation failed: {str(e)}")
            return []

    def calculate_ranking_correlation(
        self, ref_ranking: List[str], curr_ranking: List[str]
    ) -> float:
        """Calculate correlation between feature rankings"""
        try:
            # Get common features
            common_features = set(ref_ranking) & set(curr_ranking)

            if not common_features:
                return 0.0

            # Create ranking dictionaries
            ref_ranks = {
                feature: i for i, feature in enumerate(ref_ranking) if feature in common_features
            }
            curr_ranks = {
                feature: i for i, feature in enumerate(curr_ranking) if feature in common_features
            }

            # Calculate Spearman correlation
            ref_rank_values = [ref_ranks[feature] for feature in common_features]
            curr_rank_values = [curr_ranks[feature] for feature in common_features]

            correlation = np.corrcoef(ref_rank_values, curr_rank_values)[0, 1]

            return max(correlation, 0.0) if not np.isnan(correlation) else 0.0

        except Exception as e:
            logger.error(f"Ranking correlation calculation failed: {str(e)}")
            return 0.0

    async def get_importance_summary(
        self, importance: Dict[str, float], top_n: int = 10
    ) -> Dict[str, Any]:
        """Get a summary of feature importance"""
        try:
            # Sort features by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

            # Get top features
            top_features = sorted_features[:top_n]

            # Calculate statistics
            importance_values = list(importance.values())

            summary = {
                "total_features": len(importance),
                "top_features": top_features,
                "mean_importance": np.mean(importance_values),
                "std_importance": np.std(importance_values),
                "max_importance": np.max(importance_values),
                "min_importance": np.min(importance_values),
                "importance_range": np.max(importance_values) - np.min(importance_values),
            }

            return summary

        except Exception as e:
            logger.error(f"Importance summary calculation failed: {str(e)}")
            return {}
