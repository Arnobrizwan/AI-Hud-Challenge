"""
Prediction Drift Detection
Detect changes in model predictions over time
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from safety_engine.config import get_drift_config
from safety_engine.models import PredictionDriftResult
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class PredictionDriftDetector:
    """Detect drift in model predictions"""

    def __init__(self):
        self.config = get_drift_config()
        self.is_initialized = False

    async def initialize(self):
        """Initialize the prediction drift detector"""
        try:
            self.is_initialized = True
            logger.info("Prediction drift detector initialized")

        except Exception as e:
            logger.error(f"Failed to initialize prediction drift detector: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.is_initialized = False
            logger.info("Prediction drift detector cleanup completed")

        except Exception as e:
            logger.error(f"Error during prediction drift detector cleanup: {str(e)}")

    async def detect_prediction_drift(
        self, reference_predictions: pd.Series, current_predictions: pd.Series
    ) -> PredictionDriftResult:
        """Detect drift in model predictions"""

        if not self.is_initialized:
            raise RuntimeError("Prediction drift detector not initialized")

        try:
            # Method 1: Distribution-based drift detection
            distribution_drift = await self.detect_prediction_distribution_drift(
                reference_predictions, current_predictions
            )

            # Method 2: Accuracy drift detection
            accuracy_drift = await self.detect_accuracy_drift(
                reference_predictions, current_predictions
            )

            # Method 3: Calibration drift detection
            calibration_drift = await self.detect_calibration_drift(
                reference_predictions, current_predictions
            )

            # Combine results
            drift_scores = [distribution_drift, accuracy_drift, calibration_drift]
            drift_detected = any(score > self.config.confidence_threshold for score in drift_scores)
            overall_drift_score = max(drift_scores)

            # Calculate accuracy change
            accuracy_change = self.calculate_accuracy_change(
                reference_predictions, current_predictions
            )

            # Calculate confidence
            confidence = self.calculate_confidence(drift_scores)

            return PredictionDriftResult(
                drift_detected=drift_detected,
                drift_score=overall_drift_score,
                prediction_accuracy_change=accuracy_change,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Prediction drift detection failed: {str(e)}")
            return PredictionDriftResult(
                drift_detected=False,
                drift_score=0.0,
                prediction_accuracy_change=0.0,
                confidence=0.0,
            )

    async def detect_prediction_distribution_drift(
        self, ref_pred: pd.Series, curr_pred: pd.Series
    ) -> float:
        """Detect drift in prediction distributions"""
        try:
            # Convert to numeric if needed
            if not pd.api.types.is_numeric_dtype(ref_pred):
                ref_pred = pd.Categorical(ref_pred).codes
            if not pd.api.types.is_numeric_dtype(curr_pred):
                curr_pred = pd.Categorical(curr_pred).codes

            # Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = stats.ks_2samp(ref_pred, curr_pred)

            # Mann-Whitney U test
            mw_statistic, mw_p_value = stats.mannwhitneyu(
                ref_pred, curr_pred, alternative="two-sided"
            )

            # Use the more significant result
            min_p_value = min(ks_p_value, mw_p_value)

            # Convert p-value to drift score
            drift_score = 1.0 - min_p_value

            return drift_score

        except Exception as e:
            logger.error(f"Prediction distribution drift detection failed: {str(e)}")
            return 0.0

    async def detect_accuracy_drift(self, ref_pred: pd.Series, curr_pred: pd.Series) -> float:
        """Detect drift in prediction accuracy"""
        try:
            # Calculate prediction statistics
            ref_stats = self.calculate_prediction_stats(ref_pred)
            curr_stats = self.calculate_prediction_stats(curr_pred)

            # Compare statistics
            accuracy_drift = 0.0

            # Mean prediction drift
            if "mean" in ref_stats and "mean" in curr_stats:
                mean_drift = abs(ref_stats["mean"] - curr_stats["mean"])
                accuracy_drift += mean_drift * 0.3

            # Variance prediction drift
            if "var" in ref_stats and "var" in curr_stats:
                var_drift = abs(ref_stats["var"] - curr_stats["var"])
                accuracy_drift += var_drift * 0.3

            # Entropy drift (for classification)
            if "entropy" in ref_stats and "entropy" in curr_stats:
                entropy_drift = abs(ref_stats["entropy"] - curr_stats["entropy"])
                accuracy_drift += entropy_drift * 0.4

            # Normalize drift score
            drift_score = min(accuracy_drift, 1.0)

            return drift_score

        except Exception as e:
            logger.error(f"Accuracy drift detection failed: {str(e)}")
            return 0.0

    async def detect_calibration_drift(self, ref_pred: pd.Series, curr_pred: pd.Series) -> float:
        """Detect drift in prediction calibration"""
        try:
            # Convert predictions to probabilities if needed
            ref_probs = self.convert_to_probabilities(ref_pred)
            curr_probs = self.convert_to_probabilities(curr_pred)

            # Calculate calibration curves
            ref_calibration = self.calculate_calibration_curve(ref_probs)
            curr_calibration = self.calculate_calibration_curve(curr_probs)

            # Calculate calibration drift
            calibration_drift = np.mean(np.abs(ref_calibration - curr_calibration))

            # Convert to drift score
            drift_score = min(calibration_drift * 2, 1.0)

            return drift_score

        except Exception as e:
            logger.error(f"Calibration drift detection failed: {str(e)}")
            return 0.0

    def calculate_prediction_stats(self, predictions: pd.Series) -> Dict[str, float]:
        """Calculate prediction statistics"""
        try:
            stats_dict = {}

            if pd.api.types.is_numeric_dtype(predictions):
                stats_dict["mean"] = predictions.mean()
                stats_dict["var"] = predictions.var()
                stats_dict["std"] = predictions.std()
                stats_dict["min"] = predictions.min()
                stats_dict["max"] = predictions.max()
            else:
                # For categorical predictions
                value_counts = predictions.value_counts(normalize=True)
                stats_dict["entropy"] = -np.sum(value_counts * np.log2(value_counts + 1e-10))
                stats_dict["unique_count"] = len(value_counts)
                stats_dict["most_common_prob"] = value_counts.max()

            return stats_dict

        except Exception as e:
            logger.error(f"Prediction stats calculation failed: {str(e)}")
            return {}

    def convert_to_probabilities(self, predictions: pd.Series) -> np.ndarray:
        """Convert predictions to probabilities"""
        try:
            if pd.api.types.is_numeric_dtype(predictions):
                # Assume predictions are already probabilities
                return predictions.values
            else:
                # Convert categorical to probabilities
                value_counts = predictions.value_counts(normalize=True)
                return predictions.map(value_counts).values

        except Exception as e:
            logger.error(f"Probability conversion failed: {str(e)}")
            return np.array([0.5] * len(predictions))

    def calculate_calibration_curve(
        self, probabilities: np.ndarray, n_bins: int = 10
    ) -> np.ndarray:
        """Calculate calibration curve"""
        try:
            # Create bins
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            # Calculate calibration for each bin
            calibration = np.zeros(n_bins)

            for i in range(n_bins):
                in_bin = (probabilities > bin_lowers[i]) & (probabilities <= bin_uppers[i])
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    calibration[i] = prop_in_bin
                else:
                    calibration[i] = bin_lowers[i] + (bin_uppers[i] - bin_lowers[i]) / 2

            return calibration

        except Exception as e:
            logger.error(f"Calibration curve calculation failed: {str(e)}")
            return np.linspace(0, 1, n_bins)

    def calculate_accuracy_change(self, ref_pred: pd.Series, curr_pred: pd.Series) -> float:
        """Calculate accuracy change between reference and current predictions"""
        try:
            # This is a simplified version - in practice, you'd need ground truth labels
            # For now, we'll calculate a proxy based on prediction consistency

            if len(ref_pred) != len(curr_pred):
                return 0.0

            # Calculate prediction agreement
            if pd.api.types.is_numeric_dtype(ref_pred) and pd.api.types.is_numeric_dtype(curr_pred):
                # For numeric predictions, calculate correlation
                correlation = ref_pred.corr(curr_pred)
                accuracy_change = 1.0 - abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                # For categorical predictions, calculate agreement
                agreement = (ref_pred == curr_pred).mean()
                accuracy_change = 1.0 - agreement

            return accuracy_change

        except Exception as e:
            logger.error(f"Accuracy change calculation failed: {str(e)}")
            return 0.0

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
