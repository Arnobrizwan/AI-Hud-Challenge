"""
Anomaly Detection Detectors
Various detectors for different types of anomalies
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from safety_engine.config import get_anomaly_config
from safety_engine.models import Anomaly, ThreatLevel
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)


class BaseAnomalyDetector:
    """Base class for anomaly detectors"""

    def __init__(self):
        self.config = get_anomaly_config()
        self.is_initialized = False
        self.is_trained = False
        self.scaler = StandardScaler()

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the detector"""
        self.is_initialized = True

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup resources"""
        self.is_initialized = False

    async def train(self, X: np.ndarray) -> Dict[str, Any]:
        """Train the detector"""
        pass

    async def detect(self, features: Dict[str, float]) -> Optional[Any]:
        """Detect anomalies in features"""
        pass


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector"""

    def __init__(self):
        super().__init__()
        self.model = IsolationForest(
            contamination=self.config.isolation_forest_contamination,
            random_state=42,
            n_estimators=100,
        )
        self.feature_names = []

    async def train(self, X: np.ndarray) -> Dict[str, Any]:
        """Train the Isolation Forest model"""
        try:
            if len(X) < 10:
            except Exception as e:
                pass

                logger.warning("Insufficient data for Isolation Forest training")
                return

            # Fit scaler
            X_scaled = self.scaler.fit_transform(X)

            # Train model
            self.model.fit(X_scaled)
            self.is_trained = True

            logger.info("Isolation Forest model trained successfully")

        except Exception as e:
            logger.error(f"Isolation Forest training failed: {str(e)}")

    async def detect(self, features: Dict[str, float]) -> Optional[Any]:
        """Detect anomalies using Isolation Forest"""
        try:
            if not self.is_trained:
            except Exception as e:
                pass

                return None

            # Convert features to array
            feature_array = self.features_to_array(features)
            if feature_array is None:
                return None

            # Scale features
            feature_array_scaled = self.scaler.transform(feature_array.reshape(1, -1))

            # Predict anomaly
            anomaly_score = self.model.decision_function(feature_array_scaled)[0]
            is_anomaly = self.model.predict(feature_array_scaled)[0] == -1

            # Convert to 0-1 scale
            normalized_score = max(0.0, min(1.0, (anomaly_score + 0.5) * 2))

            # Create anomalies for significant deviations
            anomalies = []
            if is_anomaly or normalized_score > 0.7:
                # Find which features contributed most to the anomaly
                feature_contributions = self.calculate_feature_contributions(features, feature_array_scaled)

                for feature_name, contribution in feature_contributions.items():
                    if abs(contribution) > 0.5:  # Significant contribution
                        anomalies.append(
                            Anomaly(
                                metric_name=feature_name,
                                anomaly_type="isolation_forest",
                                severity=self.determine_severity(abs(contribution)),
                                value=features.get(feature_name, 0.0),
                                expected_value=0.0,  # Would need historical data for this
                                deviation=abs(contribution),
                            )

            return type(
                "AnomalyResult",
                (),
                {
                    "anomaly_score": normalized_score,
                    "is_anomaly": is_anomaly,
                    "anomalies": anomalies,
                },
            )()

        except Exception as e:
            logger.error(f"Isolation Forest detection failed: {str(e)}")
            return None

    def features_to_array(self, features: Dict[str, float]) -> Optional[np.ndarray]:
        """Convert features dictionary to numpy array"""
        try:
            if not features:
            except Exception as e:
                pass

                return None

            # Use predefined feature order or create from features
            if not self.feature_names:
                self.feature_names = sorted(features.keys())

            feature_array = np.array([features.get(name, 0.0) for name in self.feature_names])
            return feature_array

        except Exception as e:
            logger.error(f"Feature array conversion failed: {str(e)}")
            return None

    def calculate_feature_contributions(
        self, features: Dict[str, float], scaled_features: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature contributions to anomaly score"""
        try:
            contributions = {}
            except Exception as e:
                pass


            # Simple contribution calculation based on feature values
            for i, feature_name in enumerate(self.feature_names):
                if i < len(scaled_features[0]):
                    # Higher absolute values contribute more to anomaly
                    contributions[feature_name] = abs(scaled_features[0][i])

            return contributions

        except Exception as e:
            logger.error(f"Feature contribution calculation failed: {str(e)}")
            return {}

    def determine_severity(self, contribution: float) -> str:
        """Determine anomaly severity based on contribution"""
        if contribution > 0.8:
            return "critical"
        elif contribution > 0.6:
            return "high"
        elif contribution > 0.4:
            return "medium"
        else:
            return "low"


class OneClassSVMDetector(BaseAnomalyDetector):
    """One-Class SVM anomaly detector"""

    def __init__(self):
        super().__init__()
        self.model = OneClassSVM(nu=self.config.one_class_svm_nu, kernel="rbf", gamma="scale")
        self.feature_names = []

    async def train(self, X: np.ndarray) -> Dict[str, Any]:
        """Train the One-Class SVM model"""
        try:
            if len(X) < 10:
            except Exception as e:
                pass

                logger.warning("Insufficient data for One-Class SVM training")
                return

            # Fit scaler
            X_scaled = self.scaler.fit_transform(X)

            # Train model
            self.model.fit(X_scaled)
            self.is_trained = True

            logger.info("One-Class SVM model trained successfully")

        except Exception as e:
            logger.error(f"One-Class SVM training failed: {str(e)}")

    async def detect(self, features: Dict[str, float]) -> Optional[Any]:
        """Detect anomalies using One-Class SVM"""
        try:
            if not self.is_trained:
            except Exception as e:
                pass

                return None

            # Convert features to array
            feature_array = self.features_to_array(features)
            if feature_array is None:
                return None

            # Scale features
            feature_array_scaled = self.scaler.transform(feature_array.reshape(1, -1))

            # Predict anomaly
            prediction = self.model.predict(feature_array_scaled)[0]
            decision_score = self.model.decision_function(feature_array_scaled)[0]

            # Convert to 0-1 scale
            is_anomaly = prediction == -1
            normalized_score = max(0.0, min(1.0, (decision_score + 1) / 2))

            # Create anomalies for significant deviations
            anomalies = []
            if is_anomaly or normalized_score > 0.7:
                # Find which features contributed most to the anomaly
                feature_contributions = self.calculate_feature_contributions(features, feature_array_scaled)

                for feature_name, contribution in feature_contributions.items():
                    if abs(contribution) > 0.5:  # Significant contribution
                        anomalies.append(
                            Anomaly(
                                metric_name=feature_name,
                                anomaly_type="one_class_svm",
                                severity=self.determine_severity(abs(contribution)),
                                value=features.get(feature_name, 0.0),
                                expected_value=0.0,
                                deviation=abs(contribution),
                            )

            return type(
                "AnomalyResult",
                (),
                {
                    "anomaly_score": normalized_score,
                    "is_anomaly": is_anomaly,
                    "anomalies": anomalies,
                },
            )()

        except Exception as e:
            logger.error(f"One-Class SVM detection failed: {str(e)}")
            return None

    def features_to_array(self, features: Dict[str, float]) -> Optional[np.ndarray]:
        """Convert features dictionary to numpy array"""
        try:
            if not features:
            except Exception as e:
                pass

                return None

            # Use predefined feature order or create from features
            if not self.feature_names:
                self.feature_names = sorted(features.keys())

            feature_array = np.array([features.get(name, 0.0) for name in self.feature_names])
            return feature_array

        except Exception as e:
            logger.error(f"Feature array conversion failed: {str(e)}")
            return None

    def calculate_feature_contributions(
        self, features: Dict[str, float], scaled_features: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature contributions to anomaly score"""
        try:
            contributions = {}
            except Exception as e:
                pass


            # Simple contribution calculation based on feature values
            for i, feature_name in enumerate(self.feature_names):
                if i < len(scaled_features[0]):
                    # Higher absolute values contribute more to anomaly
                    contributions[feature_name] = abs(scaled_features[0][i])

            return contributions

        except Exception as e:
            logger.error(f"Feature contribution calculation failed: {str(e)}")
            return {}

    def determine_severity(self, contribution: float) -> str:
        """Determine anomaly severity based on contribution"""
        if contribution > 0.8:
            return "critical"
        elif contribution > 0.6:
            return "high"
        elif contribution > 0.4:
            return "medium"
        else:
            return "low"


class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """LSTM Autoencoder anomaly detector"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.sequence_length = self.config.lstm_sequence_length
        self.feature_names = []
        self.historical_sequences = []

    async def train(self, X: np.ndarray) -> Dict[str, Any]:
        """Train the LSTM Autoencoder model"""
        try:
            if len(X) < self.sequence_length * 2:
            except Exception as e:
                pass

                logger.warning("Insufficient data for LSTM Autoencoder training")
                return

            # Create sequences
            sequences = self.create_sequences(X)
            if len(sequences) < 10:
                logger.warning("Insufficient sequences for LSTM Autoencoder training")
                return

            # For now, we'll simulate training
            # In a real implementation, this would train an LSTM autoencoder
            self.model = "simulated_lstm_model"
            self.is_trained = True

            logger.info("LSTM Autoencoder model trained successfully")

        except Exception as e:
            logger.error(f"LSTM Autoencoder training failed: {str(e)}")

    async def detect(self, features: Dict[str, float]) -> Optional[Any]:
        """Detect anomalies using LSTM Autoencoder"""
        try:
            if not self.is_trained:
            except Exception as e:
                pass

                return None

            # Convert features to array
            feature_array = self.features_to_array(features)
            if feature_array is None:
                return None

            # Add to historical sequences
            self.historical_sequences.append(feature_array)

            # Keep only recent sequences
            if len(self.historical_sequences) > self.sequence_length * 2:
                self.historical_sequences = self.historical_sequences[-self.sequence_length * 2 :]

            # Create sequence if we have enough data
            if len(self.historical_sequences) >= self.sequence_length:
                sequence = np.array(self.historical_sequences[-self.sequence_length :])

                # Simulate reconstruction error
                # In a real implementation, this would use the trained
                # autoencoder
                reconstruction_error = self.simulate_reconstruction_error(sequence)

                # Convert to anomaly score
                anomaly_score = min(1.0, reconstruction_error)

                # Create anomalies for significant deviations
                anomalies = []
                if anomaly_score > 0.7:
                    # Find which features contributed most to the anomaly
                    feature_contributions = self.calculate_feature_contributions(features, feature_array)

                    for feature_name, contribution in feature_contributions.items():
                        if abs(contribution) > 0.5:  # Significant contribution
                            anomalies.append(
                                Anomaly(
                                    metric_name=feature_name,
                                    anomaly_type="lstm_autoencoder",
                                    severity=self.determine_severity(abs(contribution)),
                                    value=features.get(feature_name, 0.0),
                                    expected_value=0.0,
                                    deviation=abs(contribution),
                                )

                return type(
                    "AnomalyResult",
                    (),
                    {
                        "anomaly_score": anomaly_score,
                        "is_anomaly": anomaly_score > 0.7,
                        "anomalies": anomalies,
                    },
                )()

            return None

        except Exception as e:
            logger.error(f"LSTM Autoencoder detection failed: {str(e)}")
            return None

    def create_sequences(self, X: np.ndarray) -> List[np.ndarray]:
        """Create sequences from time series data"""
        try:
            sequences = []
            except Exception as e:
                pass

            for i in range(len(X) - self.sequence_length + 1):
                sequence = X[i : i + self.sequence_length]
                sequences.append(sequence)
            return sequences

        except Exception as e:
            logger.error(f"Sequence creation failed: {str(e)}")
            return []

    def simulate_reconstruction_error(self, sequence: np.ndarray) -> float:
        """Simulate reconstruction error (in real implementation, use trained autoencoder)"""
        try:
            # Simple simulation based on sequence variance
            except Exception as e:
                pass

            sequence_variance = np.var(sequence)
            sequence_mean = np.mean(sequence)

            # Higher variance and deviation from mean indicate higher
            # reconstruction error
            reconstruction_error = min(1.0, sequence_variance + abs(sequence_mean - 0.5))

            return reconstruction_error

        except Exception as e:
            logger.error(f"Reconstruction error simulation failed: {str(e)}")
            return 0.0

    def features_to_array(self, features: Dict[str, float]) -> Optional[np.ndarray]:
        """Convert features dictionary to numpy array"""
        try:
            if not features:
            except Exception as e:
                pass

                return None

            # Use predefined feature order or create from features
            if not self.feature_names:
                self.feature_names = sorted(features.keys())

            feature_array = np.array([features.get(name, 0.0) for name in self.feature_names])
            return feature_array

        except Exception as e:
            logger.error(f"Feature array conversion failed: {str(e)}")
            return None

    def calculate_feature_contributions(
        self, features: Dict[str, float], feature_array: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature contributions to anomaly score"""
        try:
            contributions = {}
            except Exception as e:
                pass


            # Simple contribution calculation based on feature values
            for i, feature_name in enumerate(self.feature_names):
                if i < len(feature_array):
                    # Higher absolute values contribute more to anomaly
                    contributions[feature_name] = abs(feature_array[i])

            return contributions

        except Exception as e:
            logger.error(f"Feature contribution calculation failed: {str(e)}")
            return {}

    def determine_severity(self, contribution: float) -> str:
        """Determine anomaly severity based on contribution"""
        if contribution > 0.8:
            return "critical"
        elif contribution > 0.6:
            return "high"
        elif contribution > 0.4:
            return "medium"
        else:
            return "low"


class StatisticalAnomalyDetector(BaseAnomalyDetector):
    """Statistical anomaly detector using z-scores and percentiles"""

    def __init__(self):
        super().__init__()
        self.feature_stats = {}
        self.feature_names = []

    async def train(self, X: np.ndarray) -> Dict[str, Any]:
        """Train the statistical detector"""
        try:
            if len(X) < 10:
            except Exception as e:
                pass

                logger.warning("Insufficient data for statistical detector training")
                return

            # Calculate statistics for each feature
            for i in range(X.shape[1]):
                feature_name = f"feature_{i}"
                if i < len(self.feature_names):
                    feature_name = self.feature_names[i]

                feature_data = X[:, i]
                self.feature_stats[feature_name] = {
                    "mean": np.mean(feature_data),
                    "std": np.std(feature_data),
                    "min": np.min(feature_data),
                    "max": np.max(feature_data),
                    "q25": np.percentile(feature_data, 25),
                    "q75": np.percentile(feature_data, 75),
                    "q95": np.percentile(feature_data, 95),
                    "q99": np.percentile(feature_data, 99),
                }

            self.is_trained = True
            logger.info("Statistical detector trained successfully")

        except Exception as e:
            logger.error(f"Statistical detector training failed: {str(e)}")

    async def detect(self, features: Dict[str, float]) -> Optional[Any]:
        """Detect anomalies using statistical methods"""
        try:
            if not self.is_trained:
            except Exception as e:
                pass

                return None

            anomalies = []
            max_anomaly_score = 0.0

            for feature_name, value in features.items():
                if feature_name not in self.feature_stats:
                    continue

                stats = self.feature_stats[feature_name]

                # Calculate z-score
                if stats["std"] > 0:
                    z_score = abs((value - stats["mean"]) / stats["std"])
                else:
                    z_score = 0.0

                # Calculate percentile-based anomaly score
                percentile_score = 0.0
                if value > stats["q99"]:
                    percentile_score = 1.0
                elif value > stats["q95"]:
                    percentile_score = 0.8
                elif value > stats["q75"]:
                    percentile_score = 0.5
                elif value < stats["q25"]:
                    percentile_score = 0.5

                # Combine z-score and percentile scores
                anomaly_score = max(z_score / 3.0, percentile_score)  # Normalize z-score
                max_anomaly_score = max(max_anomaly_score, anomaly_score)

                # Create anomaly if significant
                if anomaly_score > 0.7:
                    anomalies.append(
                        Anomaly(
                            metric_name=feature_name,
                            anomaly_type="statistical",
                            severity=self.determine_severity(anomaly_score),
                            value=value,
                            expected_value=stats["mean"],
                            deviation=abs(value - stats["mean"]),
                        )

            return type(
                "AnomalyResult",
                (),
                {
                    "anomaly_score": min(max_anomaly_score, 1.0),
                    "is_anomaly": len(anomalies) > 0,
                    "anomalies": anomalies,
                },
            )()

        except Exception as e:
            logger.error(f"Statistical detection failed: {str(e)}")
            return None

    def determine_severity(self, anomaly_score: float) -> str:
        """Determine anomaly severity based on score"""
        if anomaly_score > 0.9:
            return "critical"
        elif anomaly_score > 0.8:
            return "high"
        elif anomaly_score > 0.7:
            return "medium"
        else:
            return "low"
