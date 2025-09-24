"""
Anomaly Detection System
System anomaly detection and monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from safety_engine.config import get_anomaly_config
from safety_engine.models import Anomaly, AnomalyDetectionRequest, AnomalyDetectionResult

from .detectors import (
    IsolationForestDetector,
    LSTMAutoencoderDetector,
    OneClassSVMDetector,
    StatisticalAnomalyDetector,
)

logger = logging.getLogger(__name__)


class AnomalyDetectionSystem:
    """System anomaly detection and monitoring"""

    def __init__(self):
        self.config = get_anomaly_config()
        self.is_initialized = False

        # Anomaly detectors
        self.isolation_forest = IsolationForestDetector()
        self.one_class_svm = OneClassSVMDetector()
        self.lstm_autoencoder = LSTMAutoencoderDetector()
        self.statistical_detector = StatisticalAnomalyDetector()

        # Historical data for training
        self.historical_metrics = []
        self.user_behavior_history = {}

    async def initialize(self):
        """Initialize the anomaly detection system"""
        try:
            # Initialize all detectors
            await self.isolation_forest.initialize()
            await self.one_class_svm.initialize()
            await self.lstm_autoencoder.initialize()
            await self.statistical_detector.initialize()

            self.is_initialized = True
            logger.info("Anomaly detection system initialized")

        except Exception as e:
            logger.error(f"Failed to initialize anomaly detection system: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.isolation_forest.cleanup()
            await self.one_class_svm.cleanup()
            await self.lstm_autoencoder.cleanup()
            await self.statistical_detector.cleanup()

            self.is_initialized = False
            logger.info("Anomaly detection system cleanup completed")

        except Exception as e:
            logger.error(f"Error during anomaly detection system cleanup: {str(e)}")

    async def detect_anomalies(self, request: AnomalyDetectionRequest) -> AnomalyDetectionResult:
        """Detect system anomalies"""

        if not self.is_initialized:
            raise RuntimeError("Anomaly detection system not initialized")

        try:
            # Extract features from request
            features = self.extract_features(request)

            if not features:
                return AnomalyDetectionResult(
                    anomaly_score=0.0,
                    anomalies_detected=[],
                    system_health="unknown",
                    recommendations=[],
                )

            # Run anomaly detection using multiple methods
            detection_results = await asyncio.gather(
                self.isolation_forest.detect(features),
                self.one_class_svm.detect(features),
                self.lstm_autoencoder.detect(features),
                self.statistical_detector.detect(features),
                return_exceptions=True,
            )

            # Process results
            anomalies = []
            anomaly_scores = []

            for i, result in enumerate(detection_results):
                if isinstance(result, Exception):
                    logger.warning(f"Anomaly detector {i} failed: {str(result)}")
                    continue

                if result and hasattr(result, "anomalies"):
                    anomalies.extend(result.anomalies)
                if result and hasattr(result, "anomaly_score"):
                    anomaly_scores.append(result.anomaly_score)

            # Calculate overall anomaly score
            overall_anomaly_score = np.mean(anomaly_scores) if anomaly_scores else 0.0

            # Determine system health
            system_health = self.determine_system_health(overall_anomaly_score, anomalies)

            # Generate recommendations
            recommendations = self.generate_recommendations(anomalies, system_health)

            # Update historical data
            await self.update_historical_data(request, overall_anomaly_score)

            return AnomalyDetectionResult(
                anomaly_score=overall_anomaly_score,
                anomalies_detected=anomalies,
                system_health=system_health,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            raise

    def extract_features(self, request: AnomalyDetectionRequest) -> Dict[str, float]:
        """Extract features from anomaly detection request"""
        try:
            features = {}

            # System metrics
            system_metrics = request.system_metrics
            for metric_name, value in system_metrics.items():
                features[f"system_{metric_name}"] = float(value)

            # User behavior metrics
            if request.user_behavior:
                user_behavior = request.user_behavior
                for behavior_name, value in user_behavior.items():
                    if isinstance(value, (int, float)):
                        features[f"user_{behavior_name}"] = float(value)
                    elif isinstance(value, dict):
                        # Flatten nested dictionaries
                        for sub_name, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                features[f"user_{behavior_name}_{sub_name}"] = float(sub_value)

            # Add derived features
            features.update(self.calculate_derived_features(features))

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return {}

    def calculate_derived_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate derived features from base features"""
        try:
            derived_features = {}

            # System load features
            if "system_cpu_usage" in features and "system_memory_usage" in features:
                cpu_usage = features["system_cpu_usage"]
                memory_usage = features["system_memory_usage"]
                derived_features["system_load_avg"] = (cpu_usage + memory_usage) / 2
                derived_features["system_load_ratio"] = cpu_usage / max(memory_usage, 0.01)

            # Request rate features
            if "system_requests_per_second" in features:
                rps = features["system_requests_per_second"]
                derived_features["system_request_rate_log"] = np.log(rps + 1)
                derived_features["system_request_rate_sqrt"] = np.sqrt(rps)

            # Error rate features
            if "system_error_rate" in features:
                error_rate = features["system_error_rate"]
                derived_features["system_error_rate_squared"] = error_rate**2
                derived_features["system_error_rate_log"] = np.log(error_rate + 0.001)

            # Response time features
            if "system_avg_response_time" in features:
                response_time = features["system_avg_response_time"]
                derived_features["system_response_time_log"] = np.log(response_time + 1)
                derived_features["system_response_time_sqrt"] = np.sqrt(response_time)

            # User activity features
            user_features = {k: v for k, v in features.items() if k.startswith("user_")}
            if user_features:
                user_values = list(user_features.values())
                derived_features["user_activity_mean"] = np.mean(user_values)
                derived_features["user_activity_std"] = np.std(user_values)
                derived_features["user_activity_max"] = np.max(user_values)
                derived_features["user_activity_min"] = np.min(user_values)

            return derived_features

        except Exception as e:
            logger.error(f"Derived feature calculation failed: {str(e)}")
            return {}

    def determine_system_health(self, anomaly_score: float, anomalies: List[Anomaly]) -> str:
        """Determine overall system health based on anomaly score and detected anomalies"""
        try:
            # Count anomalies by severity
            critical_anomalies = sum(1 for a in anomalies if a.severity == "critical")
            high_anomalies = sum(1 for a in anomalies if a.severity == "high")
            medium_anomalies = sum(1 for a in anomalies if a.severity == "medium")

            # Determine health status
            if critical_anomalies > 0 or anomaly_score > 0.9:
                return "critical"
            elif high_anomalies > 2 or anomaly_score > 0.7:
                return "degraded"
            elif medium_anomalies > 5 or anomaly_score > 0.5:
                return "warning"
            elif anomaly_score > 0.3:
                return "monitoring"
            else:
                return "healthy"

        except Exception as e:
            logger.error(f"System health determination failed: {str(e)}")
            return "unknown"

    def generate_recommendations(self, anomalies: List[Anomaly], system_health: str) -> List[str]:
        """Generate recommendations based on detected anomalies"""
        try:
            recommendations = []

            # Health-based recommendations
            if system_health == "critical":
                recommendations.append(
                    "Immediate intervention required - system is in critical state"
                )
                recommendations.append("Consider scaling up resources or reducing load")
                recommendations.append("Review recent changes and rollback if necessary")
            elif system_health == "degraded":
                recommendations.append("System performance is degraded - monitor closely")
                recommendations.append("Consider proactive scaling or load balancing")
                recommendations.append("Review system logs for root cause")
            elif system_health == "warning":
                recommendations.append("System showing warning signs - increase monitoring")
                recommendations.append("Consider preventive maintenance")
                recommendations.append("Review system configuration")

            # Anomaly-specific recommendations
            for anomaly in anomalies:
                if anomaly.metric_name == "cpu_usage" and anomaly.value > 0.8:
                    recommendations.append(
                        "High CPU usage detected - consider scaling or optimization"
                    )
                elif anomaly.metric_name == "memory_usage" and anomaly.value > 0.8:
                    recommendations.append(
                        "High memory usage detected - consider memory optimization"
                    )
                elif anomaly.metric_name == "error_rate" and anomaly.value > 0.1:
                    recommendations.append("High error rate detected - investigate error sources")
                elif anomaly.metric_name == "response_time" and anomaly.value > 1000:
                    recommendations.append(
                        "High response time detected - optimize database queries or caching"
                    )
                elif anomaly.metric_name == "request_rate" and anomaly.value > 1000:
                    recommendations.append(
                        "High request rate detected - consider rate limiting or scaling"
                    )

            # Remove duplicates and return
            return list(set(recommendations))

        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return []

    async def update_historical_data(self, request: AnomalyDetectionRequest, anomaly_score: float):
        """Update historical data for training and analysis"""
        try:
            # Add current metrics to historical data
            current_data = {
                "timestamp": datetime.utcnow(),
                "anomaly_score": anomaly_score,
                "system_metrics": request.system_metrics.copy(),
                "user_behavior": request.user_behavior.copy() if request.user_behavior else {},
            }

            self.historical_metrics.append(current_data)

            # Keep only recent data (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.historical_metrics = [
                data for data in self.historical_metrics if data["timestamp"] > cutoff_time
            ]

            # Update user behavior history
            if request.user_behavior:
                user_id = request.user_behavior.get("user_id")
                if user_id:
                    if user_id not in self.user_behavior_history:
                        self.user_behavior_history[user_id] = []

                    self.user_behavior_history[user_id].append(
                        {
                            "timestamp": datetime.utcnow(),
                            "behavior": request.user_behavior.copy(),
                            "anomaly_score": anomaly_score,
                        }
                    )

                    # Keep only recent data for each user
                    cutoff_time = datetime.utcnow() - timedelta(hours=24)
                    self.user_behavior_history[user_id] = [
                        data
                        for data in self.user_behavior_history[user_id]
                        if data["timestamp"] > cutoff_time
                    ]

        except Exception as e:
            logger.error(f"Historical data update failed: {str(e)}")

    async def train_models(self):
        """Train anomaly detection models with historical data"""
        try:
            if len(self.historical_metrics) < 100:
                logger.warning("Insufficient historical data for training")
                return

            # Prepare training data
            training_data = []
            for data in self.historical_metrics:
                features = self.extract_features_from_historical_data(data)
                if features:
                    training_data.append(features)

            if not training_data:
                logger.warning("No valid training data available")
                return

            # Convert to numpy array
            X = np.array(training_data)

            # Train models
            await self.isolation_forest.train(X)
            await self.one_class_svm.train(X)
            await self.lstm_autoencoder.train(X)
            await self.statistical_detector.train(X)

            logger.info("Anomaly detection models trained successfully")

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")

    def extract_features_from_historical_data(self, data: Dict[str, Any]) -> List[float]:
        """Extract features from historical data"""
        try:
            features = []

            # System metrics
            system_metrics = data.get("system_metrics", {})
            for metric_name in [
                "cpu_usage",
                "memory_usage",
                "disk_usage",
                "network_usage",
                "error_rate",
                "response_time",
            ]:
                features.append(system_metrics.get(metric_name, 0.0))

            # User behavior metrics
            user_behavior = data.get("user_behavior", {})
            for behavior_name in [
                "activity_level",
                "request_frequency",
                "session_duration",
                "error_count",
            ]:
                features.append(user_behavior.get(behavior_name, 0.0))

            return features

        except Exception as e:
            logger.error(f"Feature extraction from historical data failed: {str(e)}")
            return []

    async def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get anomaly detection statistics"""
        try:
            if not self.historical_metrics:
                return {"message": "No historical data available"}

            # Calculate statistics
            anomaly_scores = [data["anomaly_score"] for data in self.historical_metrics]

            stats = {
                "total_observations": len(self.historical_metrics),
                "average_anomaly_score": np.mean(anomaly_scores),
                "max_anomaly_score": np.max(anomaly_scores),
                "min_anomaly_score": np.min(anomaly_scores),
                "anomaly_score_std": np.std(anomaly_scores),
                "high_anomaly_count": sum(1 for score in anomaly_scores if score > 0.7),
                "critical_anomaly_count": sum(1 for score in anomaly_scores if score > 0.9),
                "detectors_status": {
                    "isolation_forest": self.isolation_forest.is_trained,
                    "one_class_svm": self.one_class_svm.is_trained,
                    "lstm_autoencoder": self.lstm_autoencoder.is_trained,
                    "statistical_detector": self.statistical_detector.is_trained,
                },
            }

            return stats

        except Exception as e:
            logger.error(f"Anomaly statistics calculation failed: {str(e)}")
            return {"error": str(e)}

    async def get_system_health_trend(self, hours: int = 24) -> Dict[str, Any]:
        """Get system health trend over time"""
        try:
            if not self.historical_metrics:
                return {"message": "No historical data available"}

            # Filter data by time window
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_data = [
                data for data in self.historical_metrics if data["timestamp"] > cutoff_time
            ]

            if not recent_data:
                return {"message": "No data in specified time window"}

            # Calculate trend
            timestamps = [data["timestamp"] for data in recent_data]
            anomaly_scores = [data["anomaly_score"] for data in recent_data]

            # Calculate trend direction
            if len(anomaly_scores) > 1:
                trend_slope = np.polyfit(range(len(anomaly_scores)), anomaly_scores, 1)[0]
                trend_direction = (
                    "improving"
                    if trend_slope < -0.01
                    else "degrading" if trend_slope > 0.01 else "stable"
                )
            else:
                trend_direction = "stable"

            return {
                "time_window_hours": hours,
                "data_points": len(recent_data),
                "trend_direction": trend_direction,
                "average_anomaly_score": np.mean(anomaly_scores),
                "max_anomaly_score": np.max(anomaly_scores),
                "min_anomaly_score": np.min(anomaly_scores),
                "current_health": self.determine_system_health(np.mean(anomaly_scores), []),
            }

        except Exception as e:
            logger.error(f"System health trend calculation failed: {str(e)}")
            return {"error": str(e)}
