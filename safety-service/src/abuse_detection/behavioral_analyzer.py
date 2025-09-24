"""
Behavioral Anomaly Detector
Analyze user behavior patterns for abuse detection
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from safety_engine.models import BehavioralSignals
from safety_engine.config import get_abuse_config

logger = logging.getLogger(__name__)

class BehavioralAnomalyDetector:
    """Detect behavioral anomalies in user activity"""
    
    def __init__(self):
        self.config = get_abuse_config()
        self.is_initialized = False
        
        # Models
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.clustering_model = None
        
        # Behavioral patterns
        self.normal_patterns = {}
        self.user_histories = {}
        
    async def initialize(self):
        """Initialize the behavioral analyzer"""
        try:
            # Initialize models
            self.isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            self.clustering_model = DBSCAN(
                eps=0.5,
                min_samples=5
            )
            
            self.is_initialized = True
            logger.info("Behavioral anomaly detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize behavioral analyzer: {str(e)}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.isolation_forest = None
            self.scaler = None
            self.clustering_model = None
            self.normal_patterns.clear()
            self.user_histories.clear()
            
            self.is_initialized = False
            logger.info("Behavioral analyzer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during behavioral analyzer cleanup: {str(e)}")
    
    async def analyze_behavior(self, user_id: str, recent_activities: List[Dict[str, Any]],
                             time_window: timedelta) -> BehavioralSignals:
        """Analyze user behavior for anomalies"""
        
        if not self.is_initialized:
            raise RuntimeError("Behavioral analyzer not initialized")
        
        try:
            # Extract behavioral features
            features = self.extract_behavioral_features(recent_activities, time_window)
            
            # Check velocity anomalies
            velocity_anomaly = await self.check_velocity_anomaly(user_id, features)
            
            # Check pattern anomalies
            pattern_anomaly = await self.check_pattern_anomaly(user_id, features)
            
            # Check frequency anomalies
            frequency_anomaly = await self.check_frequency_anomaly(user_id, features)
            
            # Check time anomalies
            time_anomaly = await self.check_time_anomaly(user_id, features)
            
            # Calculate overall anomaly score
            anomaly_score = self.calculate_anomaly_score(
                velocity_anomaly, pattern_anomaly, frequency_anomaly, time_anomaly
            )
            
            return BehavioralSignals(
                anomaly_score=anomaly_score,
                velocity_anomaly=velocity_anomaly,
                pattern_anomaly=pattern_anomaly,
                frequency_anomaly=frequency_anomaly,
                time_anomaly=time_anomaly
            )
            
        except Exception as e:
            logger.error(f"Behavioral analysis failed: {str(e)}")
            return BehavioralSignals(anomaly_score=0.0)
    
    def extract_behavioral_features(self, activities: List[Dict[str, Any]], 
                                  time_window: timedelta) -> Dict[str, float]:
        """Extract behavioral features from activities"""
        try:
            if not activities:
                return self.get_default_features()
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(activities)
            
            # Ensure timestamp column exists
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.Timestamp.now()
            
            # Convert timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by time window
            cutoff_time = datetime.utcnow() - time_window
            df = df[df['timestamp'] >= cutoff_time]
            
            if len(df) == 0:
                return self.get_default_features()
            
            # Calculate features
            features = {}
            
            # Activity frequency
            features['activity_count'] = len(df)
            features['activities_per_hour'] = len(df) / max(time_window.total_seconds() / 3600, 1)
            
            # Time-based features
            if len(df) > 1:
                time_diffs = df['timestamp'].diff().dt.total_seconds().dropna()
                features['avg_time_between_activities'] = time_diffs.mean()
                features['time_variance'] = time_diffs.var()
                features['min_time_between_activities'] = time_diffs.min()
                features['max_time_between_activities'] = time_diffs.max()
            else:
                features['avg_time_between_activities'] = 0
                features['time_variance'] = 0
                features['min_time_between_activities'] = 0
                features['max_time_between_activities'] = 0
            
            # Activity type distribution
            if 'activity_type' in df.columns:
                type_counts = df['activity_type'].value_counts()
                features['activity_diversity'] = len(type_counts)
                features['most_common_activity_ratio'] = type_counts.max() / len(df)
            else:
                features['activity_diversity'] = 1
                features['most_common_activity_ratio'] = 1.0
            
            # Request size features
            if 'request_size' in df.columns:
                features['avg_request_size'] = df['request_size'].mean()
                features['max_request_size'] = df['request_size'].max()
                features['request_size_variance'] = df['request_size'].var()
            else:
                features['avg_request_size'] = 0
                features['max_request_size'] = 0
                features['request_size_variance'] = 0
            
            # Error rate
            if 'success' in df.columns:
                features['error_rate'] = 1 - df['success'].mean()
            else:
                features['error_rate'] = 0
            
            # Response time features
            if 'response_time' in df.columns:
                features['avg_response_time'] = df['response_time'].mean()
                features['max_response_time'] = df['response_time'].max()
                features['response_time_variance'] = df['response_time'].var()
            else:
                features['avg_response_time'] = 0
                features['max_response_time'] = 0
                features['response_time_variance'] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return self.get_default_features()
    
    def get_default_features(self) -> Dict[str, float]:
        """Get default feature values when no data is available"""
        return {
            'activity_count': 0,
            'activities_per_hour': 0,
            'avg_time_between_activities': 0,
            'time_variance': 0,
            'min_time_between_activities': 0,
            'max_time_between_activities': 0,
            'activity_diversity': 1,
            'most_common_activity_ratio': 1.0,
            'avg_request_size': 0,
            'max_request_size': 0,
            'request_size_variance': 0,
            'error_rate': 0,
            'avg_response_time': 0,
            'max_response_time': 0,
            'response_time_variance': 0
        }
    
    async def check_velocity_anomaly(self, user_id: str, features: Dict[str, float]) -> bool:
        """Check for velocity anomalies (unusual activity speed)"""
        try:
            # Get user's historical patterns
            user_history = self.user_histories.get(user_id, [])
            
            if len(user_history) < 5:  # Need some history
                return False
            
            # Calculate current velocity metrics
            current_velocity = features.get('activities_per_hour', 0)
            current_request_size = features.get('avg_request_size', 0)
            
            # Calculate historical averages
            historical_velocities = [h.get('activities_per_hour', 0) for h in user_history[-10:]]
            historical_request_sizes = [h.get('avg_request_size', 0) for h in user_history[-10:]]
            
            avg_velocity = np.mean(historical_velocities)
            avg_request_size = np.mean(historical_request_sizes)
            
            # Check for velocity anomalies
            velocity_ratio = current_velocity / max(avg_velocity, 1e-6)
            request_size_ratio = current_request_size / max(avg_request_size, 1e-6)
            
            # Anomaly if velocity is significantly higher than normal
            velocity_anomaly = velocity_ratio > self.config.velocity_threshold
            
            # Anomaly if request size is significantly larger than normal
            request_size_anomaly = request_size_ratio > self.config.velocity_threshold
            
            return velocity_anomaly or request_size_anomaly
            
        except Exception as e:
            logger.error(f"Velocity anomaly check failed: {str(e)}")
            return False
    
    async def check_pattern_anomaly(self, user_id: str, features: Dict[str, float]) -> bool:
        """Check for pattern anomalies (unusual activity patterns)"""
        try:
            # Get user's historical patterns
            user_history = self.user_histories.get(user_id, [])
            
            if len(user_history) < 10:  # Need more history for pattern analysis
                return False
            
            # Prepare feature vectors
            historical_features = np.array([list(h.values()) for h in user_history[-20:]])
            current_features = np.array([list(features.values())])
            
            # Normalize features
            all_features = np.vstack([historical_features, current_features])
            normalized_features = self.scaler.fit_transform(all_features)
            
            current_normalized = normalized_features[-1:].reshape(1, -1)
            historical_normalized = normalized_features[:-1]
            
            # Use isolation forest to detect anomalies
            self.isolation_forest.fit(historical_normalized)
            anomaly_score = self.isolation_forest.decision_function(current_normalized)[0]
            
            # Anomaly if score is below threshold
            pattern_anomaly = anomaly_score < -0.1  # Isolation forest threshold
            
            return pattern_anomaly
            
        except Exception as e:
            logger.error(f"Pattern anomaly check failed: {str(e)}")
            return False
    
    async def check_frequency_anomaly(self, user_id: str, features: Dict[str, float]) -> bool:
        """Check for frequency anomalies (unusual activity frequency)"""
        try:
            # Get user's historical patterns
            user_history = self.user_histories.get(user_id, [])
            
            if len(user_history) < 5:
                return False
            
            # Calculate current frequency metrics
            current_activity_count = features.get('activity_count', 0)
            current_diversity = features.get('activity_diversity', 1)
            
            # Calculate historical averages
            historical_counts = [h.get('activity_count', 0) for h in user_history[-10:]]
            historical_diversities = [h.get('activity_diversity', 1) for h in user_history[-10:]]
            
            avg_count = np.mean(historical_counts)
            avg_diversity = np.mean(historical_diversities)
            
            # Check for frequency anomalies
            count_ratio = current_activity_count / max(avg_count, 1e-6)
            diversity_ratio = current_diversity / max(avg_diversity, 1e-6)
            
            # Anomaly if frequency is significantly different from normal
            count_anomaly = count_ratio > self.config.frequency_threshold or count_ratio < (1 / self.config.frequency_threshold)
            diversity_anomaly = abs(diversity_ratio - 1.0) > 0.5  # 50% change in diversity
            
            return count_anomaly or diversity_anomaly
            
        except Exception as e:
            logger.error(f"Frequency anomaly check failed: {str(e)}")
            return False
    
    async def check_time_anomaly(self, user_id: str, features: Dict[str, float]) -> bool:
        """Check for time anomalies (unusual timing patterns)"""
        try:
            # Get user's historical patterns
            user_history = self.user_histories.get(user_id, [])
            
            if len(user_history) < 5:
                return False
            
            # Calculate current time metrics
            current_time_variance = features.get('time_variance', 0)
            current_avg_time = features.get('avg_time_between_activities', 0)
            
            # Calculate historical averages
            historical_variances = [h.get('time_variance', 0) for h in user_history[-10:]]
            historical_avg_times = [h.get('avg_time_between_activities', 0) for h in user_history[-10:]]
            
            avg_variance = np.mean(historical_variances)
            avg_time = np.mean(historical_avg_times)
            
            # Check for time anomalies
            variance_ratio = current_time_variance / max(avg_variance, 1e-6)
            time_ratio = current_avg_time / max(avg_time, 1e-6)
            
            # Anomaly if timing patterns are significantly different
            variance_anomaly = variance_ratio > self.config.time_threshold or variance_ratio < (1 / self.config.time_threshold)
            time_anomaly = time_ratio > self.config.time_threshold or time_ratio < (1 / self.config.time_threshold)
            
            return variance_anomaly or time_anomaly
            
        except Exception as e:
            logger.error(f"Time anomaly check failed: {str(e)}")
            return False
    
    def calculate_anomaly_score(self, velocity_anomaly: bool, pattern_anomaly: bool,
                              frequency_anomaly: bool, time_anomaly: bool) -> float:
        """Calculate overall anomaly score"""
        try:
            # Weight different types of anomalies
            weights = {
                'velocity': 0.3,
                'pattern': 0.4,
                'frequency': 0.2,
                'time': 0.1
            }
            
            # Calculate weighted score
            score = 0.0
            if velocity_anomaly:
                score += weights['velocity']
            if pattern_anomaly:
                score += weights['pattern']
            if frequency_anomaly:
                score += weights['frequency']
            if time_anomaly:
                score += weights['time']
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Anomaly score calculation failed: {str(e)}")
            return 0.0
    
    async def update_user_history(self, user_id: str, features: Dict[str, float]) -> None:
        """Update user's behavioral history"""
        try:
            if user_id not in self.user_histories:
                self.user_histories[user_id] = []
            
            # Add current features to history
            self.user_histories[user_id].append(features.copy())
            
            # Keep only recent history (last 100 entries)
            if len(self.user_histories[user_id]) > 100:
                self.user_histories[user_id] = self.user_histories[user_id][-100:]
            
        except Exception as e:
            logger.error(f"User history update failed: {str(e)}")
    
    async def get_behavioral_summary(self, user_id: str) -> Dict[str, Any]:
        """Get behavioral summary for a user"""
        try:
            user_history = self.user_histories.get(user_id, [])
            
            if not user_history:
                return {"message": "No behavioral history available"}
            
            # Calculate summary statistics
            recent_history = user_history[-10:]  # Last 10 activities
            
            summary = {
                "total_activities": len(user_history),
                "recent_activities": len(recent_history),
                "avg_activities_per_hour": np.mean([h.get('activities_per_hour', 0) for h in recent_history]),
                "avg_activity_diversity": np.mean([h.get('activity_diversity', 1) for h in recent_history]),
                "avg_error_rate": np.mean([h.get('error_rate', 0) for h in recent_history]),
                "behavioral_consistency": self.calculate_behavioral_consistency(recent_history)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Behavioral summary calculation failed: {str(e)}")
            return {"error": str(e)}
    
    def calculate_behavioral_consistency(self, history: List[Dict[str, float]]) -> float:
        """Calculate behavioral consistency score"""
        try:
            if len(history) < 2:
                return 1.0
            
            # Calculate coefficient of variation for key metrics
            key_metrics = ['activities_per_hour', 'activity_diversity', 'error_rate']
            consistency_scores = []
            
            for metric in key_metrics:
                values = [h.get(metric, 0) for h in history if metric in h]
                if len(values) > 1 and np.mean(values) > 0:
                    cv = np.std(values) / np.mean(values)
                    consistency_scores.append(1.0 - min(cv, 1.0))  # Convert to consistency score
            
            return np.mean(consistency_scores) if consistency_scores else 1.0
            
        except Exception as e:
            logger.error(f"Behavioral consistency calculation failed: {str(e)}")
            return 0.0
