"""
Multidimensional Drift Detector
Advanced drift detection across multiple dimensions
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from .models import (
    DriftDetectionRequest, DriftAnalysisResult, DataDriftResult,
    FeatureDriftResult, StatisticalTestResult, ConceptDriftResult,
    PredictionDriftResult, ImportanceDriftResult
)
from safety_engine.config import get_drift_config
from .statistical_detectors import (
    KolmogorovSmirnovDetector, ChiSquareDetector,
    PopulationStabilityIndexDetector, WassersteinDetector
)
from .concept_drift import ConceptDriftDetector
from .prediction_drift import PredictionDriftDetector
from .feature_importance import FeatureImportanceMonitor

logger = logging.getLogger(__name__)

class MultidimensionalDriftDetector:
    """Advanced drift detection across multiple dimensions"""
    
    def __init__(self):
        self.config = get_drift_config()
        
        # Statistical detectors
        self.statistical_detectors = {
            'ks_test': KolmogorovSmirnovDetector(),
            'chi_square': ChiSquareDetector(),
            'psi': PopulationStabilityIndexDetector(),
            'wasserstein': WassersteinDetector()
        }
        
        # Specialized detectors
        self.concept_drift_detector = ConceptDriftDetector()
        self.prediction_drift_detector = PredictionDriftDetector()
        self.feature_importance_monitor = FeatureImportanceMonitor()
        
        # State
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the drift detector"""
        try:
            # Initialize all detectors
            for detector in self.statistical_detectors.values():
                await detector.initialize()
            
            await self.concept_drift_detector.initialize()
            await self.prediction_drift_detector.initialize()
            await self.feature_importance_monitor.initialize()
            
            self.is_initialized = True
            logger.info("Multidimensional drift detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize drift detector: {str(e)}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            for detector in self.statistical_detectors.values():
                await detector.cleanup()
            
            await self.concept_drift_detector.cleanup()
            await self.prediction_drift_detector.cleanup()
            await self.feature_importance_monitor.cleanup()
            
            logger.info("Drift detector cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during drift detector cleanup: {str(e)}")
    
    async def detect_comprehensive_drift(self, 
                                       drift_request: DriftDetectionRequest) -> DriftAnalysisResult:
        """Comprehensive drift detection across all dimensions"""
        
        if not self.is_initialized:
            raise RuntimeError("Drift detector not initialized")
        
        try:
            # Data drift detection
            data_drift_results = await self.detect_data_drift(
                reference_data=drift_request.reference_data,
                current_data=drift_request.current_data,
                features=drift_request.features_to_monitor
            )
            
            # Concept drift detection
            concept_drift_results = None
            if drift_request.reference_labels is not None and drift_request.current_labels is not None:
                concept_drift_results = await self.concept_drift_detector.detect_concept_drift(
                    reference_labels=drift_request.reference_labels,
                    current_labels=drift_request.current_labels,
                    feature_data=drift_request.current_data
                )
            else:
                concept_drift_results = ConceptDriftResult(
                    drift_detected=False,
                    drift_score=0.0,
                    affected_features=[],
                    confidence=1.0
                )
            
            # Prediction drift detection
            prediction_drift_results = None
            if drift_request.reference_predictions is not None and drift_request.current_predictions is not None:
                prediction_drift_results = await self.prediction_drift_detector.detect_prediction_drift(
                    reference_predictions=drift_request.reference_predictions,
                    current_predictions=drift_request.current_predictions
                )
            else:
                prediction_drift_results = PredictionDriftResult(
                    drift_detected=False,
                    drift_score=0.0,
                    prediction_accuracy_change=0.0,
                    confidence=1.0
                )
            
            # Feature importance drift
            importance_drift_results = None
            if drift_request.reference_model is not None and drift_request.current_model is not None:
                importance_drift_results = await self.feature_importance_monitor.detect_importance_drift(
                    reference_model=drift_request.reference_model,
                    current_model=drift_request.current_model
                )
            else:
                importance_drift_results = ImportanceDriftResult(
                    drift_detected=False,
                    drift_score=0.0,
                    feature_importance_changes={},
                    confidence=1.0
                )
            
            # Calculate overall drift severity
            overall_drift_severity = self.calculate_overall_drift_severity([
                data_drift_results,
                concept_drift_results,
                prediction_drift_results,
                importance_drift_results
            ])
            
            return DriftAnalysisResult(
                data_drift=data_drift_results,
                concept_drift=concept_drift_results,
                prediction_drift=prediction_drift_results,
                importance_drift=importance_drift_results,
                overall_severity=overall_drift_severity,
                requires_action=overall_drift_severity > self.config.severity_threshold,
                analysis_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Comprehensive drift detection failed: {str(e)}")
            raise
    
    async def detect_data_drift(self, reference_data: pd.DataFrame,
                              current_data: pd.DataFrame,
                              features: List[str]) -> DataDriftResult:
        """Detect drift in data distributions"""
        
        try:
            feature_drift_results = {}
            
            for feature in features:
                if feature not in reference_data.columns or feature not in current_data.columns:
                    logger.warning(f"Feature {feature} not found in data")
                    continue
                
                ref_values = reference_data[feature].dropna()
                curr_values = current_data[feature].dropna()
                
                # Skip if insufficient data
                if len(ref_values) < self.config.min_samples or len(curr_values) < self.config.min_samples:
                    logger.warning(f"Insufficient data for feature {feature}")
                    continue
                
                # Apply appropriate statistical test based on data type
                if pd.api.types.is_numeric_dtype(ref_values):
                    # Numeric feature - use multiple tests
                    test_results = {
                        'ks_test': await self.statistical_detectors['ks_test'].test(ref_values, curr_values),
                        'wasserstein': await self.statistical_detectors['wasserstein'].test(ref_values, curr_values),
                        'psi': await self.statistical_detectors['psi'].test(ref_values, curr_values)
                    }
                else:
                    # Categorical feature
                    test_results = {
                        'chi_square': await self.statistical_detectors['chi_square'].test(ref_values, curr_values),
                        'psi': await self.statistical_detectors['psi'].test(ref_values, curr_values)
                    }
                
                # Calculate drift magnitude
                drift_magnitude = self.calculate_drift_magnitude(ref_values, curr_values)
                
                # Determine if feature has drifted
                is_drifted = any(result.is_significant for result in test_results.values())
                
                # Calculate drift score
                drift_score = max(result.drift_score for result in test_results.values())
                
                feature_drift_results[feature] = FeatureDriftResult(
                    feature_name=feature,
                    test_results=test_results,
                    drift_magnitude=drift_magnitude,
                    is_drifted=is_drifted,
                    drift_score=drift_score
                )
            
            # Calculate overall data drift score
            overall_data_drift = self.calculate_overall_data_drift(feature_drift_results)
            
            return DataDriftResult(
                feature_results=feature_drift_results,
                overall_drift_score=overall_data_drift,
                drifted_features=[f for f, r in feature_drift_results.items() if r.is_drifted]
            )
            
        except Exception as e:
            logger.error(f"Data drift detection failed: {str(e)}")
            raise
    
    def calculate_drift_magnitude(self, ref_values: pd.Series, curr_values: pd.Series) -> float:
        """Calculate drift magnitude between reference and current data"""
        try:
            if pd.api.types.is_numeric_dtype(ref_values):
                # For numeric data, use Wasserstein distance
                from scipy.stats import wasserstein_distance
                return wasserstein_distance(ref_values, curr_values)
            else:
                # For categorical data, use total variation distance
                ref_probs = ref_values.value_counts(normalize=True)
                curr_probs = curr_values.value_counts(normalize=True)
                
                # Align probabilities
                all_categories = set(ref_probs.index) | set(curr_probs.index)
                ref_probs = ref_probs.reindex(all_categories, fill_value=0)
                curr_probs = curr_probs.reindex(all_categories, fill_value=0)
                
                return 0.5 * np.sum(np.abs(ref_probs - curr_probs))
                
        except Exception as e:
            logger.error(f"Failed to calculate drift magnitude: {str(e)}")
            return 0.0
    
    def calculate_overall_data_drift(self, feature_results: Dict[str, FeatureDriftResult]) -> float:
        """Calculate overall data drift score from feature results"""
        if not feature_results:
            return 0.0
        
        # Weighted average of drift scores
        total_weight = 0
        weighted_score = 0
        
        for feature_result in feature_results.values():
            # Weight by drift magnitude
            weight = feature_result.drift_magnitude + 0.1  # Add small constant to avoid zero weight
            weighted_score += feature_result.drift_score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def calculate_overall_drift_severity(self, drift_results: List[Any]) -> float:
        """Calculate overall drift severity from all drift types"""
        scores = []
        
        for result in drift_results:
            if result is None:
                continue
                
            if hasattr(result, 'overall_drift_score'):
                scores.append(result.overall_drift_score)
            elif hasattr(result, 'drift_score'):
                scores.append(result.drift_score)
        
        if not scores:
            return 0.0
        
        # Use maximum drift score as overall severity
        return max(scores)
    
    async def get_drift_summary(self, drift_result: DriftAnalysisResult) -> Dict[str, Any]:
        """Get a summary of drift detection results"""
        return {
            "overall_severity": drift_result.overall_severity,
            "requires_action": drift_result.requires_action,
            "data_drift_score": drift_result.data_drift.overall_drift_score,
            "drifted_features": drift_result.data_drift.drifted_features,
            "concept_drift_detected": drift_result.concept_drift.drift_detected,
            "prediction_drift_detected": drift_result.prediction_drift.drift_detected,
            "importance_drift_detected": drift_result.importance_drift.drift_detected,
            "analysis_timestamp": drift_result.analysis_timestamp.isoformat()
        }
