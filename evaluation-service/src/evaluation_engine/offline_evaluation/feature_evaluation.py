"""
Feature evaluation utilities for offline evaluation
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    SelectPercentile,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import ElasticNetCV, LassoCV

logger = logging.getLogger(__name__)


class FeatureEvaluator:
    """Feature evaluation and importance analysis"""

    def __init__(self):
        self.feature_importance_methods = {
            "mutual_info": self._mutual_information_importance,
            "f_score": self._f_score_importance,
            "chi2": self._chi2_importance,
            "random_forest": self._random_forest_importance,
            "lasso": self._lasso_importance,
            "elastic_net": self._elastic_net_importance,
        }

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the feature evaluator"""
        pass

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup feature evaluator resources"""
        pass

    async def evaluate_feature_importance(self, model, dataset: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Evaluate feature importance using multiple methods"""

        if dataset is None:
            logger.warning("No dataset provided for feature importance evaluation")
            return {}

        X = dataset.get("features")
        y = dataset.get("labels")

        if X is None or y is None:
            logger.warning("Missing features or labels in dataset")
            return {}

        logger.info(f"Evaluating feature importance for {X.shape[1]} features")

        importance_scores = {}

        # Try different methods based on model type
        if hasattr(model, "feature_importances_"):
            # Tree-based models
            importance_scores["tree_importance"] = dict(zip(range(X.shape[1]), model.feature_importances_))

        if hasattr(model, "coef_"):
            # Linear models
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            importance_scores["linear_coefficients"] = dict(zip(range(X.shape[1]), np.abs(coef)))

        # Statistical methods
        try:
            if len(np.unique(y)) <= 10:  # Classification
                mi_scores = mutual_info_classif(X, y, random_state=42)
                f_scores, _ = f_classif(X, y)
                chi2_scores, _ = chi2(X, y)
            else:  # Regression
                mi_scores = mutual_info_regression(X, y, random_state=42)
                f_scores, _ = f_regression(X, y)
                # Chi2 not applicable for regression
                chi2_scores = np.zeros(X.shape[1])

            importance_scores["mutual_information"] = dict(zip(range(X.shape[1]), mi_scores))
            importance_scores["f_score"] = dict(zip(range(X.shape[1]), f_scores))
            importance_scores["chi2"] = dict(zip(range(X.shape[1]), chi2_scores))
        except Exception as e:
            logger.warning(f"Error calculating statistical importance: {str(e)}")

        # Regularization-based methods
        try:
            lasso_scores = await self._lasso_importance(X, y)
            importance_scores["lasso"] = lasso_scores

            elastic_net_scores = await self._elastic_net_importance(X, y)
            importance_scores["elastic_net"] = elastic_net_scores
        except Exception as e:
            logger.warning(f"Error calculating regularization importance: {str(e)}")

        # Random Forest importance
        try:
            rf_scores = await self._random_forest_importance(X, y)
            importance_scores["random_forest"] = rf_scores
        except Exception as e:
            logger.warning(f"Error calculating Random Forest importance: {str(e)}")

        # Calculate aggregated importance
        aggregated_importance = await self._aggregate_importance_scores(importance_scores)
        importance_scores["aggregated"] = aggregated_importance

        return importance_scores

    async def _mutual_information_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
        """Calculate mutual information importance"""
        if len(np.unique(y)) <= 10:
            scores = mutual_info_classif(X, y, random_state=42)
        else:
            scores = mutual_info_regression(X, y, random_state=42)

        return dict(zip(range(X.shape[1]), scores))

    async def _f_score_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
        """Calculate F-score importance"""
        if len(np.unique(y)) <= 10:
            scores, _ = f_classif(X, y)
        else:
            scores, _ = f_regression(X, y)

        return dict(zip(range(X.shape[1]), scores))

    async def _chi2_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
        """Calculate Chi-squared importance"""
        if len(np.unique(y)) <= 10:
            scores, _ = chi2(X, y)
        else:
            scores = np.zeros(X.shape[1])  # Chi2 not applicable for regression

        return dict(zip(range(X.shape[1]), scores))

    async def _random_forest_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
        """Calculate Random Forest importance"""
        if len(np.unique(y)) <= 10:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)

        rf.fit(X, y)
        return dict(zip(range(X.shape[1]), rf.feature_importances_))

    async def _lasso_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
        """Calculate Lasso importance"""
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X, y)
        return dict(zip(range(X.shape[1]), np.abs(lasso.coef_)))

    async def _elastic_net_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
        """Calculate Elastic Net importance"""
        elastic_net = ElasticNetCV(cv=5, random_state=42)
        elastic_net.fit(X, y)
        return dict(zip(range(X.shape[1]), np.abs(elastic_net.coef_)))

    async def _aggregate_importance_scores(self, importance_scores: Dict[str, Dict[int, float]]) -> Dict[int, float]:
        """Aggregate importance scores from multiple methods"""

        if not importance_scores:
            return {}

        # Get all feature indices
        all_features = set()
        for scores in importance_scores.values():
            all_features.update(scores.keys())

        aggregated = {}

        for feature_idx in all_features:
            scores = []
            for method, method_scores in importance_scores.items():
                if feature_idx in method_scores:
                    scores.append(method_scores[feature_idx])

            if scores:
                # Use mean of available scores
                aggregated[feature_idx] = np.mean(scores)

        return aggregated

    async def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "mutual_info",
        k: int = 10,
        percentile: int = 50,
    ) -> Dict[str, Any]:
        """Select most important features"""
        logger.info(f"Selecting features using {method}")

        if method == "mutual_info":
            if len(np.unique(y)) <= 10:
                selector = SelectKBest(mutual_info_classif, k=k)
            else:
                selector = SelectKBest(mutual_info_regression, k=k)
        elif method == "f_score":
            if len(np.unique(y)) <= 10:
                selector = SelectKBest(f_classif, k=k)
            else:
                selector = SelectKBest(f_regression, k=k)
        elif method == "chi2":
            selector = SelectKBest(chi2, k=k)
        elif method == "percentile":
            if len(np.unique(y)) <= 10:
                selector = SelectPercentile(f_classif, percentile=percentile)
            else:
                selector = SelectPercentile(f_regression, percentile=percentile)
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")

        # Fit selector
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)

        return {
            "selected_features": selected_features.tolist(),
            "X_selected": X_selected,
            "n_selected": len(selected_features),
            "method": method,
        }

    async def analyze_feature_correlations(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze feature correlations"""
        logger.info("Analyzing feature correlations")

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                corr = abs(corr_matrix[i, j])
                if corr > 0.8:  # High correlation threshold
                    high_corr_pairs.append({"feature_1": i, "feature_2": j, "correlation": corr})

        # Calculate average correlation
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_triangle = corr_matrix[mask]
        avg_correlation = np.mean(np.abs(upper_triangle))

        return {
            "correlation_matrix": corr_matrix.tolist(),
            "high_correlation_pairs": high_corr_pairs,
            "average_correlation": avg_correlation,
            "n_features": X.shape[1],
        }

    async def analyze_feature_distributions(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze feature distributions"""
        logger.info("Analyzing feature distributions")

        distributions = {}

        for i in range(X.shape[1]):
            feature = X[:, i]

            # Basic statistics
            stats = {
                "mean": np.mean(feature),
                "std": np.std(feature),
                "min": np.min(feature),
                "max": np.max(feature),
                "median": np.median(feature),
                "skewness": self._calculate_skewness(feature),
                "kurtosis": self._calculate_kurtosis(feature),
            }

            # Check for missing values
            stats["missing_count"] = np.sum(np.isnan(feature))
            stats["missing_percentage"] = stats["missing_count"] / len(feature) * 100

            # Check for outliers (using IQR method)
            Q1 = np.percentile(feature, 25)
            Q3 = np.percentile(feature, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = np.sum((feature < lower_bound) | (feature > upper_bound))
            stats["outlier_count"] = outliers
            stats["outlier_percentage"] = outliers / len(feature) * 100

            distributions[i] = stats

        return distributions

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
