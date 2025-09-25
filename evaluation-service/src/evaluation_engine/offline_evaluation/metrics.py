"""
Comprehensive metrics calculators for different model types
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)

from ..models import (
    ClassificationMetrics,
    ClusteringMetrics,
    RankingMetrics,
    RecommendationMetrics,
    RegressionMetrics,
)

logger = logging.getLogger(__name__)


class RankingMetricsCalculator:
    """Specialized metrics for ranking models"""

    async def initialize(self) -> Dict[str, Any]:
                        """Initialize the calculator"""
        pass

    async def calculate_metrics(
        self, predictions: np.ndarray, ground_truth: np.ndarray, additional_data: Dict[str, Any]
    ) -> RankingMetrics:
        """Calculate comprehensive ranking metrics"""

        logger.info("Calculating ranking metrics")

        metrics = {}

        # Precision@K for various K values
        for k in [1, 3, 5, 10, 20]:
            metrics[f"precision_at_{k}"] = self.precision_at_k(predictions, ground_truth, k)
            metrics[f"recall_at_{k}"] = self.recall_at_k(predictions, ground_truth, k)

        # NDCG@K for various K values
        for k in [5, 10, 20]:
            metrics[f"ndcg_at_{k}"] = self.ndcg_at_k(predictions, ground_truth, k)

        # Mean Reciprocal Rank
        metrics["mrr"] = self.mean_reciprocal_rank(predictions, ground_truth)

        # Mean Average Precision
        metrics["map"] = self.mean_average_precision(predictions, ground_truth)

        # Diversity metrics
        if "item_features" in additional_data:
            metrics["intra_list_diversity"] = self.calculate_diversity(predictions, additional_data["item_features"])

        # Coverage metrics
        if "catalog_size" in additional_data:
            metrics["catalog_coverage"] = self.calculate_catalog_coverage(predictions, additional_data["catalog_size"])

        # Novelty metrics
        if "popularity_scores" in additional_data:
            metrics["novelty"] = self.calculate_novelty(predictions, additional_data["popularity_scores"])

        # Spearman correlation
        metrics["spearman_correlation"] = self.spearman_correlation(predictions, ground_truth)

        # Kendall tau correlation
        metrics["kendall_tau"] = self.kendall_tau(predictions, ground_truth)

        return RankingMetrics(**metrics)

    def precision_at_k(self, predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
        """Calculate Precision@K"""
        if len(predictions) == 0:
            return 0.0

        # Get top-k predictions
        top_k_indices = np.argsort(predictions)[-k:]

        # Calculate precision
        relevant_items = np.sum(ground_truth[top_k_indices] > 0)
        return relevant_items / k

    def recall_at_k(self, predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
        """Calculate Recall@K"""
        if len(predictions) == 0:
            return 0.0

        # Get top-k predictions
        top_k_indices = np.argsort(predictions)[-k:]

        # Calculate recall
        relevant_items = np.sum(ground_truth[top_k_indices] > 0)
        total_relevant = np.sum(ground_truth > 0)

        return relevant_items / total_relevant if total_relevant > 0 else 0.0

    def ndcg_at_k(self, predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
        """Calculate NDCG@K"""
        if len(predictions) == 0:
            return 0.0

        def dcg_at_k(scores, k):
            scores = scores[:k]
            return np.sum([score / np.log2(i + 2) for i, score in enumerate(scores)])

        # Calculate DCG@K for predictions
        sorted_indices = np.argsort(predictions)[::-1]
        dcg = dcg_at_k(ground_truth[sorted_indices], k)

        # Calculate IDCG@K (perfect ranking)
        ideal_scores = np.sort(ground_truth)[::-1]
        idcg = dcg_at_k(ideal_scores, k)

        return dcg / idcg if idcg > 0 else 0.0

    def mean_reciprocal_rank(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate Mean Reciprocal Rank"""
        if len(predictions) == 0:
            return 0.0

        # Find rank of first relevant item
        sorted_indices = np.argsort(predictions)[::-1]
        for i, idx in enumerate(sorted_indices):
            if ground_truth[idx] > 0:
                return 1.0 / (i + 1)

        return 0.0

    def mean_average_precision(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate Mean Average Precision"""
        if len(predictions) == 0:
            return 0.0

        # Sort by predictions
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_ground_truth = ground_truth[sorted_indices]

        # Calculate average precision
        relevant_indices = np.where(sorted_ground_truth > 0)[0]
        if len(relevant_indices) == 0:
            return 0.0

        precision_sum = 0.0
        for i, idx in enumerate(relevant_indices):
            precision_at_i = (i + 1) / (idx + 1)
            precision_sum += precision_at_i

        return precision_sum / len(relevant_indices)

    def calculate_diversity(self, predictions: np.ndarray, item_features: np.ndarray) -> float:
        """Calculate intra-list diversity"""
        if len(predictions) == 0 or item_features is None:
            return 0.0

        # Get top-k items
        top_k = min(10, len(predictions))
        top_k_indices = np.argsort(predictions)[-top_k:]

        if len(top_k_indices) < 2:
            return 0.0

        # Calculate pairwise distances
        distances = []
        for i in range(len(top_k_indices)):
            for j in range(i + 1, len(top_k_indices)):
                dist = np.linalg.norm(item_features[top_k_indices[i]] - item_features[top_k_indices[j]])
                distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def calculate_catalog_coverage(self, predictions: np.ndarray, catalog_size: int) -> float:
        """Calculate catalog coverage"""
        if len(predictions) == 0:
            return 0.0

        # Get unique items in top-k predictions
        top_k = min(100, len(predictions))
        top_k_indices = np.argsort(predictions)[-top_k:]
        unique_items = len(np.unique(top_k_indices))

        return unique_items / catalog_size

    def calculate_novelty(self, predictions: np.ndarray, popularity_scores: np.ndarray) -> float:
        """Calculate novelty (inverse of popularity)"""
        if len(predictions) == 0 or popularity_scores is None:
            return 0.0

        # Get top-k items
        top_k = min(10, len(predictions))
        top_k_indices = np.argsort(predictions)[-top_k:]

        # Calculate average novelty (inverse of popularity)
        novelty_scores = 1.0 - popularity_scores[top_k_indices]
        return np.mean(novelty_scores)

    def spearman_correlation(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate Spearman correlation"""
        if len(predictions) == 0 or len(ground_truth) == 0:
            return 0.0

        correlation, _ = spearmanr(predictions, ground_truth)
        return correlation if not np.isnan(correlation) else 0.0

    def kendall_tau(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate Kendall tau correlation"""
        if len(predictions) == 0 or len(ground_truth) == 0:
            return 0.0

        correlation, _ = kendalltau(predictions, ground_truth)
        return correlation if not np.isnan(correlation) else 0.0


class ClassificationMetricsCalculator:
    """Specialized metrics for classification models"""

    async def initialize(self) -> Dict[str, Any]:
                        """Initialize the calculator"""
        pass

    async def calculate_metrics(
        self, predictions: np.ndarray, ground_truth: np.ndarray, additional_data: Dict[str, Any]
    ) -> ClassificationMetrics:
        """Calculate comprehensive classification metrics"""

        logger.info("Calculating classification metrics")

        # Convert predictions to binary if needed
        if predictions.ndim > 1:
            y_pred = np.argmax(predictions, axis=1)
            y_pred_proba = predictions[:, 1] if predictions.shape[1] == 2 else predictions
        else:
            y_pred = (predictions > 0.5).astype(int)
            y_pred_proba = predictions

        # Basic metrics
        accuracy = accuracy_score(ground_truth, y_pred)
        precision = precision_score(ground_truth, y_pred, average="binary", zero_division=0)
        recall = recall_score(ground_truth, y_pred, average="binary", zero_division=0)
        f1 = f1_score(ground_truth, y_pred, average="binary", zero_division=0)

        # AUC metrics
        try:
            auc_roc = roc_auc_score(ground_truth, y_pred_proba)
        except ValueError:
            auc_roc = 0.0

        try:
            auc_pr = average_precision_score(ground_truth, y_pred_proba)
        except ValueError:
            auc_pr = 0.0

        # Confusion matrix
        cm = confusion_matrix(ground_truth, y_pred)
        confusion_matrix_list = cm.tolist()

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            confusion_matrix=confusion_matrix_list,
        )


class RegressionMetricsCalculator:
    """Specialized metrics for regression models"""

    async def initialize(self) -> Dict[str, Any]:
                        """Initialize the calculator"""
        pass

    async def calculate_metrics(
        self, predictions: np.ndarray, ground_truth: np.ndarray, additional_data: Dict[str, Any]
    ) -> RegressionMetrics:
        """Calculate comprehensive regression metrics"""

        logger.info("Calculating regression metrics")

        # Mean Squared Error
        mse = mean_squared_error(ground_truth, predictions)

        # Root Mean Squared Error
        rmse = np.sqrt(mse)

        # Mean Absolute Error
        mae = mean_absolute_error(ground_truth, predictions)

        # R² Score
        r2 = r2_score(ground_truth, predictions)

        # Mean Absolute Percentage Error
        mape = self.mean_absolute_percentage_error(ground_truth, predictions)

        return RegressionMetrics(mse=mse, rmse=rmse, mae=mae, r2_score=r2, mape=mape)

    def mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        if len(y_true) == 0:
            return 0.0

        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return 0.0

        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape


class RecommendationMetricsCalculator:
    """Specialized metrics for recommendation models"""

    async def initialize(self) -> Dict[str, Any]:
                        """Initialize the calculator"""
        pass

    async def calculate_metrics(
        self, predictions: np.ndarray, ground_truth: np.ndarray, additional_data: Dict[str, Any]
    ) -> RecommendationMetrics:
        """Calculate comprehensive recommendation metrics"""

        logger.info("Calculating recommendation metrics")

        # Hit Rate
        hit_rate = self.calculate_hit_rate(predictions, ground_truth)

        # Coverage
        coverage = self.calculate_coverage(predictions, additional_data)

        # Diversity
        diversity = self.calculate_diversity(predictions, additional_data)

        # Novelty
        novelty = self.calculate_novelty(predictions, additional_data)

        # Serendipity
        serendipity = self.calculate_serendipity(predictions, ground_truth, additional_data)

        return RecommendationMetrics(
            hit_rate=hit_rate,
            coverage=coverage,
            diversity=diversity,
            novelty=novelty,
            serendipity=serendipity,
        )

    def calculate_hit_rate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate hit rate"""
        if len(predictions) == 0:
            return 0.0

        # For each user, check if any recommended item is relevant
        hits = 0
        for i in range(len(predictions)):
            # Get top-k recommendations
            top_k = min(10, len(predictions[i]))
            top_k_indices = np.argsort(predictions[i])[-top_k:]

            # Check if any recommended item is relevant
            if np.any(ground_truth[top_k_indices] > 0):
                hits += 1

        return hits / len(predictions)

    def calculate_coverage(self, predictions: np.ndarray, additional_data: Dict[str, Any]) -> float:
        """Calculate catalog coverage"""
        if len(predictions) == 0:
            return 0.0

        # Get all unique items recommended
        all_recommended_items = set()
        for user_predictions in predictions:
            top_k = min(10, len(user_predictions))
            top_k_indices = np.argsort(user_predictions)[-top_k:]
            all_recommended_items.update(top_k_indices)

        # Calculate coverage
        catalog_size = additional_data.get("catalog_size", len(predictions[0]))
        return len(all_recommended_items) / catalog_size

    def calculate_diversity(self, predictions: np.ndarray, additional_data: Dict[str, Any]) -> float:
        """Calculate diversity of recommendations"""
        if len(predictions) == 0:
            return 0.0

        # Calculate pairwise diversity across all users
        diversity_scores = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                # Get top-k for both users
                top_k = min(10, len(predictions[i]))
                top_k_i = set(np.argsort(predictions[i])[-top_k:])
                top_k_j = set(np.argsort(predictions[j])[-top_k:])

                # Calculate Jaccard diversity
                intersection = len(top_k_i.intersection(top_k_j))
                union = len(top_k_i.union(top_k_j))
                diversity = 1 - (intersection / union) if union > 0 else 0
                diversity_scores.append(diversity)

        return np.mean(diversity_scores) if diversity_scores else 0.0

    def calculate_novelty(self, predictions: np.ndarray, additional_data: Dict[str, Any]) -> float:
        """Calculate novelty of recommendations"""
        if len(predictions) == 0:
            return 0.0

        popularity_scores = additional_data.get("popularity_scores")
        if popularity_scores is None:
            return 0.0

        novelty_scores = []
        for user_predictions in predictions:
            top_k = min(10, len(user_predictions))
            top_k_indices = np.argsort(user_predictions)[-top_k:]

            # Calculate average novelty (inverse of popularity)
            user_novelty = np.mean(1.0 - popularity_scores[top_k_indices])
            novelty_scores.append(user_novelty)

        return np.mean(novelty_scores)

    def calculate_serendipity(
        self, predictions: np.ndarray, ground_truth: np.ndarray, additional_data: Dict[str, Any]
    ) -> float:
        """Calculate serendipity of recommendations"""
        if len(predictions) == 0:
            return 0.0

        # Serendipity measures how surprising but relevant recommendations are
        # This is a simplified implementation
        serendipity_scores = []

        for i in range(len(predictions)):
            top_k = min(10, len(predictions[i]))
            top_k_indices = np.argsort(predictions[i])[-top_k:]

            # Check relevance and surprise
            relevant_items = ground_truth[top_k_indices] > 0
            if np.any(relevant_items):
                # Simple serendipity: relevant items that are not too popular
                popularity_scores = additional_data.get("popularity_scores", np.ones(len(predictions[i])))
                surprise_scores = 1.0 - popularity_scores[top_k_indices]
                serendipity = np.mean(surprise_scores[relevant_items])
                serendipity_scores.append(serendipity)

        return np.mean(serendipity_scores) if serendipity_scores else 0.0


class ClusteringMetricsCalculator:
    """Specialized metrics for clustering models"""

    async def initialize(self) -> Dict[str, Any]:
                    """Initialize the calculator"""
        pass

    async def calculate_metrics(
        self, predictions: np.ndarray, ground_truth: np.ndarray, additional_data: Dict[str, Any]
    ) -> ClusteringMetrics:
        """Calculate comprehensive clustering metrics"""

        logger.info("Calculating clustering metrics")

        # Get features for clustering metrics
        features = additional_data.get("features")
        if features is None:
            # Use predictions as features if no features provided
            features = predictions.reshape(-1, 1)

        # Silhouette Score
        try:
            silhouette = silhouette_score(features, predictions)
        except ValueError:
            silhouette = 0.0

        # Calinski-Harabasz Score
        try:
            calinski_harabasz = calinski_harabasz_score(features, predictions)
        except ValueError:
            calinski_harabasz = 0.0

        # Davies-Bouldin Score
        try:
            davies_bouldin = davies_bouldin_score(features, predictions)
        except ValueError:
            davies_bouldin = 0.0

        # Inertia (within-cluster sum of squares)
        inertia = self.calculate_inertia(features, predictions)

        return ClusteringMetrics(
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            davies_bouldin_score=davies_bouldin,
            inertia=inertia,
        )

    def calculate_inertia(self, features: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares (inertia)"""
        if len(features) == 0:
            return 0.0

        inertia = 0.0
        unique_clusters = np.unique(cluster_labels)

        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            cluster_points = features[cluster_mask]

            if len(cluster_points) > 0:
                cluster_center = np.mean(cluster_points, axis=0)
                cluster_inertia = np.sum((cluster_points - cluster_center) ** 2)
                inertia += cluster_inertia

        return inertia
