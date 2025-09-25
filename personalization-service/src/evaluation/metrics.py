"""Comprehensive evaluation metrics for personalization."""

import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from sklearn.metrics import f1_score, ndcg_score, precision_score, recall_score

from ..database.postgres_client import PostgreSQLClient
from ..database.redis_client import RedisClient
from ..models.schemas import Recommendation, UserInteraction, UserProfile

logger = structlog.get_logger()


class PersonalizationMetrics:
    """Comprehensive evaluation metrics for personalization systems."""

    def __init__(self, redis_client: RedisClient, postgres_client: PostgreSQLClient):
        self.redis = redis_client
        self.postgres = postgres_client

    async def calculate_offline_metrics(
        self, user_id: str, recommendations: List[Recommendation], ground_truth: List[str]
    ) -> Dict[str, float]:
        """Calculate offline evaluation metrics."""
        if not recommendations or not ground_truth:
            return {}

        # Convert to binary relevance
        recommended_items = [rec.item_id for rec in recommendations]
        relevant_items = set(ground_truth)

        # Binary relevance for each recommendation
        y_true = [1 if item in relevant_items else 0 for item in recommended_items]
        # All recommendations are predicted as relevant
        y_pred = [1] * len(recommended_items)

        # Basic metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Ranking metrics
        ndcg = self._calculate_ndcg(recommendations, relevant_items)
        map_score = self._calculate_map(recommendations, relevant_items)

        # Coverage metrics
        coverage = self._calculate_coverage(recommendations, ground_truth)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "ndcg": ndcg,
            "map": map_score,
            "coverage": coverage,
        }

    async def calculate_online_metrics(self, user_id: str, time_window: int = 7) -> Dict[str, float]:
        """Calculate online evaluation metrics."""
        # Get user interactions in time window
        interactions = await self._get_user_interactions(user_id, time_window)

        if not interactions:
            return {}

        # Calculate engagement metrics
        click_rate = self._calculate_click_rate(interactions)
        conversion_rate = self._calculate_conversion_rate(interactions)
        dwell_time = self._calculate_avg_dwell_time(interactions)

        # Calculate diversity metrics
        topic_diversity = self._calculate_topic_diversity(interactions)
        source_diversity = self._calculate_source_diversity(interactions)

        # Calculate novelty metrics
        novelty = self._calculate_novelty(interactions)

        return {
            "click_rate": click_rate,
            "conversion_rate": conversion_rate,
            "avg_dwell_time": dwell_time,
            "topic_diversity": topic_diversity,
            "source_diversity": source_diversity,
            "novelty": novelty,
        }

    async def calculate_business_metrics(self, time_window: int = 30) -> Dict[str, float]:
        """Calculate business impact metrics."""
        # Get interactions in time window
        interactions = await self._get_interactions_in_window(time_window)

        if not interactions:
            return {}

        # Calculate engagement metrics
        total_interactions = len(interactions)
        unique_users = len(set(interaction["user_id"] for interaction in interactions))

        # Calculate content performance
        content_performance = self._calculate_content_performance(interactions)

        # Calculate user retention
        retention_rate = await self._calculate_retention_rate(time_window)

        # Calculate revenue metrics (if available)
        revenue_metrics = await self._calculate_revenue_metrics(interactions)

        return {
            "total_interactions": total_interactions,
            "unique_users": unique_users,
            "avg_interactions_per_user": (total_interactions / unique_users if unique_users > 0 else 0),
            "content_performance": content_performance,
            "retention_rate": retention_rate,
            **revenue_metrics,
        }

    def _calculate_ndcg(self, recommendations: List[Recommendation], relevant_items: set, k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not recommendations:
            return 0.0

        # Get top-k recommendations
        top_k = recommendations[:k]

        # Create relevance scores
        relevance_scores = []
        for rec in top_k:
            if rec.item_id in relevant_items:
                relevance_scores.append(1.0)
            else:
                relevance_scores.append(0.0)

        # Calculate NDCG
        if not relevance_scores:
            return 0.0

        # Ideal DCG (all relevant items at the top)
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevance_scores), len(relevant_items))))

        # Actual DCG
        actual_dcg = sum(score / np.log2(i + 2) for i, score in enumerate(relevance_scores))

        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def _calculate_map(self, recommendations: List[Recommendation], relevant_items: set) -> float:
        """Calculate Mean Average Precision."""
        if not recommendations:
            return 0.0

        # Calculate precision at each position
        precisions = []
        relevant_count = 0

        for i, rec in enumerate(recommendations):
            if rec.item_id in relevant_items:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                precisions.append(precision)

        # Calculate MAP
        if not precisions:
            return 0.0

        return sum(precisions) / len(relevant_items) if relevant_items else 0.0

    def _calculate_coverage(self, recommendations: List[Recommendation], ground_truth: List[str]) -> float:
        """Calculate coverage of recommendations."""
        if not recommendations or not ground_truth:
            return 0.0

        recommended_items = set(rec.item_id for rec in recommendations)
        ground_truth_items = set(ground_truth)

        # Coverage = intersection / ground truth
        intersection = len(recommended_items.intersection(ground_truth_items))
        return intersection / len(ground_truth_items) if ground_truth_items else 0.0

    def _calculate_click_rate(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate click rate from interactions."""
        if not interactions:
            return 0.0

        clicks = sum(1 for interaction in interactions if interaction.get("interaction_type") == "click")

        return clicks / len(interactions)

    def _calculate_conversion_rate(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate conversion rate from interactions."""
        if not interactions:
            return 0.0

        conversions = sum(
            1 for interaction in interactions if interaction.get("interaction_type") in ["share", "save", "like"]
        )

        return conversions / len(interactions)

    def _calculate_avg_dwell_time(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate average dwell time from interactions."""
        if not interactions:
            return 0.0

        # This would require session data with timestamps
        # For now, return a placeholder
        return 0.0

    def _calculate_topic_diversity(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate topic diversity from interactions."""
        if not interactions:
            return 0.0

        # Get all topics from interactions
        all_topics = []
        for interaction in interactions:
            topics = interaction.get("context", {}).get("topics", [])
            all_topics.extend(topics)

        if not all_topics:
            return 0.0

        # Calculate diversity as unique topics / total topics
        unique_topics = len(set(all_topics))
        total_topics = len(all_topics)

        return unique_topics / total_topics if total_topics > 0 else 0.0

    def _calculate_source_diversity(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate source diversity from interactions."""
        if not interactions:
            return 0.0

        # Get all sources from interactions
        sources = [
            interaction.get("context", {}).get("source")
            for interaction in interactions
            if interaction.get("context", {}).get("source")
        ]

        if not sources:
            return 0.0

        # Calculate diversity as unique sources / total sources
        unique_sources = len(set(sources))
        total_sources = len(sources)

        return unique_sources / total_sources if total_sources > 0 else 0.0

    def _calculate_novelty(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate novelty from interactions."""
        if not interactions:
            return 0.0

        # This would require comparing with user's historical preferences
        # For now, return a placeholder
        return 0.0

    def _calculate_content_performance(self, interactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate content performance metrics."""
        if not interactions:
            return {}

        # Group by content
        content_stats = defaultdict(list)
        for interaction in interactions:
            item_id = interaction.get("item_id")
            if item_id:
                content_stats[item_id].append(interaction)

        # Calculate metrics per content
        content_metrics = {}
        for item_id, item_interactions in content_stats.items():
            content_metrics[item_id] = {
                "interaction_count": len(item_interactions),
                "click_rate": self._calculate_click_rate(item_interactions),
                "conversion_rate": self._calculate_conversion_rate(item_interactions),
            }

        return content_metrics

    async def _get_user_interactions(self, user_id: str, time_window: int) -> List[Dict[str, Any]]:
        """Get user interactions in time window."""
        query = """
        SELECT * FROM user_interactions
        WHERE user_id = $1 AND timestamp > $2
        ORDER BY timestamp DESC
        """

        cutoff_date = datetime.utcnow() - timedelta(days=time_window)
        return await self.postgres.fetch_all(query, user_id, cutoff_date)

    async def _get_interactions_in_window(self, time_window: int) -> List[Dict[str, Any]]:
        """Get all interactions in time window."""
        query = """
        SELECT * FROM user_interactions
        WHERE timestamp > $1
        ORDER BY timestamp DESC
        """

        cutoff_date = datetime.utcnow() - timedelta(days=time_window)
        return await self.postgres.fetch_all(query, cutoff_date)

    async def _calculate_retention_rate(self, time_window: int) -> float:
        """Calculate user retention rate."""
        # Get users who had interactions in the first half of the window
        half_window = time_window // 2
        start_date = datetime.utcnow() - timedelta(days=time_window)
        mid_date = datetime.utcnow() - timedelta(days=half_window)

        # Users active in first half
        first_half_users = await self.postgres.fetch_all(
            """
            SELECT DISTINCT user_id FROM user_interactions
            WHERE timestamp BETWEEN $1 AND $2
            """,
            start_date,
            mid_date,
        )

        if not first_half_users:
            return 0.0

        # Users active in second half
        second_half_users = await self.postgres.fetch_all(
            """
            SELECT DISTINCT user_id FROM user_interactions
            WHERE timestamp > $1
            """,
            mid_date,
        )

        # Calculate retention
        first_half_user_ids = set(user["user_id"] for user in first_half_users)
        second_half_user_ids = set(user["user_id"] for user in second_half_users)

        retained_users = len(first_half_user_ids.intersection(second_half_user_ids))
        total_users = len(first_half_user_ids)

        return retained_users / total_users if total_users > 0 else 0.0

    async def _calculate_revenue_metrics(self, interactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate revenue-related metrics."""
        # This would require revenue data
        # For now, return placeholder metrics
        return {"revenue_per_user": 0.0, "revenue_per_interaction": 0.0, "total_revenue": 0.0}

    async def calculate_model_performance(self, model_name: str) -> Dict[str, float]:
        """Calculate performance metrics for a specific model."""
        # Get model predictions and actual outcomes
        predictions = await self._get_model_predictions(model_name)
        actual_outcomes = await self._get_actual_outcomes(model_name)

        if not predictions or not actual_outcomes:
            return {}

        # Calculate accuracy metrics
        accuracy = self._calculate_accuracy(predictions, actual_outcomes)
        precision = self._calculate_precision(predictions, actual_outcomes)
        recall = self._calculate_recall(predictions, actual_outcomes)
        f1 = self._calculate_f1_score(predictions, actual_outcomes)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

    async def _get_model_predictions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get model predictions from database."""
        query = """
        SELECT * FROM recommendation_logs
        WHERE algorithm_used = $1
        ORDER BY timestamp DESC
        LIMIT 1000
        """

        return await self.postgres.fetch_all(query, model_name)

    async def _get_actual_outcomes(self, model_name: str) -> List[Dict[str, Any]]:
        """Get actual outcomes for model predictions."""
        # This would require tracking actual user behavior after recommendations
        # For now, return empty list
        return []

    def _calculate_accuracy(self, predictions: List[Dict[str, Any]], actual_outcomes: List[Dict[str, Any]]) -> float:
        """Calculate accuracy metric."""
        # This would require matching predictions with outcomes
        # For now, return placeholder
        return 0.0

    def _calculate_precision(self, predictions: List[Dict[str, Any]], actual_outcomes: List[Dict[str, Any]]) -> float:
        """Calculate precision metric."""
        # This would require matching predictions with outcomes
        # For now, return placeholder
        return 0.0

    def _calculate_recall(self, predictions: List[Dict[str, Any]], actual_outcomes: List[Dict[str, Any]]) -> float:
        """Calculate recall metric."""
        # This would require matching predictions with outcomes
        # For now, return placeholder
        return 0.0

    def _calculate_f1_score(self, predictions: List[Dict[str, Any]], actual_outcomes: List[Dict[str, Any]]) -> float:
        """Calculate F1 score metric."""
        # This would require matching predictions with outcomes
        # For now, return placeholder
        return 0.0

    async def get_comprehensive_metrics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Get comprehensive evaluation metrics."""
        metrics = {}

        if user_id:
            # User-specific metrics
            offline_metrics = await self.calculate_offline_metrics(user_id, [], [])
            online_metrics = await self.calculate_online_metrics(user_id)
            metrics["user_metrics"] = {**offline_metrics, **online_metrics}

        # Business metrics
        business_metrics = await self.calculate_business_metrics()
        metrics["business_metrics"] = business_metrics

        # Model performance
        model_metrics = {}
        for model in ["collaborative", "content_based", "bandit", "hybrid"]:
            model_metrics[model] = await self.calculate_model_performance(model)
        metrics["model_metrics"] = model_metrics

        return metrics
