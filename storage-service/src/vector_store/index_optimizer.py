"""
Vector Index Optimizer - Optimize vector indexes for performance
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from models import IndexBuildResult, VectorIndexConfig

logger = logging.getLogger(__name__)


class VectorIndexOptimizer:
    """Optimize vector indexes for maximum performance"""

    def __init__(self):
        self._initialized = False
        self._index_performance: Dict[str, Dict[str, Any]] = {}
        self._optimization_history: List[Dict[str, Any]] = []

    async def initialize(self) -> Dict[str, Any]:
        """Initialize index optimizer"""
        self._initialized = True
        logger.info("Vector Index Optimizer initialized")

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup resources"""
        self._initialized = False
        logger.info("Vector Index Optimizer cleanup complete")

    async def optimize_index_parameters(
        self, index_name: str, current_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Optimize index parameters based on current performance"""
        if not self._initialized:
            return {}

        try:
            logger.info(f"Optimizing parameters for index {index_name}")

            # Analyze current performance
            avg_query_time = current_performance.get("avg_query_time", 0.0)
            index_size = current_performance.get("index_size", 0)
            query_count = current_performance.get("query_count", 0)

            # Determine optimization strategy
            if avg_query_time > 100:  # Slow queries
                optimization = self._optimize_for_speed()
            elif index_size > 1000000000:  # Large index (>1GB)
                optimization = self._optimize_for_size()
            elif query_count > 10000:  # High query volume
                optimization = self._optimize_for_throughput()
            else:
                optimization = self._optimize_balanced()

            # Record optimization
            self._record_optimization(
                index_name, current_performance, optimization)

            return optimization

        except Exception as e:
            logger.error(f"Failed to optimize index parameters: {e}")
            return {}

    def _optimize_for_speed(self) -> Dict[str, Any]:
        """Optimize for query speed"""
        return {
            "hnsw_m": 32,  # Higher connectivity for better recall
            "ef_construction": 400,  # Higher construction effort
            "ef_search": 200,  # Higher search effort
            "strategy": "speed_optimized",
            "description": "Optimized for maximum query speed",
        }

    def _optimize_for_size(self) -> Dict[str, Any]:
        """Optimize for index size"""
        return {
            "hnsw_m": 8,  # Lower connectivity to reduce size
            "ef_construction": 100,  # Lower construction effort
            "ef_search": 50,  # Lower search effort
            "strategy": "size_optimized",
            "description": "Optimized for minimum index size",
        }

    def _optimize_for_throughput(self) -> Dict[str, Any]:
        """Optimize for high query throughput"""
        return {
            "hnsw_m": 16,  # Balanced connectivity
            "ef_construction": 200,  # Balanced construction
            "ef_search": 100,  # Balanced search
            "strategy": "throughput_optimized",
            "description": "Optimized for high query throughput",
        }

    def _optimize_balanced(self) -> Dict[str, Any]:
        """Balanced optimization"""
        return {
            "hnsw_m": 16,
            "ef_construction": 200,
            "ef_search": 100,
            "strategy": "balanced",
            "description": "Balanced optimization for speed and size",
        }

    def _record_optimization(self,
                             index_name: str,
                             current_performance: Dict[str,
                                                       Any],
                             optimization: Dict[str,
                                                Any]):
        """Record optimization history"""
        self._optimization_history.append(
            {
                "index_name": index_name,
                "timestamp": datetime.utcnow().isoformat(),
                "current_performance": current_performance,
                "optimization": optimization,
            }
        )

        # Keep only last 100 optimizations
        if len(self._optimization_history) > 100:
            self._optimization_history = self._optimization_history[-100:]

    async def analyze_index_performance(
        self, index_name: str, query_times: List[float]
    ) -> Dict[str, Any]:
    """Analyze index performance metrics"""
        if not self._initialized:
            return {}

        try:
            if not query_times:
                return {
                    "avg_query_time": 0.0,
                    "p95_query_time": 0.0,
                    "p99_query_time": 0.0}

            # Calculate performance metrics
            avg_query_time = statistics.mean(query_times)
            p95_query_time = self._percentile(query_times, 95)
            p99_query_time = self._percentile(query_times, 99)
            min_query_time = min(query_times)
            max_query_time = max(query_times)

            # Determine performance level
            if avg_query_time < 10:
                performance_level = "excellent"
            elif avg_query_time < 50:
                performance_level = "good"
            elif avg_query_time < 100:
                performance_level = "acceptable"
            else:
                performance_level = "poor"

            performance_metrics = {
                "avg_query_time": avg_query_time,
                "p95_query_time": p95_query_time,
                "p99_query_time": p99_query_time,
                "min_query_time": min_query_time,
                "max_query_time": max_query_time,
                "query_count": len(query_times),
                "performance_level": performance_level,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Store performance data
            self._index_performance[index_name] = performance_metrics

            return performance_metrics

        except Exception as e:
            logger.error(f"Failed to analyze index performance: {e}")
            return {}

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)

        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    async def recommend_index_strategy(
        self, data_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Recommend index strategy based on data characteristics"""
        if not self._initialized:
            return {}

        try:
            vector_count = data_characteristics.get("vector_count", 0)
            vector_dimension = data_characteristics.get(
                "vector_dimension", 768)
            query_pattern = data_characteristics.get("query_pattern", "mixed")
            update_frequency = data_characteristics.get(
                "update_frequency", "low")

            # Determine optimal strategy
            if vector_count < 10000:
                strategy = "exact_search"
                config = {
                    "use_hnsw": False,
                    "use_ivfflat": True,
                    "ivfflat_lists": max(1, vector_count // 1000),
                }
            elif vector_count < 100000:
                strategy = "small_hnsw"
                config = {
                    "use_hnsw": True,
                    "hnsw_m": 8,
                    "ef_construction": 100,
                    "ef_search": 50}
            elif vector_count < 1000000:
                strategy = "medium_hnsw"
                config = {
                    "use_hnsw": True,
                    "hnsw_m": 16,
                    "ef_construction": 200,
                    "ef_search": 100}
            else:
                strategy = "large_hnsw"
                config = {
                    "use_hnsw": True,
                    "hnsw_m": 32,
                    "ef_construction": 400,
                    "ef_search": 200}

            # Adjust for query pattern
            if query_pattern == "exact":
                config["ef_search"] = config.get("ef_search", 100) * 2
            elif query_pattern == "approximate":
                config["ef_search"] = max(
                    50, config.get("ef_search", 100) // 2)

            # Adjust for update frequency
            if update_frequency == "high":
                config["ef_construction"] = max(
                    100, config.get("ef_construction", 200) // 2)

            return {
                "strategy": strategy,
                "config": config,
                "description": f"Recommended {strategy} for {vector_count} vectors",
                "estimated_performance": self._estimate_performance(
                    strategy,
                    vector_count),
            }

        except Exception as e:
            logger.error(f"Failed to recommend index strategy: {e}")
            return {}

    def _estimate_performance(
            self, strategy: str, vector_count: int) -> Dict[str, float]:
        """Estimate performance for a strategy"""
        base_times = {
            "exact_search": 0.1,
            "small_hnsw": 0.05,
            "medium_hnsw": 0.1,
            "large_hnsw": 0.2,
        }

        base_time = base_times.get(strategy, 0.1)

        # Scale with vector count (logarithmic)
        import math

        scale_factor = 1 + math.log10(max(1, vector_count / 1000))

        return {
            "estimated_query_time_ms": base_time * scale_factor * 1000,
            "estimated_index_size_mb": vector_count * 0.003,  # Rough estimate
            "estimated_build_time_minutes": vector_count * 0.0001,
        }

    async def get_optimization_history(
        self, index_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get optimization history"""
        if not self._initialized:
            return []

        if index_name:
            return [
                opt for opt in self._optimization_history if opt["index_name"] == index_name]
        else:
            return self._optimization_history

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all indexes"""
        if not self._initialized or not self._index_performance:
            return {}

        try:
            all_query_times = []
            performance_levels = {}

            for index_name, metrics in self._index_performance.items():
                all_query_times.extend(
                    [metrics["avg_query_time"]] * metrics.get("query_count", 1))
                performance_levels[index_name] = metrics.get(
                    "performance_level", "unknown")

            if not all_query_times:
                return {}

            return {
                "total_indexes": len(self._index_performance),
                "overall_avg_query_time": statistics.mean(all_query_times),
                "overall_p95_query_time": self._percentile(all_query_times, 95),
                "performance_distribution": {
                    "excellent": sum(
                        1 for level in performance_levels.values() if level == "excellent"
                    ),
                    "good": sum(1 for level in performance_levels.values() if level == "good"),
                    "acceptable": sum(
                        1 for level in performance_levels.values() if level == "acceptable"
                    ),
                    "poor": sum(1 for level in performance_levels.values() if level == "poor"),
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}
