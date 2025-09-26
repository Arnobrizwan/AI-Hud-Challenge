"""Cache functionality for evaluation engine."""

from typing import Any, Dict, Optional


class EvaluationCache:
    """Cache for evaluation results."""
    
    def __init__(self):
        """Initialize the cache."""
        self.cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get a cached value."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any):
        """Set a cached value."""
        self.cache[key] = value
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()


# Global cache instance
evaluation_cache = EvaluationCache()


def init_cache():
    """Initialize the evaluation cache."""
    global evaluation_cache
    evaluation_cache = EvaluationCache()
    return evaluation_cache