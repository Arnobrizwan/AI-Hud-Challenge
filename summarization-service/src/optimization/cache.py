"""Summary cache for optimization."""

from typing import Dict, Optional


class SummaryCache:
    """Cache for summaries."""
    
    def __init__(self):
        """Initialize the cache."""
        self.cache = {}
    
    def get(self, key: str) -> Optional[str]:
        """Get a cached summary."""
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        """Set a cached summary."""
        self.cache[key] = value
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()