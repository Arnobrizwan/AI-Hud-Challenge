"""Metrics collection for summarization service."""

from typing import Dict, Optional


class MetricsCollector:
    """Collects metrics for the summarization service."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics = {}
    
    def record_summarization_request(self, content_length: int, summary_length: int):
        """Record a summarization request."""
        pass
    
    def record_processing_time(self, processing_time: float):
        """Record processing time."""
        pass
    
    def get_metrics(self) -> Dict[str, any]:
        """Get current metrics."""
        return self.metrics