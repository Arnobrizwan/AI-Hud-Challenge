"""Monitoring for evaluation engine."""

from typing import Any, Dict


class EvaluationMonitoring:
    """Monitoring for evaluation service."""
    
    def __init__(self):
        """Initialize monitoring."""
        pass
    
    def record_evaluation(self, model_id: str, metrics: Dict[str, Any]):
        """Record evaluation metrics."""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        return {"evaluations": 0}
