"""Dependencies for evaluation engine."""

from .core import EvaluationEngine

# Global evaluation engine instance
evaluation_engine = EvaluationEngine()


def get_evaluation_engine() -> EvaluationEngine:
    """Get the evaluation engine instance."""
    return evaluation_engine