"""
Vector Store Manager - High-performance vector similarity search with pgvector
"""

from .manager import VectorStoreManager
from .postgres_vector import PostgreSQLVectorStore
from .similarity_calculator import SimilarityCalculator
from .index_optimizer import VectorIndexOptimizer

__all__ = [
    "VectorStoreManager",
    "PostgreSQLVectorStore",
    "SimilarityCalculator",
    "VectorIndexOptimizer"
]
