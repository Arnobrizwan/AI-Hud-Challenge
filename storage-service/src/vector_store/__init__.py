"""
Vector Store Manager - High-performance vector similarity search with pgvector
"""

from .index_optimizer import VectorIndexOptimizer
from .manager import VectorStoreManager
from .postgres_vector import PostgreSQLVectorStore
from .similarity_calculator import SimilarityCalculator

__all__ = [
    "VectorStoreManager",
    "PostgreSQLVectorStore",
    "SimilarityCalculator",
    "VectorIndexOptimizer",
]
