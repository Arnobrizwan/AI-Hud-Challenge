"""
Similarity Calculator - Advanced vector similarity calculations and re-ranking
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

from models import SimilarityResult, SimilaritySearchParams

logger = logging.getLogger(__name__)

class SimilarityCalculator:
    """Calculate and optimize vector similarities"""
    
    def __init__(self):
        self._initialized = False
        
    async def initialize(self):
        """Initialize similarity calculator"""
        self._initialized = True
        logger.info("Similarity Calculator initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        self._initialized = False
        logger.info("Similarity Calculator cleanup complete")
    
    def calculate_cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vector1, dtype=np.float32)
            v2 = np.array(vector2, dtype=np.float32)
            
            # Normalize vectors
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            
            # Calculate cosine similarity
            similarity = np.dot(v1_norm, v2_norm)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    def calculate_euclidean_distance(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors"""
        try:
            v1 = np.array(vector1, dtype=np.float32)
            v2 = np.array(vector2, dtype=np.float32)
            
            distance = np.linalg.norm(v1 - v2)
            return float(distance)
            
        except Exception as e:
            logger.error(f"Failed to calculate Euclidean distance: {e}")
            return float('inf')
    
    def calculate_manhattan_distance(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate Manhattan distance between two vectors"""
        try:
            v1 = np.array(vector1, dtype=np.float32)
            v2 = np.array(vector2, dtype=np.float32)
            
            distance = np.sum(np.abs(v1 - v2))
            return float(distance)
            
        except Exception as e:
            logger.error(f"Failed to calculate Manhattan distance: {e}")
            return float('inf')
    
    def calculate_dot_product(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate dot product between two vectors"""
        try:
            v1 = np.array(vector1, dtype=np.float32)
            v2 = np.array(vector2, dtype=np.float32)
            
            dot_product = np.dot(v1, v2)
            return float(dot_product)
            
        except Exception as e:
            logger.error(f"Failed to calculate dot product: {e}")
            return 0.0
    
    async def rerank_results(self, query_vector: List[float], 
                           results: List[SimilarityResult],
                           search_params: SimilaritySearchParams) -> List[SimilarityResult]:
        """Re-rank similarity results using advanced algorithms"""
        if not self._initialized:
            return results
        
        try:
            logger.info(f"Re-ranking {len(results)} results")
            
            # Convert query vector to numpy array
            query_array = np.array(query_vector, dtype=np.float32)
            
            # Re-rank using multiple similarity measures
            reranked_results = []
            
            for result in results:
                # Get the stored vector (would need to fetch from database)
                # For now, we'll use the existing similarity score
                base_score = result.similarity_score
                
                # Apply boosting factors
                boosted_score = self._apply_boosting_factors(
                    base_score, result, search_params
                )
                
                # Create new result with boosted score
                reranked_result = SimilarityResult(
                    content_id=result.content_id,
                    similarity_score=boosted_score,
                    embedding_type=result.embedding_type,
                    metadata=result.metadata
                )
                
                reranked_results.append(reranked_result)
            
            # Sort by boosted score
            reranked_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.info(f"Re-ranking completed, top score: {reranked_results[0].similarity_score if reranked_results else 0}")
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Failed to re-rank results: {e}")
            return results
    
    def _apply_boosting_factors(self, base_score: float, result: SimilarityResult,
                              search_params: SimilaritySearchParams) -> float:
        """Apply boosting factors to similarity score"""
        boosted_score = base_score
        
        # Recency boost (if metadata contains timestamp)
        if 'created_at' in result.metadata:
            try:
                from datetime import datetime, timedelta
                created_at = datetime.fromisoformat(result.metadata['created_at'])
                days_old = (datetime.utcnow() - created_at).days
                
                # Boost newer content
                if days_old < 7:  # Last week
                    boosted_score *= 1.1
                elif days_old < 30:  # Last month
                    boosted_score *= 1.05
            except Exception:
                pass
        
        # Category boost
        if 'category' in result.metadata and search_params.category_filter:
            if result.metadata['category'] == search_params.category_filter:
                boosted_score *= 1.2
        
        # Source authority boost
        if 'source_authority' in result.metadata:
            authority = result.metadata['source_authority']
            if authority > 0.8:
                boosted_score *= 1.15
            elif authority > 0.6:
                boosted_score *= 1.05
        
        # Quality indicators boost
        if 'quality_score' in result.metadata:
            quality = result.metadata['quality_score']
            boosted_score *= (1 + quality * 0.1)  # Up to 10% boost
        
        return min(boosted_score, 1.0)  # Cap at 1.0
    
    def calculate_batch_similarities(self, query_vector: List[float],
                                   candidate_vectors: List[List[float]]) -> List[float]:
        """Calculate similarities for multiple vectors efficiently"""
        try:
            query_array = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            candidates_array = np.array(candidate_vectors, dtype=np.float32)
            
            # Use sklearn for efficient batch calculation
            similarities = cosine_similarity(query_array, candidates_array)[0]
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Failed to calculate batch similarities: {e}")
            return [0.0] * len(candidate_vectors)
    
    def find_duplicates(self, vectors: List[List[float]], 
                       threshold: float = 0.95) -> List[Tuple[int, int, float]]:
        """Find duplicate vectors based on similarity threshold"""
        try:
            duplicates = []
            vectors_array = np.array(vectors, dtype=np.float32)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(vectors_array)
            
            # Find pairs above threshold
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    if similarities[i][j] >= threshold:
                        duplicates.append((i, j, similarities[i][j]))
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Failed to find duplicates: {e}")
            return []
    
    def calculate_centroid(self, vectors: List[List[float]]) -> List[float]:
        """Calculate centroid of multiple vectors"""
        try:
            vectors_array = np.array(vectors, dtype=np.float32)
            centroid = np.mean(vectors_array, axis=0)
            return centroid.tolist()
            
        except Exception as e:
            logger.error(f"Failed to calculate centroid: {e}")
            return []
    
    def calculate_diversity_score(self, vectors: List[List[float]]) -> float:
        """Calculate diversity score for a set of vectors"""
        try:
            if len(vectors) < 2:
                return 0.0
            
            vectors_array = np.array(vectors, dtype=np.float32)
            
            # Calculate pairwise distances
            distances = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    dist = np.linalg.norm(vectors_array[i] - vectors_array[j])
                    distances.append(dist)
            
            # Diversity is the average distance
            diversity = np.mean(distances) if distances else 0.0
            return float(diversity)
            
        except Exception as e:
            logger.error(f"Failed to calculate diversity score: {e}")
            return 0.0
