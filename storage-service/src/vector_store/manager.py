"""
Vector Store Manager - High-performance vector similarity search
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models import (
    IndexBuildResult,
    SearchMethod,
    SimilarityResult,
    SimilaritySearchParams,
    SimilaritySearchResult,
    VectorIndexConfig,
    VectorStorageResult,
)

from .index_optimizer import VectorIndexOptimizer
from .postgres_vector import PostgreSQLVectorStore
from .similarity_calculator import SimilarityCalculator

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage vector embeddings and similarity search"""

    def __init__(self):
        self.postgres_vector: Optional[PostgreSQLVectorStore] = None
        self.similarity_calculator: Optional[SimilarityCalculator] = None
        self.index_optimizer: Optional[VectorIndexOptimizer] = None
        self._initialized = False

    async def initialize(self):
        """Initialize vector store components"""
        if self._initialized:
            return

        logger.info("Initializing Vector Store Manager...")

        try:
            self.postgres_vector = PostgreSQLVectorStore()
            await self.postgres_vector.initialize()

            self.similarity_calculator = SimilarityCalculator()
            await self.similarity_calculator.initialize()

            self.index_optimizer = VectorIndexOptimizer()
            await self.index_optimizer.initialize()

            self._initialized = True
            logger.info("Vector Store Manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Vector Store Manager: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Vector Store Manager...")

        cleanup_tasks = []

        if self.postgres_vector:
            cleanup_tasks.append(self.postgres_vector.cleanup())
        if self.similarity_calculator:
            cleanup_tasks.append(self.similarity_calculator.cleanup())
        if self.index_optimizer:
            cleanup_tasks.append(self.index_optimizer.cleanup())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self._initialized = False
        logger.info("Vector Store Manager cleanup complete")

    async def store_embeddings(
        self, content_id: str, embeddings: Dict[str, List[float]]
    ) -> VectorStorageResult:
        """Store content embeddings optimally"""
        if not self._initialized or not self.postgres_vector:
            raise RuntimeError("Vector Store Manager not initialized")

        logger.info(f"Storing embeddings for content {content_id}")

        try:
            storage_results = []

            for embedding_type, vector in embeddings.items():
                # Convert to numpy array for processing
                vector_array = np.array(vector, dtype=np.float32)

                # Normalize vector for cosine similarity
                normalized_vector = vector_array / np.linalg.norm(vector_array)

                # Store in PostgreSQL with pgvector
                result = await self.postgres_vector.store_vector(
                    content_id=content_id,
                    embedding_type=embedding_type,
                    vector=normalized_vector.tolist(),
                    metadata={
                        "dimension": len(vector),
                        "norm": float(np.linalg.norm(vector_array)),
                        "created_at": datetime.utcnow().isoformat(),
                    },
                )

                storage_results.append(result)

            logger.info(f"Stored {len(embeddings)} embeddings for content {content_id}")

            return VectorStorageResult(
                content_id=content_id,
                stored_embeddings=list(embeddings.keys()),
                storage_results=storage_results,
            )

        except Exception as e:
            logger.error(f"Failed to store embeddings for content {content_id}: {e}")
            raise

    async def similarity_search(
        self, search_params: SimilaritySearchParams
    ) -> SimilaritySearchResult:
        """High-performance vector similarity search"""
        if not self._initialized or not self.postgres_vector:
            raise RuntimeError("Vector Store Manager not initialized")

        logger.info(f"Performing similarity search for {search_params.embedding_type}")

        start_time = time.time()

        try:
            # Convert query vector to numpy array
            query_vector = np.array(search_params.query_vector, dtype=np.float32)

            # Normalize query vector
            normalized_query = query_vector / np.linalg.norm(query_vector)

            # Use optimal search method based on parameters
            if search_params.search_method == SearchMethod.EXACT:
                results = await self._exact_similarity_search(normalized_query, search_params)
            elif search_params.search_method == SearchMethod.APPROXIMATE:
                results = await self._approximate_similarity_search(normalized_query, search_params)
            else:
                results = await self._hybrid_similarity_search(normalized_query, search_params)

            # Apply post-processing filters
            if search_params.filters:
                results = await self._apply_similarity_filters(results, search_params.filters)

            # Re-rank if needed
            if search_params.rerank and self.similarity_calculator:
                results = await self.similarity_calculator.rerank_results(
                    normalized_query.tolist(), results, search_params
                )

            search_duration = int((time.time() - start_time) * 1000)  # Convert to milliseconds

            logger.info(
                f"Similarity search completed in {search_duration}ms, found {len(results)} results"
            )

            return SimilaritySearchResult(
                query_vector=normalized_query.tolist(),
                results=results[: search_params.top_k],
                search_method=search_params.search_method,
                search_duration=search_duration,
                total_candidates=len(results),
            )

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise

    async def get_article_embeddings(self, article_id: str) -> Dict[str, List[float]]:
        """Get embeddings for an article"""
        if not self._initialized or not self.postgres_vector:
            raise RuntimeError("Vector Store Manager not initialized")

        try:
            return await self.postgres_vector.get_embeddings(article_id)
        except Exception as e:
            logger.error(f"Failed to get embeddings for article {article_id}: {e}")
            raise

    async def build_vector_index(self, index_config: VectorIndexConfig) -> IndexBuildResult:
        """Build optimized vector index"""
        if not self._initialized or not self.postgres_vector:
            raise RuntimeError("Vector Store Manager not initialized")

        logger.info(f"Building vector index {index_config.index_name}")

        start_time = time.time()

        try:
            # Create HNSW index for approximate similarity
            index_sql = f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_config.index_name}
            ON vector_embeddings 
            USING hnsw (vector vector_cosine_ops)
            WITH (m = {index_config.hnsw_m}, ef_construction = {index_config.ef_construction})
            WHERE embedding_type = '{index_config.embedding_type}'
            """

            await self.postgres_vector.execute_ddl(index_sql)

            build_duration = int((time.time() - start_time) * 1000)

            # Analyze index performance
            index_stats = await self._analyze_index_performance(index_config.index_name)

            logger.info(f"Vector index {index_config.index_name} built in {build_duration}ms")

            return IndexBuildResult(
                index_name=index_config.index_name,
                build_duration=build_duration,
                index_size=index_stats.get("index_size", 0),
                query_performance=index_stats.get("avg_query_time", 0.0),
            )

        except Exception as e:
            logger.error(f"Failed to build vector index {index_config.index_name}: {e}")
            raise

    async def refresh_indexes(self, content_id: str):
        """Refresh vector indexes for content"""
        if not self._initialized or not self.postgres_vector:
            return

        try:
            await self.postgres_vector.refresh_indexes(content_id)
        except Exception as e:
            logger.warning(f"Failed to refresh indexes for content {content_id}: {e}")

    async def _exact_similarity_search(
        self, query_vector: np.ndarray, params: SimilaritySearchParams
    ) -> List[SimilarityResult]:
        """Exact similarity search using pgvector"""
        try:
            # Use pgvector's cosine similarity operator
            query = """
            SELECT 
                content_id,
                embedding_type,
                vector,
                1 - (vector <=> $1) as similarity_score,
                metadata
            FROM vector_embeddings 
            WHERE embedding_type = $2
            AND ($3::text IS NULL OR metadata->>'category' = $3)
            ORDER BY vector <=> $1
            LIMIT $4
            """

            results = await self.postgres_vector.execute_query(
                query,
                query_vector.tolist(),
                params.embedding_type,
                params.category_filter,
                params.top_k * 2,  # Get more candidates for filtering
            )

            similarity_results = []
            for row in results:
                if row["similarity_score"] >= params.similarity_threshold:
                    similarity_results.append(
                        SimilarityResult(
                            content_id=row["content_id"],
                            similarity_score=row["similarity_score"],
                            embedding_type=row["embedding_type"],
                            metadata=row["metadata"] or {},
                        )
                    )

            return similarity_results

        except Exception as e:
            logger.error(f"Exact similarity search failed: {e}")
            raise

    async def _approximate_similarity_search(
        self, query_vector: np.ndarray, params: SimilaritySearchParams
    ) -> List[SimilarityResult]:
        """Approximate similarity search using HNSW index"""
        try:
            # Use HNSW index for approximate search
            query = """
            SELECT 
                content_id,
                embedding_type,
                vector,
                1 - (vector <=> $1) as similarity_score,
                metadata
            FROM vector_embeddings 
            WHERE embedding_type = $2
            AND ($3::text IS NULL OR metadata->>'category' = $3)
            ORDER BY vector <=> $1
            LIMIT $4
            """

            results = await self.postgres_vector.execute_query(
                query,
                query_vector.tolist(),
                params.embedding_type,
                params.category_filter,
                params.top_k * 3,  # Get more candidates for approximate search
            )

            similarity_results = []
            for row in results:
                if row["similarity_score"] >= params.similarity_threshold:
                    similarity_results.append(
                        SimilarityResult(
                            content_id=row["content_id"],
                            similarity_score=row["similarity_score"],
                            embedding_type=row["embedding_type"],
                            metadata=row["metadata"] or {},
                        )
                    )

            return similarity_results

        except Exception as e:
            logger.error(f"Approximate similarity search failed: {e}")
            raise

    async def _hybrid_similarity_search(
        self, query_vector: np.ndarray, params: SimilaritySearchParams
    ) -> List[SimilarityResult]:
        """Hybrid similarity search combining exact and approximate methods"""
        try:
            # Get approximate results first (faster)
            approximate_results = await self._approximate_similarity_search(query_vector, params)

            # If we have enough high-quality results, return them
            if len(approximate_results) >= params.top_k:
                return approximate_results[: params.top_k]

            # Otherwise, fall back to exact search for remaining slots
            exact_params = SimilaritySearchParams(
                query_vector=query_vector.tolist(),
                embedding_type=params.embedding_type,
                top_k=params.top_k - len(approximate_results),
                similarity_threshold=params.similarity_threshold,
                search_method=SearchMethod.EXACT,
                filters=params.filters,
                category_filter=params.category_filter,
                rerank=False,
            )

            exact_results = await self._exact_similarity_search(query_vector, exact_params)

            # Combine and deduplicate results
            all_results = approximate_results + exact_results
            seen_ids = set()
            combined_results = []

            for result in all_results:
                if result.content_id not in seen_ids:
                    seen_ids.add(result.content_id)
                    combined_results.append(result)

            return combined_results[: params.top_k]

        except Exception as e:
            logger.error(f"Hybrid similarity search failed: {e}")
            raise

    async def _apply_similarity_filters(
        self, results: List[SimilarityResult], filters: Dict[str, Any]
    ) -> List[SimilarityResult]:
        """Apply post-processing filters to similarity results"""
        filtered_results = []

        for result in results:
            include_result = True

            # Apply metadata filters
            for filter_key, filter_value in filters.items():
                if filter_key in result.metadata:
                    if result.metadata[filter_key] != filter_value:
                        include_result = False
                        break
                else:
                    include_result = False
                    break

            if include_result:
                filtered_results.append(result)

        return filtered_results

    async def _analyze_index_performance(self, index_name: str) -> Dict[str, Any]:
        """Analyze index performance metrics"""
        try:
            # Get index size
            size_query = """
            SELECT pg_size_pretty(pg_relation_size('vector_embeddings')) as index_size
            """
            size_result = await self.postgres_vector.execute_query(size_query)

            # Get query performance (simplified)
            performance_query = """
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch
            FROM pg_stat_user_indexes 
            WHERE indexname = $1
            """
            perf_result = await self.postgres_vector.execute_query(performance_query, index_name)

            return {
                "index_size": size_result[0]["index_size"] if size_result else "0 bytes",
                "avg_query_time": 0.0,  # Would need more complex analysis
                "index_usage": perf_result[0] if perf_result else {},
            }

        except Exception as e:
            logger.warning(f"Failed to analyze index performance: {e}")
            return {"index_size": "0 bytes", "avg_query_time": 0.0}
