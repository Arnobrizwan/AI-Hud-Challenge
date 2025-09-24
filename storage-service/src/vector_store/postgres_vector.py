"""
PostgreSQL Vector Store - pgvector integration for high-performance vector operations
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from asyncpg.pool import Pool

from config import Settings
from models import SimilarityResult

logger = logging.getLogger(__name__)


class PostgreSQLVectorStore:
    """PostgreSQL with pgvector for vector storage and similarity search"""

    def __init__(self):
        self.pool: Optional[Pool] = None
        self.settings = Settings()
        self._initialized = False

    async def initialize(self) -> Dict[str, Any]:
    """Initialize PostgreSQL connection pool"""
        if self._initialized:
            return

        logger.info("Initializing PostgreSQL Vector Store...")

        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.settings.postgres_host,
                port=self.settings.postgres_port,
                database=self.settings.postgres_database,
                user=self.settings.postgres_username,
                password=self.settings.postgres_password,
                min_size=5,
                max_size=self.settings.postgres_pool_size,
                command_timeout=60,
            )

            # Initialize database schema
            await self._initialize_schema()

            self._initialized = True
            logger.info("PostgreSQL Vector Store initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL Vector Store: {e}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
    """Cleanup connection pool"""
        if self.pool:
    await self.pool.close()
            self.pool = None

        self._initialized = False
        logger.info("PostgreSQL Vector Store cleanup complete")

    async def _initialize_schema(self) -> Dict[str, Any]:
    """Initialize database schema for vector operations"""
        try:
    async with self.pool.acquire() as conn:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # Create vector embeddings table
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vector_embeddings (
                        id SERIAL PRIMARY KEY,
                        content_id VARCHAR(255) NOT NULL,
                        embedding_type VARCHAR(100) NOT NULL,
                        vector VECTOR(768),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(content_id, embedding_type)
                    )
                """
                )

                # Create indexes for performance
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_vector_embeddings_content_id
                    ON vector_embeddings(content_id)
                """
                )

                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_vector_embeddings_type
                    ON vector_embeddings(embedding_type)
                """
                )

                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_vector_embeddings_metadata
                    ON vector_embeddings USING GIN(metadata)
                """
                )

                # Create HNSW index for approximate similarity search
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_vector_embeddings_hnsw
                    ON vector_embeddings
                    USING hnsw (vector vector_cosine_ops)
                    WITH (m = 16, ef_construction = 200)
                """
                )

                logger.info("PostgreSQL vector schema initialized")

        except Exception as e:
            logger.error(f"Failed to initialize vector schema: {e}")
            raise

    async def store_vector(self,
                           content_id: str,
                           embedding_type: str,
                           vector: List[float],
                           metadata: Dict[str,
                                          Any]) -> Dict[str, Any]:
    """Store vector embedding"""
        if not self._initialized or not self.pool:
            raise RuntimeError("PostgreSQL Vector Store not initialized")

        try:
    async with self.pool.acquire() as conn:
                # Insert or update vector
                await conn.execute(
                    """
                    INSERT INTO vector_embeddings
                    (content_id, embedding_type, vector, metadata, updated_at)
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                    ON CONFLICT (content_id, embedding_type)
                    DO UPDATE SET
                        vector = EXCLUDED.vector,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    content_id,
                    embedding_type,
                    vector,
                    json.dumps(metadata),
                )

                return {
                    "content_id": content_id,
                    "embedding_type": embedding_type,
                    "stored": True,
                    "timestamp": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            logger.error(f"Failed to store vector for {content_id}: {e}")
            raise

    async def get_embeddings(self, content_id: str) -> Dict[str, List[float]]:
        """Get all embeddings for a content ID"""
        if not self._initialized or not self.pool:
            raise RuntimeError("PostgreSQL Vector Store not initialized")

        try:
    async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT embedding_type, vector, metadata
                    FROM vector_embeddings
                    WHERE content_id = $1
                """,
                    content_id,
                )

                embeddings = {}
                for row in rows:
                    embeddings[row["embedding_type"]] = row["vector"]

                return embeddings

        except Exception as e:
            logger.error(f"Failed to get embeddings for {content_id}: {e}")
            raise

    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a query and return results"""
        if not self._initialized or not self.pool:
            raise RuntimeError("PostgreSQL Vector Store not initialized")

        try:
    async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    async def execute_ddl(self, ddl: str) -> Dict[str, Any]:
    """Execute DDL statement"""
        if not self._initialized or not self.pool:
            raise RuntimeError("PostgreSQL Vector Store not initialized")

        try:
    async with self.pool.acquire() as conn:
    await conn.execute(ddl)

        except Exception as e:
            logger.error(f"DDL execution failed: {e}")
            raise

    async def refresh_indexes(self, content_id: str) -> Dict[str, Any]:
    """Refresh indexes for a specific content ID"""
        if not self._initialized or not self.pool:
            return

        try:
    async with self.pool.acquire() as conn:
                # Update the updated_at timestamp to trigger index refresh
                await conn.execute(
                    """
                    UPDATE vector_embeddings
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE content_id = $1
                """,
                    content_id,
                )

        except Exception as e:
            logger.warning(f"Failed to refresh indexes for {content_id}: {e}")

    async def delete_embeddings(
            self,
            content_id: str,
            embedding_type: Optional[str] = None):
         -> Dict[str, Any]:"""Delete embeddings for a content ID"""
        if not self._initialized or not self.pool:
            raise RuntimeError("PostgreSQL Vector Store not initialized")

        try:
    async with self.pool.acquire() as conn:
                if embedding_type:
    await conn.execute(
                        """
                        DELETE FROM vector_embeddings
                        WHERE content_id = $1 AND embedding_type = $2
                    """,
                        content_id,
                        embedding_type,
                    )
                else:
    await conn.execute(
                        """
                        DELETE FROM vector_embeddings
                        WHERE content_id = $1
                    """,
                        content_id,
                    )

        except Exception as e:
            logger.error(f"Failed to delete embeddings for {content_id}: {e}")
            raise

    async def get_vector_stats(self) -> Dict[str, Any]:
    """Get vector store statistics"""
        if not self._initialized or not self.pool:
            raise RuntimeError("PostgreSQL Vector Store not initialized")

        try:
    async with self.pool.acquire() as conn:
                # Get total vectors count
                total_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM vector_embeddings
                """
                )

                # Get embeddings by type
                type_counts = await conn.fetch(
                    """
                    SELECT embedding_type, COUNT(*) as count
                    FROM vector_embeddings
                    GROUP BY embedding_type
                """
                )

                # Get table size
                table_size = await conn.fetchval(
                    """
                    SELECT pg_size_pretty(pg_total_relation_size('vector_embeddings'))
                """
                )

                return {
                    "total_vectors": total_count,
                    "embeddings_by_type": {
                        row["embedding_type"]: row["count"] for row in type_counts},
                    "table_size": table_size,
                    "timestamp": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            logger.error(f"Failed to get vector stats: {e}")
            raise
