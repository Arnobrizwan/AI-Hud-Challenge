"""LSH (Locality-Sensitive Hashing) index implementation."""

import asyncio
import json
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

import redis.asyncio as redis
from datasketch import MinHashLSH


class LSHIndex:
    """LSH index for efficient similarity search."""

    def __init__(
        self,
        redis_client: redis.Redis,
        threshold: float = 0.7,
        num_perm: int = 128,
        num_bands: int = 16,
        band_size: int = 8,
        index_name: str = "lsh_index",
    ):
        """Initialize LSH index.

        Args:
            redis_client: Redis client for storage
            threshold: Similarity threshold for LSH
            num_perm: Number of permutations for MinHash
            num_bands: Number of bands for LSH
            band_size: Size of each band
            index_name: Name of the index in Redis
        """
        self.redis = redis_client
        self.threshold = threshold
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.band_size = band_size
        self.index_name = index_name

        # Initialize LSH
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

        # Redis keys
        self.index_key = f"{index_name}:lsh"
        self.metadata_key = f"{index_name}:metadata"
        self.stats_key = f"{index_name}:stats"

    async def add_article(self, article_id: UUID, minhash: Any) -> None:
        """Add article to LSH index.

        Args:
            article_id: Article UUID
            minhash: MinHash signature
        """
        # Add to in-memory LSH
        self.lsh.insert(str(article_id), minhash)

        # Store in Redis
        await self._store_in_redis(article_id, minhash)

        # Update statistics
        await self._update_stats("articles_added", 1)

    async def query_candidates(self, minhash: Any) -> List[UUID]:
        """Query LSH index for similar articles.

        Args:
            minhash: MinHash signature to query

        Returns:
            List of candidate article IDs
        """
        # Query in-memory LSH
        candidates = self.lsh.query(minhash)

        # Convert to UUIDs
        candidate_ids = [UUID(candidate) for candidate in candidates]

        # Update statistics
        await self._update_stats("queries_performed", 1)

        return candidate_ids

    async def remove_article(self, article_id: UUID) -> None:
        """Remove article from LSH index.

        Args:
            article_id: Article UUID to remove
        """
        # Remove from in-memory LSH
        self.lsh.remove(str(article_id))

        # Remove from Redis
        await self._remove_from_redis(article_id)

        # Update statistics
        await self._update_stats("articles_removed", 1)

    async def batch_add_articles(self, articles: List[Tuple[UUID, Any]]) -> None:
        """Add multiple articles to LSH index in batch.

        Args:
            articles: List of (article_id, minhash) tuples
        """
        # Add to in-memory LSH
        for article_id, minhash in articles:
            self.lsh.insert(str(article_id), minhash)

        # Store in Redis in batch
        await self._batch_store_in_redis(articles)

        # Update statistics
        await self._update_stats("articles_added", len(articles))

    async def get_index_stats(self) -> Dict[str, Any]:
        """Get LSH index statistics.

        Returns:
            Dictionary with index statistics
        """
        stats = await self.redis.hgetall(self.stats_key)

        # Convert string values to appropriate types
        result = {}
        for key, value in stats.items():
            try:
                result[key.decode()] = int(value)
            except (ValueError, AttributeError):
                result[key.decode()] = value.decode()

        # Add current index size
        result["index_size"] = len(self.lsh)

        return result

    async def clear_index(self) -> None:
        """Clear the entire LSH index."""
        # Clear in-memory LSH
        self.lsh = MinHashLSH(
            threshold=self.threshold,
            num_perm=self.num_perm,
            num_bands=self.num_bands,
            band_size=self.band_size,
        )

        # Clear Redis
        await self.redis.delete(self.index_key)
        await self.redis.delete(self.metadata_key)
        await self.redis.delete(self.stats_key)

    async def rebuild_index(self, articles: List[Tuple[UUID, Any]]) -> None:
        """Rebuild the entire LSH index.

        Args:
            articles: List of (article_id, minhash) tuples
        """
        # Clear existing index
        await self.clear_index()

        # Rebuild with new articles
        await self.batch_add_articles(articles)

    async def _store_in_redis(self, article_id: UUID, minhash: Any) -> None:
        """Store article in Redis.

        Args:
            article_id: Article UUID
            minhash: MinHash signature
        """
        # Serialize MinHash
        minhash_data = pickle.dumps(minhash)

        # Store in Redis hash
        await self.redis.hset(self.index_key, str(article_id), minhash_data)

        # Store metadata
        metadata = {
            "article_id": str(article_id),
            "added_at": asyncio.get_event_loop().time(),
            "num_perm": self.num_perm,
        }

        await self.redis.hset(self.metadata_key, str(article_id), json.dumps(metadata))

    async def _batch_store_in_redis(self, articles: List[Tuple[UUID, Any]]) -> None:
        """Store multiple articles in Redis in batch.

        Args:
            articles: List of (article_id, minhash) tuples
        """
        # Prepare batch data
        index_data = {}
        metadata_data = {}
        current_time = asyncio.get_event_loop().time()

        for article_id, minhash in articles:
            # Serialize MinHash
            minhash_data = pickle.dumps(minhash)
            index_data[str(article_id)] = minhash_data

            # Prepare metadata
            metadata = {
                "article_id": str(article_id),
                "added_at": current_time,
                "num_perm": self.num_perm,
            }
            metadata_data[str(article_id)] = json.dumps(metadata)

        # Batch store in Redis
        if index_data:
            await self.redis.hset(self.index_key, mapping=index_data)

        if metadata_data:
            await self.redis.hset(self.metadata_key, mapping=metadata_data)

    async def _remove_from_redis(self, article_id: UUID) -> None:
        """Remove article from Redis.

        Args:
            article_id: Article UUID to remove
        """
        await self.redis.hdel(self.index_key, str(article_id))
        await self.redis.hdel(self.metadata_key, str(article_id))

    async def _update_stats(self, stat_name: str, increment: int = 1) -> None:
        """Update statistics in Redis.

        Args:
            stat_name: Name of the statistic
            increment: Value to increment by
        """
        await self.redis.hincrby(self.stats_key, stat_name, increment)

    async def load_from_redis(self) -> None:
        """Load LSH index from Redis."""
        # Get all articles from Redis
        articles_data = await self.redis.hgetall(self.index_key)

        # Rebuild in-memory LSH
        self.lsh = MinHashLSH(
            threshold=self.threshold,
            num_perm=self.num_perm,
            num_bands=self.num_bands,
            band_size=self.band_size,
        )

        for article_id_str, minhash_data in articles_data.items():
            try:
                article_id = UUID(article_id_str)
                minhash = pickle.loads(minhash_data)
                self.lsh.insert(article_id_str, minhash)
            except (ValueError, pickle.PickleError) as e:
                # Skip invalid entries
                print(f"Warning: Failed to load article {article_id_str}: {e}")
                continue


class LSHIndexManager:
    """Manager for multiple LSH indices."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize LSH index manager.

        Args:
            redis_client: Redis client for storage
        """
        self.redis = redis_client
        self.indices: Dict[str, LSHIndex] = {}

    async def get_index(
        self,
        index_name: str,
        threshold: float = 0.7,
        num_perm: int = 128,
        num_bands: int = 16,
        band_size: int = 8,
    ) -> LSHIndex:
        """Get or create LSH index.

        Args:
            index_name: Name of the index
            threshold: Similarity threshold
            num_perm: Number of permutations
            num_bands: Number of bands
            band_size: Size of each band

        Returns:
            LSH index instance
        """
        if index_name not in self.indices:
            self.indices[index_name] = LSHIndex(
                redis_client=self.redis,
                threshold=threshold,
                num_perm=num_perm,
                num_bands=num_bands,
                band_size=band_size,
                index_name=index_name,
            )

            # Load existing data from Redis
            await self.indices[index_name].load_from_redis()

        return self.indices[index_name]

    async def remove_index(self, index_name: str) -> None:
        """Remove LSH index.

        Args:
            index_name: Name of the index to remove
        """
        if index_name in self.indices:
            await self.indices[index_name].clear_index()
            del self.indices[index_name]

    async def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all indices.

        Returns:
            Dictionary with statistics for each index
        """
        stats = {}
        for name, index in self.indices.items():
            stats[name] = await index.get_index_stats()

        return stats
