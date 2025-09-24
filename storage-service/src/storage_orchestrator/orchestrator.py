"""
Storage Orchestrator - Main coordination class for polyglot persistence
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from cache_management import CacheCoordinator
from query_optimization import QueryOptimizer
from storage_backends.cloud_storage import MediaStorageManager
from storage_backends.elasticsearch import ElasticsearchManager
from storage_backends.postgresql import PostgreSQLManager
from storage_backends.redis import RedisManager
from storage_backends.timeseries import TimeseriesDBManager
from vector_store import VectorStoreManager

from models import (
    Article,
    RetrievalOptions,
    RetrievedArticle,
    SearchRequest,
    SearchResult,
    SimilarityResult,
    SimilaritySearchParams,
    SimilaritySearchResult,
    StorageResult,
    StorageType,
)

logger = logging.getLogger(__name__)


@dataclass
class StorageTask:
    """Represents a storage task for a specific data store"""

    store_type: StorageType
    operation: str
    data: Any
    priority: int = 1


class StorageOrchestrator:
    """Orchestrate data storage across multiple systems"""

    def __init__(self):
        self.postgres_manager: Optional[PostgreSQLManager] = None
        self.elasticsearch_manager: Optional[ElasticsearchManager] = None
        self.redis_manager: Optional[RedisManager] = None
        self.vector_store: Optional[VectorStoreManager] = None
        self.media_storage: Optional[MediaStorageManager] = None
        self.timeseries_db: Optional[TimeseriesDBManager] = None
        self.cache_coordinator: Optional[CacheCoordinator] = None
        self.query_optimizer: Optional[QueryOptimizer] = None

        self._initialized = False
        self._storage_tasks: List[StorageTask] = []

    async def initialize(self):
        """Initialize all storage managers"""
        if self._initialized:
            return

        logger.info("Initializing Storage Orchestrator...")

        try:
            # Initialize storage backends
            self.postgres_manager = PostgreSQLManager()
            await self.postgres_manager.initialize()

            self.elasticsearch_manager = ElasticsearchManager()
            await self.elasticsearch_manager.initialize()

            self.redis_manager = RedisManager()
            await self.redis_manager.initialize()

            self.vector_store = VectorStoreManager()
            await self.vector_store.initialize()

            self.media_storage = MediaStorageManager()
            await self.media_storage.initialize()

            self.timeseries_db = TimeseriesDBManager()
            await self.timeseries_db.initialize()

            # Initialize coordination services
            self.cache_coordinator = CacheCoordinator()
            await self.cache_coordinator.initialize()

            self.query_optimizer = QueryOptimizer()
            await self.query_optimizer.initialize()

            self._initialized = True
            logger.info("Storage Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Storage Orchestrator: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Storage Orchestrator...")

        cleanup_tasks = []

        if self.postgres_manager:
            cleanup_tasks.append(self.postgres_manager.cleanup())
        if self.elasticsearch_manager:
            cleanup_tasks.append(self.elasticsearch_manager.cleanup())
        if self.redis_manager:
            cleanup_tasks.append(self.redis_manager.cleanup())
        if self.vector_store:
            cleanup_tasks.append(self.vector_store.cleanup())
        if self.media_storage:
            cleanup_tasks.append(self.media_storage.cleanup())
        if self.timeseries_db:
            cleanup_tasks.append(self.timeseries_db.cleanup())
        if self.cache_coordinator:
            cleanup_tasks.append(self.cache_coordinator.cleanup())
        if self.query_optimizer:
            cleanup_tasks.append(self.query_optimizer.cleanup())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self._initialized = False
        logger.info("Storage Orchestrator cleanup complete")

    async def store_article(self, article: Article) -> StorageResult:
        """Store article across appropriate data stores"""
        if not self._initialized:
            raise RuntimeError("Storage Orchestrator not initialized")

        logger.info(f"Storing article {article.id}")

        storage_tasks = []
        stored_locations = []
        failed_stores = []

        try:
            # Store structured data in PostgreSQL
            storage_tasks.append(
                self._create_storage_task(StorageType.POSTGRESQL, "store_article_metadata", article)
            )

            # Store for full-text search in Elasticsearch
            storage_tasks.append(
                self._create_storage_task(StorageType.ELASTICSEARCH, "index_article", article)
            )

            # Store embeddings in vector database
            if article.embeddings:
                storage_tasks.append(
                    self._create_storage_task(
                        StorageType.VECTOR_STORE,
                        "store_embeddings",
                        (article.id, article.embeddings),
                    )
                )

            # Store media files in cloud storage
            if article.media_files:
                storage_tasks.append(
                    self._create_storage_task(
                        StorageType.MEDIA_STORAGE,
                        "store_media_files",
                        (article.id, article.media_files),
                    )
                )

            # Store time-series metrics
            storage_tasks.append(
                self._create_storage_task(StorageType.TIMESERIES, "record_article_metrics", article)
            )

            # Execute all storage operations
            storage_results = await self._execute_storage_tasks(storage_tasks)

            # Process results
            for i, result in enumerate(storage_results):
                task = storage_tasks[i]
                if isinstance(result, Exception):
                    failed_stores.append(f"{task.store_type.value}: {str(result)}")
                    logger.warning(f"Storage failed for {task.store_type.value}: {result}")
                else:
                    stored_locations.append(task.store_type)

            # Update cache
            if self.cache_coordinator:
                await self.cache_coordinator.cache_article(article)

            logger.info(f"Article {article.id} stored in {len(stored_locations)} locations")

            return StorageResult(
                article_id=article.id,
                stored_locations=stored_locations,
                failed_stores=failed_stores,
                storage_timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Failed to store article {article.id}: {e}")
            raise

    async def retrieve_article(
        self, article_id: str, retrieval_options: RetrievalOptions
    ) -> RetrievedArticle:
        """Intelligently retrieve article from optimal sources"""
        if not self._initialized:
            raise RuntimeError("Storage Orchestrator not initialized")

        logger.info(f"Retrieving article {article_id}")

        try:
            # Check cache first
            if retrieval_options.use_cache and self.cache_coordinator:
                cached_article = await self.cache_coordinator.get_cached_article(article_id)
                if cached_article and not self._is_cache_stale(cached_article):
                    logger.info(f"Article {article_id} retrieved from cache")
                    return cached_article

            # Determine optimal retrieval strategy
            retrieval_plan = await self._create_retrieval_plan(article_id, retrieval_options)

            # Execute retrieval plan
            article_data = {}
            retrieval_sources = []

            for step in retrieval_plan:
                try:
                    if step["store_type"] == StorageType.POSTGRESQL:
                        metadata = await self.postgres_manager.get_article_metadata(article_id)
                        article_data["metadata"] = metadata
                        retrieval_sources.append(StorageType.POSTGRESQL)

                    elif step["store_type"] == StorageType.ELASTICSEARCH:
                        content = await self.elasticsearch_manager.get_article_content(article_id)
                        article_data["content"] = content
                        retrieval_sources.append(StorageType.ELASTICSEARCH)

                    elif step["store_type"] == StorageType.VECTOR_STORE:
                        embeddings = await self.vector_store.get_article_embeddings(article_id)
                        article_data["embeddings"] = embeddings
                        retrieval_sources.append(StorageType.VECTOR_STORE)

                    elif step["store_type"] == StorageType.MEDIA_STORAGE:
                        media = await self.media_storage.get_article_media(article_id)
                        article_data["media"] = media
                        retrieval_sources.append(StorageType.MEDIA_STORAGE)

                except Exception as e:
                    logger.warning(f"Failed to retrieve from {step['store_type']}: {e}")
                    continue

            # Combine data into complete article
            retrieved_article = self._combine_article_data(article_id, article_data)
            retrieved_article.retrieval_sources = retrieval_sources
            retrieved_article.retrieval_timestamp = datetime.utcnow()

            # Update cache
            if retrieval_options.update_cache and self.cache_coordinator:
                await self.cache_coordinator.update_cache(retrieved_article)

            logger.info(f"Article {article_id} retrieved from {len(retrieval_sources)} sources")
            return retrieved_article

        except Exception as e:
            logger.error(f"Failed to retrieve article {article_id}: {e}")
            raise

    async def search_articles(self, search_request: SearchRequest) -> SearchResult:
        """Search articles using Elasticsearch"""
        if not self._initialized or not self.elasticsearch_manager:
            raise RuntimeError("Elasticsearch manager not available")

        try:
            return await self.elasticsearch_manager.search_articles(search_request)
        except Exception as e:
            logger.error(f"Article search failed: {e}")
            raise

    async def similarity_search(
        self, search_params: SimilaritySearchParams
    ) -> SimilaritySearchResult:
        """Perform vector similarity search"""
        if not self._initialized or not self.vector_store:
            raise RuntimeError("Vector store manager not available")

        try:
            return await self.vector_store.similarity_search(search_params)
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise

    async def update_search_indexes(self, article_id: str):
        """Update search indexes for an article"""
        if not self._initialized:
            return

        try:
            # Update Elasticsearch index
            if self.elasticsearch_manager:
                await self.elasticsearch_manager.refresh_index(article_id)

            # Update vector indexes
            if self.vector_store:
                await self.vector_store.refresh_indexes(article_id)

        except Exception as e:
            logger.warning(f"Failed to update indexes for article {article_id}: {e}")

    async def update_cache_warmup(self, article_id: str):
        """Warm up cache for an article"""
        if not self._initialized or not self.cache_coordinator:
            return

        try:
            # Pre-load article into cache
            retrieval_options = RetrievalOptions(use_cache=False, update_cache=True)
            await self.retrieve_article(article_id, retrieval_options)

        except Exception as e:
            logger.warning(f"Failed to warm up cache for article {article_id}: {e}")

    def _create_storage_task(
        self, store_type: StorageType, operation: str, data: Any
    ) -> StorageTask:
        """Create a storage task"""
        return StorageTask(store_type=store_type, operation=operation, data=data)

    async def _execute_storage_tasks(self, tasks: List[StorageTask]) -> List[Any]:
        """Execute storage tasks in parallel"""

        async def execute_task(task: StorageTask):
            try:
                if task.store_type == StorageType.POSTGRESQL:
                    return await self.postgres_manager.store_article_metadata(task.data)
                elif task.store_type == StorageType.ELASTICSEARCH:
                    return await self.elasticsearch_manager.index_article(task.data)
                elif task.store_type == StorageType.VECTOR_STORE:
                    article_id, embeddings = task.data
                    return await self.vector_store.store_embeddings(article_id, embeddings)
                elif task.store_type == StorageType.MEDIA_STORAGE:
                    article_id, media_files = task.data
                    return await self.media_storage.store_media_files(article_id, media_files)
                elif task.store_type == StorageType.TIMESERIES:
                    return await self.timeseries_db.record_article_metrics(task.data)
                else:
                    raise ValueError(f"Unknown storage type: {task.store_type}")
            except Exception as e:
                logger.error(f"Storage task failed for {task.store_type}: {e}")
                return e

        return await asyncio.gather(*[execute_task(task) for task in tasks], return_exceptions=True)

    async def _create_retrieval_plan(
        self, article_id: str, retrieval_options: RetrievalOptions
    ) -> List[Dict[str, Any]]:
        """Create optimal retrieval plan"""
        plan = []

        # Always try PostgreSQL first for metadata
        plan.append({"store_type": StorageType.POSTGRESQL, "priority": 1, "required": True})

        # Add Elasticsearch for content
        plan.append({"store_type": StorageType.ELASTICSEARCH, "priority": 2, "required": False})

        # Add vector store if embeddings needed
        if (
            retrieval_options.preferred_sources
            and StorageType.VECTOR_STORE in retrieval_options.preferred_sources
        ):
            plan.append({"store_type": StorageType.VECTOR_STORE, "priority": 3, "required": False})

        # Add media storage if media needed
        if (
            retrieval_options.preferred_sources
            and StorageType.MEDIA_STORAGE in retrieval_options.preferred_sources
        ):
            plan.append({"store_type": StorageType.MEDIA_STORAGE, "priority": 4, "required": False})

        return sorted(plan, key=lambda x: x["priority"])

    def _is_cache_stale(self, cached_article: RetrievedArticle) -> bool:
        """Check if cached article is stale"""
        if not cached_article.retrieval_timestamp:
            return True

        # Consider cache stale after 1 hour
        cache_age = (datetime.utcnow() - cached_article.retrieval_timestamp).total_seconds()
        return cache_age > 3600

    def _combine_article_data(
        self, article_id: str, article_data: Dict[str, Any]
    ) -> RetrievedArticle:
        """Combine data from different sources into a complete article"""
        # Start with metadata as base
        metadata = article_data.get("metadata", {})
        content = article_data.get("content", {})
        embeddings = article_data.get("embeddings", {})
        media = article_data.get("media", [])

        # Merge content data
        combined_data = {
            "id": article_id,
            "title": content.get("title", metadata.get("title", "")),
            "content": content.get("content", metadata.get("content", "")),
            "summary": content.get("summary", metadata.get("summary")),
            "author": content.get("author", metadata.get("author")),
            "source": content.get("source", metadata.get("source", "")),
            "published_at": content.get("published_at", metadata.get("published_at")),
            "categories": content.get("categories", metadata.get("categories", [])),
            "tags": content.get("tags", metadata.get("tags", [])),
            "language": content.get("language", metadata.get("language", "en")),
            "url": content.get("url", metadata.get("url", "")),
            "embeddings": embeddings,
            "media_files": media,
            "metadata": metadata.get("metadata", {}),
            "cache_hit": False,
        }

        return RetrievedArticle(**combined_data)
