"""
Core ingestion service for orchestrating content collection and processing.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.adapters.api_adapter import APIAdapter
from src.adapters.base import BaseAdapter
from src.adapters.rss_adapter import RSSAdapter
from src.adapters.web_scraping_adapter import WebScrapingAdapter
from src.models.content import (
    ContentMetrics,
    DuplicateDetection,
    NormalizedArticle,
    ProcessingBatch,
    ProcessingStatus,
    SourceConfig,
)
from src.normalizers.content_normalizer import ContentNormalizer
from src.normalizers.duplicate_detector import DuplicateDetector
from src.utils.content_parser import ContentParser
from src.utils.date_utils import DateUtils
from src.utils.http_client import HTTPClient
from src.utils.url_utils import URLUtils

logger = logging.getLogger(__name__)


class IngestionService:
    """Core service for content ingestion and processing."""

    def __init__(
        self,
        http_client: HTTPClient = None,
        content_parser: ContentParser = None,
        url_utils: URLUtils = None,
        date_utils: DateUtils = None,
    ):
        self.http_client = http_client or HTTPClient()
        self.content_parser = content_parser or ContentParser()
        self.url_utils = url_utils or URLUtils()
        self.date_utils = date_utils or DateUtils()

        # Initialize processors
        self.content_normalizer = ContentNormalizer(
            content_parser=self.content_parser, url_utils=self.url_utils, date_utils=self.date_utils
        )
        self.duplicate_detector = DuplicateDetector(self.content_normalizer)

        # Adapter registry
        self.adapters = {
            "rss_feed": RSSAdapter,
            "atom_feed": RSSAdapter,
            "json_feed": RSSAdapter,
            "api": APIAdapter,
            "web_scraping": WebScrapingAdapter,
        }

        # Processing state
        self.active_batches: Dict[str, ProcessingBatch] = {}
        self.source_adapters: Dict[str, BaseAdapter] = {}
        self.processing_metrics: Dict[str, ContentMetrics] = {}

    async def process_source(self, source_config: SourceConfig) -> ProcessingBatch:
        """Process a single content source."""
        batch_id = str(uuid.uuid4())

        # Create processing batch
        batch = ProcessingBatch(
            batch_id=batch_id,
            source_id=source_config.id,
            articles=[],
            total_count=0,
            status=ProcessingStatus.PENDING,
        )

        self.active_batches[batch_id] = batch

        try:
            # Update batch status
            batch.status = ProcessingStatus.INGESTING
            batch.started_at = datetime.utcnow()

            # Get or create adapter
            adapter = await self._get_adapter(source_config)

            # Process content
            articles = []
            async for article in adapter.fetch_content():
                if article:
                    articles.append(article)

            # Update batch with articles
            batch.articles = articles
            batch.total_count = len(articles)

            # Process articles
            batch.status = ProcessingStatus.NORMALIZING
            processed_articles = await self._process_articles(articles)

            # Update batch with processed articles
            batch.articles = processed_articles
            batch.processed_count = len(
                [a for a in processed_articles if a.processing_status == ProcessingStatus.COMPLETED]
            )
            batch.failed_count = len([a for a in processed_articles if a.processing_status == ProcessingStatus.FAILED])
            batch.duplicate_count = len(
                [a for a in processed_articles if a.processing_status == ProcessingStatus.DUPLICATE]
            )

            # Update batch status
            if batch.processed_count > 0:
                batch.status = ProcessingStatus.COMPLETED
            else:
                batch.status = ProcessingStatus.FAILED

            batch.completed_at = datetime.utcnow()

            # Update metrics
            await self._update_metrics(source_config.id, batch)

            logger.info(f"Completed processing source {source_config.id}: {batch.processed_count} articles processed")

        except Exception as e:
            batch.status = ProcessingStatus.FAILED
            batch.error_message = str(e)
            batch.completed_at = datetime.utcnow()
            logger.error(f"Error processing source {source_config.id}: {e}")

        return batch

    async def process_sources(self, source_configs: List[SourceConfig]) -> List[ProcessingBatch]:
        """Process multiple content sources concurrently."""
        # Filter enabled sources
        enabled_sources = [s for s in source_configs if s.enabled]

        if not enabled_sources:
            logger.warning("No enabled sources to process")
            return []

        # Process sources concurrently
        tasks = [self.process_source(source) for source in enabled_sources]
        batches = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_batches = []
        for batch in batches:
            if isinstance(batch, ProcessingBatch):
                valid_batches.append(batch)
            else:
                logger.error(f"Error processing source: {batch}")

        return valid_batches

    async def _get_adapter(self, source_config: SourceConfig) -> BaseAdapter:
        """Get or create adapter for source."""
        if source_config.id in self.source_adapters:
            return self.source_adapters[source_config.id]

        # Create adapter based on source type
        adapter_class = self.adapters.get(source_config.type.value)
        if not adapter_class:
            raise ValueError(f"Unknown source type: {source_config.type}")

        adapter = adapter_class(
            source_config=source_config,
            http_client=self.http_client,
            content_parser=self.content_parser,
            url_utils=self.url_utils,
            date_utils=self.date_utils,
        )

        self.source_adapters[source_config.id] = adapter
        return adapter

    async def _process_articles(self, articles: List[NormalizedArticle]) -> List[NormalizedArticle]:
        """Process articles through normalization and duplicate detection."""
        processed_articles = []

        for article in articles:
            try:
                # Normalize article
                normalized_article = await self.content_normalizer.normalize_article(article)

                # Check for duplicates
                if normalized_article.processing_status == ProcessingStatus.COMPLETED:
                    duplicates = await self.duplicate_detector.detect_duplicates(normalized_article, processed_articles)

                    if duplicates:
                        normalized_article.processing_status = ProcessingStatus.DUPLICATE
                        normalized_article.ingestion_metadata["duplicates"] = [
                            {
                                "duplicate_of": dup.duplicate_of,
                                "similarity_score": dup.similarity_score,
                            }
                            for dup in duplicates
                        ]

                processed_articles.append(normalized_article)

            except Exception as e:
                logger.warning(f"Error processing article {article.id}: {e}")
                article.processing_status = ProcessingStatus.FAILED
                article.ingestion_metadata["error_message"] = str(e)
                processed_articles.append(article)

        return processed_articles

    async def _update_metrics(self, source_id: str, batch: ProcessingBatch) -> Dict[str, Any]:
    """Update processing metrics for source."""
        if source_id not in self.processing_metrics:
            self.processing_metrics[source_id] = ContentMetrics(source_id=source_id, date=datetime.utcnow().date())

        metrics = self.processing_metrics[source_id]

        # Update counts
        metrics.total_articles += batch.total_count
        metrics.successful_articles += batch.processed_count
        metrics.failed_articles += batch.failed_count
        metrics.duplicate_articles += batch.duplicate_count

        # Update processing time
        if batch.processing_time_seconds:
            total_time = metrics.average_processing_time_ms * (metrics.total_articles - batch.total_count)
            total_time += batch.processing_time_seconds * 1000
            metrics.average_processing_time_ms = total_time / metrics.total_articles

        # Update word count
        if batch.articles:
            total_words = sum(article.word_count for article in batch.articles)
            metrics.average_word_count = total_words / len(batch.articles)

        # Update language distribution
        for article in batch.articles:
            if article.language in metrics.language_distribution:
                metrics.language_distribution[article.language] += 1
            else:
                metrics.language_distribution[article.language] = 1

        # Update content type distribution
        for article in batch.articles:
            content_type = article.content_type.value
            if content_type in metrics.content_type_distribution:
                metrics.content_type_distribution[content_type] += 1
            else:
                metrics.content_type_distribution[content_type] = 1

    async def get_source_health(self, source_id: str) -> Dict[str, Any]:
    """Get health status for a source."""
        if source_id not in self.source_adapters:
            return {"status": "not_found", "message": "Source not found"}

        adapter = self.source_adapters[source_id]
        return await adapter.health_check()

    async def get_all_source_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all sources."""
        health_status = {}

        for source_id, adapter in self.source_adapters.items():
            try:
                health_status[source_id] = await adapter.health_check()
            except Exception as e:
                health_status[source_id] = {"status": "error", "error": str(e)}

        return health_status

    async def get_processing_metrics(self, source_id: str = None) -> Dict[str, Any]:
    """Get processing metrics."""
        if source_id:
            return self.processing_metrics.get(source_id, {})

        return self.processing_metrics

    async def get_batch_status(self, batch_id: str) -> Optional[ProcessingBatch]:
        """Get status of a processing batch."""
        return self.active_batches.get(batch_id)

    async def get_active_batches(self) -> List[ProcessingBatch]:
        """Get all active processing batches."""
        return list(self.active_batches.values())

    async def cleanup_completed_batches(self, max_age_hours: int = 24) -> Dict[str, Any]:
    """Clean up completed batches older than specified age."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

        to_remove = []
        for batch_id, batch in self.active_batches.items():
            if batch.completed_at and batch.completed_at < cutoff_time:
                to_remove.append(batch_id)

        for batch_id in to_remove:
            del self.active_batches[batch_id]

        logger.info(f"Cleaned up {len(to_remove)} completed batches")

    async def test_source_connection(self, source_config: SourceConfig) -> bool:
        """Test connection to a source."""
        try:
            adapter = await self._get_adapter(source_config)
            return await adapter.test_connection()
        except Exception as e:
            logger.error(f"Error testing source connection: {e}")
            return False

    async def get_source_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a source."""
        if source_id not in self.source_adapters:
            return None

        adapter = self.source_adapters[source_id]
        return adapter.get_source_info()

    async def shutdown(self) -> Dict[str, Any]:
    """Shutdown the ingestion service."""
        # Close HTTP client
        await self.http_client.close()

        # Clean up adapters
        for adapter in self.source_adapters.values():
            if hasattr(adapter, "close"):
                await adapter.close()

        # Clear state
        self.source_adapters.clear()
        self.active_batches.clear()
        self.processing_metrics.clear()

        logger.info("Ingestion service shutdown complete")
