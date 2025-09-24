"""
Google Cloud Firestore service for data persistence.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from google.api_core import retry
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from src.config.settings import settings
from src.models.content import ContentMetrics, NormalizedArticle, ProcessingBatch, SourceConfig

logger = logging.getLogger(__name__)


class FirestoreService:
    """Google Cloud Firestore service for data persistence."""

    def __init__(self):
        self.db = firestore.Client(project=settings.GCP_PROJECT_ID)

        # Collection names
        self.articles_collection = settings.FIRESTORE_COLLECTION_ARTICLES
        self.sources_collection = settings.FIRESTORE_COLLECTION_SOURCES
        self.metadata_collection = settings.FIRESTORE_COLLECTION_METADATA

    async def save_article(self, article: NormalizedArticle) -> bool:
        """Save a single article to Firestore."""
        try:
            # Convert article to dictionary
            article_data = self._article_to_dict(article)

            # Add to Firestore
            doc_ref = self.db.collection(self.articles_collection).document(article.id)
            doc_ref.set(article_data)

            logger.debug(f"Saved article {article.id} to Firestore")
            return True

        except Exception as e:
            logger.error(f"Error saving article {article.id}: {e}")
            return False

    async def save_articles(self, articles: List[NormalizedArticle]) -> int:
        """Save multiple articles to Firestore in batch."""
        if not articles:
            return 0

        try:
            # Prepare batch
            batch = self.db.batch()

            for article in articles:
                article_data = self._article_to_dict(article)
                doc_ref = self.db.collection(self.articles_collection).document(article.id)
                batch.set(doc_ref, article_data)

            # Commit batch
            batch.commit()

            logger.info(f"Saved {len(articles)} articles to Firestore")
            return len(articles)

        except Exception as e:
            logger.error(f"Error saving articles: {e}")
            return 0

    async def get_article(self, article_id: str) -> Optional[NormalizedArticle]:
        """Get a single article from Firestore."""
        try:
            doc_ref = self.db.collection(self.articles_collection).document(article_id)
            doc = doc_ref.get()

            if doc.exists:
                article_data = doc.to_dict()
                return self._dict_to_article(article_data)

            return None

        except Exception as e:
            logger.error(f"Error getting article {article_id}: {e}")
            return None

    async def get_articles(
        self,
        limit: int = 100,
        offset: int = 0,
        source_id: str = None,
        content_type: str = None,
        language: str = None,
        date_from: datetime = None,
        date_to: datetime = None,
        order_by: str = "published_at",
        order_direction: str = "desc",
    ) -> List[NormalizedArticle]:
        """Get articles from Firestore with filtering and pagination."""
        try:
            query = self.db.collection(self.articles_collection)

            # Apply filters
            if source_id:
                query = query.where("ingestion_metadata.source_id", "==", source_id)

            if content_type:
                query = query.where("content_type", "==", content_type)

            if language:
                query = query.where("language", "==", language)

            if date_from:
                query = query.where("published_at", ">=", date_from)

            if date_to:
                query = query.where("published_at", "<=", date_to)

            # Apply ordering
            if order_direction == "desc":
                query = query.order_by(order_by, direction=firestore.Query.DESCENDING)
            else:
                query = query.order_by(order_by, direction=firestore.Query.ASCENDING)

            # Apply pagination
            if offset > 0:
                query = query.offset(offset)

            query = query.limit(limit)

            # Execute query
            docs = query.stream()

            articles = []
            for doc in docs:
                try:
                    article_data = doc.to_dict()
                    article = self._dict_to_article(article_data)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing article {doc.id}: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Error getting articles: {e}")
            return []

    async def search_articles(
        self,
        query_text: str,
        limit: int = 100,
        source_id: str = None,
        content_type: str = None,
        language: str = None,
    ) -> List[NormalizedArticle]:
        """Search articles by text content."""
        try:
            # Note: Firestore doesn't support full-text search natively
            # This is a simple implementation that searches in title and content
            # For production, consider using Algolia, Elasticsearch, or similar

            query = self.db.collection(self.articles_collection)

            # Apply filters
            if source_id:
                query = query.where("ingestion_metadata.source_id", "==", source_id)

            if content_type:
                query = query.where("content_type", "==", content_type)

            if language:
                query = query.where("language", "==", language)

            # Get all matching documents
            docs = query.stream()

            articles = []
            for doc in docs:
                try:
                    article_data = doc.to_dict()
                    article = self._dict_to_article(article_data)

                    if article and self._matches_search_query(article, query_text):
                        articles.append(article)

                        if len(articles) >= limit:
                            break

                except Exception as e:
                    logger.warning(f"Error parsing article {doc.id}: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Error searching articles: {e}")
            return []

    def _matches_search_query(self, article: NormalizedArticle, query_text: str) -> bool:
        """Check if article matches search query."""
        query_lower = query_text.lower()

        # Search in title
        if article.title and query_lower in article.title.lower():
            return True

        # Search in content
        if article.content and query_lower in article.content.lower():
            return True

        # Search in summary
        if article.summary and query_lower in article.summary.lower():
            return True

        # Search in tags
        if article.tags:
            for tag in article.tags:
                if query_lower in tag.lower():
                    return True

        return False

    async def get_duplicate_articles(
        self, article: NormalizedArticle, threshold: float = 0.8
    ) -> List[NormalizedArticle]:
        """Get articles that might be duplicates of the given article."""
        try:
            # Get articles from the same source and similar time period
            date_from = article.published_at - timedelta(days=7)
            date_to = article.published_at + timedelta(days=7)

            query = self.db.collection(self.articles_collection)
            query = query.where(
                "ingestion_metadata.source_id",
                "==",
                article.ingestion_metadata.get("source_id", ""),
            )
            query = query.where("published_at", ">=", date_from)
            query = query.where("published_at", "<=", date_to)
            query = query.where("content_hash", "!=", article.content_hash)  # Exclude exact matches

            docs = query.stream()

            potential_duplicates = []
            for doc in docs:
                try:
                    article_data = doc.to_dict()
                    candidate_article = self._dict_to_article(article_data)

                    if candidate_article:
                        potential_duplicates.append(candidate_article)

                except Exception as e:
                    logger.warning(f"Error parsing article {doc.id}: {e}")
                    continue

            return potential_duplicates

        except Exception as e:
            logger.error(f"Error getting duplicate articles: {e}")
            return []

    async def save_source_config(self, source_config: SourceConfig) -> bool:
        """Save source configuration to Firestore."""
        try:
            source_data = self._source_config_to_dict(source_config)

            doc_ref = self.db.collection(self.sources_collection).document(source_config.id)
            doc_ref.set(source_data)

            logger.debug(f"Saved source config {source_config.id} to Firestore")
            return True

        except Exception as e:
            logger.error(f"Error saving source config {source_config.id}: {e}")
            return False

    async def get_source_config(self, source_id: str) -> Optional[SourceConfig]:
        """Get source configuration from Firestore."""
        try:
            doc_ref = self.db.collection(self.sources_collection).document(source_id)
            doc = doc_ref.get()

            if doc.exists:
                source_data = doc.to_dict()
                return self._dict_to_source_config(source_data)

            return None

        except Exception as e:
            logger.error(f"Error getting source config {source_id}: {e}")
            return None

    async def get_all_source_configs(self) -> List[SourceConfig]:
        """Get all source configurations from Firestore."""
        try:
            docs = self.db.collection(self.sources_collection).stream()

            source_configs = []
            for doc in docs:
                try:
                    source_data = doc.to_dict()
                    source_config = self._dict_to_source_config(source_data)
                    if source_config:
                        source_configs.append(source_config)
                except Exception as e:
                    logger.warning(f"Error parsing source config {doc.id}: {e}")
                    continue

            return source_configs

        except Exception as e:
            logger.error(f"Error getting source configs: {e}")
            return []

    async def save_processing_batch(self, batch: ProcessingBatch) -> bool:
        """Save processing batch to Firestore."""
        try:
            batch_data = self._batch_to_dict(batch)

            doc_ref = self.db.collection(self.metadata_collection).document(
                f"batch_{batch.batch_id}"
            )
            doc_ref.set(batch_data)

            logger.debug(f"Saved batch {batch.batch_id} to Firestore")
            return True

        except Exception as e:
            logger.error(f"Error saving batch {batch.batch_id}: {e}")
            return False

    async def get_processing_batch(self, batch_id: str) -> Optional[ProcessingBatch]:
        """Get processing batch from Firestore."""
        try:
            doc_ref = self.db.collection(self.metadata_collection).document(f"batch_{batch_id}")
            doc = doc_ref.get()

            if doc.exists:
                batch_data = doc.to_dict()
                return self._dict_to_batch(batch_data)

            return None

        except Exception as e:
            logger.error(f"Error getting batch {batch_id}: {e}")
            return None

    async def save_metrics(self, metrics: ContentMetrics) -> bool:
        """Save content metrics to Firestore."""
        try:
            metrics_data = self._metrics_to_dict(metrics)

            doc_ref = self.db.collection(self.metadata_collection).document(
                f"metrics_{metrics.source_id}_{metrics.date}"
            )
            doc_ref.set(metrics_data)

            logger.debug(f"Saved metrics for source {metrics.source_id} to Firestore")
            return True

        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return False

    async def get_metrics(self, source_id: str, date: datetime = None) -> Optional[ContentMetrics]:
        """Get content metrics from Firestore."""
        try:
            if date is None:
                date = datetime.utcnow().date()

            doc_ref = self.db.collection(self.metadata_collection).document(
                f"metrics_{source_id}_{date}"
            )
            doc = doc_ref.get()

            if doc.exists:
                metrics_data = doc.to_dict()
                return self._dict_to_metrics(metrics_data)

            return None

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return None

    def _article_to_dict(self, article: NormalizedArticle) -> Dict[str, Any]:
        """Convert article to Firestore-compatible dictionary."""
        return {
            "id": article.id,
            "url": article.url,
            "canonical_url": article.canonical_url,
            "title": article.title,
            "summary": article.summary,
            "content": article.content,
            "author": article.author,
            "byline": article.byline,
            "source": article.source,
            "source_url": article.source_url,
            "published_at": article.published_at,
            "updated_at": article.updated_at,
            "language": article.language,
            "image_url": article.image_url,
            "tags": article.tags,
            "word_count": article.word_count,
            "reading_time": article.reading_time,
            "content_hash": article.content_hash,
            "content_type": article.content_type.value,
            "raw_data": article.raw_data,
            "ingestion_metadata": article.ingestion_metadata,
            "processing_status": article.processing_status.value,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

    def _dict_to_article(self, data: Dict[str, Any]) -> Optional[NormalizedArticle]:
        """Convert Firestore dictionary to article."""
        try:
            from src.models.content import ContentType, ProcessingStatus

            # Convert string enums back to enum objects
            content_type = ContentType(data.get("content_type", "article"))
            processing_status = ProcessingStatus(data.get("processing_status", "pending"))

            # Convert datetime strings back to datetime objects
            published_at = data.get("published_at")
            if isinstance(published_at, str):
                published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00"))

            updated_at = data.get("updated_at")
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

            return NormalizedArticle(
                id=data.get("id", ""),
                url=data.get("url", ""),
                canonical_url=data.get("canonical_url"),
                title=data.get("title", ""),
                summary=data.get("summary"),
                content=data.get("content"),
                author=data.get("author"),
                byline=data.get("byline"),
                source=data.get("source", ""),
                source_url=data.get("source_url", ""),
                published_at=published_at or datetime.utcnow(),
                updated_at=updated_at,
                language=data.get("language", "en"),
                image_url=data.get("image_url"),
                tags=data.get("tags", []),
                word_count=data.get("word_count", 0),
                reading_time=data.get("reading_time", 0),
                content_hash=data.get("content_hash", ""),
                content_type=content_type,
                raw_data=data.get("raw_data", {}),
                ingestion_metadata=data.get("ingestion_metadata", {}),
                processing_status=processing_status,
            )

        except Exception as e:
            logger.error(f"Error converting dict to article: {e}")
            return None

    def _source_config_to_dict(self, source_config: SourceConfig) -> Dict[str, Any]:
        """Convert source config to Firestore-compatible dictionary."""
        return {
            "id": source_config.id,
            "name": source_config.name,
            "type": source_config.type.value,
            "url": source_config.url,
            "enabled": source_config.enabled,
            "priority": source_config.priority,
            "rate_limit": source_config.rate_limit,
            "timeout": source_config.timeout,
            "retry_attempts": source_config.retry_attempts,
            "backoff_factor": source_config.backoff_factor,
            "user_agent": source_config.user_agent,
            "headers": source_config.headers,
            "auth": source_config.auth,
            "filters": source_config.filters,
            "last_checked": source_config.last_checked,
            "last_success": source_config.last_success,
            "error_count": source_config.error_count,
            "success_count": source_config.success_count,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

    def _dict_to_source_config(self, data: Dict[str, Any]) -> Optional[SourceConfig]:
        """Convert Firestore dictionary to source config."""
        try:
            from src.models.content import SourceType

            # Convert string enum back to enum object
            source_type = SourceType(data.get("type", "rss_feed"))

            # Convert datetime strings back to datetime objects
            last_checked = data.get("last_checked")
            if isinstance(last_checked, str):
                last_checked = datetime.fromisoformat(last_checked.replace("Z", "+00:00"))

            last_success = data.get("last_success")
            if isinstance(last_success, str):
                last_success = datetime.fromisoformat(last_success.replace("Z", "+00:00"))

            return SourceConfig(
                id=data.get("id", ""),
                name=data.get("name", ""),
                type=source_type,
                url=data.get("url", ""),
                enabled=data.get("enabled", True),
                priority=data.get("priority", 1),
                rate_limit=data.get("rate_limit", 60),
                timeout=data.get("timeout", 30),
                retry_attempts=data.get("retry_attempts", 3),
                backoff_factor=data.get("backoff_factor", 2.0),
                user_agent=data.get("user_agent"),
                headers=data.get("headers", {}),
                auth=data.get("auth"),
                filters=data.get("filters", {}),
                last_checked=last_checked,
                last_success=last_success,
                error_count=data.get("error_count", 0),
                success_count=data.get("success_count", 0),
            )

        except Exception as e:
            logger.error(f"Error converting dict to source config: {e}")
            return None

    def _batch_to_dict(self, batch: ProcessingBatch) -> Dict[str, Any]:
        """Convert batch to Firestore-compatible dictionary."""
        return {
            "batch_id": batch.batch_id,
            "source_id": batch.source_id,
            "total_count": batch.total_count,
            "processed_count": batch.processed_count,
            "failed_count": batch.failed_count,
            "duplicate_count": batch.duplicate_count,
            "status": batch.status.value,
            "created_at": batch.created_at,
            "started_at": batch.started_at,
            "completed_at": batch.completed_at,
            "error_message": batch.error_message,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

    def _dict_to_batch(self, data: Dict[str, Any]) -> Optional[ProcessingBatch]:
        """Convert Firestore dictionary to batch."""
        try:
            from src.models.content import ProcessingStatus

            # Convert string enum back to enum object
            status = ProcessingStatus(data.get("status", "pending"))

            # Convert datetime strings back to datetime objects
            created_at = data.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

            started_at = data.get("started_at")
            if isinstance(started_at, str):
                started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))

            completed_at = data.get("completed_at")
            if isinstance(completed_at, str):
                completed_at = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))

            return ProcessingBatch(
                batch_id=data.get("batch_id", ""),
                source_id=data.get("source_id", ""),
                articles=[],  # Articles are not stored in batch document
                total_count=data.get("total_count", 0),
                processed_count=data.get("processed_count", 0),
                failed_count=data.get("failed_count", 0),
                duplicate_count=data.get("duplicate_count", 0),
                status=status,
                created_at=created_at or datetime.utcnow(),
                started_at=started_at,
                completed_at=completed_at,
                error_message=data.get("error_message"),
            )

        except Exception as e:
            logger.error(f"Error converting dict to batch: {e}")
            return None

    def _metrics_to_dict(self, metrics: ContentMetrics) -> Dict[str, Any]:
        """Convert metrics to Firestore-compatible dictionary."""
        return {
            "source_id": metrics.source_id,
            "date": metrics.date,
            "total_articles": metrics.total_articles,
            "successful_articles": metrics.successful_articles,
            "failed_articles": metrics.failed_articles,
            "duplicate_articles": metrics.duplicate_articles,
            "average_processing_time_ms": metrics.average_processing_time_ms,
            "average_word_count": metrics.average_word_count,
            "language_distribution": metrics.language_distribution,
            "content_type_distribution": metrics.content_type_distribution,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

    def _dict_to_metrics(self, data: Dict[str, Any]) -> Optional[ContentMetrics]:
        """Convert Firestore dictionary to metrics."""
        try:
            # Convert date string back to date object
            date = data.get("date")
            if isinstance(date, str):
                date = datetime.fromisoformat(date).date()

            return ContentMetrics(
                source_id=data.get("source_id", ""),
                date=date or datetime.utcnow().date(),
                total_articles=data.get("total_articles", 0),
                successful_articles=data.get("successful_articles", 0),
                failed_articles=data.get("failed_articles", 0),
                duplicate_articles=data.get("duplicate_articles", 0),
                average_processing_time_ms=data.get("average_processing_time_ms", 0.0),
                average_word_count=data.get("average_word_count", 0.0),
                language_distribution=data.get("language_distribution", {}),
                content_type_distribution=data.get("content_type_distribution", {}),
            )

        except Exception as e:
            logger.error(f"Error converting dict to metrics: {e}")
            return None
