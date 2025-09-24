"""
Cache service using GCP Cloud Storage for content caching.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from google.cloud import firestore, storage
from loguru import logger

from ..exceptions import CacheError
from ..models.content import ExtractedContent


class CacheService:
    """Cache service using GCP Cloud Storage and Firestore."""

    def __init__(
        self,
        project_id: str,
        bucket_name: str,
        collection_name: str = "content_cache",
        ttl_hours: int = 24,
    ):
        """Initialize cache service."""
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.collection_name = collection_name
        self.ttl_hours = ttl_hours

        # Initialize GCP clients
        self.storage_client = storage.Client(project=project_id)
        self.firestore_client = firestore.Client(project=project_id)

        # Get bucket and collection references
        self.bucket = self.storage_client.bucket(bucket_name)
        self.cache_collection = self.firestore_client.collection(collection_name)

    async def get_content(self, url: str) -> Optional[ExtractedContent]:
        """
        Get cached content by URL.

        Args:
            url: Content URL

        Returns:
            Cached ExtractedContent or None if not found/expired
        """
        try:
            logger.info(f"Checking cache for URL: {url}")

            # Generate cache key
            cache_key = self._generate_cache_key(url)

            # Check Firestore for cache metadata
            doc_ref = self.cache_collection.document(cache_key)
            doc = doc_ref.get()

            if not doc.exists:
                logger.info(f"Cache miss for URL: {url}")
                return None

            cache_data = doc.to_dict()

            # Check if cache is expired
            if self._is_cache_expired(cache_data):
                logger.info(f"Cache expired for URL: {url}")
                await self._delete_cache_entry(cache_key)
                return None

            # Get content from Cloud Storage
            content_data = await self._get_content_from_storage(cache_key)
            if not content_data:
                logger.warning(f"Content not found in storage for URL: {url}")
                await self._delete_cache_entry(cache_key)
                return None

            # Deserialize content
            content = ExtractedContent.model_validate(content_data)

            logger.info(f"Cache hit for URL: {url}")
            return content

        except Exception as e:
            logger.error(f"Cache retrieval failed for {url}: {str(e)}")
            return None

    async def cache_content(self, url: str, content: ExtractedContent) -> bool:
        """
        Cache content by URL.

        Args:
            url: Content URL
            content: ExtractedContent to cache

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Caching content for URL: {url}")

            # Generate cache key
            cache_key = self._generate_cache_key(url)

            # Serialize content
            content_data = content.model_dump()

            # Store content in Cloud Storage
            await self._store_content_in_storage(cache_key, content_data)

            # Store cache metadata in Firestore
            cache_metadata = {
                "url": url,
                "cache_key": cache_key,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(hours=self.ttl_hours),
                "content_size": len(json.dumps(content_data)),
                "content_type": content.content_type,
                "word_count": content.word_count,
                "quality_score": content.quality_metrics.overall_quality,
            }

            doc_ref = self.cache_collection.document(cache_key)
            doc_ref.set(cache_metadata)

            logger.info(f"Content cached successfully for URL: {url}")
            return True

        except Exception as e:
            logger.error(f"Cache storage failed for {url}: {str(e)}")
            return False

    async def delete_content(self, url: str) -> bool:
        """
        Delete cached content by URL.

        Args:
            url: Content URL

        Returns:
            True if successful, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(url)
            return await self._delete_cache_entry(cache_key)
        except Exception as e:
            logger.error(f"Cache deletion failed for {url}: {str(e)}")
            return False

    async def clear_expired_cache(self) -> int:
        """
        Clear expired cache entries.

        Returns:
            Number of entries cleared
        """
        try:
            logger.info("Clearing expired cache entries")

            # Query expired entries
            now = datetime.utcnow()
            expired_docs = self.cache_collection.where("expires_at", "<", now).stream()

            cleared_count = 0
            for doc in expired_docs:
                cache_key = doc.id
                if await self._delete_cache_entry(cache_key):
                    cleared_count += 1

            logger.info(f"Cleared {cleared_count} expired cache entries")
            return cleared_count

        except Exception as e:
            logger.error(f"Cache cleanup failed: {str(e)}")
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
    """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            # Get total cache entries
            total_docs = self.cache_collection.stream()
            total_entries = sum(1 for _ in total_docs)

            # Get expired entries
            now = datetime.utcnow()
            expired_docs = self.cache_collection.where("expires_at", "<", now).stream()
            expired_entries = sum(1 for _ in expired_docs)

            # Get storage usage
            storage_usage = await self._get_storage_usage()

            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "active_entries": total_entries - expired_entries,
                "storage_usage_bytes": storage_usage,
                "storage_usage_mb": storage_usage / (1024 * 1024),
                "ttl_hours": self.ttl_hours,
            }

        except Exception as e:
            logger.error(f"Cache stats retrieval failed: {str(e)}")
            return {
                "total_entries": 0,
                "expired_entries": 0,
                "active_entries": 0,
                "storage_usage_bytes": 0,
                "storage_usage_mb": 0,
                "ttl_hours": self.ttl_hours,
            }

    def _generate_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _is_cache_expired(self, cache_data: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        try:
            expires_at = cache_data.get("expires_at")
            if not expires_at:
                return True

            if isinstance(expires_at, str):
                expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))

            return datetime.utcnow() > expires_at
        except Exception as e:
            logger.warning(f"Cache expiration check failed: {str(e)}")
            return True

    async def _get_content_from_storage(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get content from Cloud Storage."""
        try:
            blob_name = f"content/{cache_key}.json"
            blob = self.bucket.blob(blob_name)

            if not blob.exists():
                return None

            content_json = blob.download_as_text()
            return json.loads(content_json)

        except Exception as e:
            logger.warning(f"Content retrieval from storage failed: {str(e)}")
            return None

    async def _store_content_in_storage(self, cache_key: str, content_data: Dict[str, Any]) -> None:
        """Store content in Cloud Storage."""
        try:
            blob_name = f"content/{cache_key}.json"
            blob = self.bucket.blob(blob_name)

            content_json = json.dumps(content_data, default=str)
            blob.upload_from_string(content_json, content_type="application/json")

        except Exception as e:
            logger.error(f"Content storage failed: {str(e)}")
            raise CacheError(f"Content storage failed: {str(e)}")

    async def _delete_cache_entry(self, cache_key: str) -> bool:
        """Delete cache entry from both Firestore and Cloud Storage."""
        try:
            # Delete from Firestore
            doc_ref = self.cache_collection.document(cache_key)
            doc_ref.delete()

            # Delete from Cloud Storage
            blob_name = f"content/{cache_key}.json"
            blob = self.bucket.blob(blob_name)
            if blob.exists():
                blob.delete()

            return True

        except Exception as e:
            logger.warning(f"Cache entry deletion failed: {str(e)}")
            return False

    async def _get_storage_usage(self) -> int:
        """Get total storage usage in bytes."""
        try:
            total_size = 0
            blobs = self.bucket.list_blobs(prefix="content/")

            for blob in blobs:
                if blob.size:
                    total_size += blob.size

            return total_size

        except Exception as e:
            logger.warning(f"Storage usage calculation failed: {str(e)}")
            return 0

    async def search_cached_content(
        self,
        query: str = None,
        content_type: str = None,
        min_quality_score: float = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search cached content with filters.

        Args:
            query: Search query
            content_type: Filter by content type
            min_quality_score: Minimum quality score
            limit: Maximum results

        Returns:
            List of cached content metadata
        """
        try:
            # Build Firestore query
            firestore_query = self.cache_collection

            if content_type:
                firestore_query = firestore_query.where("content_type", "==", content_type)

            if min_quality_score is not None:
                firestore_query = firestore_query.where("quality_score", ">=", min_quality_score)

            # Execute query
            docs = firestore_query.limit(limit).stream()

            results = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                results.append(data)

            # Filter by query if provided
            if query:
                query_lower = query.lower()
                results = [result for result in results if query_lower in result.get("url", "").lower()]

            return results

        except Exception as e:
            logger.error(f"Cache search failed: {str(e)}")
            return []

    async def warm_cache(self, urls: List[str]) -> Dict[str, bool]:
        """
        Warm cache by pre-loading content for given URLs.

        Args:
            urls: List of URLs to warm cache for

        Returns:
            Dictionary mapping URLs to success status
        """
        try:
            results = {}

            for url in urls:
                try:
                    # Check if already cached
                    cached_content = await self.get_content(url)
                    if cached_content:
                        results[url] = True
                        continue

                    # Here you would typically trigger content extraction
                    # For now, just mark as not cached
                    results[url] = False

                except Exception as e:
                    logger.warning(f"Cache warming failed for {url}: {str(e)}")
                    results[url] = False

            return results

        except Exception as e:
            logger.error(f"Cache warming failed: {str(e)}")
            return {url: False for url in urls}
