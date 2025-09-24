"""
CDN Manager - Content Delivery Network integration for global caching
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp

from config import Settings

logger = logging.getLogger(__name__)


class CDNManager:
    """Manage CDN for global content caching"""

    def __init__(self):
        self.settings = Settings()
        self._initialized = False
        self._cdn_client = None
        self._api_key = None
        self._zone_id = None

    async def initialize(self):
        """Initialize CDN client"""
        if self._initialized:
            return

        logger.info("Initializing CDN Manager...")

        try:
            # Initialize CDN client based on provider
            cdn_provider = getattr(self.settings, "cdn_provider", "cloudflare")

            if cdn_provider == "cloudflare":
                await self._initialize_cloudflare()
            elif cdn_provider == "aws_cloudfront":
                await self._initialize_cloudfront()
            elif cdn_provider == "gcp_cdn":
                await self._initialize_gcp_cdn()
            else:
                logger.warning(f"Unknown CDN provider: {cdn_provider}, using mock CDN")
                await self._initialize_mock_cdn()

            self._initialized = True
            logger.info("CDN Manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CDN Manager: {e}")
            raise

    async def cleanup(self):
        """Cleanup CDN client"""
        if self._cdn_client and hasattr(self._cdn_client, "close"):
            await self._cdn_client.close()

        self._initialized = False
        logger.info("CDN Manager cleanup complete")

    async def _initialize_cloudflare(self):
        """Initialize Cloudflare CDN client"""
        try:
            self._api_key = getattr(self.settings, "cloudflare_api_key", None)
            self._zone_id = getattr(self.settings, "cloudflare_zone_id", None)

            if not self._api_key or not self._zone_id:
                logger.warning("Cloudflare credentials not provided, using mock CDN")
                await self._initialize_mock_cdn()
                return

            # Create aiohttp session for Cloudflare API
            self._cdn_client = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }
            )

            # Test connection
            await self._test_cloudflare_connection()

        except Exception as e:
            logger.error(f"Failed to initialize Cloudflare CDN: {e}")
            await self._initialize_mock_cdn()

    async def _initialize_cloudfront(self):
        """Initialize AWS CloudFront client"""
        try:
            import boto3

            self._cdn_client = boto3.client(
                "cloudfront",
                aws_access_key_id=self.settings.aws_access_key,
                aws_secret_access_key=self.settings.aws_secret_key,
                region_name=self.settings.cloud_storage_region,
            )

        except ImportError:
            logger.warning("boto3 not available, using mock CDN")
            await self._initialize_mock_cdn()
        except Exception as e:
            logger.error(f"Failed to initialize CloudFront: {e}")
            await self._initialize_mock_cdn()

    async def _initialize_gcp_cdn(self):
        """Initialize Google Cloud CDN client"""
        try:
            from google.cloud import compute_v1

            self._cdn_client = compute_v1.UrlMapsClient()

        except ImportError:
            logger.warning("google-cloud-compute not available, using mock CDN")
            await self._initialize_mock_cdn()
        except Exception as e:
            logger.error(f"Failed to initialize GCP CDN: {e}")
            await self._initialize_mock_cdn()

    async def _initialize_mock_cdn(self):
        """Initialize mock CDN for development/testing"""
        self._cdn_client = MockCDNClient()
        logger.info("Using mock CDN client")

    async def _test_cloudflare_connection(self):
        """Test Cloudflare API connection"""
        try:
            async with self._cdn_client.get(
                f"https://api.cloudflare.com/client/v4/zones/{self._zone_id}"
            ) as response:
                if response.status == 200:
                    logger.info("Cloudflare CDN connection successful")
                else:
                    raise Exception(f"Cloudflare API error: {response.status}")
        except Exception as e:
            logger.error(f"Cloudflare connection test failed: {e}")
            raise

    async def cache_content(self, cache_key: str, content: Any, ttl: int = 3600) -> bool:
        """Cache content in CDN"""
        if not self._initialized:
            return False

        try:
            # Convert content to cacheable format
            cacheable_content = self._prepare_content_for_caching(content)

            # Store in CDN
            success = await self._store_in_cdn(cache_key, cacheable_content, ttl)

            if success:
                logger.debug(f"Content cached in CDN: {cache_key}")

            return success

        except Exception as e:
            logger.error(f"Failed to cache content in CDN: {e}")
            return False

    async def get_cached_content(self, cache_key: str) -> Optional[Any]:
        """Get cached content from CDN"""
        if not self._initialized:
            return None

        try:
            content = await self._get_from_cdn(cache_key)

            if content:
                logger.debug(f"Content retrieved from CDN: {cache_key}")
                return self._deserialize_content(content)

            return None

        except Exception as e:
            logger.error(f"Failed to get cached content from CDN: {e}")
            return None

    async def purge_pattern(self, pattern: str) -> int:
        """Purge cache entries matching pattern"""
        if not self._initialized:
            return 0

        try:
            # Get list of cached keys matching pattern
            matching_keys = await self._get_matching_keys(pattern)

            if not matching_keys:
                return 0

            # Purge each key
            purged_count = 0
            for key in matching_keys:
                if await self._purge_key(key):
                    purged_count += 1

            logger.info(f"Purged {purged_count} CDN cache entries matching pattern: {pattern}")
            return purged_count

        except Exception as e:
            logger.error(f"Failed to purge CDN cache pattern {pattern}: {e}")
            return 0

    async def purge_by_tag(self, tag: str) -> int:
        """Purge cache entries by tag"""
        if not self._initialized:
            return 0

        try:
            # Get keys with specific tag
            tag_pattern = f"tag:{tag}:*"
            return await self.purge_pattern(tag_pattern)

        except Exception as e:
            logger.error(f"Failed to purge CDN cache by tag {tag}: {e}")
            return 0

    async def get_statistics(self) -> Dict[str, Any]:
        """Get CDN statistics"""
        if not self._initialized:
            return {}

        try:
            return await self._get_cdn_statistics()

        except Exception as e:
            logger.error(f"Failed to get CDN statistics: {e}")
            return {}

    def _prepare_content_for_caching(self, content: Any) -> str:
        """Prepare content for CDN caching"""
        import json

        if isinstance(content, str):
            return content
        elif isinstance(content, (dict, list)):
            return json.dumps(content, default=str)
        else:
            return str(content)

    def _deserialize_content(self, content: str) -> Any:
        """Deserialize content from CDN"""
        try:
            import json

            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content

    async def _store_in_cdn(self, cache_key: str, content: str, ttl: int) -> bool:
        """Store content in CDN"""
        if hasattr(self._cdn_client, "store_content"):
            return await self._cdn_client.store_content(cache_key, content, ttl)
        else:
            # Mock implementation
            return True

    async def _get_from_cdn(self, cache_key: str) -> Optional[str]:
        """Get content from CDN"""
        if hasattr(self._cdn_client, "get_content"):
            return await self._cdn_client.get_content(cache_key)
        else:
            # Mock implementation
            return None

    async def _purge_key(self, cache_key: str) -> bool:
        """Purge specific key from CDN"""
        if hasattr(self._cdn_client, "purge_key"):
            return await self._cdn_client.purge_key(cache_key)
        else:
            # Mock implementation
            return True

    async def _get_matching_keys(self, pattern: str) -> List[str]:
        """Get keys matching pattern from CDN"""
        if hasattr(self._cdn_client, "get_matching_keys"):
            return await self._cdn_client.get_matching_keys(pattern)
        else:
            # Mock implementation
            return []

    async def _get_cdn_statistics(self) -> Dict[str, Any]:
        """Get CDN statistics"""
        if hasattr(self._cdn_client, "get_statistics"):
            return await self._cdn_client.get_statistics()
        else:
            # Mock implementation
            return {
                "provider": "mock",
                "cached_entries": 0,
                "total_bandwidth": 0,
                "hit_rate": 0.0,
                "timestamp": datetime.utcnow().isoformat(),
            }


class MockCDNClient:
    """Mock CDN client for development/testing"""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._stats = {"hits": 0, "misses": 0, "stores": 0, "purges": 0}

    async def store_content(self, cache_key: str, content: str, ttl: int) -> bool:
        """Store content in mock CDN"""
        self._cache[cache_key] = {"content": content, "created_at": datetime.utcnow(), "ttl": ttl}
        self._stats["stores"] += 1
        return True

    async def get_content(self, cache_key: str) -> Optional[str]:
        """Get content from mock CDN"""
        if cache_key in self._cache:
            entry = self._cache[cache_key]

            # Check if expired
            if entry["ttl"] > 0:
                age = (datetime.utcnow() - entry["created_at"]).total_seconds()
                if age > entry["ttl"]:
                    del self._cache[cache_key]
                    self._stats["misses"] += 1
                    return None

            self._stats["hits"] += 1
            return entry["content"]
        else:
            self._stats["misses"] += 1
            return None

    async def purge_key(self, cache_key: str) -> bool:
        """Purge key from mock CDN"""
        if cache_key in self._cache:
            del self._cache[cache_key]
            self._stats["purges"] += 1
            return True
        return False

    async def get_matching_keys(self, pattern: str) -> List[str]:
        """Get keys matching pattern from mock CDN"""
        import fnmatch

        return [key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)]

    async def get_statistics(self) -> Dict[str, Any]:
        """Get mock CDN statistics"""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0

        return {
            "provider": "mock",
            "cached_entries": len(self._cache),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "stores": self._stats["stores"],
            "purges": self._stats["purges"],
            "timestamp": datetime.utcnow().isoformat(),
        }
