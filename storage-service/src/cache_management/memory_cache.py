"""
Memory Cache - High-performance in-memory caching with LRU eviction
"""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with metadata"""

    def __init__(self, key: str, value: Any, ttl: int = 0, tags: List[str] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.ttl = ttl
        self.tags = tags or []
        self.access_count = 0

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl <= 0:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self):
        """Update access time and count"""
        self.last_accessed = time.time()
        self.access_count += 1


class MemoryCache:
    """High-performance in-memory cache with LRU eviction"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._tag_index: Dict[str, set] = {}  # tag -> set of keys
        self._lock = asyncio.Lock()
        self._initialized = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "expired": 0}

    async def initialize(self):
        """Initialize memory cache"""
        if self._initialized:
            return

        logger.info("Initializing Memory Cache...")

        try:
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_entries())

            self._initialized = True
            logger.info(f"Memory Cache initialized with max_size={self.max_size}")

        except Exception as e:
            logger.error(f"Failed to initialize Memory Cache: {e}")
            raise

    async def cleanup(self):
        """Cleanup memory cache"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._initialized = False
        logger.info("Memory Cache cleanup complete")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._initialized:
            return None

        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check if expired
                if entry.is_expired():
                    del self._cache[key]
                    self._remove_from_tag_index(entry)
                    self._stats["expired"] += 1
                    self._stats["misses"] += 1
                    return None

                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.touch()
                self._stats["hits"] += 1

                return entry.value
            else:
                self._stats["misses"] += 1
                return None

    async def set(self, key: str, value: Any, ttl: int = None, tags: List[str] = None) -> bool:
        """Set value in cache"""
        if not self._initialized:
            return False

        try:
            async with self._lock:
                # Remove existing entry if present
                if key in self._cache:
                    old_entry = self._cache[key]
                    self._remove_from_tag_index(old_entry)
                    del self._cache[key]

                # Create new entry
                entry_ttl = ttl if ttl is not None else self.default_ttl
                entry = CacheEntry(key, value, entry_ttl, tags or [])

                # Add to cache
                self._cache[key] = entry

                # Add to tag index
                self._add_to_tag_index(entry)

                # Evict if necessary
                await self._evict_if_needed()

                return True

        except Exception as e:
            logger.error(f"Failed to set cache entry {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        if not self._initialized:
            return False

        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._remove_from_tag_index(entry)
                del self._cache[key]
                return True
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete entries matching pattern"""
        if not self._initialized:
            return 0

        import fnmatch

        deleted_count = 0
        async with self._lock:
            keys_to_delete = [key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)]

            for key in keys_to_delete:
                entry = self._cache[key]
                self._remove_from_tag_index(entry)
                del self._cache[key]
                deleted_count += 1

        return deleted_count

    async def delete_by_tag(self, tag: str) -> int:
        """Delete entries by tag"""
        if not self._initialized:
            return 0

        deleted_count = 0
        async with self._lock:
            if tag in self._tag_index:
                keys_to_delete = list(self._tag_index[tag])

                for key in keys_to_delete:
                    if key in self._cache:
                        entry = self._cache[key]
                        self._remove_from_tag_index(entry)
                        del self._cache[key]
                        deleted_count += 1

                del self._tag_index[tag]

        return deleted_count

    async def clear(self) -> int:
        """Clear all cache entries"""
        if not self._initialized:
            return 0

        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._tag_index.clear()
            return count

    async def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._initialized:
            return {}

        async with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self._stats["evictions"],
                "expired": self._stats["expired"],
                "memory_usage": self._estimate_memory_usage(),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _cleanup_expired_entries(self):
        """Background task to clean up expired entries"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute

                if not self._initialized:
                    break

                async with self._lock:
                    expired_keys = []
                    for key, entry in self._cache.items():
                        if entry.is_expired():
                            expired_keys.append(key)

                    for key in expired_keys:
                        entry = self._cache[key]
                        self._remove_from_tag_index(entry)
                        del self._cache[key]
                        self._stats["expired"] += 1

                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup task: {e}")

    async def _evict_if_needed(self):
        """Evict entries if cache is full"""
        while len(self._cache) > self.max_size:
            # Remove least recently used entry
            key, entry = self._cache.popitem(last=False)
            self._remove_from_tag_index(entry)
            self._stats["evictions"] += 1

    def _add_to_tag_index(self, entry: CacheEntry):
        """Add entry to tag index"""
        for tag in entry.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(entry.key)

    def _remove_from_tag_index(self, entry: CacheEntry):
        """Remove entry from tag index"""
        for tag in entry.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(entry.key)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]

    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        total_size = 0

        for entry in self._cache.values():
            # Rough estimation
            key_size = len(entry.key.encode("utf-8"))
            value_size = self._estimate_object_size(entry.value)
            metadata_size = 200  # Rough estimate for entry metadata

            total_size += key_size + value_size + metadata_size

        return total_size

    def _estimate_object_size(self, obj: Any) -> int:
        """Estimate size of object in bytes"""
        try:
            import sys

            return sys.getsizeof(obj)
        except Exception:
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj.encode("utf-8"))
            elif isinstance(obj, (int, float)):
                return 8
            elif isinstance(obj, dict):
                return sum(
                    self._estimate_object_size(k) + self._estimate_object_size(v)
                    for k, v in obj.items()
                )
            elif isinstance(obj, list):
                return sum(self._estimate_object_size(item) for item in obj)
            else:
                return 100  # Default estimate
