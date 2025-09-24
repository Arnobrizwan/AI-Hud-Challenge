"""
Data models for Storage, Indexing & Retrieval Service
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field


class StorageType(str, Enum):
    POSTGRESQL = "postgresql"
    ELASTICSEARCH = "elasticsearch"
    REDIS = "redis"
    VECTOR_STORE = "vector_store"
    MEDIA_STORAGE = "media_storage"
    TIMESERIES = "timeseries"


class SearchMethod(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    HYBRID = "hybrid"


class CacheLevel(str, Enum):
    MEMORY = "memory"
    REDIS = "redis"
    CDN = "cdn"


class GDPRRequestType(str, Enum):
    DATA_EXPORT = "data_export"
    DATA_DELETION = "data_deletion"
    DATA_RECTIFICATION = "data_rectification"


class RetentionPolicyType(str, Enum):
    DELETE_OLD_DATA = "delete_old_data"
    ARCHIVE_DATA = "archive_data"
    ANONYMIZE_DATA = "anonymize_data"


# Core data models
class Article(BaseModel):
    id: str
    title: str
    content: str
    summary: Optional[str] = None
    author: Optional[str] = None
    source: str
    published_at: datetime
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    language: str = "en"
    url: str
    embeddings: Optional[Dict[str, List[float]]] = None
    media_files: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StorageResult(BaseModel):
    article_id: str
    stored_locations: List[StorageType]
    failed_stores: List[str] = Field(default_factory=list)
    storage_timestamp: datetime


class RetrievedArticle(BaseModel):
    id: str
    title: str
    content: str
    summary: Optional[str] = None
    author: Optional[str] = None
    source: str
    published_at: datetime
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    language: str
    url: str
    embeddings: Optional[Dict[str, List[float]]] = None
    media_files: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieval_timestamp: datetime
    cache_hit: bool = False
    retrieval_sources: List[StorageType] = Field(default_factory=list)


# Search models
class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    date_range: Optional[Dict[str, datetime]] = None
    categories: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    offset: int = 0
    limit: int = 20
    sort_by: Optional[str] = None
    sort_order: str = "desc"
    explain: bool = False


class SearchResultItem(BaseModel):
    article_id: str
    score: float
    title: str
    summary: Optional[str] = None
    highlights: Dict[str, List[str]] = Field(default_factory=dict)
    source: str
    published_at: datetime


class SearchResult(BaseModel):
    results: List[SearchResultItem]
    total_hits: int
    aggregations: Dict[str, Any] = Field(default_factory=dict)
    search_duration: int
    query_explanation: Optional[Dict[str, Any]] = None


class SimilaritySearchParams(BaseModel):
    query_vector: List[float]
    embedding_type: str
    top_k: int = 10
    similarity_threshold: float = 0.7
    search_method: SearchMethod = SearchMethod.HYBRID
    filters: Optional[Dict[str, Any]] = None
    category_filter: Optional[str] = None
    rerank: bool = True


class SimilarityResult(BaseModel):
    content_id: str
    similarity_score: float
    embedding_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SimilaritySearchResult(BaseModel):
    query_vector: List[float]
    results: List[SimilarityResult]
    search_method: SearchMethod
    search_duration: int
    total_candidates: int


# Vector store models
class VectorStorageResult(BaseModel):
    content_id: str
    stored_embeddings: List[str]
    storage_results: List[Dict[str, Any]] = Field(default_factory=list)


class VectorIndexConfig(BaseModel):
    index_name: str
    embedding_type: str
    dimension: int
    hnsw_m: int = 16
    ef_construction: int = 200
    similarity_threshold: float = 0.7


class IndexBuildResult(BaseModel):
    index_name: str
    build_duration: int
    index_size: int
    query_performance: float


# Cache models
class CacheConfig(BaseModel):
    use_memory_cache: bool = True
    use_redis_cache: bool = True
    use_cdn_cache: bool = False
    memory_ttl: int = 300  # 5 minutes
    redis_ttl: int = 3600  # 1 hour
    cdn_ttl: int = 86400  # 24 hours
    is_public: bool = False


class CachedData(BaseModel):
    data: Any
    cache_level: Optional[CacheLevel] = None
    cache_hit: bool = False
    ttl: Optional[int] = None


class CacheResult(BaseModel):
    cache_key: str
    cached_levels: List[CacheLevel] = Field(default_factory=list)
    cache_timestamp: datetime


class CacheInvalidationRequest(BaseModel):
    key_patterns: Optional[List[str]] = None
    cache_tags: Optional[List[str]] = None
    user_id: Optional[str] = None


class InvalidationResult(BaseModel):
    invalidated_keys: int
    invalidation_timestamp: datetime


# Data lifecycle models
class RetentionPolicy(BaseModel):
    policy_id: str
    policy_type: RetentionPolicyType
    data_type: str
    retention_period_days: int
    conditions: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class RetentionResult(BaseModel):
    policies_applied: int
    retention_results: List[Dict[str, Any]] = Field(default_factory=list)
    execution_timestamp: datetime


class GDPRRequest(BaseModel):
    request_id: str
    user_id: str
    request_type: GDPRRequestType
    corrections: Optional[Dict[str, Any]] = None
    request_timestamp: datetime = Field(default_factory=datetime.utcnow)


class GDPRResponse(BaseModel):
    request_id: str
    user_id: str
    request_type: GDPRRequestType
    status: str
    data: Optional[Dict[str, Any]] = None
    processing_timestamp: datetime


class GDPRDeletionResult(BaseModel):
    user_id: str
    deletion_results: List[Dict[str, Any]] = Field(default_factory=list)
    verification_result: Dict[str, Any] = Field(default_factory=dict)
    deletion_timestamp: datetime


# Query optimization models
class MultiStoreQuery(BaseModel):
    query_id: str
    query_type: str
    data_stores: List[StorageType]
    query_components: List[Dict[str, Any]]
    performance_requirements: Dict[str, Any] = Field(default_factory=dict)


class OptimizedQuery(BaseModel):
    original_query: MultiStoreQuery
    optimized_strategy: Dict[str, Any]
    estimated_cost: float
    estimated_duration: int


class RetrievalOptions(BaseModel):
    use_cache: bool = True
    update_cache: bool = True
    preferred_sources: Optional[List[StorageType]] = None
    performance_mode: str = "balanced"  # fast, balanced, complete


# Health and monitoring models
class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    version: str = "1.0.0"


class ServiceStatus(BaseModel):
    name: str
    status: str
    response_time: Optional[float] = None
    last_check: datetime
    error_message: Optional[str] = None


# Performance models
class PerformanceMetrics(BaseModel):
    query_count: int
    avg_response_time: float
    cache_hit_ratio: float
    error_rate: float
    throughput: float
    timestamp: datetime


class QueryPerformance(BaseModel):
    query_id: str
    execution_time: float
    data_stores_used: List[StorageType]
    rows_processed: int
    cache_hits: int
    optimization_applied: bool


# Configuration models
class DatabaseConfig(BaseModel):
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30


class RedisConfig(BaseModel):
    host: str
    port: int
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 100
    socket_timeout: int = 5


class ElasticsearchConfig(BaseModel):
    hosts: List[str]
    username: Optional[str] = None
    password: Optional[str] = None
    verify_certs: bool = True
    timeout: int = 30


class CloudStorageConfig(BaseModel):
    provider: str  # aws, gcp, azure
    bucket_name: str
    region: str
    access_key: Optional[str] = None
    secret_key: Optional[str] = None


# Error models
class StorageError(BaseModel):
    error_type: str
    message: str
    storage_type: StorageType
    timestamp: datetime
    retry_count: int = 0


class ValidationError(BaseModel):
    field: str
    message: str
    value: Any


class ServiceError(BaseModel):
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
