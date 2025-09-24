"""
Configuration management for Storage Service
"""

from typing import List, Optional

from pydantic import BaseSettings, Field

from models import CloudStorageConfig, DatabaseConfig, ElasticsearchConfig, RedisConfig


class Settings(BaseSettings):
    """Application settings"""

    # Service configuration
    service_name: str = "storage-service"
    service_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # API configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")

    # Database configurations
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_database: str = Field(default="storage_db", env="POSTGRES_DATABASE")
    postgres_username: str = Field(default="postgres", env="POSTGRES_USERNAME")
    postgres_password: str = Field(default="password", env="POSTGRES_PASSWORD")
    postgres_pool_size: int = Field(default=20, env="POSTGRES_POOL_SIZE")
    postgres_max_overflow: int = Field(default=30, env="POSTGRES_MAX_OVERFLOW")

    # Redis configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_max_connections: int = Field(default=100, env="REDIS_MAX_CONNECTIONS")

    # Elasticsearch configuration
    elasticsearch_hosts: List[str] = Field(default=["localhost:9200"], env="ELASTICSEARCH_HOSTS")
    elasticsearch_username: Optional[str] = Field(default=None, env="ELASTICSEARCH_USERNAME")
    elasticsearch_password: Optional[str] = Field(default=None, env="ELASTICSEARCH_PASSWORD")
    elasticsearch_verify_certs: bool = Field(default=True, env="ELASTICSEARCH_VERIFY_CERTS")

    # Cloud storage configuration
    cloud_storage_provider: str = Field(default="aws", env="CLOUD_STORAGE_PROVIDER")
    cloud_storage_bucket: str = Field(default="storage-bucket", env="CLOUD_STORAGE_BUCKET")
    cloud_storage_region: str = Field(default="us-east-1", env="CLOUD_STORAGE_REGION")
    aws_access_key: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")

    # TimescaleDB configuration
    timescale_host: str = Field(default="localhost", env="TIMESCALE_HOST")
    timescale_port: int = Field(default=5432, env="TIMESCALE_PORT")
    timescale_database: str = Field(default="timescale_db", env="TIMESCALE_DATABASE")
    timescale_username: str = Field(default="timescale", env="TIMESCALE_USERNAME")
    timescale_password: str = Field(default="password", env="TIMESCALE_PASSWORD")

    # Performance configuration
    max_concurrent_operations: int = Field(default=100, env="MAX_CONCURRENT_OPERATIONS")
    query_timeout: int = Field(default=30, env="QUERY_TIMEOUT")
    cache_default_ttl: int = Field(default=3600, env="CACHE_DEFAULT_TTL")

    # Vector search configuration
    vector_dimension: int = Field(default=768, env="VECTOR_DIMENSION")
    vector_similarity_threshold: float = Field(default=0.7, env="VECTOR_SIMILARITY_THRESHOLD")
    vector_index_m: int = Field(default=16, env="VECTOR_INDEX_M")
    vector_index_ef_construction: int = Field(default=200, env="VECTOR_INDEX_EF_CONSTRUCTION")

    # Cache configuration
    memory_cache_size: int = Field(default=1000, env="MEMORY_CACHE_SIZE")
    memory_cache_ttl: int = Field(default=300, env="MEMORY_CACHE_TTL")
    redis_cache_ttl: int = Field(default=3600, env="REDIS_CACHE_TTL")
    cdn_cache_ttl: int = Field(default=86400, env="CDN_CACHE_TTL")

    # Data lifecycle configuration
    retention_check_interval: int = Field(default=3600, env="RETENTION_CHECK_INTERVAL")
    max_retention_days: int = Field(default=365, env="MAX_RETENTION_DAYS")
    archive_after_days: int = Field(default=90, env="ARCHIVE_AFTER_DAYS")

    # Monitoring configuration
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")

    # Security configuration
    secret_key: str = Field(default="your-secret-key", env="SECRET_KEY")
    jwt_secret: str = Field(default="your-jwt-secret", env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration: int = Field(default=3600, env="JWT_EXPIRATION")

    # Rate limiting
    rate_limit_requests: int = Field(default=1000, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")

    class Config:
        env_file = ".env"
        case_sensitive = False

    def get_postgres_config(self) -> DatabaseConfig:
        """Get PostgreSQL configuration"""
        return DatabaseConfig(
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_database,
            username=self.postgres_username,
            password=self.postgres_password,
            pool_size=self.postgres_pool_size,
            max_overflow=self.postgres_max_overflow,
        )

    def get_redis_config(self) -> RedisConfig:
        """Get Redis configuration"""
        return RedisConfig(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            db=self.redis_db,
            max_connections=self.redis_max_connections,
        )

    def get_elasticsearch_config(self) -> ElasticsearchConfig:
        """Get Elasticsearch configuration"""
        return ElasticsearchConfig(
            hosts=self.elasticsearch_hosts,
            username=self.elasticsearch_username,
            password=self.elasticsearch_password,
            verify_certs=self.elasticsearch_verify_certs,
        )

    def get_cloud_storage_config(self) -> CloudStorageConfig:
        """Get cloud storage configuration"""
        return CloudStorageConfig(
            provider=self.cloud_storage_provider,
            bucket_name=self.cloud_storage_bucket,
            region=self.cloud_storage_region,
            access_key=self.aws_access_key,
            secret_key=self.aws_secret_key,
        )

    def get_timescale_config(self) -> DatabaseConfig:
        """Get TimescaleDB configuration"""
        return DatabaseConfig(
            host=self.timescale_host,
            port=self.timescale_port,
            database=self.timescale_database,
            username=self.timescale_username,
            password=self.timescale_password,
            pool_size=self.postgres_pool_size,
            max_overflow=self.postgres_max_overflow,
        )


# Global settings instance
settings = Settings()
