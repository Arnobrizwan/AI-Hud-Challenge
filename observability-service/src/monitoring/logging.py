"""
Centralized logging with structured format
"""

import asyncio
import json
import logging
import logging.handlers
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import redis
import structlog
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Structured log entry"""

    timestamp: datetime
    level: LogLevel
    service: str
    component: str
    message: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None


@dataclass
class LoggingConfig:
    """Logging configuration"""

    elasticsearch_enabled: bool = True
    elasticsearch_host: str = "localhost"
    elasticsearch_port: int = 9200
    elasticsearch_index_prefix: str = "observability-logs"
    redis_enabled: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    log_level: str = "INFO"
    structured_logging: bool = True
    log_rotation: bool = True
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class StructuredLogger:
    """Structured logger with multiple outputs"""

    def __init__(self, config: LoggingConfig):
        self.config = config
        self.elasticsearch = None
        self.redis = None
        self.logger = None
        self.is_initialized = False

    async def initialize(self) -> Dict[str, Any]:
    """Initialize structured logging"""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Initialize outputs
        if self.config.elasticsearch_enabled:
    await self._initialize_elasticsearch()

        if self.config.redis_enabled:
    await self._initialize_redis()

        # Set up file logging
        self._setup_file_logging()

        self.is_initialized = True
        logger.info("Structured logging initialized")

    async def _initialize_elasticsearch(self) -> Dict[str, Any]:
    """Initialize Elasticsearch connection"""
        try:
            self.elasticsearch = Elasticsearch(
                [{"host": self.config.elasticsearch_host, "port": self.config.elasticsearch_port}]
            )

            # Test connection
            if self.elasticsearch.ping():
                logger.info("Elasticsearch connection established")
            else:
                logger.warning("Elasticsearch connection failed")
                self.elasticsearch = None

        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch: {str(e)}")
            self.elasticsearch = None

    async def _initialize_redis(self) -> Dict[str, Any]:
    """Initialize Redis connection"""
        try:
            self.redis = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True,
            )

            # Test connection
            self.redis.ping()
            logger.info("Redis connection established")

        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
            self.redis = None

    def _setup_file_logging(self):
        """Set up file logging with rotation"""

        # Create logger
        self.logger = logging.getLogger("observability")
        self.logger.setLevel(getattr(logging, self.config.log_level))

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # File handler with rotation
        if self.config.log_rotation:
            file_handler = logging.handlers.RotatingFileHandler(
                "logs/observability.log",
                maxBytes=self.config.max_log_size,
                backupCount=self.config.backup_count,
            )
        else:
            file_handler = logging.FileHandler("logs/observability.log")

        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    async def log(self, log_entry: LogEntry) -> Dict[str, Any]:
    """Log a structured entry"""
        if not self.is_initialized:
            logger.warning("Logging not initialized, skipping log entry")
            return

        try:
            # Convert to dict
            log_data = asdict(log_entry)
            log_data["timestamp"] = log_entry.timestamp.isoformat()
            log_data["level"] = log_entry.level.value

            # Log to file
            if self.logger:
                log_message = json.dumps(log_data)
                self.logger.log(
                    getattr(
                        logging,
                        log_entry.level.value),
                    log_message)

            # Send to Elasticsearch
            if self.elasticsearch:
    await self._send_to_elasticsearch(log_data)

            # Send to Redis
            if self.redis:
    await self._send_to_redis(log_data)

        except Exception as e:
            logger.error(f"Failed to log entry: {str(e)}")

    async def _send_to_elasticsearch(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send log data to Elasticsearch"""
        try:
            index_name = (
                f"{self.config.elasticsearch_index_prefix}-{datetime.utcnow().strftime('%Y.%m.%d')}"
            )

            doc = {"_index": index_name, "_source": log_data}

            self.elasticsearch.index(index=index_name, body=log_data)

        except Exception as e:
            logger.error(f"Failed to send to Elasticsearch: {str(e)}")

    async def _send_to_redis(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send log data to Redis"""
        try:
            # Store in Redis list for real-time access
            key = f"logs:{log_data['service']}:{log_data['level']}"
            self.redis.lpush(key, json.dumps(log_data))

            # Keep only last 1000 entries
            self.redis.ltrim(key, 0, 999)

        except Exception as e:
            logger.error(f"Failed to send to Redis: {str(e)}")

    async def search_logs(
        self, query: Dict[str, Any], time_range: Optional[Dict[str, str]] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search logs using Elasticsearch"""

        if not self.elasticsearch:
            return []

        try:
            search_body = {
                "query": {"bool": {"must": []}},
                "size": limit,
                "sort": [{"timestamp": {"order": "desc"}}],
            }

            # Add query conditions
            for field, value in query.items():
                search_body["query"]["bool"]["must"].append(
                    {"match": {field: value}})

            # Add time range
            if time_range:
                search_body["query"]["bool"]["must"].append(
                    {
                        "range": {
                            "timestamp": {
                                "gte": time_range.get("from"),
                                "lte": time_range.get("to"),
                            }
                        }
                    }
                )

            # Search across all indices
            index_pattern = f"{self.config.elasticsearch_index_prefix}-*"
            response = self.elasticsearch.search(
                index=index_pattern, body=search_body)

            return [hit["_source"] for hit in response["hits"]["hits"]]

        except Exception as e:
            logger.error(f"Failed to search logs: {str(e)}")
            return []

    async def get_log_statistics(
            self, time_window: int = 3600) -> Dict[str, Any]:
    """Get log statistics for a time window"""
        if not self.elasticsearch:
            return {}

        try:
            # Calculate time range
            to_time = datetime.utcnow()
            from_time = datetime.utcnow().replace(second=0, microsecond=0) - \
                timedelta(seconds=time_window)

            search_body = {
                "query": {
                    "range": {
                        "timestamp": {
                            "gte": from_time.isoformat(), "lte": to_time.isoformat()}}}, "aggs": {
                    "by_level": {
                        "terms": {
                            "field": "level.keyword"}}, "by_service": {
                                "terms": {
                                    "field": "service.keyword"}}, "by_component": {
                                        "terms": {
                                            "field": "component.keyword"}}, }, }

            index_pattern = f"{self.config.elasticsearch_index_prefix}-*"
            response = self.elasticsearch.search(
                index=index_pattern, body=search_body, size=0)

            return {
                "total_logs": response["hits"]["total"]["value"],
                "by_level": response["aggregations"]["by_level"]["buckets"],
                "by_service": response["aggregations"]["by_service"]["buckets"],
                "by_component": response["aggregations"]["by_component"]["buckets"],
                "time_window": time_window,
                "generated_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get log statistics: {str(e)}")
            return {}


class LogAggregator:
    """Centralized log aggregation system"""

    def __init__(self):
        self.structured_logger = None
        self.is_initialized = False

    async def initialize(self, config: LoggingConfig) -> None:
        """Initialize log aggregation"""

        self.structured_logger = StructuredLogger(config)
        await self.structured_logger.initialize()

        self.is_initialized = True
        logger.info("Log aggregator initialized")

    async def log_info(
            self,
            service: str,
            component: str,
            message: str,
            **kwargs):
         -> Dict[str, Any]:"""Log info message"""
        await self._log(LogLevel.INFO, service, component, message, **kwargs)

    async def log_warning(
            self,
            service: str,
            component: str,
            message: str,
            **kwargs):
         -> Dict[str, Any]:"""Log warning message"""
        await self._log(LogLevel.WARNING, service, component, message, **kwargs)

    async def log_error(
            self,
            service: str,
            component: str,
            message: str,
            **kwargs):
         -> Dict[str, Any]:"""Log error message"""
        await self._log(LogLevel.ERROR, service, component, message, **kwargs)

    async def log_critical(
            self,
            service: str,
            component: str,
            message: str,
            **kwargs):
         -> Dict[str, Any]:"""Log critical message"""
        await self._log(LogLevel.CRITICAL, service, component, message, **kwargs)

    async def _log(
            self,
            level: LogLevel,
            service: str,
            component: str,
            message: str,
            **kwargs):
         -> Dict[str, Any]:"""Log a message with structured data"""

        if not self.is_initialized or not self.structured_logger:
            logger.warning("Log aggregator not initialized")
            return

        log_entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            service=service,
            component=component,
            message=message,
            trace_id=kwargs.get("trace_id"),
            span_id=kwargs.get("span_id"),
            user_id=kwargs.get("user_id"),
            request_id=kwargs.get("request_id"),
            metadata=kwargs.get("metadata"),
            error_details=kwargs.get("error_details"),
        )

        await self.structured_logger.log(log_entry)

    async def search_logs(
        self, query: Dict[str, Any], time_range: Optional[Dict[str, str]] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search logs"""

        if not self.is_initialized or not self.structured_logger:
            return []

        return await self.structured_logger.search_logs(query, time_range, limit)

    async def get_log_statistics(
            self, time_window: int = 3600) -> Dict[str, Any]:
    """Get log statistics"""
        if not self.is_initialized or not self.structured_logger:
            return {}

        return await self.structured_logger.get_log_statistics(time_window)

    async def cleanup(self) -> Dict[str, Any]:
    """Cleanup log aggregator"""
        self.is_initialized = False
        logger.info("Log aggregator cleaned up")
