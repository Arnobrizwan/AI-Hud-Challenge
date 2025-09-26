"""
Structured logging configuration with correlation IDs and JSON formatting.
"""

import logging
import logging.config
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

import structlog
from pythonjsonlogger import jsonlogger

from src.config.settings import settings

# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to the log record."""
        record.correlation_id = correlation_id.get() or str(uuid.uuid4())
        return True


class JSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""

    def add_fields(
        self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]
    ) -> None:
        """Add custom fields to the log record."""
        super().add_fields(log_record, record, message_dict)

        # Add standard fields
        log_record["service"] = settings.APP_NAME
        log_record["version"] = settings.APP_VERSION
        log_record["environment"] = settings.ENVIRONMENT

        # Add correlation ID
        log_record["correlation_id"] = getattr(record, "correlation_id", None)

        # Add structured data if present
        if hasattr(record, "user_id"):
            log_record["user_id"] = record.user_id
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        if hasattr(record, "client_ip"):
            log_record["client_ip"] = record.client_ip
        if hasattr(record, "method"):
            log_record["method"] = record.method
        if hasattr(record, "path"):
            log_record["path"] = record.path
        if hasattr(record, "status_code"):
            log_record["status_code"] = record.status_code
        if hasattr(record, "duration"):
            log_record["duration"] = record.duration


def configure_logging() -> None:
    """Configure application logging."""

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            (
                structlog.processors.JSONRenderer()
                if settings.LOG_FORMAT == "json"
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JSONFormatter,
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            },
            "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
        },
        "filters": {
            "correlation": {
                "()": CorrelationFilter,
            }
        },
        "handlers": {
            "default": {
                "level": settings.LOG_LEVEL,
                "formatter": settings.LOG_FORMAT,
                "filters": ["correlation"],
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
            }
        },
        "loggers": {
            "": {"handlers": ["default"], "level": settings.LOG_LEVEL, "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }

    # Add file handler if log file is configured
    # Skip file handler for now to avoid configuration issues
    pass

    logging.config.dictConfig(config)


def get_logger(name: Optional[str] = None) -> Any:
    """Get a logger instance."""
    return structlog.get_logger(name)


def set_correlation_id(value: str) -> None:
    """Set correlation ID for the current context."""
    correlation_id.set(value)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id.get()


def log_request(
    logger: structlog.BoundLogger,
    method: str,
    path: str,
    client_ip: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    """Log incoming request."""
    logger.info(
        "Incoming request",
        method=method,
        path=path,
        client_ip=client_ip,
        user_id=user_id,
        request_id=request_id,
        event="request_start",
    )


def log_response(
    logger: structlog.BoundLogger,
    method: str,
    path: str,
    status_code: int,
    duration: float,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    """Log outgoing response."""
    logger.info(
        "Request completed",
        method=method,
        path=path,
        status_code=status_code,
        duration=duration,
        user_id=user_id,
        request_id=request_id,
        event="request_complete",
    )


def log_error(
    logger: structlog.BoundLogger, error: Exception, context: Optional[Dict[str, Any]] = None
) -> None:
    """Log an error with context."""
    logger.error(
        "Error occurred",
        error=str(error),
        error_type=type(error).__name__,
        **(context or {}),
        event="error",
    )


def log_security_event(
    logger: structlog.BoundLogger,
    event_type: str,
    user_id: Optional[str] = None,
    client_ip: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log security-related events."""
    # Build the log message with all data
    log_message = f"Security event: {event_type}"

    # Create a clean dictionary without conflicting keys
    extra_data = {
        "event_type": event_type,
        "user_id": user_id,
        "client_ip": client_ip,
        "security_event": "security",  # Use different key name
    }

    # Add details if provided
    if details:
        extra_data.update(details)

    logger.warning(log_message, **extra_data)


def log_performance(
    logger: structlog.BoundLogger,
    operation: str,
    duration: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log performance metrics."""
    logger.info(
        f"Performance: {operation}",
        operation=operation,
        duration=duration,
        event="performance",
        **(metadata or {}),
    )


# Configure logging on module import
configure_logging()
