"""
Logging configuration for MLOps Orchestration Service
"""

import logging
import sys
from typing import Optional
from datetime import datetime
import structlog
from loguru import logger

def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    enable_console: bool = True,
    enable_file: bool = True,
    log_file: str = "/app/logs/mlops-service.log"
) -> None:
    """Setup logging configuration"""
    
    # Remove default loguru handler
    logger.remove()
    
    # Configure log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Console logging
    if enable_console:
        if log_format == "json":
            logger.add(
                sys.stdout,
                level=level,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
                serialize=True,
                backtrace=True,
                diagnose=True
            )
        else:
            logger.add(
                sys.stdout,
                level=level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                backtrace=True,
                diagnose=True
            )
    
    # File logging
    if enable_file:
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            serialize=True,
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
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
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logger.bind(service=name)

class MLOpsLogger:
    """Custom logger for MLOps service"""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)
    
    def log_pipeline_event(self, pipeline_id: str, event: str, **kwargs):
        """Log pipeline-specific event"""
        self.logger.info(
            f"Pipeline {pipeline_id}: {event}",
            pipeline_id=pipeline_id,
            event=event,
            **kwargs
        )
    
    def log_training_event(self, model_name: str, event: str, **kwargs):
        """Log training-specific event"""
        self.logger.info(
            f"Training {model_name}: {event}",
            model_name=model_name,
            event=event,
            **kwargs
        )
    
    def log_deployment_event(self, deployment_id: str, event: str, **kwargs):
        """Log deployment-specific event"""
        self.logger.info(
            f"Deployment {deployment_id}: {event}",
            deployment_id=deployment_id,
            event=event,
            **kwargs
        )
    
    def log_monitoring_event(self, model_name: str, event: str, **kwargs):
        """Log monitoring-specific event"""
        self.logger.info(
            f"Monitoring {model_name}: {event}",
            model_name=model_name,
            event=event,
            **kwargs
        )
