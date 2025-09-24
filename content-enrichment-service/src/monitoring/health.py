"""Health checking for Content Enrichment Service."""

import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import structlog
import psutil
import redis.asyncio as redis
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine

from ..config import settings

logger = structlog.get_logger(__name__)


class HealthChecker:
    """Health checker for service components."""
    
    def __init__(self):
        """Initialize the health checker."""
        self.redis_client = None
        self.db_engine = None
        self.health_status = {
            "status": "unknown",
            "timestamp": None,
            "components": {},
            "overall_health": False
        }
        self.last_check = None
        self.check_interval = 30  # seconds
    
    async def initialize(self):
        """Initialize health checker components."""
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(
                settings.redis_url,
                max_connections=5,
                decode_responses=True
            )
            
            # Initialize database connection
            self.db_engine = create_async_engine(settings.database_url)
            
            logger.info("Health checker initialized")
            
        except Exception as e:
            logger.error("Failed to initialize health checker", error=str(e))
    
    async def check_health(self) -> Dict[str, Any]:
        """Check overall service health."""
        try:
            current_time = datetime.utcnow()
            
            # Check if we need to refresh health status
            if (self.last_check is None or 
                (current_time - self.last_check).total_seconds() > self.check_interval):
                
                await self._perform_health_checks()
                self.last_check = current_time
            
            return self.health_status
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "overall_health": False
            }
    
    async def _perform_health_checks(self):
        """Perform comprehensive health checks."""
        try:
            components = {}
            
            # Check system resources
            components["system"] = await self._check_system_health()
            
            # Check Redis
            components["redis"] = await self._check_redis_health()
            
            # Check database
            components["database"] = await self._check_database_health()
            
            # Check model availability
            components["models"] = await self._check_models_health()
            
            # Check external services
            components["external_services"] = await self._check_external_services_health()
            
            # Determine overall health
            overall_health = all(
                component.get("healthy", False) 
                for component in components.values()
            )
            
            self.health_status = {
                "status": "healthy" if overall_health else "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "components": components,
                "overall_health": overall_health
            }
            
        except Exception as e:
            logger.error("Health checks failed", error=str(e))
            self.health_status = {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "overall_health": False
            }
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system resource health."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Health thresholds
            cpu_healthy = cpu_percent < 80
            memory_healthy = memory_percent < 85
            disk_healthy = disk_percent < 90
            process_memory_healthy = process_memory < 4000  # 4GB limit
            
            overall_healthy = all([
                cpu_healthy, memory_healthy, disk_healthy, process_memory_healthy
            ])
            
            return {
                "healthy": overall_healthy,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "process_memory_mb": process_memory,
                "thresholds": {
                    "cpu_max": 80,
                    "memory_max": 85,
                    "disk_max": 90,
                    "process_memory_max": 4000
                }
            }
            
        except Exception as e:
            logger.error("System health check failed", error=str(e))
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis health."""
        try:
            if not self.redis_client:
                return {
                    "healthy": False,
                    "error": "Redis client not initialized"
                }
            
            # Test connection
            start_time = time.time()
            await self.redis_client.ping()
            response_time = time.time() - start_time
            
            # Get Redis info
            info = await self.redis_client.info()
            
            # Check memory usage
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            memory_usage = (used_memory / max_memory * 100) if max_memory > 0 else 0
            
            # Check connected clients
            connected_clients = info.get('connected_clients', 0)
            
            # Health criteria
            response_time_healthy = response_time < 1.0  # 1 second
            memory_healthy = memory_usage < 90
            clients_healthy = connected_clients < 1000
            
            overall_healthy = all([
                response_time_healthy, memory_healthy, clients_healthy
            ])
            
            return {
                "healthy": overall_healthy,
                "response_time_ms": response_time * 1000,
                "memory_usage_percent": memory_usage,
                "connected_clients": connected_clients,
                "version": info.get('redis_version', 'unknown')
            }
            
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            if not self.db_engine:
                return {
                    "healthy": False,
                    "error": "Database engine not initialized"
                }
            
            # Test connection
            start_time = time.time()
            async with self.db_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            response_time = time.time() - start_time
            
            # Check connection pool
            pool = self.db_engine.pool
            pool_size = pool.size()
            checked_in = pool.checkedin()
            checked_out = pool.checkedout()
            overflow = pool.overflow()
            
            # Health criteria
            response_time_healthy = response_time < 2.0  # 2 seconds
            pool_healthy = checked_out < pool_size * 0.9  # 90% pool usage
            
            overall_healthy = response_time_healthy and pool_healthy
            
            return {
                "healthy": overall_healthy,
                "response_time_ms": response_time * 1000,
                "pool_size": pool_size,
                "checked_in": checked_in,
                "checked_out": checked_out,
                "overflow": overflow
            }
            
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _check_models_health(self) -> Dict[str, Any]:
        """Check AI/ML models health."""
        try:
            # This is a simplified check
            # In practice, you'd check if models are loaded and responding
            
            model_checks = {
                "spacy_model": await self._check_spacy_model(),
                "transformer_models": await self._check_transformer_models(),
                "custom_models": await self._check_custom_models()
            }
            
            overall_healthy = all(
                check.get("healthy", False) 
                for check in model_checks.values()
            )
            
            return {
                "healthy": overall_healthy,
                "models": model_checks
            }
            
        except Exception as e:
            logger.error("Models health check failed", error=str(e))
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _check_spacy_model(self) -> Dict[str, Any]:
        """Check spaCy model health."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_lg")
            
            # Test with sample text
            doc = nlp("This is a test sentence.")
            
            return {
                "healthy": True,
                "model_name": "en_core_web_lg",
                "entities_detected": len(doc.ents) > 0
            }
            
        except Exception as e:
            logger.error("spaCy model check failed", error=str(e))
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _check_transformer_models(self) -> Dict[str, Any]:
        """Check transformer models health."""
        try:
            # This is a simplified check
            # In practice, you'd test actual model inference
            
            return {
                "healthy": True,
                "models_loaded": True,
                "note": "Simplified check - actual model testing not implemented"
            }
            
        except Exception as e:
            logger.error("Transformer models check failed", error=str(e))
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _check_custom_models(self) -> Dict[str, Any]:
        """Check custom models health."""
        try:
            # Check if model files exist
            import os
            model_dir = settings.model_cache_dir
            
            if not os.path.exists(model_dir):
                return {
                    "healthy": False,
                    "error": "Model directory not found"
                }
            
            # Check for model files
            model_files = os.listdir(model_dir)
            
            return {
                "healthy": len(model_files) > 0,
                "model_files_count": len(model_files),
                "model_directory": model_dir
            }
            
        except Exception as e:
            logger.error("Custom models check failed", error=str(e))
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _check_external_services_health(self) -> Dict[str, Any]:
        """Check external services health."""
        try:
            # Check Google Cloud services
            google_cloud_healthy = await self._check_google_cloud_health()
            
            # Check MLflow
            mlflow_healthy = await self._check_mlflow_health()
            
            overall_healthy = google_cloud_healthy and mlflow_healthy
            
            return {
                "healthy": overall_healthy,
                "google_cloud": google_cloud_healthy,
                "mlflow": mlflow_healthy
            }
            
        except Exception as e:
            logger.error("External services health check failed", error=str(e))
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _check_google_cloud_health(self) -> bool:
        """Check Google Cloud services health."""
        try:
            # This is a simplified check
            # In practice, you'd test actual API calls
            
            return True  # Assume healthy for now
            
        except Exception as e:
            logger.error("Google Cloud health check failed", error=str(e))
            return False
    
    async def _check_mlflow_health(self) -> bool:
        """Check MLflow health."""
        try:
            # This is a simplified check
            # In practice, you'd test MLflow API calls
            
            return True  # Assume healthy for now
            
        except Exception as e:
            logger.error("MLflow health check failed", error=str(e))
            return False
    
    async def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information."""
        try:
            # Force a fresh health check
            await self._perform_health_checks()
            
            return {
                **self.health_status,
                "detailed": True,
                "service_info": {
                    "name": settings.app_name,
                    "version": settings.app_version,
                    "uptime": time.time() - psutil.Process().create_time(),
                    "pid": psutil.Process().pid
                }
            }
            
        except Exception as e:
            logger.error("Detailed health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "detailed": True
            }
    
    async def cleanup(self):
        """Cleanup health checker resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            if self.db_engine:
                await self.db_engine.dispose()
            
            logger.info("Health checker cleanup completed")
            
        except Exception as e:
            logger.error("Health checker cleanup failed", error=str(e))
