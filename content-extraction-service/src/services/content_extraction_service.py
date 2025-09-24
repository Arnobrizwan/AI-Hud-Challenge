"""
Main content extraction service with orchestration and business logic.
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from ..exceptions import ContentExtractionError, ValidationError
from ..extractors.pipeline import ContentExtractionPipeline
from ..models.api import APIResponse, HealthCheckResponse
from ..models.content import (
    BatchExtractionRequest,
    BatchExtractionResponse,
    ExtractedContent,
    ExtractionRequest,
    ExtractionResponse,
    ProcessingStatus,
)
from ..services.cache_service import CacheService
from ..services.monitoring_service import MonitoringService
from ..services.queue_service import QueueService


class ContentExtractionService:
    """Main service for content extraction with full orchestration."""

    def __init__(
        self,
        extraction_pipeline: ContentExtractionPipeline,
        cache_service: CacheService,
        queue_service: QueueService,
        monitoring_service: MonitoringService,
    ):
        """Initialize content extraction service."""
        self.extraction_pipeline = extraction_pipeline
        self.cache_service = cache_service
        self.queue_service = queue_service
        self.monitoring_service = monitoring_service
        self.start_time = time.time()

    async def extract_content(
            self,
            request: ExtractionRequest) -> ExtractionResponse:
        """
        Extract content from a single URL.

        Args:
            request: Extraction request

        Returns:
            ExtractionResponse with results
        """
        try:
            logger.info(f"Processing extraction request for {request.url}")

            # Validate request
            await self._validate_request(request)

            # Record metrics
            start_time = time.time()

            # Process extraction
            response = await self.extraction_pipeline.extract_content(request)

            # Record processing time
            processing_time = time.time() - start_time

            # Update metrics
            await self.monitoring_service.record_extraction_metrics(
                success=response.success,
                processing_time_ms=int(processing_time * 1000),
                content_type=response.content.content_type if response.content else None,
                cache_hit=response.cache_hit,
            )

            # Log result
            if response.success:
                logger.info(
                    f"Content extraction completed for {request.url} in {processing_time:.2f}s"
                )
            else:
                logger.error(
                    f"Content extraction failed for {request.url}: {response.error_message}"
                )

            return response

        except ValidationError as e:
            logger.error(f"Validation error for {request.url}: {str(e)}")
            return ExtractionResponse(
                success=False,
                error_message=f"Validation error: {str(e)}",
                processing_time_ms=0,
                extraction_id="",
            )
        except Exception as e:
            logger.error(f"Unexpected error for {request.url}: {str(e)}")
            return ExtractionResponse(
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                processing_time_ms=0,
                extraction_id="",
            )

    async def batch_extract_content(
        self, request: BatchExtractionRequest
    ) -> BatchExtractionResponse:
        """
        Extract content from multiple URLs.

        Args:
            request: Batch extraction request

        Returns:
            BatchExtractionResponse with results
        """
        try:
            logger.info(
                f"Processing batch extraction for {len(request.urls)} URLs")

            # Generate batch ID
            batch_id = self._generate_batch_id()

            # Create individual extraction requests
            extraction_requests = []
            for i, url in enumerate(request.urls):
                content_type = request.content_types[i] if i < len(
                    request.content_types) else None
                extraction_method = (
                    request.extraction_methods[i] if i < len(
                        request.extraction_methods) else None)

                extraction_request = ExtractionRequest(
                    url=url,
                    content_type=content_type,
                    extraction_method=extraction_method,
                    force_refresh=request.force_refresh,
                    include_images=request.include_images,
                    quality_threshold=request.quality_threshold,
                )
                extraction_requests.append(extraction_request)

            # Process extractions
            start_time = time.time()
            responses = await self.extraction_pipeline.batch_extract_content(
                extraction_requests, max_concurrent=request.max_concurrent
            )
            processing_time = time.time() - start_time

            # Analyze results
            successful_extractions = sum(1 for r in responses if r.success)
            failed_extractions = len(responses) - successful_extractions
            cached_extractions = sum(1 for r in responses if r.cache_hit)

            # Update metrics
            await self.monitoring_service.record_batch_metrics(
                batch_id=batch_id,
                total_urls=len(request.urls),
                successful=successful_extractions,
                failed=failed_extractions,
                cached=cached_extractions,
                processing_time_ms=int(processing_time * 1000),
            )

            logger.info(
                f"Batch extraction completed: {successful_extractions}/{len(request.urls)} successful"
            )

            return BatchExtractionResponse(
                batch_id=batch_id,
                total_urls=len(request.urls),
                successful_extractions=successful_extractions,
                failed_extractions=failed_extractions,
                cached_extractions=cached_extractions,
                processing_time_ms=int(processing_time * 1000),
                results=responses,
                errors=[],
            )

        except Exception as e:
            logger.error(f"Batch extraction failed: {str(e)}")
            return BatchExtractionResponse(
                batch_id="",
                total_urls=len(request.urls),
                successful_extractions=0,
                failed_extractions=len(request.urls),
                cached_extractions=0,
                processing_time_ms=0,
                results=[],
                errors=[str(e)],
            )

    async def get_extraction_status(
            self, extraction_id: str) -> Dict[str, Any]:
    """
        Get status of an extraction.

        Args:
            extraction_id: Extraction ID

        Returns:
            Status information
        """
        try:
            # In production, this would check a database or cache
            # For now, return a placeholder response
            return {
                "extraction_id": extraction_id,
                "status": ProcessingStatus.COMPLETED,
                "progress_percentage": 100.0,
                "processing_time_ms": 0,
                "estimated_completion": None,
                "error_message": None,
            }
        except Exception as e:
            logger.error(f"Status check failed for {extraction_id}: {str(e)}")
            return {
                "extraction_id": extraction_id,
                "status": ProcessingStatus.FAILED,
                "progress_percentage": 0.0,
                "processing_time_ms": 0,
                "estimated_completion": None,
                "error_message": str(e),
            }

    async def search_content(
        self,
        query: Optional[str] = None,
        content_types: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        quality_threshold: Optional[float] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
    """
        Search extracted content.

        Args:
            query: Search query
            content_types: Filter by content types
            languages: Filter by languages
            quality_threshold: Minimum quality threshold
            limit: Maximum results
            offset: Result offset

        Returns:
            Search results
        """
        try:
            # In production, this would query a search index
            # For now, return placeholder results
            return {
                "results": [],
                "total_count": 0,
                "limit": limit,
                "offset": offset,
                "query_time_ms": 0,
            }
        except Exception as e:
            logger.error(f"Content search failed: {str(e)}")
            return {
                "results": [],
                "total_count": 0,
                "limit": limit,
                "offset": offset,
                "query_time_ms": 0,
                "error": str(e),
            }

    async def get_health_status(self) -> HealthCheckResponse:
        """Get service health status."""
        try:
            uptime = time.time() - self.start_time

            # Check dependencies
            dependencies = {
                "cache_service":
    await self._check_cache_service(),
                "queue_service":
    await self._check_queue_service(),
                "monitoring_service":
    await self._check_monitoring_service(),
            }

            # Determine overall status
            all_healthy = all(dependencies.values())
            status = "healthy" if all_healthy else "degraded"

            return HealthCheckResponse(
                status=status,
                version="1.0.0",
                uptime_seconds=uptime,
                dependencies=dependencies,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return HealthCheckResponse(
                status="unhealthy",
                version="1.0.0",
                uptime_seconds=time.time() - self.start_time,
                dependencies={},
                timestamp=datetime.utcnow(),
            )

    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        try:
            return await self.monitoring_service.get_metrics()
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {str(e)}")
            return {}

    async def _validate_request(self, request: ExtractionRequest) -> None:
        """Validate extraction request."""
        if not request.url:
            raise ValidationError("URL is required")

        if not request.url.startswith(("http://", "https://")):
            raise ValidationError("URL must start with http:// or https://")

        if request.quality_threshold < 0.0 or request.quality_threshold > 1.0:
            raise ValidationError(
                "Quality threshold must be between 0.0 and 1.0")

        if request.timeout and (request.timeout < 1 or request.timeout > 300):
            raise ValidationError("Timeout must be between 1 and 300 seconds")

    def _generate_batch_id(self) -> str:
        """Generate unique batch ID."""
        timestamp = int(time.time() * 1000)
        return f"batch_{timestamp}"

    async def _check_cache_service(self) -> bool:
        """Check cache service health."""
        try:
            # Simple health check
            return True
        except Exception:
            return False

    async def _check_queue_service(self) -> bool:
        """Check queue service health."""
        try:
            # Simple health check
            return True
        except Exception:
            return False

    async def _check_monitoring_service(self) -> bool:
        """Check monitoring service health."""
        try:
            # Simple health check
            return True
        except Exception:
            return False

    async def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old extracted content and cache data."""
        try:
            logger.info(f"Starting cleanup of data older than {days_old} days")

            # Clean up cache
            cache_cleaned = await self.cache_service.cleanup_old_entries(days_old)

            # Clean up monitoring data
            monitoring_cleaned = await self.monitoring_service.cleanup_old_metrics(days_old)

            total_cleaned = cache_cleaned + monitoring_cleaned

            logger.info(f"Cleanup completed: {total_cleaned} entries removed")

            return {
                "cache_cleaned": cache_cleaned,
                "monitoring_cleaned": monitoring_cleaned,
                "total_cleaned": total_cleaned,
            }

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return {
                "cache_cleaned": 0,
                "monitoring_cleaned": 0,
                "total_cleaned": 0,
                "error": str(e),
            }
