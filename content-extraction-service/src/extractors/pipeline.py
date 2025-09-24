"""
Multi-stage content extraction and cleanup pipeline.
"""

import asyncio
import hashlib
import time
from datetime import datetime
from typing import List, Optional

from loguru import logger

from ..exceptions import ContentExtractionError
from ..models.api import ExtractionRequest, ExtractionResponse
from ..models.content import (
    ContentMetadata,
    ContentType,
    ExtractedContent,
    ExtractionMethod,
    ExtractionStats,
    LanguageInfo,
    ProcessedImage,
    ProcessingStatus,
    QualityMetrics,
)
from ..services.cache_service import CacheService
from ..services.document_ai_service import DocumentAIService
from ..utils.html_cleaner import HTMLCleaner
from ..utils.image_processor import ImageProcessor
from ..utils.language_detector import LanguageDetector
from ..utils.metadata_extractor import MetadataExtractor
from ..utils.quality_analyzer import QualityAnalyzer
from ..utils.readability_extractor import ReadabilityExtractor


class ContentExtractionPipeline:
    """Multi-stage content extraction and cleanup pipeline."""

    def __init__(
        self,
        html_cleaner: HTMLCleaner,
        readability_extractor: ReadabilityExtractor,
        metadata_extractor: MetadataExtractor,
        image_processor: ImageProcessor,
        quality_analyzer: QualityAnalyzer,
        language_detector: LanguageDetector,
        cache_service: CacheService,
        document_ai_service: DocumentAIService,
    ):
        """Initialize the extraction pipeline."""
        self.html_cleaner = html_cleaner
        self.readability_extractor = readability_extractor
        self.metadata_extractor = metadata_extractor
        self.image_processor = image_processor
        self.quality_analyzer = quality_analyzer
        self.language_detector = language_detector
        self.cache_service = cache_service
        self.document_ai_service = document_ai_service

    async def extract_content(self, request: ExtractionRequest, raw_html: Optional[str] = None) -> ExtractionResponse:
        """
        Main extraction pipeline with fallback strategies.

        Args:
            request: Extraction request parameters
            raw_html: Pre-fetched HTML content (optional)

        Returns:
            ExtractionResponse with extracted content or error
        """
        start_time = time.time()
        extraction_id = self._generate_extraction_id(request.url)

        try:
            logger.info(f"Starting content extraction for {request.url} (ID: {extraction_id})")

            # Check cache first
            if not request.force_refresh:
                cached_content = await self.cache_service.get_content(request.url)
                if cached_content:
                    logger.info(f"Cache hit for {request.url}")
                    return ExtractionResponse(
                        success=True,
                        content=cached_content,
                        processing_time_ms=int((time.time() - start_time) * 1000),
                        cache_hit=True,
                        extraction_id=extraction_id,
                    )

            # Stage 1: Raw HTML acquisition
            html_content = raw_html or await self._fetch_html(request)
            html_fetch_time = time.time()

            # Stage 2: Metadata extraction
            metadata = await self._extract_metadata(html_content, request.url)
            metadata_time = time.time()

            # Stage 3: Content extraction with readability
            main_content = await self._extract_main_content(html_content, request)
            content_extraction_time = time.time()

            # Stage 4: Content cleaning and sanitization
            clean_content = await self._sanitize_content(main_content, request)
            sanitization_time = time.time()

            # Stage 5: Image processing and optimization
            processed_images = await self._process_images(clean_content, request)
            image_processing_time = time.time()

            # Stage 6: Quality scoring and validation
            quality_metrics = await self._score_content_quality(clean_content, request)
            quality_analysis_time = time.time()

            # Stage 7: Language and readability analysis
            language_info = await self._analyze_language(clean_content, request)
            language_analysis_time = time.time()

            # Create extraction statistics
            extraction_stats = self._create_extraction_stats(
                start_time,
                html_fetch_time,
                content_extraction_time,
                sanitization_time,
                image_processing_time,
                quality_analysis_time,
                language_analysis_time,
                html_content,
                processed_images,
            )

            # Create extracted content object
            extracted_content = self._create_extracted_content(
                request,
                clean_content,
                metadata,
                processed_images,
                quality_metrics,
                language_info,
                extraction_stats,
            )

            # Cache the result
            await self.cache_service.cache_content(request.url, extracted_content)

            processing_time = int((time.time() - start_time) * 1000)
            logger.info(f"Content extraction completed for {request.url} in {processing_time}ms")

            return ExtractionResponse(
                success=True,
                content=extracted_content,
                processing_time_ms=processing_time,
                cache_hit=False,
                extraction_id=extraction_id,
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            error_msg = f"Content extraction failed for {request.url}: {str(e)}"
            logger.error(error_msg)

            return ExtractionResponse(
                success=False,
                content=None,
                error_message=error_msg,
                processing_time_ms=processing_time,
                cache_hit=False,
                extraction_id=extraction_id,
            )

    async def _fetch_html(self, request: ExtractionRequest) -> str:
        """Stage 1: Raw HTML acquisition."""
        try:
            # Determine extraction method based on content type and request
            if request.content_type == ContentType.PDF:
                return await self._fetch_pdf_content(request)
            elif request.content_type in [ContentType.DOC, ContentType.DOCX]:
                return await self._fetch_document_content(request)
        else:
                return await self._fetch_web_content(request)
        except Exception as e:
            logger.error(f"HTML fetch failed for {request.url}: {str(e)}")
            raise ContentExtractionError(f"Failed to fetch content: {str(e)}")

    async def _fetch_web_content(self, request: ExtractionRequest) -> str:
        """Fetch web content using Playwright or BeautifulSoup."""
        try:
            # Use Playwright for JavaScript-heavy sites
            if request.extraction_method == ExtractionMethod.PLAYWRIGHT:
                from ..services.playwright_service import PlaywrightService

                playwright_service = PlaywrightService()
                return await playwright_service.fetch_html(request.url, timeout=request.timeout)
            else:
                # Use BeautifulSoup for simple HTML
                from ..services.http_service import HTTPService

                http_service = HTTPService()
                return await http_service.fetch_html(request.url, timeout=request.timeout)
        except Exception as e:
            logger.error(f"Web content fetch failed for {request.url}: {str(e)}")
            raise

    async def _fetch_pdf_content(self, request: ExtractionRequest) -> str:
        """Fetch PDF content using Document AI."""
        try:
            return await self.document_ai_service.extract_text_from_pdf(request.url)
        except Exception as e:
            logger.error(f"PDF content fetch failed for {request.url}: {str(e)}")
            raise

    async def _fetch_document_content(self, request: ExtractionRequest) -> str:
        """Fetch document content using Document AI."""
        try:
            return await self.document_ai_service.extract_text_from_document(request.url)
        except Exception as e:
            logger.error(f"Document content fetch failed for {request.url}: {str(e)}")
            raise

    async def _extract_metadata(self, html_content: str, url: str) -> ContentMetadata:
        """Stage 2: Metadata extraction."""
        try:
            return await self.metadata_extractor.extract_metadata(html_content, url)
        except Exception as e:
            logger.error(f"Metadata extraction failed for {url}: {str(e)}")
            # Return minimal metadata on failure
            return ContentMetadata()

    async def _extract_main_content(self, html_content: str, request: ExtractionRequest) -> str:
        """Stage 3: Content extraction with readability."""
        try:
            # Use custom selectors if provided
            if request.custom_selectors:
                return await self._extract_with_custom_selectors(html_content, request)

            # Use readability algorithm
            return await self.readability_extractor.extract_main_content(
                html_content, url=request.url, min_text_length=250
            )
        except Exception as e:
            logger.error(f"Main content extraction failed for {request.url}: {str(e)}")
            # Fallback to basic HTML cleaning
            return await self.html_cleaner.clean_html(html_content)

    async def _extract_with_custom_selectors(self, html_content: str, request: ExtractionRequest) -> str:
        """Extract content using custom CSS selectors."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, "html.parser")

            content_parts = []
            for selector_name, selector in request.custom_selectors.items():
                elements = soup.select(selector)
                for element in elements:
                    content_parts.append(element.get_text(strip=True))

            return "\n\n".join(content_parts)
        except Exception as e:
            logger.error(f"Custom selector extraction failed: {str(e)}")
            raise

    async def _sanitize_content(self, content: str, request: ExtractionRequest) -> str:
        """Stage 4: Content cleaning and sanitization."""
        try:
            return await self.html_cleaner.sanitize_text(content)
        except Exception as e:
            logger.error(f"Content sanitization failed for {request.url}: {str(e)}")
            # Return basic cleaned content
            return content.strip()

    async def _process_images(self, content: str, request: ExtractionRequest) -> List[ProcessedImage]:
        """Stage 5: Image processing and optimization."""
        if not request.include_images:
            return []

        try:
            return await self.image_processor.process_images_from_content(content, request.url)
        except Exception as e:
            logger.error(f"Image processing failed for {request.url}: {str(e)}")
            return []

    async def _score_content_quality(self, content: str, request: ExtractionRequest) -> QualityMetrics:
        """Stage 6: Quality scoring and validation."""
        try:
            return await self.quality_analyzer.analyze_content_quality(
                content, url=request.url, language_hint=request.language_hint
            )
        except Exception as e:
            logger.error(f"Quality analysis failed for {request.url}: {str(e)}")
            # Return default quality metrics
            return QualityMetrics(
                readability_score=50.0,
                word_count=len(content.split()),
                character_count=len(content),
                sentence_count=content.count("."),
                paragraph_count=content.count("\n\n"),
                average_sentence_length=0.0,
                average_word_length=0.0,
                image_to_text_ratio=0.0,
                link_density=0.0,
                spam_score=0.0,
                duplicate_score=0.0,
                content_freshness=0.0,
                overall_quality=50.0,
            )

    async def _analyze_language(self, content: str, request: ExtractionRequest) -> LanguageInfo:
        """Stage 7: Language and readability analysis."""
        try:
            return await self.language_detector.detect_language(content, language_hint=request.language_hint)
        except Exception as e:
            logger.error(f"Language analysis failed for {request.url}: {str(e)}")
            # Return default language info
            return LanguageInfo(detected_language="en", confidence=0.5, charset="utf-8", is_reliable=False)

    def _create_extraction_stats(
        self,
        start_time: float,
        html_fetch_time: float,
        content_extraction_time: float,
        sanitization_time: float,
        image_processing_time: float,
        quality_analysis_time: float,
        language_analysis_time: float,
        html_content: str,
        processed_images: List[ProcessedImage],
    ) -> ExtractionStats:
        """Create extraction statistics."""
        total_time = time.time() - start_time

        return ExtractionStats(
            extraction_time_ms=int(total_time * 1000),
            html_fetch_time_ms=int((html_fetch_time - start_time) * 1000),
            content_extraction_time_ms=int((content_extraction_time - html_fetch_time) * 1000),
            image_processing_time_ms=int((image_processing_time - sanitization_time) * 1000),
            quality_analysis_time_ms=int((quality_analysis_time - image_processing_time) * 1000),
            total_processing_time_ms=int(total_time * 1000),
            bytes_processed=len(html_content.encode("utf-8")),
            images_processed=len(processed_images),
            links_found=html_content.count("<a "),
            scripts_removed=html_content.count("<script"),
            styles_removed=html_content.count("<style"),
            ads_removed=0,  # This would be calculated by the HTML cleaner
        )

    def _create_extracted_content(
        self,
        request: ExtractionRequest,
        clean_content: str,
        metadata: ContentMetadata,
        processed_images: List[ProcessedImage],
        quality_metrics: QualityMetrics,
        language_info: LanguageInfo,
        extraction_stats: ExtractionStats,
    ) -> ExtractedContent:
        """Create the final extracted content object."""
        # Calculate derived fields
        word_count = quality_metrics.word_count
        reading_time = max(1, word_count // 200)  # 200 words per minute
        content_hash = hashlib.sha256(
            f"{metadata.og_title or ''}|{clean_content}|{request.url}".encode("utf-8")
        ).hexdigest()

        # Determine content type
        content_type = request.content_type or self._detect_content_type(clean_content)

        # Determine extraction method
        extraction_method = request.extraction_method or ExtractionMethod.READABILITY

        return ExtractedContent(
            url=request.url,
            canonical_url=metadata.canonical_link,
            title=metadata.og_title or "Untitled",
            content=clean_content,
            summary=metadata.og_description,
            author=metadata.author,
            publish_date=None,  # This would be extracted from metadata
            images=processed_images,
            videos=[],  # This would be extracted from content
            metadata=metadata,
            quality_metrics=quality_metrics,
            language_info=language_info,
            word_count=word_count,
            reading_time=reading_time,
            extraction_method=extraction_method,
            content_type=content_type,
            content_hash=content_hash,
            extraction_timestamp=datetime.utcnow(),
            processing_status=ProcessingStatus.COMPLETED,
            extraction_stats=extraction_stats,
            raw_html=None,  # Not storing raw HTML by default
            cleaned_html=None,  # Not storing cleaned HTML by default
            error_message=None,
        )

    def _detect_content_type(self, content: str) -> ContentType:
        """Detect content type based on content analysis."""
        # Simple heuristics for content type detection
        if any(keyword in content.lower() for keyword in ["pdf", "document", "download"]):
            return ContentType.PDF
        elif any(keyword in content.lower() for keyword in ["spreadsheet", "excel", "worksheet"]):
            return ContentType.XLSX
        elif any(keyword in content.lower() for keyword in ["presentation", "slides", "powerpoint"]):
            return ContentType.PPTX
        else:
            return ContentType.HTML

    def _generate_extraction_id(self, url: str) -> str:
        """Generate unique extraction ID."""
        timestamp = str(int(time.time() * 1000))
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
        return f"ext_{timestamp}_{url_hash}"

    async def batch_extract_content(
        self, requests: List[ExtractionRequest], max_concurrent: int = 10
    ) -> List[ExtractionResponse]:
        """
        Extract content from multiple URLs concurrently.

        Args:
            requests: List of extraction requests
            max_concurrent: Maximum concurrent extractions

        Returns:
            List of extraction responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_with_semaphore(request: ExtractionRequest) -> ExtractionResponse:
            async with semaphore:
                return await self.extract_content(request)

        tasks = [extract_with_semaphore(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
