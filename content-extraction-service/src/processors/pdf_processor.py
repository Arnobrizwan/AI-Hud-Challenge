"""
PDF content processor using GCP Document AI and fallback methods.
"""

import asyncio
import io
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from loguru import logger

from ..exceptions import ContentProcessingError
from ..models.content import ContentType, ExtractionMethod, ProcessedImage
from ..services.document_ai_service import DocumentAIService


class PDFProcessor:
    """PDF content processor with multiple extraction methods."""

    def __init__(self, document_ai_service: DocumentAIService):
        """Initialize PDF processor."""
        self.document_ai_service = document_ai_service

    async def process_pdf(
        self,
        pdf_url: str,
        include_images: bool = True,
        include_tables: bool = True,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
    """
        Process PDF content and extract text, images, and metadata.

        Args:
            pdf_url: URL or path to PDF file
            include_images: Whether to extract images
            include_tables: Whether to extract tables
            include_metadata: Whether to extract metadata

        Returns:
            Dictionary with processed PDF data
        """
        try:
            logger.info(f"Processing PDF content from {pdf_url}")

            # Try Document AI first (most accurate)
            try:
                return await self._process_with_document_ai(
                    pdf_url, include_images, include_tables, include_metadata
                )
            except Exception as e:
                logger.warning(
                    f"Document AI processing failed, trying fallback: {str(e)}")

                # Fallback to PyPDF2
                try:
                    return await self._process_with_pypdf2(
                        pdf_url, include_images, include_metadata
                    )
                except Exception as e2:
                    logger.warning(
                        f"PyPDF2 processing failed, trying pdfplumber: {str(e2)}")

                    # Final fallback to pdfplumber
                    return await self._process_with_pdfplumber(
                        pdf_url, include_images, include_tables, include_metadata
                    )

        except Exception as e:
            logger.error(f"PDF processing failed for {pdf_url}: {str(e)}")
            raise ContentProcessingError(f"PDF processing failed: {str(e)}")

    async def _process_with_document_ai(self,
                                        pdf_url: str,
                                        include_images: bool,
                                        include_tables: bool,
                                        include_metadata: bool) -> Dict[str, Any]:
    """Process PDF using GCP Document AI."""
        try:
            # Extract text and structure
            document_data = await self.document_ai_service.process_document(
                pdf_url,
                extract_text=True,
                extract_images=include_images,
                extract_tables=include_tables,
                extract_metadata=include_metadata,
            )

            # Process extracted content
            content = document_data.get("text", "")
            images = []
            tables = []
            metadata = {}

            if include_images and "images" in document_data:
                images = await self._process_document_ai_images(document_data["images"])

            if include_tables and "tables" in document_data:
                tables = self._process_document_ai_tables(
                    document_data["tables"])

            if include_metadata and "metadata" in document_data:
                metadata = document_data["metadata"]

            return {
                "content": content,
                "images": images,
                "tables": tables,
                "metadata": metadata,
                "content_type": ContentType.PDF,
                "extraction_method": ExtractionMethod.DOCUMENT_AI,
                "pages": document_data.get("pages", 0),
                "confidence": document_data.get("confidence", 0.0),
            }

        except Exception as e:
            logger.error(f"Document AI processing failed: {str(e)}")
            raise

    async def _process_with_pypdf2(
        self, pdf_url: str, include_images: bool, include_metadata: bool
    ) -> Dict[str, Any]:
    """Process PDF using PyPDF2 as fallback."""
        try:
            import PyPDF2
            import requests

            # Download PDF content
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()

            # Create PDF reader
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))

            # Extract text
            content_parts = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    content_parts.append(text)

            content = "\n\n".join(content_parts)

            # Extract metadata
            metadata = {}
            if include_metadata and pdf_reader.metadata:
                metadata = {
                    "title": pdf_reader.metadata.get(
                        "/Title",
                        ""),
                    "author": pdf_reader.metadata.get(
                        "/Author",
                        ""),
                    "subject": pdf_reader.metadata.get(
                        "/Subject",
                        ""),
                    "creator": pdf_reader.metadata.get(
                        "/Creator",
                        ""),
                    "producer": pdf_reader.metadata.get(
                        "/Producer",
                        ""),
                    "creation_date": str(
                        pdf_reader.metadata.get(
                            "/CreationDate",
                            "")),
                    "modification_date": str(
                        pdf_reader.metadata.get(
                            "/ModDate",
                            "")),
                }

            return {
                "content": content,
                "images": [],  # PyPDF2 doesn't extract images well
                "tables": [],  # PyPDF2 doesn't extract tables
                "metadata": metadata,
                "content_type": ContentType.PDF,
                "extraction_method": ExtractionMethod.PYPDF2,
                "pages": len(pdf_reader.pages),
                "confidence": 0.7,  # Lower confidence for PyPDF2
            }

        except Exception as e:
            logger.error(f"PyPDF2 processing failed: {str(e)}")
            raise

    async def _process_with_pdfplumber(self,
                                       pdf_url: str,
                                       include_images: bool,
                                       include_tables: bool,
                                       include_metadata: bool) -> Dict[str, Any]:
    """Process PDF using pdfplumber as final fallback."""
        try:
            import pdfplumber
            import requests

            # Download PDF content
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()

            content_parts = []
            images = []
            tables = []
            metadata = {}

            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                # Extract text from each page
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        content_parts.append(text)

                    # Extract tables
                    if include_tables:
                        page_tables = page.extract_tables()
                        for table in page_tables:
                            if table:
                                tables.append(
                                    {"page": page_num + 1, "data": table})

                    # Extract images
                    if include_images:
                        page_images = page.images
                        for img in page_images:
                            images.append(
                                {
                                    "page": page_num + 1,
                                    "bbox": img["bbox"],
                                    "width": img["width"],
                                    "height": img["height"],
                                }
                            )

                # Extract metadata
                if include_metadata and pdf.metadata:
                    metadata = {
                        "title": pdf.metadata.get(
                            "Title", ""), "author": pdf.metadata.get(
                            "Author", ""), "subject": pdf.metadata.get(
                            "Subject", ""), "creator": pdf.metadata.get(
                            "Creator", ""), "producer": pdf.metadata.get(
                            "Producer", ""), "creation_date": str(
                            pdf.metadata.get(
                                "CreationDate", "")), "modification_date": str(
                                    pdf.metadata.get(
                                        "ModDate", "")), }

            content = "\n\n".join(content_parts)

            return {
                "content": content,
                "images": images,
                "tables": tables,
                "metadata": metadata,
                "content_type": ContentType.PDF,
                "extraction_method": ExtractionMethod.PDFPLUMBER,
                "pages": len(pdf.pages),
                "confidence": 0.8,  # Higher confidence for pdfplumber
            }

        except Exception as e:
            logger.error(f"pdfplumber processing failed: {str(e)}")
            raise

    async def _process_document_ai_images(
            self, images_data: List[Dict]) -> List[ProcessedImage]:
        """Process images extracted by Document AI."""
        processed_images = []

        for img_data in images_data:
            try:
                processed_image = ProcessedImage(
                    url=img_data.get("url", ""),
                    width=img_data.get("width", 0),
                    height=img_data.get("height", 0),
                    file_size=img_data.get("file_size", 0),
                    format=img_data.get("format", "Unknown"),
                    alt_text=img_data.get("alt_text", ""),
                    caption=img_data.get("caption", ""),
                    is_optimized=False,
                    quality_score=img_data.get("quality_score", 0.0),
                )

                processed_images.append(processed_image)

            except Exception as e:
                logger.warning(
                    f"Failed to process Document AI image: {str(e)}")
                continue

        return processed_images

    def _process_document_ai_tables(
            self, tables_data: List[Dict]) -> List[Dict[str, Any]]:
        """Process tables extracted by Document AI."""
        processed_tables = []

        for table_data in tables_data:
            try:
                processed_table = {
                    "page": table_data.get("page", 0),
                    "headers": table_data.get("headers", []),
                    "rows": table_data.get("rows", []),
                    "confidence": table_data.get("confidence", 0.0),
                    "bbox": table_data.get("bbox", {}),
                }

                processed_tables.append(processed_table)

            except Exception as e:
                logger.warning(
                    f"Failed to process Document AI table: {str(e)}")
                continue

        return processed_tables

    async def extract_text_only(self, pdf_url: str) -> str:
        """Extract only text content from PDF."""
        try:
            result = await self.process_pdf(
                pdf_url, include_images=False, include_tables=False, include_metadata=False
            )
            return result.get("content", "")
        except Exception as e:
            logger.error(f"Text extraction failed for {pdf_url}: {str(e)}")
            raise ContentProcessingError(f"Text extraction failed: {str(e)}")

    async def extract_metadata_only(self, pdf_url: str) -> Dict[str, Any]:
    """Extract only metadata from PDF."""
        try:
            result = await self.process_pdf(
                pdf_url, include_images=False, include_tables=False, include_metadata=True
            )
            return result.get("metadata", {})
        except Exception as e:
            logger.error(f"Metadata extraction failed for {pdf_url}: {str(e)}")
            raise ContentProcessingError(
                f"Metadata extraction failed: {str(e)}")

    def _is_valid_pdf_url(self, url: str) -> bool:
        """Check if URL points to a valid PDF file."""
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            return path.endswith(".pdf") or "pdf" in path
        except Exception:
            return False

    async def get_pdf_info(self, pdf_url: str) -> Dict[str, Any]:
    """Get basic information about PDF without full processing."""
        try:
            import requests

            # Download first few bytes to check PDF header
            response = requests.get(pdf_url, stream=True, timeout=10)
            response.raise_for_status()

            # Read first 1024 bytes
            first_chunk = response.raw.read(1024)

            # Check PDF header
            is_pdf = first_chunk.startswith(b"%PDF-")

            # Get content length
            content_length = response.headers.get("content-length")
            file_size = int(content_length) if content_length else 0

            return {
                "is_pdf": is_pdf,
                "file_size": file_size,
                "content_type": response.headers.get("content-type", ""),
                "accessible": True,
            }

        except Exception as e:
            logger.error(f"PDF info check failed for {pdf_url}: {str(e)}")
            return {
                "is_pdf": False,
                "file_size": 0,
                "content_type": "",
                "accessible": False,
                "error": str(e),
            }
