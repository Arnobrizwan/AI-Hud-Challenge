"""
GCP Document AI service for PDF and document processing.
"""

import asyncio
import io
from typing import Dict, Any, List, Optional, BinaryIO
from google.cloud import documentai
from google.cloud import storage
from loguru import logger

from ..exceptions import DocumentProcessingError


class DocumentAIService:
    """GCP Document AI service for document processing."""

    def __init__(
        self,
        project_id: str,
        location: str = "us",
        processor_id: str = None,
        bucket_name: str = None
    ):
        """Initialize Document AI service."""
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        self.bucket_name = bucket_name
        
        # Initialize clients
        self.documentai_client = documentai.DocumentProcessorServiceClient()
        self.storage_client = storage.Client(project=project_id)
        
        # Processor name
        self.processor_name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    async def process_document(
        self,
        document_path: str,
        extract_text: bool = True,
        extract_images: bool = False,
        extract_tables: bool = False,
        extract_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Process document using Document AI.
        
        Args:
            document_path: Path to document (GCS URI or local path)
            extract_text: Whether to extract text
            extract_images: Whether to extract images
            extract_tables: Whether to extract tables
            extract_metadata: Whether to extract metadata
            
        Returns:
            Dictionary with extracted content
        """
        try:
            logger.info(f"Processing document: {document_path}")
            
            # Prepare document
            document = await self._prepare_document(document_path)
            
            # Configure processing request
            request = documentai.ProcessRequest(
                name=self.processor_name,
                document=document
            )
            
            # Process document
            result = self.documentai_client.process_document(request=request)
            document = result.document
            
            # Extract content based on configuration
            extracted_content = {
                'text': '',
                'images': [],
                'tables': [],
                'metadata': {},
                'pages': len(document.pages),
                'confidence': 0.0
            }
            
            if extract_text:
                extracted_content['text'] = await self._extract_text(document)
            
            if extract_images:
                extracted_content['images'] = await self._extract_images(document)
            
            if extract_tables:
                extracted_content['tables'] = await self._extract_tables(document)
            
            if extract_metadata:
                extracted_content['metadata'] = await self._extract_metadata(document)
            
            # Calculate overall confidence
            extracted_content['confidence'] = await self._calculate_confidence(document)
            
            return extracted_content
            
        except Exception as e:
            logger.error(f"Document processing failed for {document_path}: {str(e)}")
            raise DocumentProcessingError(f"Document processing failed: {str(e)}")

    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using Document AI.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            result = await self.process_document(
                pdf_path,
                extract_text=True,
                extract_images=False,
                extract_tables=False,
                extract_metadata=False
            )
            return result.get('text', '')
        except Exception as e:
            logger.error(f"PDF text extraction failed for {pdf_path}: {str(e)}")
            raise DocumentProcessingError(f"PDF text extraction failed: {str(e)}")

    async def extract_text_from_document(self, document_path: str) -> str:
        """
        Extract text from document using Document AI.
        
        Args:
            document_path: Path to document file
            
        Returns:
            Extracted text content
        """
        try:
            result = await self.process_document(
                document_path,
                extract_text=True,
                extract_images=False,
                extract_tables=False,
                extract_metadata=False
            )
            return result.get('text', '')
        except Exception as e:
            logger.error(f"Document text extraction failed for {document_path}: {str(e)}")
            raise DocumentProcessingError(f"Document text extraction failed: {str(e)}")

    async def _prepare_document(self, document_path: str) -> documentai.Document:
        """Prepare document for processing."""
        try:
            if document_path.startswith('gs://'):
                # GCS URI
                return documentai.Document(
                    gcs_source=documentai.GcsSource(uri=document_path)
                )
            else:
                # Local file
                with open(document_path, 'rb') as f:
                    content = f.read()
                
                return documentai.Document(
                    content=content,
                    mime_type=self._detect_mime_type(document_path)
                )
        except Exception as e:
            logger.error(f"Document preparation failed: {str(e)}")
            raise DocumentProcessingError(f"Document preparation failed: {str(e)}")

    async def _extract_text(self, document: documentai.Document) -> str:
        """Extract text from processed document."""
        try:
            text_parts = []
            
            for page in document.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        paragraph_text = self._get_text_from_layout(paragraph.layout, document.text)
                        if paragraph_text.strip():
                            text_parts.append(paragraph_text)
            
            return '\n\n'.join(text_parts)
        except Exception as e:
            logger.warning(f"Text extraction failed: {str(e)}")
            return ""

    async def _extract_images(self, document: documentai.Document) -> List[Dict[str, Any]]:
        """Extract images from processed document."""
        try:
            images = []
            
            for page in document.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for element in paragraph.layout.text_anchor.text_segments:
                            # Check if this element contains an image
                            if hasattr(element, 'image') and element.image:
                                image_data = {
                                    'page': page.page_number,
                                    'bbox': {
                                        'x': element.bounding_poly.vertices[0].x,
                                        'y': element.bounding_poly.vertices[0].y,
                                        'width': element.bounding_poly.vertices[2].x - element.bounding_poly.vertices[0].x,
                                        'height': element.bounding_poly.vertices[2].y - element.bounding_poly.vertices[0].y
                                    },
                                    'confidence': element.confidence if hasattr(element, 'confidence') else 0.0
                                }
                                images.append(image_data)
            
            return images
        except Exception as e:
            logger.warning(f"Image extraction failed: {str(e)}")
            return []

    async def _extract_tables(self, document: documentai.Document) -> List[Dict[str, Any]]:
        """Extract tables from processed document."""
        try:
            tables = []
            
            for page in document.pages:
                for table in page.tables:
                    table_data = {
                        'page': page.page_number,
                        'headers': [],
                        'rows': [],
                        'confidence': table.confidence if hasattr(table, 'confidence') else 0.0,
                        'bbox': {
                            'x': table.bounding_poly.vertices[0].x,
                            'y': table.bounding_poly.vertices[0].y,
                            'width': table.bounding_poly.vertices[2].x - table.bounding_poly.vertices[0].x,
                            'height': table.bounding_poly.vertices[2].y - table.bounding_poly.vertices[0].y
                        }
                    }
                    
                    # Extract table rows
                    for row in table.body_rows:
                        row_data = []
                        for cell in row.cells:
                            cell_text = self._get_text_from_layout(cell.layout, document.text)
                            row_data.append(cell_text.strip())
                        table_data['rows'].append(row_data)
                    
                    # Extract header rows
                    for row in table.header_rows:
                        header_data = []
                        for cell in row.cells:
                            cell_text = self._get_text_from_layout(cell.layout, document.text)
                            header_data.append(cell_text.strip())
                        table_data['headers'].append(header_data)
                    
                    tables.append(table_data)
            
            return tables
        except Exception as e:
            logger.warning(f"Table extraction failed: {str(e)}")
            return []

    async def _extract_metadata(self, document: documentai.Document) -> Dict[str, Any]:
        """Extract metadata from processed document."""
        try:
            metadata = {
                'page_count': len(document.pages),
                'text_length': len(document.text) if document.text else 0,
                'language': document.language if hasattr(document, 'language') else 'en',
                'mime_type': document.mime_type if hasattr(document, 'mime_type') else 'unknown'
            }
            
            # Extract entity information if available
            if hasattr(document, 'entities') and document.entities:
                entities = []
                for entity in document.entities:
                    entity_data = {
                        'text': entity.text_anchor.content if hasattr(entity, 'text_anchor') else '',
                        'type': entity.type_ if hasattr(entity, 'type_') else 'unknown',
                        'confidence': entity.confidence if hasattr(entity, 'confidence') else 0.0
                    }
                    entities.append(entity_data)
                metadata['entities'] = entities
            
            return metadata
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {str(e)}")
            return {}

    async def _calculate_confidence(self, document: documentai.Document) -> float:
        """Calculate overall confidence score for the document."""
        try:
            total_confidence = 0.0
            element_count = 0
            
            for page in document.pages:
                for block in page.blocks:
                    if hasattr(block, 'confidence') and block.confidence:
                        total_confidence += block.confidence
                        element_count += 1
                
                for table in page.tables:
                    if hasattr(table, 'confidence') and table.confidence:
                        total_confidence += table.confidence
                        element_count += 1
            
            if element_count > 0:
                return total_confidence / element_count
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.0

    def _get_text_from_layout(self, layout: Any, document_text: str) -> str:
        """Extract text from layout element."""
        try:
            if not layout or not hasattr(layout, 'text_anchor'):
                return ""
            
            text_anchor = layout.text_anchor
            if not text_anchor or not text_anchor.text_segments:
                return ""
            
            text_segments = text_anchor.text_segments
            text_parts = []
            
            for segment in text_segments:
                start_index = segment.start_index if hasattr(segment, 'start_index') else 0
                end_index = segment.end_index if hasattr(segment, 'end_index') else len(document_text)
                
                if start_index < len(document_text) and end_index <= len(document_text):
                    text_parts.append(document_text[start_index:end_index])
            
            return ''.join(text_parts)
        except Exception as e:
            logger.warning(f"Text extraction from layout failed: {str(e)}")
            return ""

    def _detect_mime_type(self, file_path: str) -> str:
        """Detect MIME type from file extension."""
        extension = file_path.lower().split('.')[-1]
        
        mime_types = {
            'pdf': 'application/pdf',
            'doc': 'application/msword',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'txt': 'text/plain',
            'rtf': 'application/rtf',
            'odt': 'application/vnd.oasis.opendocument.text',
            'xls': 'application/vnd.ms-excel',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'ppt': 'application/vnd.ms-powerpoint',
            'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        }
        
        return mime_types.get(extension, 'application/octet-stream')

    async def batch_process_documents(
        self,
        document_paths: List[str],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Process multiple documents concurrently."""
        try:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(path: str) -> Dict[str, Any]:
                async with semaphore:
                    return await self.process_document(path)
            
            tasks = [process_with_semaphore(path) for path in document_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            processed_documents = []
            for result in results:
                if isinstance(result, dict):
                    processed_documents.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Document processing failed: {str(result)}")
                    processed_documents.append({'error': str(result)})
            
            return processed_documents
            
        except Exception as e:
            logger.error(f"Batch document processing failed: {str(e)}")
            return []

    async def upload_document_to_gcs(
        self,
        local_file_path: str,
        gcs_path: str
    ) -> str:
        """Upload document to Google Cloud Storage."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(gcs_path)
            
            with open(local_file_path, 'rb') as f:
                blob.upload_from_file(f)
            
            gcs_uri = f"gs://{self.bucket_name}/{gcs_path}"
            logger.info(f"Document uploaded to {gcs_uri}")
            
            return gcs_uri
            
        except Exception as e:
            logger.error(f"Document upload to GCS failed: {str(e)}")
            raise DocumentProcessingError(f"Document upload failed: {str(e)}")

    async def download_document_from_gcs(
        self,
        gcs_uri: str,
        local_file_path: str
    ) -> str:
        """Download document from Google Cloud Storage."""
        try:
            # Parse GCS URI
            if not gcs_uri.startswith('gs://'):
                raise ValueError("Invalid GCS URI format")
            
            path_parts = gcs_uri[5:].split('/', 1)
            bucket_name = path_parts[0]
            blob_name = path_parts[1]
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            blob.download_to_filename(local_file_path)
            logger.info(f"Document downloaded from {gcs_uri} to {local_file_path}")
            
            return local_file_path
            
        except Exception as e:
            logger.error(f"Document download from GCS failed: {str(e)}")
            raise DocumentProcessingError(f"Document download failed: {str(e)}")
