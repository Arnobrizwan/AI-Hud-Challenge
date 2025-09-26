"""Custom exceptions for the content extraction service."""


class ContentExtractionError(Exception):
    """Base exception for content extraction errors."""
    pass


class QualityAnalysisError(ContentExtractionError):
    """Exception raised when quality analysis fails."""
    pass


class DocumentProcessingError(ContentExtractionError):
    """Exception raised when document processing fails."""
    pass


class MetadataExtractionError(ContentExtractionError):
    """Exception raised when metadata extraction fails."""
    pass


class ImageProcessingError(ContentExtractionError):
    """Exception raised when image processing fails."""
    pass
