"""
Image processing and optimization utilities.
"""

import asyncio
import io
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
from PIL import Image, ImageOps, ImageEnhance
import aiohttp
from loguru import logger

from ..models.content import ProcessedImage
from ..exceptions import ImageProcessingError


class ImageProcessor:
    """Advanced image processing and optimization system."""

    def __init__(
        self,
        max_width: int = 1200,
        max_height: int = 800,
        quality: int = 85,
        supported_formats: List[str] = None
    ):
        """Initialize image processor."""
        self.max_width = max_width
        self.max_height = max_height
        self.quality = quality
        self.supported_formats = supported_formats or ['JPEG', 'PNG', 'WebP', 'GIF']
        self.cache_dir = "/tmp/image_cache"

    async def process_images_from_content(
        self,
        content: str,
        base_url: str
    ) -> List[ProcessedImage]:
        """
        Process all images found in content.
        
        Args:
            content: HTML content containing images
            base_url: Base URL for resolving relative image URLs
            
        Returns:
            List of processed images
        """
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            processed_images = []
            img_tags = soup.find_all('img')
            
            for img_tag in img_tags:
                try:
                    src = img_tag.get('src')
                    if not src:
                        continue
                    
                    # Resolve relative URLs
                    img_url = urljoin(base_url, src)
                    
                    # Process individual image
                    processed_image = await self.process_single_image(
                        img_url,
                        alt_text=img_tag.get('alt', ''),
                        title=img_tag.get('title', '')
                    )
                    
                    if processed_image:
                        processed_images.append(processed_image)
                        
                except Exception as e:
                    logger.warning(f"Failed to process image {src}: {str(e)}")
                    continue
            
            return processed_images
            
        except Exception as e:
            logger.error(f"Image processing from content failed: {str(e)}")
            raise ImageProcessingError(f"Image processing failed: {str(e)}")

    async def process_single_image(
        self,
        image_url: str,
        alt_text: str = "",
        title: str = "",
        optimize: bool = True
    ) -> Optional[ProcessedImage]:
        """
        Process a single image with optimization.
        
        Args:
            image_url: URL of the image
            alt_text: Alt text for the image
            title: Title attribute for the image
            optimize: Whether to optimize the image
            
        Returns:
            ProcessedImage object or None if processing failed
        """
        try:
            logger.info(f"Processing image: {image_url}")
            
            # Download image
            image_data = await self._download_image(image_url)
            if not image_data:
                return None
            
            # Load image with PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Get original dimensions
            original_width, original_height = image.size
            
            # Process image
            if optimize:
                processed_image = await self._optimize_image(image)
                optimized_data = await self._encode_image(processed_image)
            else:
                processed_image = image
                optimized_data = image_data
            
            # Calculate quality score
            quality_score = await self._calculate_image_quality(processed_image)
            
            # Generate optimized URL (in production, this would upload to cloud storage)
            optimized_url = await self._generate_optimized_url(image_url, optimized_data)
            
            # Create processed image object
            processed_image_obj = ProcessedImage(
                url=image_url,
                local_path=None,  # Would be set in production
                optimized_url=optimized_url,
                width=processed_image.width,
                height=processed_image.height,
                file_size=len(optimized_data),
                format=processed_image.format or 'JPEG',
                alt_text=alt_text,
                caption=title,
                is_optimized=optimize,
                quality_score=quality_score
            )
            
            return processed_image_obj
            
        except Exception as e:
            logger.error(f"Single image processing failed for {image_url}: {str(e)}")
            return None

    async def _download_image(self, image_url: str) -> Optional[bytes]:
        """Download image from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=30) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        logger.warning(f"Failed to download image {image_url}: HTTP {response.status}")
                        return None
        except Exception as e:
            logger.warning(f"Image download failed for {image_url}: {str(e)}")
            return None

    async def _optimize_image(self, image: Image.Image) -> Image.Image:
        """Optimize image for web delivery."""
        try:
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparent images
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            if image.width > self.max_width or image.height > self.max_height:
                image = await self._resize_image(image, self.max_width, self.max_height)
            
            # Enhance image quality
            image = await self._enhance_image(image)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image optimization failed: {str(e)}")
            return image

    async def _resize_image(
        self,
        image: Image.Image,
        max_width: int,
        max_height: int
    ) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        try:
            # Calculate new dimensions
            width, height = image.size
            aspect_ratio = width / height
            
            if width > max_width:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
            elif height > max_height:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
            else:
                return image
            
            # Resize image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return resized_image
            
        except Exception as e:
            logger.warning(f"Image resize failed: {str(e)}")
            return image

    async def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image quality."""
        try:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Enhance color
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.05)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}")
            return image

    async def _encode_image(self, image: Image.Image) -> bytes:
        """Encode image to bytes with optimization."""
        try:
            output = io.BytesIO()
            
            # Save with optimization
            image.save(
                output,
                format='JPEG',
                quality=self.quality,
                optimize=True,
                progressive=True
            )
            
            return output.getvalue()
            
        except Exception as e:
            logger.warning(f"Image encoding failed: {str(e)}")
            # Fallback to basic encoding
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=75)
            return output.getvalue()

    async def _calculate_image_quality(self, image: Image.Image) -> float:
        """Calculate image quality score (0-1)."""
        try:
            # Basic quality metrics
            width, height = image.size
            total_pixels = width * height
            
            # Size score (prefer larger images)
            size_score = min(1.0, total_pixels / (800 * 600))
            
            # Aspect ratio score (prefer reasonable aspect ratios)
            aspect_ratio = width / height
            if 0.5 <= aspect_ratio <= 2.0:
                aspect_score = 1.0
            else:
                aspect_score = max(0.3, 1.0 - abs(aspect_ratio - 1.0) * 0.5)
            
            # Resolution score
            if width >= 400 and height >= 300:
                resolution_score = 1.0
            elif width >= 200 and height >= 150:
                resolution_score = 0.7
            else:
                resolution_score = 0.4
            
            # Overall quality score
            quality_score = (size_score * 0.4 + aspect_score * 0.3 + resolution_score * 0.3)
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Quality calculation failed: {str(e)}")
            return 0.5

    async def _generate_optimized_url(self, original_url: str, image_data: bytes) -> str:
        """Generate optimized image URL (placeholder implementation)."""
        try:
            # In production, this would upload to cloud storage and return the URL
            # For now, return a placeholder URL
            url_hash = hashlib.md5(image_data).hexdigest()[:8]
            parsed_url = urlparse(original_url)
            base_name = parsed_url.path.split('/')[-1]
            name, ext = base_name.rsplit('.', 1) if '.' in base_name else (base_name, 'jpg')
            
            return f"https://optimized-images.example.com/{name}_{url_hash}.{ext}"
            
        except Exception as e:
            logger.warning(f"Optimized URL generation failed: {str(e)}")
            return original_url

    async def create_thumbnail(
        self,
        image_url: str,
        thumbnail_size: Tuple[int, int] = (200, 200)
    ) -> Optional[ProcessedImage]:
        """Create thumbnail for an image."""
        try:
            # Download image
            image_data = await self._download_image(image_url)
            if not image_data:
                return None
            
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Create thumbnail
            image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            
            # Encode thumbnail
            thumbnail_data = await self._encode_image(image)
            
            # Generate thumbnail URL
            thumbnail_url = await self._generate_thumbnail_url(image_url, thumbnail_data)
            
            return ProcessedImage(
                url=image_url,
                optimized_url=thumbnail_url,
                width=image.width,
                height=image.height,
                file_size=len(thumbnail_data),
                format='JPEG',
                alt_text="Thumbnail",
                is_optimized=True,
                quality_score=0.8
            )
            
        except Exception as e:
            logger.error(f"Thumbnail creation failed for {image_url}: {str(e)}")
            return None

    async def _generate_thumbnail_url(self, original_url: str, thumbnail_data: bytes) -> str:
        """Generate thumbnail URL."""
        try:
            url_hash = hashlib.md5(thumbnail_data).hexdigest()[:8]
            parsed_url = urlparse(original_url)
            base_name = parsed_url.path.split('/')[-1]
            name, ext = base_name.rsplit('.', 1) if '.' in base_name else (base_name, 'jpg')
            
            return f"https://thumbnails.example.com/{name}_{url_hash}_thumb.{ext}"
            
        except Exception as e:
            logger.warning(f"Thumbnail URL generation failed: {str(e)}")
            return original_url

    async def detect_image_type(self, image_data: bytes) -> str:
        """Detect image type from binary data."""
        try:
            image = Image.open(io.BytesIO(image_data))
            return image.format or 'Unknown'
        except Exception as e:
            logger.warning(f"Image type detection failed: {str(e)}")
            return 'Unknown'

    async def get_image_metadata(self, image_data: bytes) -> Dict[str, Any]:
        """Extract metadata from image."""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            metadata = {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height,
                'has_transparency': image.mode in ('RGBA', 'LA', 'P'),
                'is_animated': getattr(image, 'is_animated', False)
            }
            
            # Extract EXIF data if available
            if hasattr(image, '_getexif') and image._getexif():
                exif_data = image._getexif()
                metadata['exif'] = {
                    'orientation': exif_data.get(274),  # Orientation tag
                    'datetime': exif_data.get(306),    # DateTime tag
                    'make': exif_data.get(271),        # Make tag
                    'model': exif_data.get(272)        # Model tag
                }
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {str(e)}")
            return {}

    async def validate_image(self, image_url: str) -> Dict[str, Any]:
        """Validate image and return validation results."""
        try:
            # Download image
            image_data = await self._download_image(image_url)
            if not image_data:
                return {
                    'valid': False,
                    'error': 'Failed to download image',
                    'size': 0,
                    'format': 'Unknown'
                }
            
            # Load and validate
            image = Image.open(io.BytesIO(image_data))
            
            return {
                'valid': True,
                'size': len(image_data),
                'format': image.format,
                'dimensions': image.size,
                'mode': image.mode,
                'file_size_mb': len(image_data) / (1024 * 1024)
            }
            
        except Exception as e:
            logger.warning(f"Image validation failed for {image_url}: {str(e)}")
            return {
                'valid': False,
                'error': str(e),
                'size': 0,
                'format': 'Unknown'
            }

    async def batch_process_images(
        self,
        image_urls: List[str],
        max_concurrent: int = 5
    ) -> List[ProcessedImage]:
        """Process multiple images concurrently."""
        try:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(url: str) -> Optional[ProcessedImage]:
                async with semaphore:
                    return await self.process_single_image(url)
            
            tasks = [process_with_semaphore(url) for url in image_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None results and exceptions
            processed_images = []
            for result in results:
                if isinstance(result, ProcessedImage):
                    processed_images.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Image processing failed: {str(result)}")
            
            return processed_images
            
        except Exception as e:
            logger.error(f"Batch image processing failed: {str(e)}")
            return []
