"""
Content normalization and processing logic.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from src.models.content import NormalizedArticle, ContentType, ProcessingStatus
from src.utils.content_parser import ContentParser
from src.utils.url_utils import URLUtils
from src.utils.date_utils import DateUtils

logger = logging.getLogger(__name__)


class ContentNormalizer:
    """Content normalization and processing utilities."""
    
    def __init__(
        self,
        content_parser: ContentParser = None,
        url_utils: URLUtils = None,
        date_utils: DateUtils = None
    ):
        self.content_parser = content_parser or ContentParser()
        self.url_utils = url_utils or URLUtils()
        self.date_utils = date_utils or DateUtils()
        
        # Content quality thresholds
        self.min_word_count = 50
        self.max_word_count = 50000
        self.min_title_length = 10
        self.max_title_length = 200
        
        # Duplicate detection settings
        self.duplicate_threshold = 0.8
        self.content_hash_algorithm = 'sha256'
    
    async def normalize_article(self, article: NormalizedArticle) -> NormalizedArticle:
        """Normalize and process an article."""
        try:
            # Update processing status
            article.processing_status = ProcessingStatus.NORMALIZING
            
            # Normalize basic fields
            article = await self._normalize_basic_fields(article)
            
            # Process content
            article = await self._process_content(article)
            
            # Detect content type
            article = await self._detect_content_type(article)
            
            # Calculate metrics
            article = await self._calculate_metrics(article)
            
            # Generate content hash
            article = await self._generate_content_hash(article)
            
            # Validate article
            article = await self._validate_article(article)
            
            # Update processing status
            article.processing_status = ProcessingStatus.COMPLETED
            
            return article
        
        except Exception as e:
            logger.error(f"Error normalizing article {article.id}: {e}")
            article.processing_status = ProcessingStatus.FAILED
            article.ingestion_metadata['error_message'] = str(e)
            return article
    
    async def _normalize_basic_fields(self, article: NormalizedArticle) -> NormalizedArticle:
        """Normalize basic article fields."""
        # Normalize title
        if article.title:
            article.title = self._clean_text(article.title)
            article.title = article.title[:self.max_title_length]
        
        # Normalize URL
        if article.url:
            article.url = self.url_utils.normalize_url(article.url)
            article.canonical_url = self.url_utils.normalize_url(article.url)
        
        # Normalize author
        if article.author:
            article.author = self._clean_text(article.author)
        
        # Normalize summary
        if article.summary:
            article.summary = self._clean_text(article.summary)
        
        # Normalize source
        if article.source:
            article.source = self._clean_text(article.source)
        
        # Normalize tags
        if article.tags:
            article.tags = [self._clean_text(tag) for tag in article.tags if tag and self._clean_text(tag)]
            article.tags = list(set(article.tags))  # Remove duplicates
        
        # Normalize published date
        if article.published_at:
            article.published_at = self.date_utils.normalize_date(article.published_at)
        
        # Normalize updated date
        if article.updated_at:
            article.updated_at = self.date_utils.normalize_date(article.updated_at)
        
        return article
    
    async def _process_content(self, article: NormalizedArticle) -> NormalizedArticle:
        """Process article content."""
        if not article.content:
            return article
        
        # Extract text from HTML if needed
        if self._is_html_content(article.content):
            article.content = self.content_parser.extract_text_from_html(article.content)
        
        # Clean content
        article.content = self._clean_text(article.content)
        
        # Extract additional metadata from content
        if not article.summary and article.content:
            article.summary = self._extract_summary(article.content)
        
        # Detect language
        text_for_language = article.content or article.summary or article.title
        if text_for_language:
            article.language = self.content_parser.detect_language(text_for_language)
        
        return article
    
    async def _detect_content_type(self, article: NormalizedArticle) -> NormalizedArticle:
        """Detect content type based on content analysis."""
        content_type = ContentType.ARTICLE  # Default
        
        # Analyze title and content for content type
        title_lower = article.title.lower() if article.title else ""
        content_lower = article.content.lower() if article.content else ""
        text_to_analyze = f"{title_lower} {content_lower}"
        
        # Blog post indicators
        blog_indicators = ['blog', 'post', 'opinion', 'thoughts', 'personal']
        if any(indicator in text_to_analyze for indicator in blog_indicators):
            content_type = ContentType.BLOG_POST
        
        # News item indicators
        news_indicators = ['breaking', 'news', 'report', 'update', 'alert']
        if any(indicator in text_to_analyze for indicator in news_indicators):
            content_type = ContentType.NEWS_ITEM
        
        # Press release indicators
        press_release_indicators = ['press release', 'announcement', 'statement', 'official']
        if any(indicator in text_to_analyze for indicator in press_release_indicators):
            content_type = ContentType.PRESS_RELEASE
        
        # Opinion indicators
        opinion_indicators = ['opinion', 'editorial', 'viewpoint', 'analysis', 'commentary']
        if any(indicator in text_to_analyze for indicator in opinion_indicators):
            content_type = ContentType.OPINION
        
        # Analysis indicators
        analysis_indicators = ['analysis', 'research', 'study', 'investigation', 'deep dive']
        if any(indicator in text_to_analyze for indicator in analysis_indicators):
            content_type = ContentType.ANALYSIS
        
        # Interview indicators
        interview_indicators = ['interview', 'q&a', 'conversation', 'discussion']
        if any(indicator in text_to_analyze for indicator in interview_indicators):
            content_type = ContentType.INTERVIEW
        
        # Review indicators
        review_indicators = ['review', 'rating', 'evaluation', 'assessment']
        if any(indicator in text_to_analyze for indicator in review_indicators):
            content_type = ContentType.REVIEW
        
        article.content_type = content_type
        return article
    
    async def _calculate_metrics(self, article: NormalizedArticle) -> NormalizedArticle:
        """Calculate article metrics."""
        if article.content:
            # Calculate reading metrics
            reading_metrics = self.content_parser.calculate_reading_metrics(article.content)
            article.word_count = reading_metrics['word_count']
            article.reading_time = reading_metrics['reading_time_minutes']
        else:
            article.word_count = 0
            article.reading_time = 0
        
        return article
    
    async def _generate_content_hash(self, article: NormalizedArticle) -> NormalizedArticle:
        """Generate content hash for duplicate detection."""
        # Create content for hashing
        content_for_hash = f"{article.title}|{article.content or ''}|{article.url}"
        
        # Generate hash
        if self.content_hash_algorithm == 'sha256':
            article.content_hash = hashlib.sha256(content_for_hash.encode('utf-8')).hexdigest()
        else:
            article.content_hash = hashlib.md5(content_for_hash.encode('utf-8')).hexdigest()
        
        return article
    
    async def _validate_article(self, article: NormalizedArticle) -> NormalizedArticle:
        """Validate article quality and completeness."""
        # Check minimum requirements
        if article.word_count < self.min_word_count:
            article.processing_status = ProcessingStatus.SKIPPED
            article.ingestion_metadata['error_message'] = f"Article too short: {article.word_count} words"
            return article
        
        if article.word_count > self.max_word_count:
            article.processing_status = ProcessingStatus.SKIPPED
            article.ingestion_metadata['error_message'] = f"Article too long: {article.word_count} words"
            return article
        
        if not article.title or len(article.title) < self.min_title_length:
            article.processing_status = ProcessingStatus.SKIPPED
            article.ingestion_metadata['error_message'] = "Title too short or missing"
            return article
        
        if len(article.title) > self.max_title_length:
            article.title = article.title[:self.max_title_length]
        
        # Check URL validity
        if not self.url_utils.is_valid_url(article.url):
            article.processing_status = ProcessingStatus.SKIPPED
            article.ingestion_metadata['error_message'] = "Invalid URL"
            return article
        
        # Check if URL should be skipped
        if self.url_utils.should_skip_url(article.url):
            article.processing_status = ProcessingStatus.SKIPPED
            article.ingestion_metadata['error_message'] = "URL should be skipped"
            return article
        
        # Check date validity
        if article.published_at and not self.date_utils.is_valid_date(article.published_at):
            article.processing_status = ProcessingStatus.SKIPPED
            article.ingestion_metadata['error_message'] = "Invalid publication date"
            return article
        
        return article
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Decode HTML entities
        import html
        text = html.unescape(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!!', text)
        text = re.sub(r'[?]{2,}', '??', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[''']', "'", text)
        
        return text.strip()
    
    def _is_html_content(self, content: str) -> bool:
        """Check if content contains HTML."""
        if not content:
            return False
        
        # Simple HTML detection
        html_patterns = [
            r'<[^>]+>',  # HTML tags
            r'&[a-zA-Z]+;',  # HTML entities
            r'&#[0-9]+;',  # Numeric HTML entities
        ]
        
        for pattern in html_patterns:
            if re.search(pattern, content):
                return True
        
        return False
    
    def _extract_summary(self, content: str, max_length: int = 200) -> str:
        """Extract summary from content."""
        if not content:
            return ""
        
        # Clean content
        clean_content = self._clean_text(content)
        
        # Extract first paragraph or sentence
        sentences = re.split(r'[.!?]+', clean_content)
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) <= max_length:
                return first_sentence
            else:
                return first_sentence[:max_length-3] + "..."
        
        # Fallback to truncated content
        if len(clean_content) <= max_length:
            return clean_content
        else:
            return clean_content[:max_length-3] + "..."
    
    async def detect_duplicates(
        self, 
        article: NormalizedArticle, 
        existing_articles: List[NormalizedArticle]
    ) -> List[Tuple[NormalizedArticle, float]]:
        """Detect duplicate articles."""
        duplicates = []
        
        for existing_article in existing_articles:
            # Check content hash first (exact match)
            if article.content_hash == existing_article.content_hash:
                duplicates.append((existing_article, 1.0))
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(article, existing_article)
            
            if similarity >= self.duplicate_threshold:
                duplicates.append((existing_article, similarity))
        
        # Sort by similarity (highest first)
        duplicates.sort(key=lambda x: x[1], reverse=True)
        
        return duplicates
    
    def _calculate_similarity(self, article1: NormalizedArticle, article2: NormalizedArticle) -> float:
        """Calculate similarity between two articles."""
        # Title similarity
        title_similarity = self._text_similarity(article1.title, article2.title)
        
        # Content similarity
        content_similarity = self._text_similarity(article1.content or "", article2.content or "")
        
        # URL similarity
        url_similarity = self._url_similarity(article1.url, article2.url)
        
        # Weighted average
        similarity = (
            title_similarity * 0.4 +
            content_similarity * 0.4 +
            url_similarity * 0.2
        )
        
        return similarity
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap."""
        if not text1 or not text2:
            return 0.0
        
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _url_similarity(self, url1: str, url2: str) -> float:
        """Calculate URL similarity."""
        if not url1 or not url2:
            return 0.0
        
        # Check if URLs are from the same domain
        domain1 = self.url_utils.extract_domain(url1)
        domain2 = self.url_utils.extract_domain(url2)
        
        if domain1 == domain2:
            return 0.8  # High similarity for same domain
        
        # Check if URLs are similar
        if url1 == url2:
            return 1.0
        
        # Check if one URL is a variation of the other
        if self._are_url_variations(url1, url2):
            return 0.6
        
        return 0.0
    
    def _are_url_variations(self, url1: str, url2: str) -> bool:
        """Check if URLs are variations of each other."""
        # Remove common tracking parameters
        clean_url1 = self.url_utils.normalize_url(url1)
        clean_url2 = self.url_utils.normalize_url(url2)
        
        # Check if they're the same after cleaning
        if clean_url1 == clean_url2:
            return True
        
        # Check if one is a subdirectory of the other
        if clean_url1.startswith(clean_url2) or clean_url2.startswith(clean_url1):
            return True
        
        return False
    
    async def enhance_article(self, article: NormalizedArticle) -> NormalizedArticle:
        """Enhance article with additional processing."""
        try:
            # Extract additional metadata
            if article.content:
                # Extract additional tags from content
                additional_tags = self._extract_tags_from_content(article.content)
                article.tags.extend(additional_tags)
                article.tags = list(set(article.tags))  # Remove duplicates
            
            # Extract additional author information
            if not article.author and article.content:
                article.author = self.content_parser.extract_author(article.content)
            
            # Extract additional image URLs
            if not article.image_url and article.content:
                article.image_url = self.content_parser.extract_image_url(article.content, article.url)
            
            # Extract additional summary
            if not article.summary and article.content:
                article.summary = self._extract_summary(article.content)
            
            return article
        
        except Exception as e:
            logger.warning(f"Error enhancing article {article.id}: {e}")
            return article
    
    def _extract_tags_from_content(self, content: str) -> List[str]:
        """Extract additional tags from content."""
        if not content:
            return []
        
        # Common tag patterns
        tag_patterns = [
            r'#(\w+)',  # Hashtags
            r'@(\w+)',  # Mentions
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Title case phrases
        ]
        
        tags = []
        for pattern in tag_patterns:
            matches = re.findall(pattern, content)
            tags.extend(matches)
        
        # Clean and filter tags
        cleaned_tags = []
        for tag in tags:
            tag = tag.strip().lower()
            if len(tag) > 2 and len(tag) < 50:  # Reasonable tag length
                cleaned_tags.append(tag)
        
        return list(set(cleaned_tags))  # Remove duplicates
