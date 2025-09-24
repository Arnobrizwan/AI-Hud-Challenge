"""
Content parsing utilities for HTML, XML, and text processing.
"""

import re
import html
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urljoin, urlparse
from datetime import datetime
import chardet
from bs4 import BeautifulSoup, Comment
import html2text
from readability import Document
import langdetect
from textstat import flesch_reading_ease, flesch_kincaid_grade
import logging

logger = logging.getLogger(__name__)


class ContentParser:
    """Content parsing utilities."""
    
    def __init__(self):
        self.html2text_converter = html2text.HTML2Text()
        self.html2text_converter.ignore_links = False
        self.html2text_converter.ignore_images = False
        self.html2text_converter.ignore_emphasis = False
        self.html2text_converter.body_width = 0  # Don't wrap lines
    
    def detect_encoding(self, content: bytes) -> str:
        """Detect content encoding."""
        try:
            result = chardet.detect(content)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            # If confidence is low, try common encodings
            if confidence < 0.7:
                for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        content.decode(enc)
                        return enc
                    except UnicodeDecodeError:
                        continue
            
            return encoding
        except Exception as e:
            logger.warning(f"Error detecting encoding: {e}")
            return 'utf-8'
    
    def decode_content(self, content: bytes, encoding: str = None) -> str:
        """Decode content with proper encoding."""
        if encoding is None:
            encoding = self.detect_encoding(content)
        
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            # Fallback to utf-8 with error handling
            return content.decode('utf-8', errors='ignore')
    
    def clean_html(self, html_content: str) -> str:
        """Clean HTML content by removing scripts, styles, and comments."""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Remove empty elements
        for element in soup.find_all():
            if not element.get_text(strip=True) and not element.find(['img', 'br', 'hr']):
                element.decompose()
        
        return str(soup)
    
    def extract_text_from_html(self, html_content: str, use_readability: bool = True) -> str:
        """Extract clean text from HTML content."""
        if use_readability:
            try:
                # Use readability to extract main content
                doc = Document(html_content)
                main_content = doc.summary()
                if main_content:
                    # Convert to text
                    text = self.html2text_converter.handle(main_content)
                    return self.clean_text(text)
            except Exception as e:
                logger.warning(f"Readability extraction failed: {e}")
        
        # Fallback to simple HTML to text conversion
        text = self.html2text_converter.handle(html_content)
        return self.clean_text(text)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Decode HTML entities
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
    
    def extract_title(self, html_content: str, fallback: str = None) -> str:
        """Extract title from HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Try different title sources
        title = None
        
        # 1. <title> tag
        title_tag = soup.find('title')
        if title_tag and title_tag.get_text(strip=True):
            title = title_tag.get_text(strip=True)
        
        # 2. Open Graph title
        if not title:
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                title = og_title.get('content').strip()
        
        # 3. Twitter title
        if not title:
            twitter_title = soup.find('meta', name='twitter:title')
            if twitter_title and twitter_title.get('content'):
                title = twitter_title.get('content').strip()
        
        # 4. H1 tag
        if not title:
            h1_tag = soup.find('h1')
            if h1_tag and h1_tag.get_text(strip=True):
                title = h1_tag.get_text(strip=True)
        
        # Clean and return title
        if title:
            return self.clean_text(title)
        
        return fallback or "Untitled"
    
    def extract_description(self, html_content: str) -> Optional[str]:
        """Extract description from HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Try different description sources
        description = None
        
        # 1. Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            description = meta_desc.get('content').strip()
        
        # 2. Open Graph description
        if not description:
            og_desc = soup.find('meta', property='og:description')
            if og_desc and og_desc.get('content'):
                description = og_desc.get('content').strip()
        
        # 3. Twitter description
        if not description:
            twitter_desc = soup.find('meta', name='twitter:description')
            if twitter_desc and twitter_desc.get('content'):
                description = twitter_desc.get('content').strip()
        
        # 4. First paragraph
        if not description:
            first_p = soup.find('p')
            if first_p and first_p.get_text(strip=True):
                description = first_p.get_text(strip=True)
        
        return self.clean_text(description) if description else None
    
    def extract_author(self, html_content: str) -> Optional[str]:
        """Extract author from HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Try different author sources
        author = None
        
        # 1. Meta author
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author and meta_author.get('content'):
            author = meta_author.get('content').strip()
        
        # 2. Open Graph author
        if not author:
            og_author = soup.find('meta', property='article:author')
            if og_author and og_author.get('content'):
                author = og_author.get('content').strip()
        
        # 3. JSON-LD author
        if not author:
            json_ld = soup.find('script', type='application/ld+json')
            if json_ld:
                try:
                    import json
                    data = json.loads(json_ld.string)
                    if isinstance(data, dict) and 'author' in data:
                        author_data = data['author']
                        if isinstance(author_data, dict) and 'name' in author_data:
                            author = author_data['name']
                        elif isinstance(author_data, str):
                            author = author_data
                except (json.JSONDecodeError, KeyError):
                    pass
        
        # 4. Byline class
        if not author:
            byline = soup.find(class_=re.compile(r'byline|author', re.I))
            if byline and byline.get_text(strip=True):
                author = byline.get_text(strip=True)
        
        return self.clean_text(author) if author else None
    
    def extract_image_url(self, html_content: str, base_url: str = None) -> Optional[str]:
        """Extract main image URL from HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Try different image sources
        image_url = None
        
        # 1. Open Graph image
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            image_url = og_image.get('content').strip()
        
        # 2. Twitter image
        if not image_url:
            twitter_image = soup.find('meta', name='twitter:image')
            if twitter_image and twitter_image.get('content'):
                image_url = twitter_image.get('content').strip()
        
        # 3. First img tag
        if not image_url:
            img_tag = soup.find('img')
            if img_tag and img_tag.get('src'):
                image_url = img_tag.get('src')
        
        # 4. Article image
        if not image_url:
            article_img = soup.find('img', class_=re.compile(r'article|main|featured', re.I))
            if article_img and article_img.get('src'):
                image_url = article_img.get('src')
        
        # Convert relative URL to absolute
        if image_url and base_url:
            image_url = urljoin(base_url, image_url)
        
        return image_url
    
    def extract_tags(self, html_content: str) -> List[str]:
        """Extract tags/categories from HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')
        tags = []
        
        # 1. Meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords and meta_keywords.get('content'):
            keywords = meta_keywords.get('content').split(',')
            tags.extend([tag.strip() for tag in keywords if tag.strip()])
        
        # 2. Open Graph tags
        og_tags = soup.find_all('meta', property='article:tag')
        for tag in og_tags:
            if tag.get('content'):
                tags.append(tag.get('content').strip())
        
        # 3. Schema.org tags
        json_ld = soup.find('script', type='application/ld+json')
        if json_ld:
            try:
                import json
                data = json.loads(json_ld.string)
                if isinstance(data, dict) and 'keywords' in data:
                    if isinstance(data['keywords'], list):
                        tags.extend(data['keywords'])
                    elif isinstance(data['keywords'], str):
                        tags.extend(data['keywords'].split(','))
            except (json.JSONDecodeError, KeyError):
                pass
        
        # 4. Class-based tags
        tag_elements = soup.find_all(class_=re.compile(r'tag|category|topic', re.I))
        for element in tag_elements:
            text = element.get_text(strip=True)
            if text:
                tags.append(text)
        
        # Clean and deduplicate tags
        cleaned_tags = []
        for tag in tags:
            cleaned = self.clean_text(tag)
            if cleaned and cleaned not in cleaned_tags:
                cleaned_tags.append(cleaned)
        
        return cleaned_tags[:10]  # Limit to 10 tags
    
    def detect_language(self, text: str) -> str:
        """Detect language of text content."""
        if not text or len(text.strip()) < 10:
            return 'en'  # Default to English
        
        try:
            # Use langdetect for language detection
            detected_lang = langdetect.detect(text)
            return detected_lang
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'en'  # Default to English
    
    def calculate_reading_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate reading metrics for text content."""
        if not text:
            return {
                'word_count': 0,
                'character_count': 0,
                'sentence_count': 0,
                'reading_time_minutes': 0,
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0
            }
        
        # Basic counts
        word_count = len(text.split())
        character_count = len(text)
        sentence_count = len(re.findall(r'[.!?]+', text))
        
        # Reading time (average 200 words per minute)
        reading_time_minutes = max(1, word_count // 200)
        
        # Flesch reading ease and grade level
        try:
            flesch_ease = flesch_reading_ease(text)
            flesch_grade = flesch_kincaid_grade(text)
        except Exception:
            flesch_ease = 0
            flesch_grade = 0
        
        return {
            'word_count': word_count,
            'character_count': character_count,
            'sentence_count': sentence_count,
            'reading_time_minutes': reading_time_minutes,
            'flesch_reading_ease': round(flesch_ease, 2),
            'flesch_kincaid_grade': round(flesch_grade, 2)
        }
    
    def extract_publication_date(self, html_content: str) -> Optional[datetime]:
        """Extract publication date from HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Try different date sources
        date_str = None
        
        # 1. Meta publication date
        meta_date = soup.find('meta', property='article:published_time')
        if meta_date and meta_date.get('content'):
            date_str = meta_date.get('content')
        elif not date_str:
            meta_date = soup.find('meta', attrs={'name': 'publication_date'})
            if meta_date and meta_date.get('content'):
                date_str = meta_date.get('content')
        
        # 2. JSON-LD date
        if not date_str:
            json_ld = soup.find('script', type='application/ld+json')
            if json_ld:
                try:
                    import json
                    data = json.loads(json_ld.string)
                    if isinstance(data, dict) and 'datePublished' in data:
                        date_str = data['datePublished']
                except (json.JSONDecodeError, KeyError):
                    pass
        
        # 3. Time tag
        if not date_str:
            time_tag = soup.find('time')
            if time_tag and time_tag.get('datetime'):
                date_str = time_tag.get('datetime')
        
        # Parse date string
        if date_str:
            try:
                from dateutil import parser
                return parser.parse(date_str)
            except Exception as e:
                logger.warning(f"Date parsing failed: {e}")
        
        return None
    
    def extract_canonical_url(self, html_content: str, base_url: str = None) -> Optional[str]:
        """Extract canonical URL from HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Look for canonical link
        canonical_link = soup.find('link', rel='canonical')
        if canonical_link and canonical_link.get('href'):
            canonical_url = canonical_link.get('href')
            
            # Convert relative URL to absolute
            if base_url and not canonical_url.startswith(('http://', 'https://')):
                canonical_url = urljoin(base_url, canonical_url)
            
            return canonical_url
        
        return None


# Global content parser instance
content_parser = ContentParser()
