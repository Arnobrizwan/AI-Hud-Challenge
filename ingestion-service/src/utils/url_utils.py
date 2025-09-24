"""
URL utilities for canonicalization, validation, and domain handling.
"""

import re
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode
from urllib.robotparser import RobotFileParser
import tldextract
import logging

logger = logging.getLogger(__name__)


class URLUtils:
    """URL processing utilities."""
    
    def __init__(self):
        # Common URL patterns to clean
        self.clean_patterns = [
            # Remove common tracking parameters
            (r'[?&](utm_[^&]+)', ''),
            (r'[?&](fbclid=[^&]+)', ''),
            (r'[?&](gclid=[^&]+)', ''),
            (r'[?&](ref=[^&]+)', ''),
            (r'[?&](source=[^&]+)', ''),
            (r'[?&](campaign=[^&]+)', ''),
            (r'[?&](medium=[^&]+)', ''),
            (r'[?&](_ga=[^&]+)', ''),
            (r'[?&](_gid=[^&]+)', ''),
            (r'[?&](mc_[^&]+)', ''),
            (r'[?&](igshid=[^&]+)', ''),
            (r'[?&](twclid=[^&]+)', ''),
            # Remove session IDs
            (r'[?&](sessionid=[^&]+)', ''),
            (r'[?&](jsessionid=[^&]+)', ''),
            (r'[?&](phpsessid=[^&]+)', ''),
            # Remove other common parameters
            (r'[?&](timestamp=[^&]+)', ''),
            (r'[?&](time=[^&]+)', ''),
            (r'[?&](date=[^&]+)', ''),
        ]
        
        # URL patterns to skip
        self.skip_patterns = [
            r'\.(pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|tar|gz)$',
            r'\.(jpg|jpeg|png|gif|bmp|svg|webp|ico)$',
            r'\.(mp3|mp4|avi|mov|wmv|flv|webm)$',
            r'\.(css|js|xml|rss|atom)$',
            r'#.*$',  # Remove fragments
        ]
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        if not url or not isinstance(url, str):
            return False
        
        # Basic URL pattern check
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return bool(url_pattern.match(url))
    
    def normalize_url(self, url: str, base_url: str = None) -> str:
        """Normalize URL by cleaning parameters and converting to absolute URL."""
        if not self.is_valid_url(url):
            return url
        
        # Parse URL
        parsed = urlparse(url)
        
        # Convert to absolute URL if needed
        if base_url and not parsed.netloc:
            url = urljoin(base_url, url)
            parsed = urlparse(url)
        
        # Clean the URL
        cleaned_url = self._clean_url_parameters(url)
        
        # Parse cleaned URL
        parsed = urlparse(cleaned_url)
        
        # Normalize components
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        
        # Remove port if it's the default port
        if scheme == 'http' and parsed.port == 80:
            netloc = netloc.replace(':80', '')
        elif scheme == 'https' and parsed.port == 443:
            netloc = netloc.replace(':443', '')
        
        # Normalize path
        path = parsed.path
        if not path:
            path = '/'
        
        # Remove trailing slash for non-root paths
        if path != '/' and path.endswith('/'):
            path = path[:-1]
        
        # Rebuild URL
        normalized = urlunparse((
            scheme,
            netloc,
            path,
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        
        return normalized
    
    def _clean_url_parameters(self, url: str) -> str:
        """Clean URL by removing tracking and unnecessary parameters."""
        for pattern, replacement in self.clean_patterns:
            url = re.sub(pattern, replacement, url)
        
        # Clean up multiple ? and & characters
        url = re.sub(r'\?+', '?', url)
        url = re.sub(r'&+', '&', url)
        url = re.sub(r'\?&', '?', url)
        url = re.sub(r'&$', '', url)
        url = re.sub(r'\?$', '', url)
        
        return url
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if not self.is_valid_url(url):
            return ""
        
        parsed = urlparse(url)
        return parsed.netloc.lower()
    
    def extract_domain_info(self, url: str) -> Dict[str, str]:
        """Extract detailed domain information."""
        if not self.is_valid_url(url):
            return {}
        
        extracted = tldextract.extract(url)
        
        return {
            'domain': extracted.domain,
            'suffix': extracted.suffix,
            'subdomain': extracted.subdomain,
            'registered_domain': f"{extracted.domain}.{extracted.suffix}",
            'full_domain': f"{extracted.subdomain}.{extracted.domain}.{extracted.suffix}" if extracted.subdomain else f"{extracted.domain}.{extracted.suffix}"
        }
    
    def is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs are from the same domain."""
        domain1 = self.extract_domain(url1)
        domain2 = self.extract_domain(url2)
        return domain1 == domain2
    
    def is_subdomain(self, url: str, parent_domain: str) -> bool:
        """Check if URL is a subdomain of parent domain."""
        domain_info = self.extract_domain_info(url)
        return domain_info.get('registered_domain', '').endswith(parent_domain)
    
    def should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped based on patterns."""
        for pattern in self.skip_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False
    
    def extract_url_parameters(self, url: str) -> Dict[str, List[str]]:
        """Extract URL parameters."""
        if not self.is_valid_url(url):
            return {}
        
        parsed = urlparse(url)
        return parse_qs(parsed.query)
    
    def build_url_with_parameters(self, base_url: str, parameters: Dict[str, Any]) -> str:
        """Build URL with parameters."""
        if not self.is_valid_url(base_url):
            return base_url
        
        parsed = urlparse(base_url)
        query_params = parse_qs(parsed.query)
        
        # Add new parameters
        for key, value in parameters.items():
            if isinstance(value, list):
                query_params[key] = value
            else:
                query_params[key] = [str(value)]
        
        # Build query string
        query_string = urlencode(query_params, doseq=True)
        
        # Rebuild URL
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            query_string,
            parsed.fragment
        ))
    
    def get_url_depth(self, url: str) -> int:
        """Get URL depth (number of path segments)."""
        if not self.is_valid_url(url):
            return 0
        
        parsed = urlparse(url)
        path_segments = [seg for seg in parsed.path.split('/') if seg]
        return len(path_segments)
    
    def is_deep_url(self, url: str, max_depth: int = 3) -> bool:
        """Check if URL is too deep."""
        return self.get_url_depth(url) > max_depth
    
    def extract_base_url(self, url: str) -> str:
        """Extract base URL (scheme + netloc)."""
        if not self.is_valid_url(url):
            return ""
        
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def is_secure_url(self, url: str) -> bool:
        """Check if URL uses HTTPS."""
        if not self.is_valid_url(url):
            return False
        
        parsed = urlparse(url)
        return parsed.scheme.lower() == 'https'
    
    def convert_to_https(self, url: str) -> str:
        """Convert HTTP URL to HTTPS if possible."""
        if not self.is_valid_url(url):
            return url
        
        parsed = urlparse(url)
        if parsed.scheme.lower() == 'http':
            return urlunparse((
                'https',
                parsed.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
        
        return url
    
    def extract_path_segments(self, url: str) -> List[str]:
        """Extract path segments from URL."""
        if not self.is_valid_url(url):
            return []
        
        parsed = urlparse(url)
        return [seg for seg in parsed.path.split('/') if seg]
    
    def is_relative_url(self, url: str) -> bool:
        """Check if URL is relative."""
        return not url.startswith(('http://', 'https://'))
    
    def resolve_relative_url(self, relative_url: str, base_url: str) -> str:
        """Resolve relative URL against base URL."""
        if not self.is_relative_url(relative_url):
            return relative_url
        
        return urljoin(base_url, relative_url)
    
    def get_url_fingerprint(self, url: str) -> str:
        """Get URL fingerprint for duplicate detection."""
        normalized = self.normalize_url(url)
        parsed = urlparse(normalized)
        
        # Create fingerprint from normalized components
        fingerprint_parts = [
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params
        ]
        
        return '|'.join(fingerprint_parts)
    
    def is_news_url(self, url: str) -> bool:
        """Check if URL appears to be a news article."""
        if not self.is_valid_url(url):
            return False
        
        # News URL patterns
        news_patterns = [
            r'/article/',
            r'/news/',
            r'/story/',
            r'/post/',
            r'/blog/',
            r'/\d{4}/\d{2}/\d{2}/',  # Date pattern
            r'/\d{4}/\d{2}/',  # Year/month pattern
            r'\.html$',
            r'\.php$',
            r'\.asp$',
            r'\.aspx$',
        ]
        
        for pattern in news_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        
        return False
    
    def extract_article_id(self, url: str) -> Optional[str]:
        """Extract article ID from URL."""
        if not self.is_valid_url(url):
            return None
        
        # Common ID patterns
        id_patterns = [
            r'/(\d+)/?$',  # Numeric ID at end
            r'/article/(\d+)',  # /article/123
            r'/news/(\d+)',  # /news/123
            r'/story/(\d+)',  # /story/123
            r'id=(\d+)',  # ?id=123
            r'/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})',  # UUID
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None


# Global URL utils instance
url_utils = URLUtils()
