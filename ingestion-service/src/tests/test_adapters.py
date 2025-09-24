"""
Unit tests for content adapters.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from src.adapters.rss_adapter import RSSAdapter
from src.adapters.api_adapter import APIAdapter
from src.models.content import SourceConfig, SourceType


class TestRSSAdapter:
    """Test cases for RSS adapter."""
    
    @pytest.fixture
    def source_config(self):
        """Create a test source config."""
        return SourceConfig(
            id="test-rss",
            name="Test RSS Feed",
            type=SourceType.RSS_FEED,
            url="https://example.com/feed.xml"
        )
    
    @pytest.fixture
    def rss_adapter(self, source_config):
        """Create a test RSS adapter."""
        with patch('src.adapters.rss_adapter.HTTPClient'):
            return RSSAdapter(source_config)
    
    def test_adapter_initialization(self, rss_adapter, source_config):
        """Test adapter initialization."""
        assert rss_adapter.source_config == source_config
        assert rss_adapter.is_processing is False
        assert rss_adapter.error_count == 0
        assert rss_adapter.success_count == 0
    
    @pytest.mark.asyncio
    async def test_test_connection(self, rss_adapter):
        """Test connection testing."""
        with patch.object(rss_adapter.http_client, 'head') as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_head.return_value = mock_response
            
            result = await rss_adapter.test_connection()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_test_connection_failure(self, rss_adapter):
        """Test connection testing failure."""
        with patch.object(rss_adapter.http_client, 'head') as mock_head:
            mock_head.side_effect = Exception("Connection failed")
            
            result = await rss_adapter.test_connection()
            assert result is False
    
    def test_get_source_info(self, rss_adapter):
        """Test getting source information."""
        info = rss_adapter.get_source_info()
        
        assert info['type'] == 'RSS/Atom Feed'
        assert info['url'] == rss_adapter.source_config.url
        assert info['rate_limit'] == rss_adapter.source_config.rate_limit


class TestAPIAdapter:
    """Test cases for API adapter."""
    
    @pytest.fixture
    def api_source_config(self):
        """Create a test API source config."""
        return SourceConfig(
            id="test-api",
            name="Test API",
            type=SourceType.API,
            url="https://api.example.com/articles",
            filters={
                'api_config': {
                    'articles_path': 'data',
                    'field_mapping': {
                        'title': 'title',
                        'url': 'url',
                        'content': 'content'
                    },
                    'pagination': {
                        'type': 'page_based',
                        'params': {
                            'page': '{page}',
                            'pageSize': 20
                        }
                    }
                }
            }
        )
    
    @pytest.fixture
    def api_adapter(self, api_source_config):
        """Create a test API adapter."""
        with patch('src.adapters.api_adapter.HTTPClient'):
            return APIAdapter(api_source_config)
    
    def test_adapter_initialization(self, api_adapter, api_source_config):
        """Test adapter initialization."""
        assert api_adapter.source_config == api_source_config
        assert api_adapter.current_page == 1
        assert api_adapter.has_more_pages is True
    
    def test_build_request_url(self, api_adapter):
        """Test URL building with pagination."""
        api_adapter.current_page = 2
        
        url = api_adapter._build_request_url()
        
        assert 'page=2' in url
        assert 'pageSize=20' in url
    
    def test_prepare_headers(self, api_adapter):
        """Test header preparation."""
        api_adapter.source_config.auth = {
            'type': 'bearer',
            'token': 'test-token'
        }
        
        headers = api_adapter._prepare_headers()
        
        assert 'Authorization' in headers
        assert headers['Authorization'] == 'Bearer test-token'
    
    def test_extract_articles_from_response(self, api_adapter):
        """Test article extraction from API response."""
        response_data = {
            'data': [
                {'title': 'Article 1', 'url': 'https://example.com/1'},
                {'title': 'Article 2', 'url': 'https://example.com/2'}
            ]
        }
        
        articles = api_adapter._extract_articles_from_response(response_data)
        
        assert len(articles) == 2
        assert articles[0]['title'] == 'Article 1'
        assert articles[1]['title'] == 'Article 2'
    
    def test_extract_field(self, api_adapter):
        """Test field extraction using dot notation."""
        data = {
            'author': {
                'name': 'John Doe',
                'email': 'john@example.com'
            }
        }
        
        # Test simple field
        title = api_adapter._extract_field(data, 'title')
        assert title is None
        
        # Test nested field
        name = api_adapter._extract_field(data, 'author.name')
        assert name == 'John Doe'
    
    @pytest.mark.asyncio
    async def test_has_more_pages(self, api_adapter):
        """Test pagination logic."""
        # Test page-based pagination
        articles = [{'title': f'Article {i}'} for i in range(20)]
        api_adapter.pagination_config = {'type': 'page_based', 'page_size': 20}
        
        has_more = await api_adapter._has_more_pages(articles)
        assert has_more is True
        
        # Test no more pages
        articles = [{'title': f'Article {i}'} for i in range(10)]
        has_more = await api_adapter._has_more_pages(articles)
        assert has_more is False


class TestAdapterBase:
    """Test cases for base adapter functionality."""
    
    @pytest.fixture
    def source_config(self):
        """Create a test source config."""
        return SourceConfig(
            id="test-source",
            name="Test Source",
            type=SourceType.RSS_FEED,
            url="https://example.com/feed.xml"
        )
    
    def test_create_ingestion_metadata(self, source_config):
        """Test ingestion metadata creation."""
        from src.adapters.base import BaseAdapter
        
        with patch('src.adapters.base.HTTPClient'):
            adapter = BaseAdapter(source_config)
            
            metadata = adapter._create_ingestion_metadata()
            
            assert metadata['source_id'] == source_config.id
            assert metadata['source_type'] == source_config.type
            assert metadata['source_url'] == source_config.url
            assert metadata['retry_count'] == 0
            assert metadata['robots_txt_respected'] is True
    
    def test_generate_article_id(self, source_config):
        """Test article ID generation."""
        from src.adapters.base import BaseAdapter
        
        with patch('src.adapters.base.HTTPClient'):
            adapter = BaseAdapter(source_config)
            
            article_id = adapter._generate_article_id(
                "https://example.com/article",
                "Test Article"
            )
            
            assert len(article_id) == 16
            assert article_id.isalnum()
    
    def test_calculate_content_hash(self, source_config):
        """Test content hash calculation."""
        from src.adapters.base import BaseAdapter
        
        with patch('src.adapters.base.HTTPClient'):
            adapter = BaseAdapter(source_config)
            
            content = "Test content for hashing"
            hash_value = adapter._calculate_content_hash(content)
            
            assert len(hash_value) == 64  # SHA256 hash length
            assert hash_value.isalnum()
    
    def test_is_valid_article(self, source_config):
        """Test article validation."""
        from src.adapters.base import BaseAdapter
        from src.models.content import NormalizedArticle
        
        with patch('src.adapters.base.HTTPClient'):
            adapter = BaseAdapter(source_config)
            
            # Valid article
            valid_article = NormalizedArticle(
                id="test-article",
                url="https://example.com/article",
                title="Test Article Title",
                source="Test Source",
                source_url="https://example.com",
                published_at=datetime.utcnow(),
                content_hash="abc123",
                word_count=100
            )
            
            assert adapter._is_valid_article(valid_article) is True
            
            # Invalid article (too short)
            invalid_article = NormalizedArticle(
                id="test-article",
                url="https://example.com/article",
                title="Short",
                source="Test Source",
                source_url="https://example.com",
                published_at=datetime.utcnow(),
                content_hash="abc123",
                word_count=10
            )
            
            assert adapter._is_valid_article(invalid_article) is False
    
    def test_apply_content_filters(self, source_config):
        """Test content filtering."""
        from src.adapters.base import BaseAdapter
        from src.models.content import NormalizedArticle
        
        with patch('src.adapters.base.HTTPClient'):
            adapter = BaseAdapter(source_config)
            
            # Set up filters
            adapter.source_config.filters = {
                'min_word_count': 50,
                'max_word_count': 1000,
                'languages': ['en'],
                'title_keywords': ['test', 'article']
            }
            
            # Valid article
            valid_article = NormalizedArticle(
                id="test-article",
                url="https://example.com/article",
                title="Test Article Title",
                source="Test Source",
                source_url="https://example.com",
                published_at=datetime.utcnow(),
                content_hash="abc123",
                word_count=100,
                language="en"
            )
            
            assert adapter._apply_content_filters(valid_article) is True
            
            # Invalid article (wrong language)
            invalid_article = NormalizedArticle(
                id="test-article",
                url="https://example.com/article",
                title="Test Article Title",
                source="Test Source",
                source_url="https://example.com",
                published_at=datetime.utcnow(),
                content_hash="abc123",
                word_count=100,
                language="es"
            )
            
            assert adapter._apply_content_filters(invalid_article) is False
