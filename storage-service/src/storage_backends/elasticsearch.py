"""
Elasticsearch Manager - Advanced full-text search and analytics
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from models import (
    Article, SearchRequest, SearchResult, SearchResultItem,
    IndexingResult
)
from config import Settings

logger = logging.getLogger(__name__)

class ElasticsearchManager:
    """Manage Elasticsearch for full-text search"""
    
    def __init__(self):
        self.es_client: Optional[AsyncElasticsearch] = None
        self.settings = Settings()
        self._initialized = False
        self._index_templates = {}
        
    async def initialize(self):
        """Initialize Elasticsearch client and indexes"""
        if self._initialized:
            return
            
        logger.info("Initializing Elasticsearch Manager...")
        
        try:
            # Create Elasticsearch client
            es_config = self.settings.get_elasticsearch_config()
            
            self.es_client = AsyncElasticsearch(
                hosts=es_config.hosts,
                basic_auth=(es_config.username, es_config.password) if es_config.username else None,
                verify_certs=es_config.verify_certs,
                timeout=es_config.timeout
            )
            
            # Test connection
            await self.es_client.ping()
            
            # Initialize index templates
            await self._initialize_index_templates()
            
            # Create default indexes
            await self._create_default_indexes()
            
            self._initialized = True
            logger.info("Elasticsearch Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch Manager: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup Elasticsearch client"""
        if self.es_client:
            await self.es_client.close()
            self.es_client = None
        
        self._initialized = False
        logger.info("Elasticsearch Manager cleanup complete")
    
    async def _initialize_index_templates(self):
        """Initialize Elasticsearch index templates"""
        try:
            # Article index template
            article_template = {
                "index_patterns": ["articles-*"],
                "template": {
                    "settings": {
                        "number_of_shards": 2,
                        "number_of_replicas": 1,
                        "analysis": {
                            "analyzer": {
                                "custom_text_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "stop", "snowball"]
                                },
                                "title_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "stop"]
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "id": {"type": "keyword"},
                            "title": {
                                "type": "text",
                                "analyzer": "title_analyzer",
                                "fields": {
                                    "keyword": {"type": "keyword"},
                                    "suggest": {"type": "completion"}
                                }
                            },
                            "content": {
                                "type": "text",
                                "analyzer": "custom_text_analyzer"
                            },
                            "summary": {
                                "type": "text",
                                "analyzer": "custom_text_analyzer"
                            },
                            "author": {
                                "type": "text",
                                "fields": {
                                    "keyword": {"type": "keyword"}
                                }
                            },
                            "source": {
                                "type": "keyword"
                            },
                            "published_at": {"type": "date"},
                            "categories": {"type": "keyword"},
                            "tags": {"type": "keyword"},
                            "language": {"type": "keyword"},
                            "url": {"type": "keyword"},
                            "title_boost": {"type": "float"},
                            "summary_boost": {"type": "float"},
                            "source_authority": {"type": "float"},
                            "indexed_at": {"type": "date"}
                        }
                    }
                }
            }
            
            await self.es_client.indices.put_index_template(
                name="articles_template",
                body=article_template
            )
            
            logger.info("Elasticsearch index templates initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize index templates: {e}")
            raise
    
    async def _create_default_indexes(self):
        """Create default indexes"""
        try:
            # Create current month index
            current_date = datetime.utcnow()
            index_name = f"articles-{current_date.strftime('%Y-%m')}"
            
            if not await self.es_client.indices.exists(index=index_name):
                await self.es_client.indices.create(index=index_name)
                logger.info(f"Created index: {index_name}")
            
        except Exception as e:
            logger.error(f"Failed to create default indexes: {e}")
            raise
    
    async def index_article(self, article: Article) -> IndexingResult:
        """Index article for full-text search"""
        if not self._initialized or not self.es_client:
            raise RuntimeError("Elasticsearch Manager not initialized")
        
        logger.info(f"Indexing article {article.id}")
        
        try:
            # Prepare document for indexing
            doc = {
                'id': article.id,
                'title': article.title,
                'content': article.content,
                'summary': article.summary,
                'author': article.author,
                'source': article.source,
                'published_at': article.published_at.isoformat() if article.published_at else None,
                'categories': article.categories,
                'tags': article.tags,
                'language': article.language,
                'url': article.url,
                'indexed_at': datetime.utcnow().isoformat()
            }
            
            # Add boost factors for ranking
            doc['title_boost'] = 2.0
            doc['summary_boost'] = 1.5
            doc['source_authority'] = await self._get_source_authority(article.source)
            
            # Determine index name based on publication date
            index_name = self._get_index_name(article.published_at)
            
            # Index in Elasticsearch
            result = await self.es_client.index(
                index=index_name,
                id=article.id,
                body=doc,
                refresh='wait_for'
            )
            
            logger.info(f"Article {article.id} indexed in {index_name}")
            
            return IndexingResult(
                article_id=article.id,
                index_name=index_name,
                operation=result['result'],
                version=result['_version']
            )
            
        except Exception as e:
            logger.error(f"Failed to index article {article.id}: {e}")
            raise
    
    async def search_articles(self, search_request: SearchRequest) -> SearchResult:
        """Advanced full-text search with ranking"""
        if not self._initialized or not self.es_client:
            raise RuntimeError("Elasticsearch Manager not initialized")
        
        logger.info(f"Searching articles with query: {search_request.query}")
        
        try:
            # Build complex query
            query = await self._build_search_query(search_request)
            
            # Add aggregations for faceted search
            aggregations = await self._build_aggregations(search_request)
            
            # Execute search
            search_body = {
                'query': query,
                'aggs': aggregations,
                'highlight': {
                    'fields': {
                        'title': {'number_of_fragments': 1},
                        'content': {'number_of_fragments': 3, 'fragment_size': 150}
                    }
                },
                'sort': self._build_sort_criteria(search_request),
                'from': search_request.offset,
                'size': search_request.limit
            }
            
            # Determine indices to search
            indices = self._get_search_indices(search_request.date_range)
            
            response = await self.es_client.search(
                index=indices,
                body=search_body
            )
            
            # Process results
            search_results = []
            for hit in response['hits']['hits']:
                search_results.append(SearchResultItem(
                    article_id=hit['_id'],
                    score=hit['_score'],
                    title=hit['_source']['title'],
                    summary=hit['_source'].get('summary'),
                    highlights=hit.get('highlight', {}),
                    source=hit['_source']['source'],
                    published_at=datetime.fromisoformat(hit['_source']['published_at'].replace('Z', '+00:00'))
                ))
            
            logger.info(f"Search completed, found {len(search_results)} results")
            
            return SearchResult(
                results=search_results,
                total_hits=response['hits']['total']['value'],
                aggregations=self._process_aggregations(response.get('aggregations', {})),
                search_duration=response['took'],
                query_explanation=self._explain_query(query) if search_request.explain else None
            )
            
        except Exception as e:
            logger.error(f"Article search failed: {e}")
            raise
    
    async def get_article_content(self, article_id: str) -> Dict[str, Any]:
        """Get article content from Elasticsearch"""
        if not self._initialized or not self.es_client:
            raise RuntimeError("Elasticsearch Manager not initialized")
        
        try:
            # Search across all article indices
            response = await self.es_client.search(
                index="articles-*",
                body={
                    "query": {
                        "term": {
                            "id": article_id
                        }
                    }
                }
            )
            
            if response['hits']['total']['value'] > 0:
                return response['hits']['hits'][0]['_source']
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get article content {article_id}: {e}")
            raise
    
    async def refresh_index(self, article_id: str):
        """Refresh index for an article"""
        if not self._initialized or not self.es_client:
            return
        
        try:
            # Refresh all article indices
            await self.es_client.indices.refresh(index="articles-*")
            
        except Exception as e:
            logger.warning(f"Failed to refresh index for article {article_id}: {e}")
    
    async def delete_article(self, article_id: str):
        """Delete article from all indices"""
        if not self._initialized or not self.es_client:
            raise RuntimeError("Elasticsearch Manager not initialized")
        
        try:
            # Delete from all article indices
            await self.es_client.delete_by_query(
                index="articles-*",
                body={
                    "query": {
                        "term": {
                            "id": article_id
                        }
                    }
                }
            )
            
            logger.info(f"Article {article_id} deleted from Elasticsearch")
            
        except Exception as e:
            logger.error(f"Failed to delete article {article_id}: {e}")
            raise
    
    def _get_index_name(self, published_at: Optional[datetime]) -> str:
        """Get index name based on publication date"""
        if published_at:
            return f"articles-{published_at.strftime('%Y-%m')}"
        else:
            return f"articles-{datetime.utcnow().strftime('%Y-%m')}"
    
    def _get_search_indices(self, date_range: Optional[Dict[str, datetime]]) -> str:
        """Get indices to search based on date range"""
        if not date_range:
            return "articles-*"
        
        start_date = date_range.get('start')
        end_date = date_range.get('end')
        
        if start_date and end_date:
            # Generate list of indices for date range
            indices = []
            current = start_date.replace(day=1)
            end = end_date.replace(day=1)
            
            while current <= end:
                indices.append(f"articles-{current.strftime('%Y-%m')}")
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
            
            return ",".join(indices)
        
        return "articles-*"
    
    async def _build_search_query(self, search_request: SearchRequest) -> Dict[str, Any]:
        """Build complex search query"""
        # Base query
        query_parts = []
        
        # Main text search
        if search_request.query:
            query_parts.append({
                "multi_match": {
                    "query": search_request.query,
                    "fields": [
                        "title^2.0",
                        "summary^1.5", 
                        "content^1.0"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            })
        
        # Filters
        filters = []
        
        if search_request.categories:
            filters.append({
                "terms": {
                    "categories": search_request.categories
                }
            })
        
        if search_request.sources:
            filters.append({
                "terms": {
                    "source": search_request.sources
                }
            })
        
        if search_request.languages:
            filters.append({
                "terms": {
                    "language": search_request.languages
                }
            })
        
        if search_request.date_range:
            date_filter = {"range": {"published_at": {}}}
            if search_request.date_range.get('start'):
                date_filter["range"]["published_at"]["gte"] = search_request.date_range['start'].isoformat()
            if search_request.date_range.get('end'):
                date_filter["range"]["published_at"]["lte"] = search_request.date_range['end'].isoformat()
            filters.append(date_filter)
        
        # Combine query and filters
        if query_parts and filters:
            query = {
                "bool": {
                    "must": query_parts,
                    "filter": filters
                }
            }
        elif query_parts:
            query = query_parts[0] if len(query_parts) == 1 else {"bool": {"must": query_parts}}
        elif filters:
            query = {"bool": {"filter": filters}}
        else:
            query = {"match_all": {}}
        
        return query
    
    async def _build_aggregations(self, search_request: SearchRequest) -> Dict[str, Any]:
        """Build aggregations for faceted search"""
        aggs = {
            "categories": {
                "terms": {
                    "field": "categories",
                    "size": 20
                }
            },
            "sources": {
                "terms": {
                    "field": "source",
                    "size": 20
                }
            },
            "languages": {
                "terms": {
                    "field": "language",
                    "size": 10
                }
            },
            "published_dates": {
                "date_histogram": {
                    "field": "published_at",
                    "calendar_interval": "month",
                    "min_doc_count": 1
                }
            }
        }
        
        return aggs
    
    def _build_sort_criteria(self, search_request: SearchRequest) -> List[Dict[str, Any]]:
        """Build sort criteria"""
        if search_request.sort_by == "published_at":
            return [{"published_at": {"order": search_request.sort_order}}]
        elif search_request.sort_by == "relevance":
            return ["_score"]
        else:
            # Default: relevance with recency boost
            return [
                "_score",
                {"published_at": {"order": "desc"}}
            ]
    
    def _process_aggregations(self, aggregations: Dict[str, Any]) -> Dict[str, Any]:
        """Process aggregation results"""
        processed = {}
        
        for agg_name, agg_data in aggregations.items():
            if 'buckets' in agg_data:
                processed[agg_name] = [
                    {
                        'key': bucket['key'],
                        'doc_count': bucket['doc_count']
                    }
                    for bucket in agg_data['buckets']
                ]
            else:
                processed[agg_name] = agg_data
        
        return processed
    
    def _explain_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Explain query structure"""
        return {
            "query_type": type(query).__name__,
            "query_structure": query,
            "explanation": "Query explanation would be generated by Elasticsearch"
        }
    
    async def _get_source_authority(self, source: str) -> float:
        """Get source authority score"""
        # This would typically be stored in a database or cache
        # For now, return a default value
        authority_scores = {
            'reuters': 0.9,
            'ap': 0.9,
            'bbc': 0.85,
            'cnn': 0.8,
            'nytimes': 0.85,
            'washingtonpost': 0.8
        }
        
        return authority_scores.get(source.lower(), 0.5)
