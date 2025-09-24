"""
PostgreSQL Manager - Structured data storage with connection pooling
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import asyncpg
from asyncpg.pool import Pool

from models import Article, StorageType
from config import Settings

logger = logging.getLogger(__name__)

class PostgreSQLManager:
    """Manage PostgreSQL for structured data storage"""
    
    def __init__(self):
        self.pool: Optional[Pool] = None
        self.settings = Settings()
        self._initialized = False
        
    async def initialize(self):
        """Initialize PostgreSQL connection pool"""
        if self._initialized:
            return
            
        logger.info("Initializing PostgreSQL Manager...")
        
        try:
            # Create connection pool
            pg_config = self.settings.get_postgres_config()
            
            self.pool = await asyncpg.create_pool(
                host=pg_config.host,
                port=pg_config.port,
                database=pg_config.database,
                user=pg_config.username,
                password=pg_config.password,
                min_size=5,
                max_size=pg_config.pool_size,
                command_timeout=60
            )
            
            # Initialize database schema
            await self._initialize_schema()
            
            self._initialized = True
            logger.info("PostgreSQL Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL Manager: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None
        
        self._initialized = False
        logger.info("PostgreSQL Manager cleanup complete")
    
    async def _initialize_schema(self):
        """Initialize database schema"""
        try:
            async with self.pool.acquire() as conn:
                # Create articles table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS articles (
                        id VARCHAR(255) PRIMARY KEY,
                        title TEXT NOT NULL,
                        content TEXT,
                        summary TEXT,
                        author VARCHAR(255),
                        source VARCHAR(255) NOT NULL,
                        published_at TIMESTAMP,
                        categories TEXT[],
                        tags TEXT[],
                        language VARCHAR(10) DEFAULT 'en',
                        url VARCHAR(500),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for performance
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_articles_source 
                    ON articles(source)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_articles_published_at 
                    ON articles(published_at)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_articles_categories 
                    ON articles USING GIN(categories)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_articles_tags 
                    ON articles USING GIN(tags)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_articles_metadata 
                    ON articles USING GIN(metadata)
                """)
                
                # Create full-text search index
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_articles_fulltext 
                    ON articles USING GIN(
                        to_tsvector('english', title || ' ' || COALESCE(content, '') || ' ' || COALESCE(summary, ''))
                    )
                """)
                
                logger.info("PostgreSQL schema initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL schema: {e}")
            raise
    
    async def store_article_metadata(self, article: Article) -> Dict[str, Any]:
        """Store article metadata in PostgreSQL"""
        if not self._initialized or not self.pool:
            raise RuntimeError("PostgreSQL Manager not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                # Insert or update article
                await conn.execute("""
                    INSERT INTO articles 
                    (id, title, content, summary, author, source, published_at, 
                     categories, tags, language, url, metadata, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, CURRENT_TIMESTAMP)
                    ON CONFLICT (id)
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        summary = EXCLUDED.summary,
                        author = EXCLUDED.author,
                        source = EXCLUDED.source,
                        published_at = EXCLUDED.published_at,
                        categories = EXCLUDED.categories,
                        tags = EXCLUDED.tags,
                        language = EXCLUDED.language,
                        url = EXCLUDED.url,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """, 
                article.id,
                article.title,
                article.content,
                article.summary,
                article.author,
                article.source,
                article.published_at,
                article.categories,
                article.tags,
                article.language,
                article.url,
                json.dumps(article.metadata)
                )
                
                logger.info(f"Article {article.id} metadata stored in PostgreSQL")
                
                return {
                    'article_id': article.id,
                    'stored': True,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to store article metadata {article.id}: {e}")
            raise
    
    async def get_article_metadata(self, article_id: str) -> Dict[str, Any]:
        """Get article metadata from PostgreSQL"""
        if not self._initialized or not self.pool:
            raise RuntimeError("PostgreSQL Manager not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT id, title, content, summary, author, source, published_at,
                           categories, tags, language, url, metadata, created_at, updated_at
                    FROM articles
                    WHERE id = $1
                """, article_id)
                
                if row:
                    return {
                        'id': row['id'],
                        'title': row['title'],
                        'content': row['content'],
                        'summary': row['summary'],
                        'author': row['author'],
                        'source': row['source'],
                        'published_at': row['published_at'],
                        'categories': row['categories'] or [],
                        'tags': row['tags'] or [],
                        'language': row['language'],
                        'url': row['url'],
                        'metadata': row['metadata'] or {},
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    }
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Failed to get article metadata {article_id}: {e}")
            raise
    
    async def search_articles(self, query: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Search articles using full-text search"""
        if not self._initialized or not self.pool:
            raise RuntimeError("PostgreSQL Manager not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, title, content, summary, author, source, published_at,
                           categories, tags, language, url, metadata,
                           ts_rank(to_tsvector('english', title || ' ' || COALESCE(content, '') || ' ' || COALESCE(summary, '')), 
                                   plainto_tsquery('english', $1)) as rank
                    FROM articles
                    WHERE to_tsvector('english', title || ' ' || COALESCE(content, '') || ' ' || COALESCE(summary, '')) 
                          @@ plainto_tsquery('english', $1)
                    ORDER BY rank DESC, published_at DESC
                    LIMIT $2 OFFSET $3
                """, query, limit, offset)
                
                results = []
                for row in rows:
                    results.append({
                        'id': row['id'],
                        'title': row['title'],
                        'content': row['content'],
                        'summary': row['summary'],
                        'author': row['author'],
                        'source': row['source'],
                        'published_at': row['published_at'],
                        'categories': row['categories'] or [],
                        'tags': row['tags'] or [],
                        'language': row['language'],
                        'url': row['url'],
                        'metadata': row['metadata'] or {},
                        'rank': float(row['rank'])
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search articles: {e}")
            raise
    
    async def get_articles_by_category(self, category: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get articles by category"""
        if not self._initialized or not self.pool:
            raise RuntimeError("PostgreSQL Manager not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, title, content, summary, author, source, published_at,
                           categories, tags, language, url, metadata
                    FROM articles
                    WHERE $1 = ANY(categories)
                    ORDER BY published_at DESC
                    LIMIT $2
                """, category, limit)
                
                results = []
                for row in rows:
                    results.append({
                        'id': row['id'],
                        'title': row['title'],
                        'content': row['content'],
                        'summary': row['summary'],
                        'author': row['author'],
                        'source': row['source'],
                        'published_at': row['published_at'],
                        'categories': row['categories'] or [],
                        'tags': row['tags'] or [],
                        'language': row['language'],
                        'url': row['url'],
                        'metadata': row['metadata'] or {}
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get articles by category {category}: {e}")
            raise
    
    async def get_articles_by_source(self, source: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get articles by source"""
        if not self._initialized or not self.pool:
            raise RuntimeError("PostgreSQL Manager not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, title, content, summary, author, source, published_at,
                           categories, tags, language, url, metadata
                    FROM articles
                    WHERE source = $1
                    ORDER BY published_at DESC
                    LIMIT $2
                """, source, limit)
                
                results = []
                for row in rows:
                    results.append({
                        'id': row['id'],
                        'title': row['title'],
                        'content': row['content'],
                        'summary': row['summary'],
                        'author': row['author'],
                        'source': row['source'],
                        'published_at': row['published_at'],
                        'categories': row['categories'] or [],
                        'tags': row['tags'] or [],
                        'language': row['language'],
                        'url': row['url'],
                        'metadata': row['metadata'] or {}
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get articles by source {source}: {e}")
            raise
    
    async def delete_article(self, article_id: str):
        """Delete article from PostgreSQL"""
        if not self._initialized or not self.pool:
            raise RuntimeError("PostgreSQL Manager not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    DELETE FROM articles WHERE id = $1
                """, article_id)
                
                logger.info(f"Article {article_id} deleted from PostgreSQL")
                
        except Exception as e:
            logger.error(f"Failed to delete article {article_id}: {e}")
            raise
    
    async def get_article_stats(self) -> Dict[str, Any]:
        """Get article statistics"""
        if not self._initialized or not self.pool:
            raise RuntimeError("PostgreSQL Manager not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                # Total articles
                total_articles = await conn.fetchval("SELECT COUNT(*) FROM articles")
                
                # Articles by source
                source_stats = await conn.fetch("""
                    SELECT source, COUNT(*) as count
                    FROM articles
                    GROUP BY source
                    ORDER BY count DESC
                """)
                
                # Articles by category
                category_stats = await conn.fetch("""
                    SELECT unnest(categories) as category, COUNT(*) as count
                    FROM articles
                    WHERE categories IS NOT NULL
                    GROUP BY category
                    ORDER BY count DESC
                    LIMIT 20
                """)
                
                # Recent articles
                recent_articles = await conn.fetchval("""
                    SELECT COUNT(*) FROM articles
                    WHERE published_at >= CURRENT_DATE - INTERVAL '7 days'
                """)
                
                return {
                    'total_articles': total_articles,
                    'articles_by_source': {row['source']: row['count'] for row in source_stats},
                    'articles_by_category': {row['category']: row['count'] for row in category_stats},
                    'recent_articles_7_days': recent_articles,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get article stats: {e}")
            raise
