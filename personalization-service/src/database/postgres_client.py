"""PostgreSQL client for data persistence."""

import asyncio
from typing import Any, Optional, List, Dict, Union
import asyncpg
import structlog

from ..config.settings import settings

logger = structlog.get_logger()


class PostgreSQLClient:
    """PostgreSQL client for data persistence."""
    
    def __init__(self):
        self.database_url = settings.database_url
        self.pool: Optional[asyncpg.Pool] = None
        
    async def connect(self) -> None:
        """Connect to PostgreSQL and create connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60,
                server_settings={
                    'jit': 'off'
                }
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            
            logger.info("Connected to PostgreSQL successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self.pool:
            await self.pool.close()
            logger.info("Disconnected from PostgreSQL")
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query and return the result."""
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        async with self.pool.acquire() as conn:
            try:
                result = await conn.execute(query, *args)
                return result
            except Exception as e:
                logger.error(f"Error executing query: {e}")
                raise
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch one row from the database."""
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        async with self.pool.acquire() as conn:
            try:
                row = await conn.fetchrow(query, *args)
                if row:
                    return dict(row)
                return None
            except Exception as e:
                logger.error(f"Error fetching one row: {e}")
                raise
    
    async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch all rows from the database."""
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Error fetching all rows: {e}")
                raise
    
    async def fetch_val(self, query: str, *args) -> Any:
        """Fetch a single value from the database."""
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        async with self.pool.acquire() as conn:
            try:
                return await conn.fetchval(query, *args)
            except Exception as e:
                logger.error(f"Error fetching value: {e}")
                raise
    
    async def transaction(self):
        """Get a transaction context manager."""
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        return self.pool.acquire()
    
    async def execute_many(self, query: str, args_list: List[tuple]) -> None:
        """Execute a query with many parameter sets."""
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        async with self.pool.acquire() as conn:
            try:
                await conn.executemany(query, args_list)
            except Exception as e:
                logger.error(f"Error executing many queries: {e}")
                raise
    
    async def copy_records(self, table: str, records: List[Dict[str, Any]], 
                          columns: List[str]) -> None:
        """Copy records to a table efficiently."""
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        if not records:
            return
        
        async with self.pool.acquire() as conn:
            try:
                # Prepare data for copy
                data = []
                for record in records:
                    row = tuple(record.get(col) for col in columns)
                    data.append(row)
                
                # Copy records
                await conn.copy_records_to_table(
                    table, 
                    records=data, 
                    columns=columns
                )
            except Exception as e:
                logger.error(f"Error copying records: {e}")
                raise
    
    async def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table information."""
        query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = $1
        ORDER BY ordinal_position
        """
        
        return await self.fetch_all(query, table_name)
    
    async def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get table statistics."""
        query = """
        SELECT 
            schemaname,
            tablename,
            attname,
            n_distinct,
            correlation
        FROM pg_stats
        WHERE tablename = $1
        """
        
        stats = await self.fetch_all(query, table_name)
        
        return {
            'table_name': table_name,
            'columns': stats
        }
    
    async def get_database_size(self) -> Dict[str, Any]:
        """Get database size information."""
        query = """
        SELECT 
            pg_database_size(current_database()) as database_size,
            pg_size_pretty(pg_database_size(current_database())) as database_size_pretty
        """
        
        result = await self.fetch_one(query)
        return result or {}
    
    async def get_table_sizes(self) -> List[Dict[str, Any]]:
        """Get table sizes."""
        query = """
        SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
            pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """
        
        return await self.fetch_all(query)
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        query = """
        SELECT 
            current_database() as database_name,
            current_user as user_name,
            inet_server_addr() as server_address,
            inet_server_port() as server_port,
            version() as version
        """
        
        return await self.fetch_one(query) or {}
    
    async def get_health(self) -> Dict[str, Any]:
        """Get database health status."""
        try:
            # Test basic connectivity
            await self.fetch_val('SELECT 1')
            
            # Get connection info
            conn_info = await self.get_connection_info()
            
            # Get database size
            db_size = await self.get_database_size()
            
            return {
                "status": "healthy",
                "database": conn_info.get("database_name"),
                "user": conn_info.get("user_name"),
                "version": conn_info.get("version"),
                "size": db_size.get("database_size_pretty"),
                "pool_size": self.pool.get_size() if self.pool else 0
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def create_index_if_not_exists(self, table: str, index_name: str, 
                                       columns: str, unique: bool = False) -> None:
        """Create index if it doesn't exist."""
        unique_keyword = "UNIQUE" if unique else ""
        query = f"""
        CREATE INDEX IF NOT EXISTS {index_name} 
        ON {table} {unique_keyword} ({columns})
        """
        
        await self.execute(query)
    
    async def drop_table_if_exists(self, table: str) -> None:
        """Drop table if it exists."""
        query = f"DROP TABLE IF EXISTS {table} CASCADE"
        await self.execute(query)
    
    async def truncate_table(self, table: str) -> None:
        """Truncate table."""
        query = f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"
        await self.execute(query)
    
    async def analyze_table(self, table: str) -> None:
        """Analyze table for query optimization."""
        query = f"ANALYZE {table}"
        await self.execute(query)
    
    async def vacuum_table(self, table: str) -> None:
        """Vacuum table to reclaim space."""
        query = f"VACUUM {table}"
        await self.execute(query)
