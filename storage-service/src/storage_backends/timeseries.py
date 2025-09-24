"""
TimescaleDB Manager - Time-series analytics and metrics storage
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from asyncpg.pool import Pool

from config import Settings
from models import Article

logger = logging.getLogger(__name__)


class TimeseriesDBManager:
    """Manage TimescaleDB for time-series analytics"""

    def __init__(self):
        self.pool: Optional[Pool] = None
        self.settings = Settings()
        self._initialized = False

    async def initialize(self) -> Dict[str, Any]:
    """Initialize TimescaleDB connection pool"""
        if self._initialized:
            return

        logger.info("Initializing TimescaleDB Manager...")

        try:
            # Create connection pool
            timescale_config = self.settings.get_timescale_config()

            self.pool = await asyncpg.create_pool(
                host=timescale_config.host,
                port=timescale_config.port,
                database=timescale_config.database,
                user=timescale_config.username,
                password=timescale_config.password,
                min_size=5,
                max_size=timescale_config.pool_size,
                command_timeout=60,
            )

            # Initialize database schema
            await self._initialize_schema()

            self._initialized = True
            logger.info("TimescaleDB Manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB Manager: {e}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
    """Cleanup connection pool"""
        if self.pool:
    await self.pool.close()
            self.pool = None

        self._initialized = False
        logger.info("TimescaleDB Manager cleanup complete")

    async def _initialize_schema(self) -> Dict[str, Any]:
    """Initialize TimescaleDB schema"""
        try:
    async with self.pool.acquire() as conn:
                # Enable TimescaleDB extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

                # Create article metrics table
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS article_metrics (
                        time TIMESTAMPTZ NOT NULL,
                        article_id VARCHAR(255) NOT NULL,
                        source VARCHAR(255),
                        category VARCHAR(100),
                        language VARCHAR(10),
                        word_count INTEGER,
                        reading_time INTEGER,
                        engagement_score FLOAT,
                        quality_score FLOAT,
                        sentiment_score FLOAT,
                        metadata JSONB
                    )
                """
                )

                # Create hypertable
                await conn.execute(
                    """
                    SELECT create_hypertable('article_metrics', 'time',
                                           if_not_exists => TRUE)
                """
                )

                # Create indexes for performance
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_article_metrics_article_id
                    ON article_metrics(article_id, time DESC)
                """
                )

                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_article_metrics_source
                    ON article_metrics(source, time DESC)
                """
                )

                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_article_metrics_category
                    ON article_metrics(category, time DESC)
                """
                )

                # Create continuous aggregates for common queries
                await self._create_continuous_aggregates()

                logger.info("TimescaleDB schema initialized")

        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB schema: {e}")
            raise

    async def _create_continuous_aggregates(self) -> Dict[str, Any]:
    """Create continuous aggregates for common queries"""
        try:
    async with self.pool.acquire() as conn:
                # Daily article metrics
                await conn.execute(
                    """
                    CREATE MATERIALIZED VIEW IF NOT EXISTS daily_article_metrics
                    WITH (timescaledb.continuous) AS
                    SELECT
                        time_bucket('1 day', time) AS day,
                        source,
                        category,
                        language,
                        COUNT(*) as article_count,
                        AVG(word_count) as avg_word_count,
                        AVG(reading_time) as avg_reading_time,
                        AVG(engagement_score) as avg_engagement_score,
                        AVG(quality_score) as avg_quality_score,
                        AVG(sentiment_score) as avg_sentiment_score
                    FROM article_metrics
                    GROUP BY day, source, category, language
                """
                )

                # Hourly article metrics
                await conn.execute(
                    """
                    CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_article_metrics
                    WITH (timescaledb.continuous) AS
                    SELECT
                        time_bucket('1 hour', time) AS hour,
                        source,
                        COUNT(*) as article_count,
                        AVG(engagement_score) as avg_engagement_score,
                        AVG(quality_score) as avg_quality_score
                    FROM article_metrics
                    GROUP BY hour, source
                """
                )

                # Set refresh policies
                await conn.execute(
                    """
                    SELECT add_continuous_aggregate_policy('daily_article_metrics',
                        start_offset => INTERVAL '7 days',
                        end_offset => INTERVAL '1 hour',
                        schedule_interval => INTERVAL '1 hour')
                """
                )

                await conn.execute(
                    """
                    SELECT add_continuous_aggregate_policy('hourly_article_metrics',
                        start_offset => INTERVAL '1 day',
                        end_offset => INTERVAL '1 hour',
                        schedule_interval => INTERVAL '15 minutes')
                """
                )

                logger.info("TimescaleDB continuous aggregates created")

        except Exception as e:
            logger.warning(f"Failed to create continuous aggregates: {e}")

    async def record_article_metrics(self, article: Article) -> Dict[str, Any]:
    """Record metrics for an article"""
        if not self._initialized or not self.pool:
            raise RuntimeError("TimescaleDB Manager not initialized")

        try:
            # Calculate metrics
            metrics = self._calculate_article_metrics(article)

            async with self.pool.acquire() as conn:
    await conn.execute(
                    """
                    INSERT INTO article_metrics
                    (time, article_id, source, category, language, word_count,
                     reading_time, engagement_score, quality_score, sentiment_score, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                    datetime.utcnow(),
                    article.id,
                    article.source,
                    article.categories[0] if article.categories else None,
                    article.language,
                    metrics["word_count"],
                    metrics["reading_time"],
                    metrics["engagement_score"],
                    metrics["quality_score"],
                    metrics["sentiment_score"],
                    json.dumps(metrics["metadata"]),
                )

            logger.debug(f"Recorded metrics for article {article.id}")

            return {
                "article_id": article.id,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(
                f"Failed to record metrics for article {article.id}: {e}")
            raise

    def _calculate_article_metrics(self, article: Article) -> Dict[str, Any]:
    """Calculate various metrics for an article"""
        # Word count
        word_count = len(article.content.split()) if article.content else 0

        # Reading time (average 200 words per minute)
        reading_time = max(1, word_count // 200)

        # Engagement score (simplified calculation)
        engagement_score = min(
            1.0,
            max(
                0.0,
                (len(article.tags) * 0.1)  # More tags = higher engagement
                + (1.0 if article.summary else 0.0)  # Has summary
                + (min(1.0, word_count / 1000)),  # Length factor
            ),
        )

        # Quality score (simplified calculation)
        quality_score = min(
            1.0,
            max(
                0.0,
                (1.0 if article.author else 0.0)  # Has author
                + (1.0 if article.summary else 0.0)  # Has summary
                + (min(1.0, word_count / 500))  # Length factor
                + (0.1 if len(article.categories)
                   > 0 else 0.0),  # Has categories
            ),
        )

        # Sentiment score (placeholder - would use actual sentiment analysis)
        sentiment_score = 0.5  # Neutral by default

        return {
            "word_count": word_count,
            "reading_time": reading_time,
            "engagement_score": engagement_score,
            "quality_score": quality_score,
            "sentiment_score": sentiment_score,
            "metadata": {
                "has_embeddings": bool(article.embeddings),
                "has_media": bool(article.media_files),
                "categories_count": len(article.categories),
                "tags_count": len(article.tags),
            },
        }

    async def get_article_metrics(
        self,
        article_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get metrics for a specific article"""
        if not self._initialized or not self.pool:
            raise RuntimeError("TimescaleDB Manager not initialized")

        try:
    async with self.pool.acquire() as conn:
                query = """
                    SELECT time, article_id, source, category, language, word_count,
                           reading_time, engagement_score, quality_score, sentiment_score, metadata
                    FROM article_metrics
                    WHERE article_id = $1
                """
                params = [article_id]

                if start_time:
                    query += " AND time >= $2"
                    params.append(start_time)
                    if end_time:
                        query += " AND time <= $3"
                        params.append(end_time)
                elif end_time:
                    query += " AND time <= $2"
                    params.append(end_time)

                query += " ORDER BY time DESC"

                rows = await conn.fetch(query, *params)

                metrics = []
                for row in rows:
                    metrics.append(
                        {
                            "time": row["time"],
                            "article_id": row["article_id"],
                            "source": row["source"],
                            "category": row["category"],
                            "language": row["language"],
                            "word_count": row["word_count"],
                            "reading_time": row["reading_time"],
                            "engagement_score": row["engagement_score"],
                            "quality_score": row["quality_score"],
                            "sentiment_score": row["sentiment_score"],
                            "metadata": row["metadata"],
                        }
                    )

                return metrics

        except Exception as e:
            logger.error(
                f"Failed to get metrics for article {article_id}: {e}")
            raise

    async def get_daily_metrics(
        self, start_date: datetime, end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get daily aggregated metrics"""
        if not self._initialized or not self.pool:
            raise RuntimeError("TimescaleDB Manager not initialized")

        try:
    async with self.pool.acquire() as conn:
                query = """
                    SELECT day, source, category, language, article_count,
                           avg_word_count, avg_reading_time, avg_engagement_score,
                           avg_quality_score, avg_sentiment_score
                    FROM daily_article_metrics
                    WHERE day >= $1
                """
                params = [start_date]

                if end_date:
                    query += " AND day <= $2"
                    params.append(end_date)

                query += " ORDER BY day DESC"

                rows = await conn.fetch(query, *params)

                metrics = []
                for row in rows:
                    metrics.append(
                        {
                            "day": row["day"],
                            "source": row["source"],
                            "category": row["category"],
                            "language": row["language"],
                            "article_count": row["article_count"],
                            "avg_word_count": (
                                float(row["avg_word_count"]) if row["avg_word_count"] else 0
                            ),
                            "avg_reading_time": (
                                float(row["avg_reading_time"]) if row["avg_reading_time"] else 0
                            ),
                            "avg_engagement_score": (
                                float(row["avg_engagement_score"])
                                if row["avg_engagement_score"]
                                else 0
                            ),
                            "avg_quality_score": (
                                float(row["avg_quality_score"]) if row["avg_quality_score"] else 0
                            ),
                            "avg_sentiment_score": (
                                float(row["avg_sentiment_score"])
                                if row["avg_sentiment_score"]
                                else 0
                            ),
                        }
                    )

                return metrics

        except Exception as e:
            logger.error(f"Failed to get daily metrics: {e}")
            raise

    async def get_hourly_metrics(
        self, start_time: datetime, end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get hourly aggregated metrics"""
        if not self._initialized or not self.pool:
            raise RuntimeError("TimescaleDB Manager not initialized")

        try:
    async with self.pool.acquire() as conn:
                query = """
                    SELECT hour, source, article_count, avg_engagement_score, avg_quality_score
                    FROM hourly_article_metrics
                    WHERE hour >= $1
                """
                params = [start_time]

                if end_time:
                    query += " AND hour <= $2"
                    params.append(end_time)

                query += " ORDER BY hour DESC"

                rows = await conn.fetch(query, *params)

                metrics = []
                for row in rows:
                    metrics.append(
                        {
                            "hour": row["hour"],
                            "source": row["source"],
                            "article_count": row["article_count"],
                            "avg_engagement_score": (
                                float(row["avg_engagement_score"])
                                if row["avg_engagement_score"]
                                else 0
                            ),
                            "avg_quality_score": (
                                float(row["avg_quality_score"]) if row["avg_quality_score"] else 0
                            ),
                        }
                    )

                return metrics

        except Exception as e:
            logger.error(f"Failed to get hourly metrics: {e}")
            raise

    async def get_source_performance(
            self, source: str, days: int = 30) -> Dict[str, Any]:
    """Get performance metrics for a source"""
        if not self._initialized or not self.pool:
            raise RuntimeError("TimescaleDB Manager not initialized")

        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            async with self.pool.acquire() as conn:
                # Get aggregated metrics for the source
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total_articles,
                        AVG(word_count) as avg_word_count,
                        AVG(reading_time) as avg_reading_time,
                        AVG(engagement_score) as avg_engagement_score,
                        AVG(quality_score) as avg_quality_score,
                        AVG(sentiment_score) as avg_sentiment_score
                    FROM article_metrics
                    WHERE source = $1 AND time >= $2
                """,
                    source,
                    start_date,
                )

                if row:
                    return {
                        "source": source,
                        "period_days": days,
                        "total_articles": row["total_articles"],
                        "avg_word_count": (
                            float(row["avg_word_count"]) if row["avg_word_count"] else 0
                        ),
                        "avg_reading_time": (
                            float(row["avg_reading_time"]) if row["avg_reading_time"] else 0
                        ),
                        "avg_engagement_score": (
                            float(row["avg_engagement_score"]) if row["avg_engagement_score"] else 0
                        ),
                        "avg_quality_score": (
                            float(row["avg_quality_score"]) if row["avg_quality_score"] else 0
                        ),
                        "avg_sentiment_score": (
                            float(row["avg_sentiment_score"]) if row["avg_sentiment_score"] else 0
                        ),
                    }
                else:
                    return {
                        "source": source,
                        "period_days": days,
                        "total_articles": 0,
                        "avg_word_count": 0,
                        "avg_reading_time": 0,
                        "avg_engagement_score": 0,
                        "avg_quality_score": 0,
                        "avg_sentiment_score": 0,
                    }

        except Exception as e:
            logger.error(f"Failed to get source performance for {source}: {e}")
            raise

    async def get_top_sources(self, limit: int = 10,
                              days: int = 7) -> List[Dict[str, Any]]:
        """Get top performing sources"""
        if not self._initialized or not self.pool:
            raise RuntimeError("TimescaleDB Manager not initialized")

        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        source,
                        COUNT(*) as article_count,
                        AVG(engagement_score) as avg_engagement_score,
                        AVG(quality_score) as avg_quality_score
                    FROM article_metrics
                    WHERE time >= $1
                    GROUP BY source
                    ORDER BY avg_engagement_score DESC, article_count DESC
                    LIMIT $2
                """,
                    start_date,
                    limit,
                )

                sources = []
                for row in rows:
                    sources.append(
                        {
                            "source": row["source"],
                            "article_count": row["article_count"],
                            "avg_engagement_score": (
                                float(row["avg_engagement_score"])
                                if row["avg_engagement_score"]
                                else 0
                            ),
                            "avg_quality_score": (
                                float(row["avg_quality_score"]) if row["avg_quality_score"] else 0
                            ),
                        }
                    )

                return sources

        except Exception as e:
            logger.error(f"Failed to get top sources: {e}")
            raise
