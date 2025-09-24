"""Database models using SQLAlchemy."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, 
    JSON, LargeBinary, String, Text, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID, INTERVAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Article(Base):
    """Article database model."""
    __tablename__ = "articles"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text)
    url = Column(Text, unique=True, nullable=False)
    source = Column(String(255), nullable=False)
    published_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    quality_score = Column(Float, default=0.0)
    content_hash = Column(String(64))
    title_hash = Column(String(64))
    embedding = Column(LargeBinary)  # Vector will be stored as binary
    entities = Column(JSON, default=list)
    topics = Column(JSON, default=list)
    locations = Column(JSON, default=list)
    language = Column(String(10), default="en")
    word_count = Column(Integer, default=0)
    reading_time = Column(Integer, default=0)

    # Relationships
    clusters = relationship("ArticleCluster", back_populates="article")
    duplicates = relationship("Duplicate", foreign_keys="Duplicate.article_id", back_populates="article")
    duplicate_of = relationship("Duplicate", foreign_keys="Duplicate.duplicate_of_id", back_populates="duplicate_article")
    lsh_entries = relationship("LSHIndex", back_populates="article")
    processing_queue = relationship("ProcessingQueue", back_populates="article")

    # Indexes
    __table_args__ = (
        Index("idx_articles_published_at", "published_at"),
        Index("idx_articles_source", "source"),
        Index("idx_articles_content_hash", "content_hash"),
        Index("idx_articles_title_hash", "title_hash"),
        Index("idx_articles_quality_score", "quality_score"),
        Index("idx_articles_language", "language"),
    )


class Cluster(Base):
    """Cluster database model."""
    __tablename__ = "clusters"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    representative_article_id = Column(UUID(as_uuid=True), ForeignKey("articles.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    article_count = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)
    topics = Column(JSON, default=list)
    entities = Column(JSON, default=list)
    locations = Column(JSON, default=list)
    time_span = Column(INTERVAL)
    is_active = Column(Boolean, default=True)

    # Relationships
    representative_article = relationship("Article", foreign_keys=[representative_article_id])
    articles = relationship("ArticleCluster", back_populates="cluster")

    # Indexes
    __table_args__ = (
        Index("idx_clusters_created_at", "created_at"),
        Index("idx_clusters_quality_score", "quality_score"),
        Index("idx_clusters_is_active", "is_active"),
    )


class ArticleCluster(Base):
    """Article-Cluster many-to-many relationship."""
    __tablename__ = "article_clusters"

    article_id = Column(UUID(as_uuid=True), ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True)
    cluster_id = Column(UUID(as_uuid=True), ForeignKey("clusters.id", ondelete="CASCADE"), primary_key=True)
    similarity_score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    article = relationship("Article", back_populates="clusters")
    cluster = relationship("Cluster", back_populates="articles")

    # Indexes
    __table_args__ = (
        Index("idx_article_clusters_article_id", "article_id"),
        Index("idx_article_clusters_cluster_id", "cluster_id"),
        Index("idx_article_clusters_similarity", "similarity_score"),
    )


class Duplicate(Base):
    """Duplicate relationship model."""
    __tablename__ = "duplicates"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    article_id = Column(UUID(as_uuid=True), ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    duplicate_of_id = Column(UUID(as_uuid=True), ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    similarity_score = Column(Float, nullable=False)
    similarity_type = Column(String(50), nullable=False)  # 'lsh', 'semantic', 'content'
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    article = relationship("Article", foreign_keys=[article_id], back_populates="duplicates")
    duplicate_article = relationship("Article", foreign_keys=[duplicate_of_id], back_populates="duplicate_of")

    # Constraints
    __table_args__ = (
        UniqueConstraint("article_id", "duplicate_of_id", name="uq_duplicates_article_duplicate"),
        Index("idx_duplicates_article_id", "article_id"),
        Index("idx_duplicates_duplicate_of_id", "duplicate_of_id"),
        Index("idx_duplicates_similarity", "similarity_score"),
    )


class LSHIndex(Base):
    """LSH index model."""
    __tablename__ = "lsh_index"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    article_id = Column(UUID(as_uuid=True), ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    minhash_signature = Column(LargeBinary, nullable=False)
    content_fingerprint = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    article = relationship("Article", back_populates="lsh_entries")

    # Indexes
    __table_args__ = (
        Index("idx_lsh_index_article_id", "article_id"),
        Index("idx_lsh_index_content_fingerprint", "content_fingerprint"),
    )


class ProcessingQueue(Base):
    """Processing queue model."""
    __tablename__ = "processing_queue"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    article_id = Column(UUID(as_uuid=True), ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), default="pending")  # 'pending', 'processing', 'completed', 'failed'
    priority = Column(Integer, default=0)
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True))

    # Relationships
    article = relationship("Article", back_populates="processing_queue")

    # Indexes
    __table_args__ = (
        Index("idx_processing_queue_status", "status"),
        Index("idx_processing_queue_priority", "priority", "created_at"),
        Index("idx_processing_queue_created_at", "created_at"),
    )


class SystemMetrics(Base):
    """System metrics model."""
    __tablename__ = "system_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(20), nullable=False)  # 'counter', 'gauge', 'histogram'
    labels = Column(JSON, default=dict)

    # Indexes
    __table_args__ = (
        Index("idx_system_metrics_timestamp", "timestamp"),
        Index("idx_system_metrics_name", "metric_name"),
        Index("idx_system_metrics_type", "metric_type"),
    )
