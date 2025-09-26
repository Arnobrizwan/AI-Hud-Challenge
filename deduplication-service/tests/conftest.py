"""Test configuration and fixtures."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

import pytest
import redis.asyncio as redis
from faker import Faker

from src.algorithms.lsh.lsh_index import LSHIndexManager
from src.algorithms.lsh.minhash import ContentFingerprinter, MinHashGenerator
from src.algorithms.similarity.combined import (
    CombinedSimilarityCalculator,
    SimilarityThresholdManager,
)
from src.algorithms.similarity.semantic import (
    ContentSimilarityCalculator,
    SemanticSimilarityCalculator,
)
from src.clustering.event_grouping import EventGroupingEngine
from src.clustering.incremental_dbscan import IncrementalDBSCAN
from src.deduplication.pipeline import DeduplicationPipeline, DeduplicationService
from src.models.schemas import Entity, Location, NormalizedArticle, Topic
from src.monitoring.metrics import MetricsCollector
from src.utils.cache import CacheManager

fake = Faker()


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def redis_client() -> Dict[str, Any]:
    """Create Redis client for tests."""
    client = redis.from_url("redis://localhost:6379/1")  # Use test database
    await client.flushdb()  # Clear test database
    yield client
    await client.flushdb()
    await client.close()


@pytest.fixture
def minhash_generator():
    """Create MinHash generator."""
    return MinHashGenerator(num_perm=64, seed=42)


@pytest.fixture
def content_fingerprinter(minhash_generator):
    """Create content fingerprinter."""
    return ContentFingerprinter(minhash_generator)


@pytest.fixture
def semantic_calculator():
    """Create semantic similarity calculator."""
    return SemanticSimilarityCalculator(model_name="all-MiniLM-L6-v2", embedding_dimension=384)


@pytest.fixture
def content_calculator(minhash_generator):
    """Create content similarity calculator."""
    return ContentSimilarityCalculator(minhash_generator)


@pytest.fixture
def combined_calculator(semantic_calculator, content_calculator):
    """Create combined similarity calculator."""
    return CombinedSimilarityCalculator(
        semantic_calculator,
        content_calculator,
        title_weight=0.4,
        content_weight=0.4,
        entity_weight=0.2,
    )


@pytest.fixture
def threshold_manager():
    """Create similarity threshold manager."""
    return SimilarityThresholdManager(
        default_threshold=0.85, lsh_threshold=0.7, content_threshold=0.8, title_threshold=0.9
    )


@pytest.fixture
async def lsh_manager(redis_client) -> Dict[str, Any]:
    """Create LSH index manager."""
    return LSHIndexManager(redis_client)


@pytest.fixture
async def deduplication_pipeline(redis_client, lsh_manager, combined_calculator, threshold_manager) -> Dict[str, Any]:
    """Create deduplication pipeline."""
    pipeline = DeduplicationPipeline(
        redis_client=redis_client,
        lsh_index_manager=lsh_manager,
        similarity_calculator=combined_calculator,
        threshold_manager=threshold_manager,
    )
    await pipeline.initialize()
    return pipeline


@pytest.fixture
def deduplication_service(deduplication_pipeline):
    """Create deduplication service."""
    return DeduplicationService(deduplication_pipeline)


@pytest.fixture
def clustering_engine():
    """Create incremental DBSCAN clustering engine."""
    return IncrementalDBSCAN(eps=0.3, min_samples=2, max_cluster_size=100)


@pytest.fixture
def event_grouping_engine(clustering_engine, semantic_calculator):
    """Create event grouping engine."""
    return EventGroupingEngine(
        clustering_engine=clustering_engine,
        semantic_calculator=semantic_calculator,
        time_window_hours=24,
        min_cluster_size=2,
        max_cluster_size=100,
    )


@pytest.fixture
async def metrics_collector(redis_client) -> Dict[str, Any]:
    """Create metrics collector."""
    collector = MetricsCollector(redis_client)
    await collector.initialize()
    return collector


@pytest.fixture
async def cache_manager(redis_client) -> Dict[str, Any]:
    """Create cache manager."""
    manager = CacheManager(redis_client)
    await manager.initialize()
    return manager


@pytest.fixture
def sample_article():
    """Create sample article."""
    return NormalizedArticle(
        id=uuid4(),
        title="Breaking: Major earthquake hits California",
        content="A powerful earthquake measuring 7.2 on the Richter scale struck California today, causing widespread damage and power outages across the region. Emergency services are responding to multiple incidents.",
        summary="Major earthquake hits California, causing widespread damage",
        url="https://example.com/earthquake-california",
        source="reuters",
        published_at=datetime.now(timezone.utc),
        quality_score=0.9,
        entities=[
            Entity(text="California", label="GPE", confidence=0.95, start=0, end=10),
            Entity(text="earthquake", label="EVENT", confidence=0.9, start=0, end=10),
        ],
        topics=[
            Topic(name="Natural Disasters", confidence=0.9, category="Environment"),
            Topic(name="California", confidence=0.8, category="Geography"),
        ],
        locations=[Location(name="California", country="USA", confidence=0.95)],
        language="en",
        word_count=45,
        reading_time=1,
    )


@pytest.fixture
def sample_articles():
    """Create sample articles."""
    articles = []

    # Similar articles about the same event
    for i in range(5):
        article = NormalizedArticle(
            id=uuid4(),
            title=f"Earthquake in California - Update {i+1}",
            content=f"Breaking news about the earthquake in California. Update {i+1} with latest developments and damage assessment.",
            summary=f"California earthquake update {i+1}",
            url=f"https://example.com/earthquake-update-{i+1}",
            source=f"source_{i+1}",
            published_at=datetime.now(timezone.utc),
            quality_score=0.8 + i * 0.02,
            entities=[
                Entity(text="California", label="GPE", confidence=0.9, start=0, end=10),
                Entity(text="earthquake", label="EVENT", confidence=0.85, start=0, end=10),
            ],
            topics=[
                Topic(name="Natural Disasters", confidence=0.85, category="Environment"),
                Topic(name="California", confidence=0.8, category="Geography"),
            ],
            locations=[Location(name="California", country="USA", confidence=0.9)],
            language="en",
            word_count=30 + i * 5,
            reading_time=1,
        )
        articles.append(article)

    # Different articles about different topics
    different_topics = [
        ("Technology", "AI breakthrough announced", "Major AI breakthrough in machine learning"),
        ("Sports", "World Cup final results", "Exciting World Cup final match results"),
        ("Politics", "Election results announced", "Latest election results and analysis"),
        ("Business", "Stock market update", "Stock market shows strong performance"),
    ]

    for topic, title, content in different_topics:
        article = NormalizedArticle(
            id=uuid4(),
            title=title,
            content=content,
            summary=content[:50],
            url=f"https://example.com/{topic.lower().replace(' ', '-')}",
            source="news_source",
            published_at=datetime.now(timezone.utc),
            quality_score=0.7,
            entities=[],
            topics=[Topic(name=topic, confidence=0.8, category="General")],
            locations=[],
            language="en",
            word_count=20,
            reading_time=1,
        )
        articles.append(article)

    return articles


@pytest.fixture
def duplicate_articles():
    """Create duplicate articles."""
    base_article = NormalizedArticle(
        id=uuid4(),
        title="Original article title",
        content="This is the original article content with important information about the topic.",
        summary="Original article summary",
        url="https://example.com/original",
        source="original_source",
        published_at=datetime.now(timezone.utc),
        quality_score=0.9,
        entities=[],
        topics=[],
        locations=[],
        language="en",
        word_count=20,
        reading_time=1,
    )

    # Exact duplicate
    exact_duplicate = NormalizedArticle(
        id=uuid4(),
        title="Original article title",
        content="This is the original article content with important information about the topic.",
        summary="Original article summary",
        url="https://example.com/duplicate",
        source="duplicate_source",
        published_at=datetime.now(timezone.utc),
        quality_score=0.8,
        entities=[],
        topics=[],
        locations=[],
        language="en",
        word_count=20,
        reading_time=1,
    )

    # Near duplicate with minor changes
    near_duplicate = NormalizedArticle(
        id=uuid4(),
        title="Original article title - Updated",
        content="This is the original article content with important information about the topic. Some additional details added.",
        summary="Original article summary with updates",
        url="https://example.com/near-duplicate",
        source="near_duplicate_source",
        published_at=datetime.now(timezone.utc),
        quality_score=0.85,
        entities=[],
        topics=[],
        locations=[],
        language="en",
        word_count=25,
        reading_time=1,
    )

    return [base_article, exact_duplicate, near_duplicate]


@pytest.fixture
def mock_articles_for_clustering():
    """Create mock articles for clustering tests."""
    articles = []

    # Group 1: Technology articles
    tech_articles = [
        (
            "AI breakthrough in machine learning",
            "New AI model achieves state-of-the-art performance",
        ),
        ("Machine learning advances", "Latest developments in ML algorithms"),
        ("Artificial intelligence news", "AI technology continues to evolve"),
    ]

    for title, content in tech_articles:
        article = NormalizedArticle(
            id=uuid4(),
            title=title,
            content=content,
            summary=content[:50],
            url=f"https://example.com/tech/{uuid4()}",
            source="tech_news",
            published_at=datetime.now(timezone.utc),
            quality_score=0.8,
            entities=[],
            topics=[Topic(name="Technology", confidence=0.9, category="Tech")],
            locations=[],
            language="en",
            word_count=len(content.split()),
            reading_time=1,
        )
        articles.append(article)

    # Group 2: Sports articles
    sports_articles = [
        ("World Cup final results", "Exciting final match ends with victory"),
        ("Championship game highlights", "Key moments from the championship"),
        ("Sports news update", "Latest sports results and analysis"),
    ]

    for title, content in sports_articles:
        article = NormalizedArticle(
            id=uuid4(),
            title=title,
            content=content,
            summary=content[:50],
            url=f"https://example.com/sports/{uuid4()}",
            source="sports_news",
            published_at=datetime.now(timezone.utc),
            quality_score=0.75,
            entities=[],
            topics=[Topic(name="Sports", confidence=0.9, category="Sports")],
            locations=[],
            language="en",
            word_count=len(content.split()),
            reading_time=1,
        )
        articles.append(article)

    return articles
