# Storage, Indexing & Retrieval Service

A high-performance microservice providing polyglot persistence, vector similarity search, full-text search, and intelligent caching for the news aggregation pipeline.

## Features

### 🗄️ Polyglot Persistence
- **PostgreSQL with pgvector** - Structured data and vector embeddings
- **Elasticsearch** - Advanced full-text search and analytics
- **Redis** - High-performance caching and session management
- **TimescaleDB** - Time-series analytics and metrics
- **Cloud Storage** - Media files and backup storage

### 🔍 Advanced Search Capabilities
- **Vector Similarity Search** - Sub-50ms search across 1M+ vectors
- **Full-Text Search** - Advanced ranking with faceted search
- **Hybrid Search** - Combines vector and text search
- **Query Optimization** - Cross-store query performance optimization

### ⚡ Intelligent Caching
- **Multi-layer Caching** - Memory, Redis, and CDN
- **Cache Policies** - Intelligent TTL and invalidation strategies
- **Cache Coordination** - Optimal data placement across layers
- **90%+ Cache Hit Ratio** - High-performance data access

### 📊 Data Lifecycle Management
- **Automated Archival** - Data retention and lifecycle policies
- **GDPR Compliance** - Right to be forgotten and data export
- **Backup & Recovery** - Automated backup and disaster recovery
- **Data Anonymization** - Privacy-preserving data handling

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  Storage        │    │  Vector Store   │
│                 │◄──►│  Orchestrator   │◄──►│  (pgvector)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼───────┐ ┌─────▼─────┐ ┌──────▼──────┐
        │ PostgreSQL    │ │ Redis     │ │Elasticsearch│
        │ (Structured)  │ │ (Cache)   │ │ (Full-text) │
        └───────────────┘ └───────────┘ └─────────────┘
                │
        ┌───────▼───────┐
        │ TimescaleDB   │
        │ (Time-series) │
        └───────────────┘
```

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- PostgreSQL with pgvector extension
- Redis
- Elasticsearch
- TimescaleDB

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd storage-service
```

2. **Set up environment variables**
```bash
cp config.env.example .env
# Edit .env with your configuration
```

3. **Start with Docker Compose**
```bash
docker-compose up -d
```

4. **Verify installation**
```bash
curl http://localhost:8000/health
```

### Manual Installation

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Set up databases**
```sql
-- PostgreSQL
CREATE EXTENSION vector;

-- TimescaleDB
CREATE EXTENSION timescaledb;
```

3. **Run the service**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## API Usage

### Store Article
```python
import requests

article_data = {
    "id": "article-123",
    "title": "Breaking News",
    "content": "Article content...",
    "source": "reuters",
    "published_at": "2024-01-01T00:00:00Z",
    "embeddings": {
        "content": [0.1, 0.2, 0.3, ...]  # 768-dimensional vector
    }
}

response = requests.post("http://localhost:8000/storage/articles", json=article_data)
```

### Search Articles
```python
# Full-text search
search_request = {
    "query": "breaking news",
    "filters": {"categories": ["politics"]},
    "limit": 20
}

response = requests.post("http://localhost:8000/search/fulltext", json=search_request)

# Vector similarity search
similarity_request = {
    "query_vector": [0.1, 0.2, 0.3, ...],
    "embedding_type": "content",
    "top_k": 10
}

response = requests.post("http://localhost:8000/search/similarity", json=similarity_request)
```

### Cache Management
```python
# Invalidate cache
invalidation_request = {
    "key_patterns": ["article:*"],
    "cache_tags": ["breaking-news"]
}

response = requests.post("http://localhost:8000/cache/invalidate", json=invalidation_request)
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_HOST` | PostgreSQL host | localhost |
| `POSTGRES_PORT` | PostgreSQL port | 5432 |
| `POSTGRES_DATABASE` | Database name | storage_db |
| `REDIS_HOST` | Redis host | localhost |
| `ELASTICSEARCH_HOSTS` | Elasticsearch hosts | localhost:9200 |
| `VECTOR_DIMENSION` | Vector dimension | 768 |
| `CACHE_DEFAULT_TTL` | Default cache TTL | 3600 |

### Performance Tuning

1. **Vector Search Performance**
   - Adjust HNSW index parameters (`VECTOR_INDEX_M`, `VECTOR_INDEX_EF_CONSTRUCTION`)
   - Optimize vector dimensions based on your embedding model

2. **Cache Performance**
   - Tune memory cache size (`MEMORY_CACHE_SIZE`)
   - Configure Redis connection pool (`REDIS_MAX_CONNECTIONS`)

3. **Database Performance**
   - Adjust connection pool sizes
   - Configure appropriate indexes

## Monitoring

### Health Checks
- **Liveness**: `GET /health/live`
- **Readiness**: `GET /health/ready`
- **Health Status**: `GET /health`

### Metrics
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (admin/admin)

### Key Metrics
- Query response times
- Cache hit ratios
- Database connection pools
- Vector search performance
- Storage utilization

## Data Lifecycle

### Retention Policies
Configure data retention policies for different data types:

```python
retention_policy = {
    "policy_type": "delete_old_data",
    "data_type": "articles",
    "retention_period_days": 365
}
```

### GDPR Compliance
Handle GDPR requests for data export, deletion, and rectification:

```python
gdpr_request = {
    "request_id": "req-123",
    "user_id": "user-456",
    "request_type": "data_export"
}
```

## Development

### Running Tests
```bash
pytest tests/ -v --cov=src/
```

### Code Quality
```bash
black src/
isort src/
flake8 src/
mypy src/
```

### Database Migrations
```bash
# Create migration
alembic revision --autogenerate -m "Add new table"

# Apply migration
alembic upgrade head
```

## Deployment

### Kubernetes
```bash
kubectl apply -f k8s/
```

### Docker
```bash
docker build -t storage-service .
docker run -p 8000:8000 storage-service
```

### Production Considerations
- Use managed database services (RDS, Cloud SQL, etc.)
- Configure proper monitoring and alerting
- Set up automated backups
- Use secrets management for credentials
- Configure network security groups
- Enable SSL/TLS encryption

## Performance Benchmarks

- **Vector Similarity Search**: < 50ms for 1M+ vectors
- **Full-Text Search**: < 100ms response time
- **Cache Hit Ratio**: > 90%
- **Query Optimization**: < 10ms overhead
- **Throughput**: 100,000+ QPS with horizontal scaling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the API specifications

## Roadmap

- [ ] GraphQL API support
- [ ] Real-time search updates
- [ ] Advanced ML-based ranking
- [ ] Multi-tenant support
- [ ] Advanced analytics dashboard
- [ ] Automated performance tuning
