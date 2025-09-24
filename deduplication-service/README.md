# Deduplication & Event Grouping Microservice

A high-performance microservice for identifying near-duplicate content and grouping related articles into coherent news events using advanced clustering algorithms.

## Features

### Core Functionality
- **Near-duplicate detection** using LSH (Locality-Sensitive Hashing)
- **Semantic similarity** with sentence embeddings
- **Incremental clustering** for streaming data
- **Event grouping** with temporal and topical coherence
- **Representative article selection** for clusters
- **Real-time duplicate detection** API
- **Batch processing** for historical data

### Advanced Algorithms
- **MinHash and LSH** for efficient similarity detection
- **Sentence-BERT embeddings** for semantic similarity
- **DBSCAN clustering** with custom distance metrics
- **Incremental clustering** algorithms
- **Temporal decay** for relevance scoring

### Performance Optimizations
- **LSH index sharding** for horizontal scaling
- **Bloom filters** for fast negative lookups
- **Vector quantization** for memory efficiency
- **Incremental index updates**
- **Batch processing** optimizations
- **Parallel similarity** computations

### Monitoring & Metrics
- **Duplicate detection accuracy** (precision/recall)
- **Clustering quality metrics** (silhouette score)
- **Processing latency** and throughput
- **Index size** and memory usage
- **False positive/negative** rates

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Redis         │    │   PostgreSQL    │
│   Application   │◄──►│   (LSH Cache)   │◄──►│   (pgvector)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Deduplication │
│   Pipeline      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Event         │
│   Grouping      │
└─────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.9+
- Redis 6.0+
- PostgreSQL 13+ with pgvector extension
- Docker & Docker Compose

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd deduplication-service
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp env.example .env
# Edit .env with your configuration
```

4. **Start services with Docker Compose**
```bash
docker-compose up -d
```

5. **Run database migrations**
```bash
# The init.sql script will be automatically executed
```

6. **Start the service**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
```http
GET /health
```

### Deduplication
```http
POST /deduplicate
Content-Type: application/json

{
  "articles": [
    {
      "id": "uuid",
      "title": "Article title",
      "content": "Article content",
      "url": "https://example.com/article",
      "source": "reuters",
      "published_at": "2024-01-01T00:00:00Z"
    }
  ],
  "batch_id": "optional-batch-id",
  "force_reprocess": false,
  "similarity_threshold": 0.85
}
```

### Event Grouping
```http
POST /group-events
Content-Type: application/json

{
  "articles": [...],
  "time_window_hours": 24,
  "min_cluster_size": 2,
  "max_cluster_size": 100
}
```

### Similarity Search
```http
GET /similarity/{article_id}?threshold=0.85&max_results=10
```

### Metrics
```http
GET /metrics
GET /stats
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string |
| `REDIS_URL` | - | Redis connection string |
| `SIMILARITY_THRESHOLD` | 0.85 | Similarity threshold for duplicates |
| `LSH_THRESHOLD` | 0.7 | LSH threshold for candidate filtering |
| `CLUSTERING_EPS` | 0.3 | DBSCAN epsilon parameter |
| `CLUSTERING_MIN_SAMPLES` | 2 | DBSCAN min samples parameter |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `BATCH_SIZE` | 100 | Processing batch size |
| `MAX_WORKERS` | 4 | Number of worker processes |

### Similarity Thresholds

- **Content Similarity**: 0.8 (80%)
- **Title Similarity**: 0.9 (90%)
- **LSH Threshold**: 0.7 (70%)
- **Combined Threshold**: 0.85 (85%)

## Performance Tuning

### Memory Optimization
- Adjust `BATCH_SIZE` based on available memory
- Configure Redis memory limits
- Use vector quantization for embeddings

### CPU Optimization
- Increase `MAX_WORKERS` for CPU-bound workloads
- Enable parallel processing for batch operations
- Use async/await for I/O operations

### Database Optimization
- Create appropriate indexes
- Use connection pooling
- Configure pgvector parameters

## Monitoring

### Metrics Available
- Articles processed per second
- Duplicate detection rate
- Clustering quality scores
- Processing latency (P50, P95, P99)
- Memory and CPU usage
- Redis performance metrics

### Health Checks
- Service health endpoint
- Database connectivity
- Redis connectivity
- LSH index status

## Development

### Running Tests
```bash
pytest tests/ -v --cov=src
```

### Code Quality
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### Building Docker Image
```bash
docker build -t deduplication-service .
```

## Deployment

### Kubernetes
```bash
kubectl apply -f k8s/
```

### Docker Compose
```bash
docker-compose up -d
```

### Cloud Run
```bash
gcloud run deploy deduplication-service \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10
```

## Algorithm Details

### LSH Implementation
- Uses MinHash with 128 permutations
- 16 bands with 8 rows each
- Configurable similarity threshold
- Incremental index updates

### Semantic Similarity
- Sentence-BERT embeddings (384 dimensions)
- Cosine similarity for comparison
- Weighted combination of multiple metrics
- Entity and topic overlap analysis

### Clustering Algorithm
- Incremental DBSCAN for streaming data
- Multi-dimensional feature vectors
- Temporal decay for relevance
- Quality-based representative selection

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce batch size
   - Check Redis memory limits
   - Monitor embedding cache

2. **Slow Processing**
   - Increase worker count
   - Optimize database queries
   - Check Redis performance

3. **Low Accuracy**
   - Adjust similarity thresholds
   - Tune clustering parameters
   - Review feature engineering

### Debug Mode
```bash
LOG_LEVEL=DEBUG uvicorn src.main:app
```

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
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide
