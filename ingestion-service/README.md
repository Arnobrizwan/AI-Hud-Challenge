# News Ingestion & Normalization Microservice

A scalable, production-ready microservice for ingesting and normalizing news content from multiple sources including RSS/Atom feeds, REST APIs, and web scraping. Built with FastAPI, Google Cloud services, and comprehensive monitoring.

## 🚀 Features

### Core Capabilities
- **Multi-source Content Ingestion**: RSS/Atom feeds, JSON feeds, REST APIs, web scraping
- **Content Normalization**: Unified schema with language detection, content type classification
- **Duplicate Detection**: Advanced similarity algorithms with configurable thresholds
- **Rate Limiting**: Per-domain rate limiting with exponential backoff
- **Robots.txt Compliance**: Respects website crawling policies
- **Real-time Processing**: Async processing with concurrent task management

### Technical Features
- **FastAPI Framework**: High-performance async web framework
- **Google Cloud Integration**: Pub/Sub, Firestore, Cloud Run
- **Comprehensive Monitoring**: Prometheus metrics, structured logging
- **Docker Deployment**: Multi-stage optimized containers
- **Auto-scaling**: Cloud Run with CPU/memory-based scaling
- **Health Checks**: Dependency monitoring and status reporting

## 📋 Requirements

### Performance Requirements
- Process 50,000+ articles per hour
- P95 ingestion latency < 5 seconds
- Memory usage < 1GB per instance
- Support 10,000+ concurrent requests

### Dependencies
- Python 3.11+
- Google Cloud Platform account
- Redis (for rate limiting and caching)
- Docker (for containerization)

## 🛠️ Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ingestion-service
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Playwright browsers**
   ```bash
   playwright install chromium
   ```

5. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

6. **Run the service**
   ```bash
   python -m uvicorn src.main:app --reload
   ```

### Docker Deployment

1. **Build the image**
   ```bash
   docker build -t ingestion-service .
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

## ⚙️ Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Application
APP_NAME="News Ingestion & Normalization Service"
DEBUG=False
ENVIRONMENT=production

# Google Cloud
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1

# Pub/Sub
PUBSUB_TOPIC_INGESTION=news-ingestion
PUBSUB_TOPIC_NORMALIZATION=news-normalization

# Firestore
FIRESTORE_COLLECTION_ARTICLES=articles
FIRESTORE_COLLECTION_SOURCES=sources

# Redis
REDIS_URL=redis://localhost:6379/0

# Content Processing
MAX_CONTENT_LENGTH=10485760
MIN_WORD_COUNT=50
MAX_WORD_COUNT=50000
DUPLICATE_THRESHOLD=0.8

# Rate Limiting
DEFAULT_RATE_LIMIT=60
RATE_LIMIT_BACKOFF_FACTOR=2.0
```

### Source Configuration

Configure content sources in `config/sources.yaml`:

```yaml
sources:
  - id: "techcrunch_rss"
    name: "TechCrunch"
    type: "rss_feed"
    url: "https://techcrunch.com/feed/"
    enabled: true
    priority: 1
    rate_limit: 60
    filters:
      min_word_count: 100
      languages: ["en"]
      title_keywords: ["tech", "startup", "innovation"]
```

## 📊 API Endpoints

### Health & Monitoring
- `GET /health` - Service health check
- `GET /metrics` - Prometheus metrics
- `GET /docs` - API documentation

### Source Management
- `GET /sources` - List all sources
- `GET /sources/{source_id}` - Get source details
- `PUT /sources/{source_id}` - Update source configuration

### Content Ingestion
- `POST /ingest` - Start ingestion for a source
- `POST /ingest/all` - Start ingestion for all enabled sources
- `GET /batches/{batch_id}` - Get batch status
- `GET /batches` - List active batches

### Content Search
- `POST /search` - Search articles with filters
- `GET /metrics/sources` - Get processing metrics

## 🏗️ Architecture

### Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  Content Adapters │    │  Normalizers    │
│                 │    │                 │    │                 │
│ • REST API      │───▶│ • RSS/Atom      │───▶│ • Content       │
│ • Health Checks │    │ • JSON Feeds    │    │ • Duplicate     │
│ • Metrics       │    │ • REST APIs     │    │ • Language      │
│ • Batch Mgmt    │    │ • Web Scraping  │    │ • Type Detection│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Google Cloud   │    │   Redis Cache   │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • Pub/Sub       │    │ • Rate Limiting │    │ • Prometheus    │
│ • Firestore     │    │ • Caching       │    │ • Grafana       │
│ • Cloud Run     │    │ • Session Mgmt  │    │ • Logging       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Ingestion**: Sources are polled based on configuration
2. **Processing**: Content is normalized and validated
3. **Deduplication**: Duplicate articles are detected and filtered
4. **Storage**: Articles are stored in Firestore
5. **Publishing**: Processed articles are published to Pub/Sub
6. **Monitoring**: Metrics are collected and exposed

## 🔧 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest src/tests/test_models.py
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Lint code
flake8 src/
mypy src/
```

### Adding New Adapters

1. Create adapter class inheriting from `BaseAdapter`
2. Implement required methods: `fetch_content()`, `test_connection()`, `get_source_info()`
3. Register adapter in `IngestionService.adapters`
4. Add tests in `src/tests/test_adapters.py`

## 🚀 Deployment

### Google Cloud Run

1. **Build and push image**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/ingestion-service
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy ingestion-service \
     --image gcr.io/PROJECT_ID/ingestion-service \
     --platform managed \
     --region us-central1 \
     --memory 1Gi \
     --cpu 1 \
     --max-instances 100
   ```

### Kubernetes

1. **Apply configurations**
   ```bash
   kubectl apply -f deployment/k8s/
   ```

2. **Check deployment**
   ```bash
   kubectl get pods -l app=ingestion-service
   ```

## 📈 Monitoring

### Metrics

Key metrics exposed at `/metrics`:

- `ingestion_articles_processed_total` - Total articles processed
- `ingestion_articles_processing_duration_seconds` - Processing time
- `ingestion_source_health_status` - Source health status
- `ingestion_duplicates_detected_total` - Duplicate articles detected
- `ingestion_rate_limit_exceeded_total` - Rate limit violations

### Dashboards

Grafana dashboards available at `http://localhost:3000`:
- Service Overview
- Source Health
- Processing Metrics
- Error Analysis

### Alerts

Configured alerts for:
- High error rates
- Source health degradation
- Processing delays
- Resource utilization

## 🔒 Security

### Security Features
- Input validation and sanitization
- Rate limiting and DDoS protection
- Secure headers (HSTS, CSP, etc.)
- Robots.txt compliance
- Content type validation

### Best Practices
- Use environment variables for secrets
- Enable Cloud Run authentication
- Monitor for suspicious activity
- Regular security updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation at `/docs`
- Review the health status at `/health`

## 🔄 Changelog

### v1.0.0
- Initial release
- Multi-source content ingestion
- Content normalization and deduplication
- Google Cloud integration
- Comprehensive monitoring
- Docker deployment support
