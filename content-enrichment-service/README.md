# AI-Powered Content Enrichment Service

A comprehensive microservice for AI-powered content enrichment using FastAPI, Vertex AI, spaCy, and Cloud SQL. This service performs named entity recognition, topic classification, sentiment analysis, and signal extraction to enhance content understanding.

## 🚀 Features

### Core AI/ML Capabilities
- **Named Entity Recognition (NER)** with entity linking to knowledge bases
- **Multi-label Topic Classification** with hierarchical categories
- **Sentiment Analysis** and emotion detection
- **Geographic Entity Extraction** and geocoding
- **Organization and Person Entity Disambiguation**
- **Content Quality Signals** and trustworthiness scoring
- **Real-time and Batch Processing** modes
- **Entity Knowledge Base Integration**
- **Custom Model Training** and deployment
- **A/B Testing** for model performance

### Technical Specifications
- **Vertex AI** for large language models (PaLM 2, Gemini)
- **spaCy** for NLP pipeline and entity recognition
- **Transformers** library for custom model fine-tuning
- **PostgreSQL** with pgvector for entity embeddings
- **Redis** for caching and rate limiting
- **Cloud Storage** for model artifacts
- **MLflow** for experiment tracking
- **Prometheus** for performance monitoring
- **OpenTelemetry** for distributed tracing

## 📋 Requirements

- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- Redis 7+
- Docker and Docker Compose
- Google Cloud Platform account (for Vertex AI)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd content-enrichment-service
```

### 2. Set Up Environment
```bash
cp env.example .env
# Edit .env with your configuration
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy Models
```bash
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm
```

### 5. Set Up Database
```bash
# Start PostgreSQL with pgvector
docker-compose up -d postgres

# Run database initialization
psql -h localhost -p 5433 -U postgres -d content_enrichment -f init.sql
```

### 6. Start the Service
```bash
# Development
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Production with Docker
docker-compose up -d
```

## 🏗️ Architecture

```
src/
├── main.py                 # FastAPI application
├── enrichment/            # Core enrichment pipeline
│   ├── pipeline.py        # Main enrichment orchestrator
│   └── __init__.py
├── models/                # AI/ML models and wrappers
│   ├── content.py         # Data models
│   └── __init__.py
├── entities/              # Entity extraction and linking
│   ├── extractor.py       # NER with spaCy
│   └── __init__.py
├── topics/                # Topic classification
│   ├── classifier.py      # Hierarchical topic classification
│   └── __init__.py
├── sentiment/             # Sentiment and emotion analysis
│   ├── analyzer.py        # Multilingual sentiment analysis
│   └── __init__.py
├── signals/               # Content quality signals
│   ├── extractor.py       # Quality and trustworthiness scoring
│   └── __init__.py
├── knowledge_base/        # Entity knowledge base
│   ├── entity_kb.py       # Entity linking and disambiguation
│   └── __init__.py
├── utils/                 # NLP utilities and helpers
│   ├── language_detector.py
│   ├── text_processor.py
│   └── __init__.py
├── middleware/            # FastAPI middleware
│   ├── rate_limiter.py    # Rate limiting
│   ├── auth.py           # Authentication
│   └── __init__.py
├── monitoring/            # Observability
│   ├── metrics.py        # Prometheus metrics
│   ├── health.py         # Health checks
│   └── __init__.py
└── config/               # Configuration
    ├── settings.py       # Application settings
    └── __init__.py
```

## 🔧 API Endpoints

### Core Endpoints

#### Enrich Single Content
```http
POST /api/v1/enrich
Content-Type: application/json

{
  "content": {
    "title": "AI Revolution in Healthcare",
    "content": "Artificial intelligence is transforming healthcare...",
    "summary": "Overview of AI applications in healthcare",
    "url": "https://example.com/article",
    "source": "Tech News",
    "author": "Dr. Jane Smith",
    "published_date": "2024-01-15T10:00:00Z",
    "content_type": "article",
    "language": "en"
  },
  "processing_mode": "realtime",
  "include_entities": true,
  "include_topics": true,
  "include_sentiment": true,
  "include_signals": true,
  "include_trust_score": true
}
```

#### Enrich Batch Content
```http
POST /api/v1/enrich/batch
Content-Type: application/json

{
  "contents": [
    {
      "title": "Article 1",
      "content": "Content 1...",
      "content_type": "article"
    },
    {
      "title": "Article 2", 
      "content": "Content 2...",
      "content_type": "blog_post"
    }
  ],
  "processing_mode": "batch"
}
```

### Monitoring Endpoints

#### Health Check
```http
GET /health
```

#### Metrics
```http
GET /metrics
```

#### Service Info
```http
GET /api/v1/info
```

#### Model Info
```http
GET /api/v1/models
```

## 📊 Content Enrichment Pipeline

The service processes content through a comprehensive pipeline:

### 1. Entity Extraction
- **spaCy NER** for base entity recognition
- **Custom patterns** for specialized entities (emails, phones, URLs, etc.)
- **Entity linking** to Wikidata knowledge base
- **Disambiguation** and deduplication
- **Confidence scoring** for each entity

### 2. Topic Classification
- **Hierarchical taxonomy** with multiple levels
- **Multi-label classification** with confidence scores
- **Custom topic models** trained on domain data
- **Keyword-based fallback** for unsupported languages

### 3. Sentiment Analysis
- **Multilingual sentiment** using transformer models
- **Emotion detection** (joy, sadness, anger, fear, etc.)
- **Subjectivity analysis** (objective vs subjective)
- **Polarity scoring** (-1 to +1 scale)

### 4. Content Quality Signals
- **Readability scores** (Flesch-Kincaid, SMOG, etc.)
- **Factual claim detection** and counting
- **Citation analysis** and source verification
- **Bias detection** and political leaning
- **Engagement prediction** and virality potential
- **Authority scoring** and expertise indicators

### 5. Trustworthiness Assessment
- **Source reliability** evaluation
- **Fact-checking quality** assessment
- **Citation quality** analysis
- **Author credibility** scoring
- **Content quality** metrics
- **Bias indicators** and warning flags

## 🎯 Performance Requirements

- **Processing Speed**: 500+ articles per minute
- **Latency**: P95 enrichment latency < 30 seconds
- **Model Inference**: < 5 seconds per article
- **Languages**: Support for 50+ languages
- **Memory**: < 4GB per instance
- **Concurrency**: 100+ concurrent requests

## 📈 Monitoring and Observability

### Metrics Collected
- **Request metrics**: Total requests, duration, success rate
- **Enrichment metrics**: Processing time, entity counts, topic counts
- **Model metrics**: Inference time, accuracy, confidence scores
- **System metrics**: CPU, memory, disk usage
- **Error metrics**: Error rates by component and type

### Health Checks
- **System resources**: CPU, memory, disk usage
- **Database connectivity**: PostgreSQL connection and query performance
- **Redis connectivity**: Cache performance and memory usage
- **Model availability**: AI/ML model loading and inference
- **External services**: Google Cloud, MLflow connectivity

### Dashboards
- **Grafana dashboards** for visualization
- **Prometheus metrics** for alerting
- **Real-time monitoring** of service health
- **Performance tracking** and optimization

## 🔒 Security

### Authentication
- **JWT tokens** for user authentication
- **API key** support for service-to-service communication
- **Role-based access control** (RBAC)
- **Permission-based** endpoint protection

### Rate Limiting
- **Redis-based** distributed rate limiting
- **Per-user** and per-IP rate limits
- **Configurable** request windows and limits
- **Graceful degradation** under load

### Data Protection
- **Input validation** and sanitization
- **Content filtering** for sensitive information
- **Audit logging** for compliance
- **Secure configuration** management

## 🧪 Testing

### Test Suite
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_entities.py
pytest tests/test_sentiment.py
pytest tests/test_integration.py
```

### Test Categories
- **Unit tests**: Individual component testing
- **Integration tests**: End-to-end pipeline testing
- **Performance tests**: Load and stress testing
- **Accuracy tests**: Model performance validation

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Scale the service
docker-compose up -d --scale content-enrichment-service=3
```

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=content-enrichment-service
```

### Google Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy content-enrichment-service \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 📚 Configuration

### Environment Variables
See `env.example` for all available configuration options:

- **Database**: PostgreSQL connection settings
- **Redis**: Cache and rate limiting configuration
- **Google Cloud**: Vertex AI and Cloud Storage settings
- **MLflow**: Experiment tracking configuration
- **Performance**: Timeout and concurrency settings
- **Monitoring**: Metrics and health check settings

### Model Configuration
- **Confidence thresholds** for entity and topic classification
- **Processing limits** for content length and batch size
- **Language support** configuration
- **Custom model paths** and versions

## 🔄 A/B Testing

The service includes built-in A/B testing capabilities:

- **Traffic splitting** for model variants
- **Performance comparison** between models
- **Statistical significance** testing
- **Automatic rollback** for poor-performing variants

## 📖 Usage Examples

### Python Client
```python
import requests

# Enrich single content
response = requests.post(
    "http://localhost:8000/api/v1/enrich",
    json={
        "content": {
            "title": "AI in Healthcare",
            "content": "Artificial intelligence is revolutionizing healthcare...",
            "content_type": "article"
        }
    }
)

enriched = response.json()
print(f"Entities: {len(enriched['enriched_content']['entities'])}")
print(f"Topics: {len(enriched['enriched_content']['topics'])}")
print(f"Sentiment: {enriched['enriched_content']['sentiment']['sentiment']}")
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/api/v1/enrich" \
  -H "Content-Type: application/json" \
  -d '{
    "content": {
      "title": "Machine Learning Advances",
      "content": "Recent advances in machine learning...",
      "content_type": "article"
    }
  }'
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the API examples
- Contact the development team

## 🔮 Roadmap

- [ ] **Multi-modal content** support (images, videos)
- [ ] **Real-time streaming** processing
- [ ] **Advanced A/B testing** with statistical analysis
- [ ] **Custom model training** UI
- [ ] **GraphQL API** support
- [ ] **Webhook notifications** for processing completion
- [ ] **Content similarity** detection
- [ ] **Automated model retraining** pipelines
