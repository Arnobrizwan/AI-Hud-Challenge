# AI-Powered Summarization & Headline Generation Service

A comprehensive microservice for AI-powered content summarization and headline generation using FastAPI, Vertex AI, and advanced NLP models. This service provides extractive and abstractive summarization, dynamic headline generation, quality validation, bias detection, and multi-language support.

## 🚀 Features

### Core Capabilities
- **Extractive Summarization**: BERT-based extractive summarization with advanced scoring
- **Abstractive Summarization**: PaLM 2 powered abstractive summarization via Vertex AI
- **Hybrid Summarization**: Intelligent combination of extractive and abstractive methods
- **Dynamic Headline Generation**: T5-based headline generation with multiple style variants
- **Multi-length Summaries**: Support for 50, 120, 300 word summaries and custom lengths

### Quality & Validation
- **Comprehensive Quality Metrics**: ROUGE, BERTScore, factual consistency validation
- **Bias Detection**: Political, gender, racial, and sentiment bias analysis
- **Factual Consistency**: Entity, numerical, temporal, and semantic consistency checking
- **Readability Optimization**: Flesch-Kincaid scoring and readability assessment

### Advanced Features
- **A/B Testing Framework**: Built-in A/B testing for different approaches
- **Multi-language Support**: Translation and processing in 10+ languages
- **Real-time & Batch Processing**: Both real-time and batch processing modes
- **Performance Optimization**: GPU batching, caching, and parallel processing
- **Comprehensive Monitoring**: Metrics collection and performance monitoring

## 🏗️ Architecture

```
src/
├── main.py                 # FastAPI application
├── config/                 # Configuration management
├── summarization/          # Core summarization logic
│   ├── engine.py          # Main summarization engine
│   ├── extractive.py      # BERT-based extractive summarization
│   ├── abstractive.py     # Vertex AI abstractive summarization
│   ├── hybrid.py          # Hybrid summarization approach
│   ├── preprocessing.py   # Content preprocessing
│   ├── quality.py         # Quality scoring
│   ├── bias.py           # Bias detection
│   ├── consistency.py    # Factual consistency checking
│   ├── ab_testing.py     # A/B testing framework
│   └── translation.py    # Multi-language translation
├── headline_generation/    # Headline generation
├── quality_validation/     # Quality validation system
├── optimization/          # Performance optimizations
└── monitoring/            # Metrics and monitoring
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- Google Cloud Project with Vertex AI enabled
- Redis (for caching)

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd summarization-service
```

2. **Set up environment variables**
```bash
cp env.example .env
# Edit .env with your configuration
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required models**
```bash
python -m spacy download en_core_web_lg
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

5. **Run with Docker Compose**
```bash
docker-compose up -d
```

## 📖 API Usage

### Basic Summarization

```python
import requests

# Summarize content
response = requests.post("http://localhost:8000/summarize", json={
    "content": {
        "text": "Your article content here...",
        "title": "Article Title",
        "language": "en",
        "content_type": "news_article"
    },
    "target_lengths": [50, 120, 300],
    "methods": ["hybrid"],
    "headline_styles": ["news", "engaging"]
})

result = response.json()
print(f"Summary: {result['result']['summary']['text']}")
print(f"Headline: {result['result']['headline']['text']}")
```

### Batch Processing

```python
# Batch summarization
response = requests.post("http://localhost:8000/summarize/batch", json={
    "requests": [
        {
            "content": {"text": "First article..."},
            "target_lengths": [120]
        },
        {
            "content": {"text": "Second article..."},
            "target_lengths": [120]
        }
    ]
})
```

### Quality Validation

```python
# Validate summary quality
response = requests.post("http://localhost:8000/quality/validate", json={
    "original": "Original article text...",
    "summary": "Generated summary..."
})

quality_metrics = response.json()
print(f"ROUGE-1 F1: {quality_metrics['quality_metrics']['rouge1_f1']}")
print(f"Overall Score: {quality_metrics['overall_score']}")
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_CLOUD_PROJECT` | Google Cloud Project ID | Required |
| `VERTEX_AI_ENDPOINT` | Vertex AI endpoint URL | Required |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `USE_GPU` | Enable GPU acceleration | `true` |
| `BATCH_SIZE` | Processing batch size | `8` |
| `CACHE_TTL` | Cache time-to-live (seconds) | `3600` |

### Quality Thresholds

```python
MIN_QUALITY_SCORE=0.7      # Minimum quality score
MIN_CONSISTENCY_SCORE=0.8  # Minimum consistency score
MAX_BIAS_SCORE=0.3         # Maximum allowed bias score
```

## 📊 Monitoring

### Health Checks
- **Health**: `GET /health`
- **Readiness**: `GET /health/ready`
- **Metrics**: `GET /metrics`

### Prometheus Metrics
The service exposes comprehensive metrics at `/metrics`:
- Request counts and rates
- Processing times
- Quality scores
- Cache hit rates
- System resource usage

### Grafana Dashboard
Access Grafana at `http://localhost:3000` (admin/admin) for visualization.

## 🧪 A/B Testing

### Create A/B Test

```python
# Create test with variants
test_id = await ab_test_manager.create_test(
    test_name="Summarization Method Test",
    variants=[
        A/BTestVariant(
            variant_id="extractive",
            name="Extractive Only",
            parameters={"method": "extractive"},
            traffic_percentage=0.5
        ),
        A/BTestVariant(
            variant_id="hybrid",
            name="Hybrid Method",
            parameters={"method": "hybrid"},
            traffic_percentage=0.5
        )
    ]
)
```

### Record Metrics

```python
# Record test metrics
await ab_test_manager.record_metrics(
    test_id=test_id,
    variant_id="hybrid",
    metrics=TestMetrics(
        quality_score=0.85,
        processing_time=2.3,
        user_satisfaction=0.9
    )
)
```

## 🌍 Multi-language Support

### Supported Languages
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)

### Translation Example

```python
# Translate content
translated_content = await translation_service.translate_content(
    content=original_content,
    target_language=Language.SPANISH
)
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run
docker build -t summarization-service .
docker run -p 8000:8000 --env-file .env summarization-service
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/
```

### Google Cloud Run

```bash
# Deploy to Cloud Run
gcloud run deploy summarization-service \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 📈 Performance Optimization

### GPU Acceleration
- Automatic GPU detection and utilization
- Batch processing for improved throughput
- Model quantization for faster inference

### Caching
- Redis-based result caching
- Intelligent cache invalidation
- Local cache for hot data

### Parallel Processing
- Async/await throughout
- Controlled concurrency limits
- Background task processing

## 🔒 Security

### Rate Limiting
- Configurable rate limits per endpoint
- IP-based throttling
- Request size limits

### Authentication
- JWT token validation (optional)
- API key authentication (optional)
- CORS configuration

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_summarization.py
```

### Test Coverage
- Unit tests for all components
- Integration tests for API endpoints
- Performance tests for load testing

## 📚 API Documentation

### Interactive Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/summarize` | POST | Generate summary and headlines |
| `/summarize/batch` | POST | Batch summarization |
| `/headlines/generate` | POST | Generate headlines only |
| `/quality/validate` | POST | Validate summary quality |
| `/metrics` | GET | Get service metrics |
| `/health` | GET | Health check |

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
- Create an issue on GitHub
- Check the documentation
- Review the API examples

## 🔄 Changelog

### v1.0.0
- Initial release
- Core summarization functionality
- Headline generation
- Quality validation
- Multi-language support
- A/B testing framework
- Comprehensive monitoring
