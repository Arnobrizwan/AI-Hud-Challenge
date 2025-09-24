# High-Performance Ranking Microservice

A production-ready content ranking microservice built with FastAPI, Redis, and advanced ML algorithms. This service provides real-time content ranking with personalization, incorporating recency, relevance, authority, and user preferences for optimal content discovery.

## 🚀 Features

### Core Capabilities
- **Multi-objective ranking** with configurable weights
- **Real-time personalization** using user behavior
- **Content freshness** and trending detection
- **Source authority** and credibility scoring
- **Topic-based ranking** with user preferences
- **Geographic and temporal** relevance
- **A/B testing** for ranking algorithms
- **Learning-to-rank** with online updates
- **Caching and precomputation** optimization
- **Sub-100ms response times** at scale

### Technical Specifications
- **LightGBM** for learning-to-rank models
- **Redis** for feature caching and precomputed rankings
- **Real-time feature** computation pipeline
- **Multi-armed bandits** for exploration/exploitation
- **Vector similarity** for content-based filtering
- **Collaborative filtering** for behavior-based ranking
- **Time-decay functions** for content freshness
- **Geographic distance** calculations
- **Comprehensive A/B testing** framework

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  Ranking Engine │    │  Personalization│
│                 │◄──►│                 │◄──►│     Engine      │
│  - REST API     │    │  - LightGBM     │    │  - Collaborative│
│  - Health Check │    │  - Heuristics   │    │  - Content-based│
│  - Metrics      │    │  - Hybrid       │    │  - Topic-based  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Feature Store  │    │  A/B Testing    │    │   Monitoring    │
│                 │    │                 │    │                 │
│  - Content      │    │  - Experiments  │    │  - Prometheus   │
│  - Freshness    │    │  - Variants     │    │  - Health Check │
│  - Authority    │    │  - Assignment   │    │  - Performance  │
│  - Personal     │    │  - Analysis     │    │  - Alerts       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│  Redis Cache    │
│                 │
│  - Features     │
│  - Rankings     │
│  - Models       │
│  - Sessions     │
└─────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.11+
- Redis 6.0+
- Docker (optional)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI-Hud-Challenge
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Redis**
   ```bash
   redis-server
   ```

4. **Set environment variables**
   ```bash
   cp config.env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python -m uvicorn src.main:app --reload
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access the services**
   - API: http://localhost:8000
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

### Kubernetes Deployment

1. **Deploy to Kubernetes**
   ```bash
   kubectl apply -f kubernetes/
   ```

2. **Check deployment status**
   ```bash
   kubectl get pods
   kubectl get services
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `WORKERS` | Number of worker processes | `1` |
| `LOG_LEVEL` | Logging level | `info` |
| `PROMETHEUS_PORT` | Prometheus metrics port | `8001` |

### Ranking Configuration

The ranking engine supports various configuration options:

```python
# Default ranking weights
DEFAULT_WEIGHTS = {
    'relevance': 0.3,
    'freshness': 0.25,
    'authority': 0.2,
    'personalization': 0.15,
    'diversity': 0.1
}

# Feature cache TTLs (seconds)
FEATURE_TTLS = {
    'content': 3600,      # 1 hour
    'freshness': 300,     # 5 minutes
    'authority': 1800,    # 30 minutes
    'personalization': 600,  # 10 minutes
    'contextual': 300,    # 5 minutes
    'interaction': 60,    # 1 minute
}
```

## 📚 API Documentation

### Core Endpoints

#### Rank Content
```http
POST /rank
Content-Type: application/json

{
  "user_id": "user123",
  "query": "artificial intelligence",
  "limit": 20,
  "enable_personalization": true,
  "content_types": ["article"],
  "topics": ["technology", "AI"],
  "location": {"lat": 40.7128, "lng": -74.0060}
}
```

**Response:**
```json
{
  "articles": [
    {
      "article": {
        "id": "article_1",
        "title": "AI Breakthrough",
        "content": "...",
        "published_at": "2024-01-01T00:00:00Z",
        "source": {...},
        "author": {...}
      },
      "rank": 1,
      "score": 0.95,
      "personalized_score": 0.92,
      "explanation": "matches your topic interests, from a source you like"
    }
  ],
  "total_count": 100,
  "algorithm_variant": "ml_ranker",
  "processing_time_ms": 45.2,
  "features_computed": 20,
  "cache_hit_rate": 0.85
}
```

#### Get User Profile
```http
GET /users/{user_id}/profile
```

#### Update User Profile
```http
PUT /users/{user_id}/profile
Content-Type: application/json

{
  "user_id": "user123",
  "topic_preferences": {
    "technology": 0.9,
    "AI": 0.8,
    "science": 0.7
  },
  "source_preferences": {
    "source_1": 0.8,
    "source_2": 0.6
  },
  "reading_patterns": {
    "preferred_hours": [9, 10, 11, 14, 15, 16]
  }
}
```

### Monitoring Endpoints

#### Health Check
```http
GET /health
```

#### Performance Metrics
```http
GET /metrics/performance?time_window=60
```

#### Algorithm Comparison
```http
GET /metrics/algorithm-comparison
```

#### Cache Statistics
```http
GET /cache/stats
```

### A/B Testing Endpoints

#### Get Experiments
```http
GET /experiments
```

#### Get Experiment Stats
```http
GET /experiments/{experiment_id}/stats
```

#### Create Experiment
```http
POST /experiments
Content-Type: application/json

{
  "experiment_id": "ranking_algorithm_v5",
  "name": "New Ranking Algorithm",
  "variants": [
    {
      "variant_id": "ml_ranker_v2",
      "name": "ML Ranker V2",
      "weight": 0.5,
      "config": {
        "algorithm": "lightgbm",
        "model_version": "v4"
      }
    },
    {
      "variant_id": "hybrid_v2",
      "name": "Hybrid V2",
      "weight": 0.5,
      "config": {
        "algorithm": "hybrid",
        "ml_weight": 0.8,
        "heuristic_weight": 0.2
      }
    }
  ],
  "start_date": "2024-01-01T00:00:00Z",
  "is_active": true
}
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_ranking_engine.py

# Run with verbose output
pytest -v
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **API Tests**: Endpoint testing
- **Performance Tests**: Load and stress testing

## 📊 Monitoring and Observability

### Metrics

The service exposes comprehensive metrics via Prometheus:

- **Request Metrics**: Total requests, response times, error rates
- **Performance Metrics**: Feature computation time, ranking time
- **Cache Metrics**: Hit rates, miss rates, memory usage
- **System Metrics**: CPU, memory, Redis connections
- **Business Metrics**: User engagement, algorithm performance

### Health Checks

The service provides detailed health checks:

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "checks": {
    "response_time": {
      "status": "healthy",
      "value": 45.2,
      "threshold": 100
    },
    "error_rate": {
      "status": "healthy",
      "value": 0.01,
      "threshold": 0.05
    },
    "cpu_usage": {
      "status": "healthy",
      "value": 65.0,
      "threshold": 80
    }
  }
}
```

### Alerting

Configure alerts for:
- Response time > 100ms
- Error rate > 5%
- CPU usage > 80%
- Memory usage > 80%
- Cache hit rate < 50%

## 🚀 Performance Optimization

### Caching Strategy

1. **Feature Caching**: Cache computed features with appropriate TTLs
2. **Result Caching**: Cache ranking results for identical requests
3. **Model Caching**: Cache ML model predictions
4. **Precomputation**: Precompute common features and rankings

### Performance Targets

- **P95 Response Time**: < 100ms
- **Throughput**: 10,000+ QPS
- **Memory Usage**: < 2GB per instance
- **Feature Computation**: < 50ms per article
- **Model Inference**: < 20ms for 100 articles

### Scaling

- **Horizontal Scaling**: Multiple service instances
- **Load Balancing**: Distribute requests across instances
- **Redis Clustering**: Scale cache layer
- **Database Sharding**: Scale data storage

## 🔒 Security

### Authentication
- API key authentication
- JWT token support
- Rate limiting

### Data Protection
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CORS configuration

### Privacy
- User data anonymization
- GDPR compliance
- Data retention policies

## 🛠️ Development

### Code Structure
```
src/
├── main.py                 # FastAPI application
├── schemas.py             # Data models and schemas
├── ranking/               # Core ranking algorithms
│   └── engine.py
├── personalization/       # User personalization
│   └── engine.py
├── features/              # Feature extraction
│   └── extractor.py
├── optimization/          # Caching and optimization
│   └── cache.py
├── testing/               # A/B testing framework
│   └── ab_framework.py
└── monitoring/            # Performance and quality metrics
    └── metrics.py
```

### Adding New Features

1. **Create feature branch**
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Implement feature**
   - Add tests
   - Update documentation
   - Follow coding standards

3. **Run tests and linting**
   ```bash
   pytest
   black src/
   isort src/
   mypy src/
   ```

4. **Create pull request**

### Code Quality

- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **pytest**: Testing
- **pytest-cov**: Coverage

## 📈 Roadmap

### Phase 1 (Current)
- ✅ Core ranking engine
- ✅ Personalization system
- ✅ A/B testing framework
- ✅ Monitoring and metrics
- ✅ Docker deployment

### Phase 2 (Next)
- [ ] Advanced ML models
- [ ] Real-time learning
- [ ] Multi-tenant support
- [ ] Advanced analytics

### Phase 3 (Future)
- [ ] Graph-based ranking
- [ ] Federated learning
- [ ] Edge deployment
- [ ] AutoML integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: [API Docs](http://localhost:8000/docs)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

## 🙏 Acknowledgments

- FastAPI for the web framework
- LightGBM for machine learning
- Redis for caching
- Prometheus for monitoring
- The open-source community