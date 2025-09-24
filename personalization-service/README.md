# Personalization Service

A sophisticated Personalization Logic microservice using FastAPI, Redis, PostgreSQL, and ML models. This service provides real-time content personalization using collaborative filtering, content-based filtering, and contextual bandits for optimal user engagement.

## Features

### Core Personalization Algorithms
- **Hybrid Recommendation System**: Combines collaborative filtering and content-based filtering
- **Real-time User Profile Updates**: Dynamic learning from user interactions
- **Contextual Multi-armed Bandits**: Thompson Sampling, UCB, and Epsilon-Greedy algorithms
- **Cold Start Handling**: Demographic-based initialization and popularity fallback
- **Preference Inference**: Learning from implicit feedback signals
- **Topic and Source Personalization**: Multi-dimensional preference modeling
- **Temporal Preference Modeling**: Time-aware recommendation patterns
- **Diversity and Serendipity Optimization**: Balanced recommendation exploration
- **A/B Testing Framework**: Algorithm performance comparison
- **Privacy-preserving Personalization**: Differential privacy and data anonymization

### Technical Specifications
- **Implicit Matrix Factorization**: For collaborative filtering
- **TF-IDF and Embeddings**: For content-based filtering
- **Thompson Sampling**: For contextual bandits
- **Redis**: Real-time feature serving and caching
- **PostgreSQL**: User profiles and interactions storage
- **Online Learning**: Mini-batch updates for real-time adaptation
- **Feature Engineering**: Comprehensive user context modeling
- **Privacy-differential Techniques**: Secure data processing
- **Comprehensive Evaluation Metrics**: Offline and online performance tracking

## Architecture

```
src/
├── main.py                 # FastAPI application
├── personalization/       # Core personalization logic
├── collaborative/         # Collaborative filtering
├── content_based/         # Content-based filtering
├── bandits/               # Contextual bandit algorithms
├── profiles/              # User profile management
├── diversity/             # Diversity optimization
├── cold_start/           # Cold start handling
├── privacy/              # Privacy preservation
├── ab_testing/           # A/B testing framework
├── evaluation/            # Personalization metrics
├── database/              # Database clients
└── config/                # Configuration settings
```

## Quick Start

### Using Docker Compose

1. **Clone the repository**
```bash
git clone <repository-url>
cd personalization-service
```

2. **Set up environment variables**
```bash
cp env.example .env
# Edit .env with your configuration
```

3. **Start the services**
```bash
docker-compose up -d
```

4. **Access the service**
- API: http://localhost:8000
- Metrics: http://localhost:9090
- API Documentation: http://localhost:8000/docs

### Using Kubernetes

1. **Apply Kubernetes manifests**
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
```

2. **Check deployment status**
```bash
kubectl get pods -n personalization-service
kubectl get services -n personalization-service
```

## API Endpoints

### Core Endpoints

- `POST /personalize` - Get personalized content recommendations
- `POST /interaction` - Record user interactions for learning
- `GET /profile/{user_id}` - Get user profile information
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics

### Management Endpoints

- `GET /metrics/models` - Get model performance metrics
- `POST /retrain` - Retrain all models
- `GET /ab/experiments` - Get A/B testing experiments
- `POST /ab/experiments` - Create new A/B test
- `GET /evaluation/metrics` - Get comprehensive evaluation metrics

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:password@localhost:5432/personalization` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `ENVIRONMENT` | Environment (development/production) | `development` |
| `DEBUG` | Enable debug mode | `false` |
| `COLLABORATIVE_FACTORS` | Matrix factorization factors | `100` |
| `COLLABORATIVE_REGULARIZATION` | Regularization parameter | `0.01` |
| `CONTENT_EMBEDDING_DIM` | Content embedding dimension | `384` |
| `TFIDF_MAX_FEATURES` | TF-IDF max features | `10000` |
| `BANDIT_ALPHA` | Thompson Sampling alpha | `1.0` |
| `BANDIT_BETA` | Thompson Sampling beta | `1.0` |
| `EPSILON` | Epsilon-Greedy epsilon | `0.1` |
| `DIVERSITY_THRESHOLD` | Diversity threshold | `0.3` |
| `SERENDIPITY_WEIGHT` | Serendipity weight | `0.2` |
| `PRIVACY_EPSILON` | Differential privacy epsilon | `1.0` |
| `PRIVACY_DELTA` | Differential privacy delta | `1e-5` |

## Usage Examples

### Basic Personalization Request

```python
import requests

# Personalize content for a user
response = requests.post("http://localhost:8000/personalize", json={
    "user_id": "user123",
    "candidates": [
        {
            "id": "article1",
            "title": "Machine Learning Trends",
            "content": "Latest trends in ML...",
            "topics": ["technology", "ai"],
            "source": "tech-news"
        }
    ],
    "context": {
        "device_type": "mobile",
        "time_of_day": "morning",
        "location": "US"
    },
    "diversity_params": {
        "enable_diversity": True,
        "topic_diversity_threshold": 0.3,
        "max_results": 10
    }
})

recommendations = response.json()
```

### Recording User Interactions

```python
# Record user interaction
interaction = {
    "user_id": "user123",
    "item_id": "article1",
    "interaction_type": "click",
    "rating": 4.5,
    "context": {
        "topics": ["technology", "ai"],
        "source": "tech-news"
    },
    "session_id": "session456",
    "device_type": "mobile"
}

response = requests.post("http://localhost:8000/interaction", json=interaction)
```

## Monitoring and Observability

### Prometheus Metrics

The service exposes comprehensive metrics:

- `personalization_requests_total` - Total personalization requests
- `personalization_request_duration_seconds` - Request duration
- `personalization_active_users` - Number of active users
- `personalization_recommendations_total` - Total recommendations generated

### Health Checks

- **Liveness Probe**: `/health` endpoint
- **Readiness Probe**: Database connectivity check
- **Metrics Endpoint**: `/metrics` for Prometheus

### Logging

Structured logging with JSON format including:
- Request/response logging
- Error tracking
- Performance metrics
- User interaction events

## Development

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker (optional)

### Local Development

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Set up databases**
```bash
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Run database migrations
python -c "from src.database.postgres_client import PostgreSQLClient; import asyncio; asyncio.run(PostgreSQLClient().connect())"
```

3. **Run the service**
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Performance Considerations

### Caching Strategy
- User profiles cached in Redis (1 hour TTL)
- Model predictions cached (1 hour TTL)
- Content embeddings cached (24 hours TTL)

### Database Optimization
- Indexed user interactions by user_id and timestamp
- GIN indexes on JSONB fields
- Vector indexes for content embeddings
- Connection pooling for PostgreSQL

### Scalability
- Horizontal pod autoscaling (3-10 replicas)
- Redis clustering for high availability
- PostgreSQL read replicas for analytics
- Load balancing with ingress controller

## Security

### Privacy Protection
- Differential privacy for user data
- Data anonymization techniques
- Secure aggregation methods
- Privacy budget management

### Authentication & Authorization
- JWT token validation
- Rate limiting per user
- Input validation and sanitization
- CORS configuration

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
- Check the API documentation at `/docs`
- Review the monitoring dashboards
