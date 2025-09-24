# Feedback & Human-in-the-Loop Service

A comprehensive microservice for collecting user feedback, managing editorial workflows, enabling human oversight of AI decisions, and creating continuous learning loops for model improvement.

## Features

### Multi-modal Feedback Collection
- **Explicit Feedback**: Direct user ratings, comments, and reports
- **Implicit Feedback**: Click tracking, dwell time, sharing behavior
- **Crowdsourced Feedback**: External annotation and validation
- **Editorial Feedback**: Professional content review and approval

### Editorial Workflow Management
- **Approval Chains**: Multi-level content approval workflows
- **Task Assignment**: Intelligent reviewer assignment based on expertise
- **Priority Management**: Urgent task escalation and handling
- **Audit Trails**: Complete workflow history and version control

### Human Oversight & Annotation
- **Interactive Annotation Tools**: Real-time annotation interface
- **Quality Control**: Inter-annotator agreement and validation
- **Guidelines Management**: Dynamic annotation guidelines and instructions
- **Progress Tracking**: Task completion monitoring and reporting

### Active Learning & Model Improvement
- **Uncertainty Detection**: Identify predictions needing human review
- **Training Data Generation**: Convert feedback into ML training data
- **Model Retraining**: Automated model updates based on new feedback
- **Performance Tracking**: Monitor model improvement over time

### Quality Assurance & Moderation
- **Automated Quality Scoring**: Multi-dimensional content quality assessment
- **Bias Detection**: Identify and flag biased content
- **Fact Checking**: Automated fact verification and accuracy scoring
- **Spam Detection**: Low-quality content filtering and removal

### Real-time Processing
- **WebSocket Support**: Real-time feedback processing and notifications
- **Stream Processing**: High-throughput event processing
- **Live Dashboards**: Real-time metrics and status updates
- **Instant Alerts**: Immediate notification of critical issues

### Analytics & Insights
- **Feedback Analytics**: Comprehensive feedback analysis and reporting
- **Trend Analysis**: Identify patterns and trends in feedback data
- **Performance Metrics**: Track system and model performance
- **Actionable Insights**: Generate recommendations for improvement

## Architecture

### Technology Stack
- **Backend**: FastAPI with Python 3.11+
- **Database**: PostgreSQL with async SQLAlchemy
- **Cache**: Redis for real-time data and session management
- **Search**: Elasticsearch for content indexing and search
- **Message Queue**: Apache Kafka for event streaming
- **ML/AI**: scikit-learn, transformers, spaCy for NLP
- **Monitoring**: Prometheus + Grafana for metrics and alerting
- **Deployment**: Docker + Kubernetes for container orchestration

### Core Components

```
src/
├── main.py                 # FastAPI application with WebSocket support
├── feedback_collection/   # Core feedback collection and processing
├── editorial_workflow/    # Editorial workflow management
├── annotation/            # Annotation tools and interfaces
├── active_learning/       # Active learning and model updates
├── quality_assurance/     # Automated QA and moderation
├── crowdsourcing/         # Crowdsourced feedback management
├── analytics/             # Feedback analytics and insights
├── realtime/              # Real-time processing and WebSockets
├── models/                # Database models and schemas
├── api/                   # REST API endpoints
├── middleware/            # Authentication and monitoring
├── notification/          # Notification services
└── rbac/                  # Role-based access control
```

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- PostgreSQL 15+
- Redis 7+

### Local Development

1. **Clone and setup**:
```bash
cd feedback-service
pip install -r requirements.txt
```

2. **Start dependencies**:
```bash
docker-compose up -d postgres redis elasticsearch kafka
```

3. **Initialize database**:
```bash
psql -h localhost -U feedback_user -d feedback_db -f init.sql
```

4. **Run the service**:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Access the service**:
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- WebSocket: ws://localhost:8000/ws/feedback/{user_id}

### Docker Deployment

1. **Build and run**:
```bash
docker-compose up --build
```

2. **Access services**:
- Feedback Service: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Kubernetes Deployment

1. **Apply configurations**:
```bash
kubectl apply -f k8s/
```

2. **Check deployment**:
```bash
kubectl get pods -n feedback-service
kubectl get services -n feedback-service
```

## API Usage

### Submit Feedback

```python
import requests

# Submit explicit feedback
feedback_data = {
    "content_id": "123e4567-e89b-12d3-a456-426614174000",
    "feedback_type": "explicit",
    "rating": 4.5,
    "comment": "Great content, very informative!",
    "metadata": {"source": "web", "user_agent": "Mozilla/5.0..."}
}

response = requests.post(
    "http://localhost:8000/api/v1/feedback/",
    json=feedback_data,
    headers={"Authorization": "Bearer your-token"}
)
```

### Real-time WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/feedback/user123');

ws.onopen = function() {
    console.log('Connected to feedback service');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

// Send feedback
ws.send(JSON.stringify({
    content_id: "123e4567-e89b-12d3-a456-426614174000",
    feedback_type: "implicit",
    signal_type: "click",
    metadata: {"timestamp": new Date().toISOString()}
}));
```

### Create Annotation Task

```python
annotation_task = {
    "content_id": "123e4567-e89b-12d3-a456-426614174000",
    "annotation_type": "sentiment",
    "annotator_ids": ["user1", "user2", "user3"],
    "guidelines": "Please annotate the sentiment of this content...",
    "deadline": "2024-01-15T23:59:59Z"
}

response = requests.post(
    "http://localhost:8000/api/v1/annotation/tasks",
    json=annotation_task
)
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/feedback_db

# Redis
REDIS_URL=redis://localhost:6379

# Elasticsearch
ELASTICSEARCH_URL=http://localhost:9200

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Security
SECRET_KEY=your-secret-key
ALGORITHM=HS256

# Performance
MAX_FEEDBACK_PER_SECOND=10000
WEBSOCKET_MAX_CONNECTIONS=1000

# Quality Thresholds
QUALITY_THRESHOLD_LOW=0.3
QUALITY_THRESHOLD_MEDIUM=0.6
QUALITY_THRESHOLD_HIGH=0.8
```

## Monitoring

### Metrics
- **Feedback Processing Rate**: Requests per second
- **Response Time**: P95 latency < 50ms
- **WebSocket Connections**: Active real-time connections
- **Annotation Quality**: Inter-annotator agreement scores
- **Model Performance**: Accuracy and confidence metrics

### Health Checks
- `/health` - Service health status
- `/metrics` - Prometheus metrics endpoint

### Alerts
- High error rates
- Slow response times
- Low annotation quality
- Model performance degradation

## Performance Requirements

- **Throughput**: 10,000+ feedback events per second
- **Latency**: P95 < 50ms for feedback processing
- **Annotation Interface**: < 100ms response time
- **Concurrent Users**: 1,000+ simultaneous annotators
- **Availability**: 99.9% uptime

## Security

- **Authentication**: JWT-based token authentication
- **Authorization**: Role-based access control (RBAC)
- **Data Encryption**: TLS for all communications
- **Audit Logging**: Complete action audit trails
- **Input Validation**: Comprehensive input sanitization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For questions and support:
- Create an issue in the repository
- Check the documentation at `/docs`
- Contact the development team
