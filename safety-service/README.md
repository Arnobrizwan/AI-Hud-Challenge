# Safety Service

A comprehensive Drift, Abuse, and Safety microservice built with FastAPI, Redis, PostgreSQL, and ML models. This service detects data drift, prevents abuse, implements content safety measures, and maintains system integrity through automated monitoring and response mechanisms.

## Features

### Multi-dimensional Drift Detection
- **Data Drift**: Detects changes in data distributions using statistical tests (KS, Chi-Square, PSI, Wasserstein)
- **Concept Drift**: Identifies changes in the relationship between features and labels
- **Prediction Drift**: Monitors changes in model predictions over time
- **Feature Importance Drift**: Tracks changes in feature importance rankings

### Real-time Abuse Detection
- **Behavioral Analysis**: ML-based anomaly detection for user behavior patterns
- **Graph-based Detection**: Network analysis to identify abuse patterns and collusion
- **Rule-based Detection**: Configurable rules for known abuse patterns
- **Reputation System**: Dynamic user reputation scoring and management
- **CAPTCHA Challenges**: Automated challenge generation for suspicious users

### Content Safety and Moderation
- **Toxicity Detection**: AI-powered toxicity and hate speech detection
- **Spam Detection**: Advanced spam and low-quality content filtering
- **Misinformation Detection**: Fact-checking and misinformation identification
- **Adult Content Detection**: NSFW content detection and filtering
- **Violence Detection**: Violence and harmful content identification
- **External API Integration**: Integration with third-party moderation services

### Advanced Rate Limiting
- **Sliding Window**: Time-based rate limiting with configurable windows
- **Token Bucket**: Burst-capable rate limiting algorithm
- **Adaptive Limiting**: Dynamic rate limits based on system load and user behavior
- **Geolocation-based**: Location-aware rate limiting and blocking
- **DDoS Protection**: Advanced protection against distributed attacks

### Compliance Monitoring
- **GDPR Compliance**: Automated GDPR compliance checking and reporting
- **Content Policy Monitoring**: Real-time content policy violation detection
- **Privacy Monitoring**: Privacy-preserving monitoring techniques
- **Audit Trails**: Comprehensive logging and audit trail management

### Automated Incident Response
- **Incident Classification**: AI-powered incident severity and type classification
- **Response Orchestration**: Automated execution of response actions
- **Escalation Management**: Multi-level escalation with stakeholder notification
- **Communication Management**: Automated stakeholder communication and updates

## Architecture

```
src/
├── main.py                    # FastAPI application entry point
├── safety_engine/            # Core safety monitoring engine
├── drift_detection/          # Data and concept drift detection
├── abuse_detection/          # Abuse pattern detection
├── content_moderation/       # Content safety and moderation
├── rate_limiting/           # Advanced rate limiting
├── compliance/              # Regulatory compliance monitoring
├── incident_response/       # Automated incident handling
├── anomaly_detection/       # System anomaly detection
└── audit/                   # Audit logging and reporting
```

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker and Docker Compose (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd safety-service
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

5. **Or run locally**
   ```bash
   # Start PostgreSQL and Redis
   # Update .env with your database URLs
   
   # Run the application
   uvicorn src.main:app --reload
   ```

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/safety_db
REDIS_URL=redis://localhost:6379/0

# Application
ENVIRONMENT=development
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key

# ML Models
MODEL_PATH=/app/models
ENABLE_ML_MODELS=true

# External APIs
PERSPECTIVE_API_KEY=your-perspective-key
OPENAI_API_KEY=your-openai-key

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
```

## Usage Examples

### Safety Monitoring

```python
from safety_engine import SafetyMonitoringEngine

# Initialize the safety engine
safety_engine = SafetyMonitoringEngine()
await safety_engine.initialize()

# Create a monitoring request
request = SafetyMonitoringRequest(
    user_id="user123",
    content="This is a test message",
    features={"text_length": 20, "sentiment": 0.5}
)

# Perform safety monitoring
status = await safety_engine.monitor_system_safety(request)
print(f"Safety Score: {status.overall_score}")
print(f"Requires Intervention: {status.requires_intervention}")
```

### Drift Detection

```python
from drift_detection import MultidimensionalDriftDetector

# Initialize drift detector
drift_detector = MultidimensionalDriftDetector()
await drift_detector.initialize()

# Detect drift
drift_request = DriftDetectionRequest(
    reference_data=reference_df,
    current_data=current_df,
    features_to_monitor=["feature1", "feature2"]
)

result = await drift_detector.detect_comprehensive_drift(drift_request)
print(f"Drift Severity: {result.overall_severity}")
print(f"Drifted Features: {result.data_drift.drifted_features}")
```

### Content Moderation

```python
from content_moderation import ContentModerationEngine

# Initialize content moderator
moderator = ContentModerationEngine()
await moderator.initialize()

# Moderate content
content = ContentItem(
    id="content123",
    text_content="This is potentially harmful content",
    user_id="user123"
)

result = await moderator.moderate_content(content)
print(f"Safety Score: {result.overall_safety_score}")
print(f"Violations: {result.violations}")
```

## Performance Requirements

- **Real-time drift detection**: < 1 second
- **Abuse detection response**: < 500ms
- **Content moderation**: < 2 seconds
- **Rate limiting check**: < 10ms
- **Support**: 100k+ safety checks per minute

## Monitoring and Observability

### Metrics
- Safety check success/failure rates
- Drift detection accuracy and performance
- Abuse detection precision and recall
- Content moderation effectiveness
- Rate limiting effectiveness
- System performance metrics

### Dashboards
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### Logging
- Structured logging with correlation IDs
- Audit trails for all safety checks
- Performance and error logging
- Security event logging

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Features

1. **Create new modules** in the appropriate package
2. **Add data models** in the models.py files
3. **Implement business logic** following the existing patterns
4. **Add tests** for new functionality
5. **Update documentation** and API docs

## Deployment

### Production Deployment

1. **Set up infrastructure**
   - PostgreSQL cluster
   - Redis cluster
   - Load balancer
   - Monitoring stack

2. **Configure environment**
   - Set production environment variables
   - Configure SSL/TLS
   - Set up monitoring and alerting

3. **Deploy application**
   - Build Docker image
   - Deploy to Kubernetes or Docker Swarm
   - Configure health checks and auto-scaling

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: safety-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: safety-service
  template:
    metadata:
      labels:
        app: safety-service
    spec:
      containers:
      - name: safety-service
        image: safety-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: safety-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: safety-secrets
              key: redis-url
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
- Create an issue in the repository
- Contact the development team
- Check the documentation and examples
