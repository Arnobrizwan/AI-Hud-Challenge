# Observability & Runbooks Service

A comprehensive observability and monitoring microservice built with FastAPI, providing complete system monitoring, alerting, troubleshooting guides, and automated remediation for the entire news aggregation pipeline.

## Features

### 🔍 Comprehensive Monitoring
- **Metrics Collection**: Prometheus-based metrics collection with custom business metrics
- **Distributed Tracing**: OpenTelemetry integration for end-to-end request tracing
- **Centralized Logging**: Structured logging with ELK stack integration
- **Health Monitoring**: Real-time health checks for all services and components

### 🚨 Intelligent Alerting
- **Smart Alert Routing**: Intelligent alert correlation and suppression
- **Escalation Policies**: Automated escalation based on severity and time
- **Multi-channel Notifications**: Slack, email, PagerDuty, and webhook support
- **Alert Management**: Full lifecycle management of alerts and incidents

### 📚 Automated Runbooks
- **Incident Response**: Automated runbook execution for common incidents
- **Approval Workflows**: Multi-level approval for critical operations
- **Step-by-step Execution**: Detailed execution tracking and rollback capabilities
- **Template Library**: Pre-built runbooks for common scenarios

### 📊 SLO/SLI Monitoring
- **Service Level Objectives**: Comprehensive SLO monitoring and tracking
- **Error Budget Management**: Real-time error budget tracking and burn rate analysis
- **Performance Metrics**: Detailed performance analysis and trend monitoring
- **Compliance Reporting**: Automated compliance and SLA reporting

### 💰 Cost Monitoring
- **Multi-cloud Support**: AWS, GCP, Azure cost monitoring
- **Resource Optimization**: Automated cost optimization recommendations
- **Budget Alerts**: Proactive budget monitoring and alerting
- **Cost Analytics**: Detailed cost breakdown and trend analysis

### 🧪 Chaos Engineering
- **Reliability Testing**: Automated chaos experiments for system resilience
- **Failure Simulation**: Network, infrastructure, and service failure testing
- **Recovery Testing**: Automated recovery and failover testing
- **Reliability Metrics**: Comprehensive reliability scoring and reporting

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Observability Service                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Metrics   │  │   Tracing   │  │   Logging   │        │
│  │  Collector  │  │   Manager   │  │ Aggregator  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Alerting   │  │  Runbooks   │  │    SLO      │        │
│  │   System    │  │   Engine    │  │  Monitor    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Incident   │  │    Cost     │  │   Chaos     │        │
│  │  Manager    │  │  Monitor    │  │  Engine     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Elasticsearch 8+

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd observability-service
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start the services**
```bash
docker-compose up -d
```

4. **Access the services**
- Observability Service: http://localhost:8000
- Grafana Dashboard: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Jaeger UI: http://localhost:16686
- Kibana: http://localhost:5601

### Development Setup

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the service**
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

### Health Endpoints
- `GET /health` - Basic health check
- `GET /health/detailed` - Comprehensive system health
- `GET /health/services` - All services health status

### Metrics Endpoints
- `GET /metrics` - All metrics
- `GET /metrics/prometheus` - Prometheus format metrics
- `GET /metrics/summary` - Metrics summary
- `GET /metrics/business` - Business metrics

### Alerting Endpoints
- `POST /alerts` - Create alert
- `GET /alerts` - List alerts
- `PUT /alerts/{id}/acknowledge` - Acknowledge alert
- `POST /alerts/rules` - Create alert rule

### Runbooks Endpoints
- `POST /runbooks` - Create runbook
- `GET /runbooks` - List runbooks
- `POST /runbooks/{id}/execute` - Execute runbook
- `GET /runbooks/templates` - Get runbook templates

### Incident Management
- `POST /incidents` - Create incident
- `GET /incidents` - List incidents
- `PUT /incidents/{id}` - Update incident
- `POST /incidents/{id}/post-mortem` - Create post-mortem

### SLO Monitoring
- `GET /slo/status` - SLO status
- `GET /slo/error-budgets` - Error budget information
- `POST /slo/definitions` - Create SLO definition

### Cost Monitoring
- `GET /cost/summary` - Cost summary
- `GET /cost/analysis` - Cost analysis
- `GET /cost/optimization` - Optimization recommendations

### Chaos Engineering
- `POST /chaos/experiments` - Create experiment
- `POST /chaos/experiments/{id}/execute` - Execute experiment
- `GET /chaos/reliability` - Reliability summary

## Configuration

### Environment Variables

```bash
# Service Configuration
SERVICE_NAME=observability-service
DEBUG=false
PORT=8000

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/observability
REDIS_URL=redis://localhost:6379/0

# Monitoring
PROMETHEUS_PORT=9090
JAEGER_HOST=localhost
JAEGER_PORT=14268
ELASTICSEARCH_URL=http://localhost:9200

# Security
SECRET_KEY=your-secret-key
JWT_SECRET=your-jwt-secret
JWT_ALGORITHM=HS256

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### Alert Rules Configuration

Create `config/alert_rules.json`:

```json
[
  {
    "id": "high_error_rate",
    "name": "High Error Rate",
    "description": "Alert when error rate exceeds 5%",
    "condition": "error_rate > 0.05",
    "severity": "high",
    "threshold": 0.05,
    "evaluation_window": 300,
    "notification_channels": ["slack", "email"]
  }
]
```

### SLO Definitions

Create `config/slo_definitions.json`:

```json
[
  {
    "id": "availability_slo",
    "name": "Service Availability",
    "description": "99.9% availability SLO",
    "target_percentage": 99.9,
    "evaluation_window": 3600,
    "sli_definitions": [
      {
        "id": "availability_sli",
        "name": "Availability SLI",
        "type": "availability",
        "query": "up{job=\"observability-service\"}",
        "target_percentage": 99.9
      }
    ]
  }
]
```

## Monitoring Stack

### Prometheus
- Metrics collection and storage
- Alert rule evaluation
- Service discovery

### Grafana
- Visualization dashboards
- Alert management
- Custom panels and queries

### Jaeger
- Distributed tracing
- Request flow analysis
- Performance bottleneck identification

### Elasticsearch + Kibana
- Centralized logging
- Log analysis and search
- Real-time log monitoring

## Best Practices

### Alerting
- Use appropriate severity levels
- Implement alert suppression rules
- Set up escalation policies
- Regular alert rule review

### Runbooks
- Test runbooks in staging
- Keep runbooks up to date
- Document manual steps clearly
- Implement approval workflows

### SLO Management
- Set realistic SLO targets
- Monitor error budgets closely
- Regular SLO review and adjustment
- Document SLO rationale

### Cost Optimization
- Regular cost analysis
- Implement budget alerts
- Use cost optimization recommendations
- Monitor cost trends

## Troubleshooting

### Common Issues

1. **Service not starting**
   - Check environment variables
   - Verify database connectivity
   - Check port availability

2. **Metrics not appearing**
   - Verify Prometheus configuration
   - Check service endpoints
   - Review metric collection logs

3. **Alerts not firing**
   - Check alert rule syntax
   - Verify notification channels
   - Review alert manager logs

4. **Runbooks failing**
   - Check execution permissions
   - Verify target service connectivity
   - Review runbook step logs

### Logs

Service logs are available at:
- Application logs: `logs/observability.log`
- Docker logs: `docker-compose logs observability-service`
- System logs: Check system journal

### Health Checks

- Service health: `GET /health`
- Detailed health: `GET /health/detailed`
- Metrics health: `GET /metrics/summary`

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
- Review the troubleshooting guide
