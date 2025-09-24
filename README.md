# 🚀 News Hub Pipeline - Complete Orchestration

A comprehensive 16-service news aggregation pipeline with single-command local development and production deployment.

## 🎯 Quick Start

### **Local Development (Single Command)**
```bash
# Clone and start everything
git clone <your-repo>
cd news-hub-pipeline
make dev-up
```

### **Production Deployment (Single Command)**
```bash
# Deploy to production via GitHub Actions
gh workflow run deploy-production.yml --ref main
```

## 🏗️ Architecture

### **16 Microservices**
1. **Foundations & Guards** (Safety Service) - Port 8001
2. **Ingestion & Normalization** (Ingestion Service) - Port 8002
3. **Content Extraction** (Content Extraction Service) - Port 8003
4. **Enrichment** (Content Enrichment Service) - Port 8004
5. **Deduplication** (Deduplication Service) - Port 8005
6. **Ranking** (Main Service) - Port 8006
7. **Summarization** (Summarization Service) - Port 8007
8. **Personalization** (Personalization Service) - Port 8008
9. **Notification Decisioning** (Notification Service) - Port 8009
10. **Feedback & Human Loop** (Feedback Service) - Port 8010
11. **Evaluation Suite** (Evaluation Service) - Port 8011
12. **MLOps Orchestration** (MLOps Service) - Port 8012
13. **Drift & Abuse Safety** (Safety Service) - Port 8013
14. **Storage & Indexing** (Storage Service) - Port 8014
15. **Realtime Interfaces** (Realtime Service) - Port 8015
16. **Observability & Runbooks** (Observability Service) - Port 8016

### **Infrastructure Services**
- **API Gateway** - Port 8000
- **Admin UI** - Port 3000
- **Grafana** - Port 3001
- **MLflow** - Port 5000
- **Prometheus** - Port 9090
- **PostgreSQL** - Port 5432
- **Redis** - Port 6379
- **Elasticsearch** - Port 9200

## 🚀 Commands

### **Local Development**
```bash
make dev-up          # 🚀 Start all 16 services locally
make dev-logs        # 📊 View all service logs  
make dev-test        # 🧪 Run comprehensive tests
make dev-down        # 🛑 Stop all services
make dev-restart     # 🔄 Restart all services
make dev-clean       # 🧹 Clean up Docker resources
```

### **Production**
```bash
make build           # 🏗️ Build production images
make deploy          # 🌍 Deploy to GCP production
```

## 🌐 Access Points

### **Local Development**
- **API Gateway**: http://localhost:8000
- **Admin UI**: http://localhost:3000
- **Grafana**: http://localhost:3001 (admin/admin)
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090

### **Service Health Checks**
- All services expose `/health` endpoints
- Health check script: `./scripts/wait-for-services.sh`
- Smoke tests: `python scripts/smoke_tests.py --environment=local`

## 🔧 Configuration

### **Environment Variables**
```bash
# Database
DATABASE_URL=postgresql://newshub:dev_password@postgres:5432/newshub
REDIS_URL=redis://redis:6379

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# Elasticsearch
ELASTICSEARCH_URL=http://elasticsearch:9200

# Prometheus
PROMETHEUS_URL=http://prometheus:9090
```

### **Service Dependencies**
- All services depend on PostgreSQL and Redis
- ML services depend on MLflow
- Search services depend on Elasticsearch
- Monitoring services depend on Prometheus

## 🧪 Testing

### **Unit Tests**
```bash
# Run tests for specific service
cd safety-service && python -m pytest tests/ -v

# Run all tests
make dev-test
```

### **Integration Tests**
```bash
# Run integration tests with test database
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### **Smoke Tests**
```bash
# Local environment
python scripts/smoke_tests.py --environment=local

# Production environment
python scripts/smoke_tests.py --environment=production
```

## 📊 Monitoring

### **Grafana Dashboards**
- **News Hub Pipeline Dashboard**: Service health, request rates, response times, error rates
- **Infrastructure Dashboard**: Database, Redis, Elasticsearch metrics
- **ML Dashboard**: Model performance, training metrics

### **Prometheus Metrics**
- Service health and availability
- Request rates and response times
- Error rates and status codes
- Resource usage (CPU, memory)

### **Alerting**
- Service down alerts
- High error rate alerts
- Resource usage alerts
- Performance degradation alerts

## 🚀 Deployment

### **GitHub Actions Workflow**
The `.github/workflows/deploy-production.yml` workflow:
1. **Builds** all 16 services
2. **Tests** all services
3. **Deploys** infrastructure via Terraform
4. **Deploys** services to Cloud Run
5. **Runs** post-deployment checks
6. **Sends** deployment notifications

### **Manual Deployment**
```bash
# Build all services
./scripts/build-all-services.sh

# Deploy to GCP
./scripts/deploy-to-gcp.sh
```

## 🔧 Development

### **Adding New Services**
1. Create service directory
2. Add `Dockerfile.dev`
3. Update `docker-compose.dev.yml`
4. Add to GitHub Actions workflow
5. Update health check script

### **Service Template**
```python
# src/main.py
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Service Name")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "service-name"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 📁 Project Structure
```
news-hub-pipeline/
├── Makefile                    # Single-command orchestration
├── docker-compose.dev.yml     # Local development stack
├── docker-compose.test.yml    # Test environment
├── .github/workflows/         # GitHub Actions
├── scripts/                   # Utility scripts
├── monitoring/                # Monitoring configuration
├── safety-service/            # Service 1: Foundations & Guards
├── ingestion-service/         # Service 2: Ingestion & Normalization
├── content-extraction-service/ # Service 3: Content Extraction
├── content-enrichment-service/ # Service 4: Enrichment
├── deduplication-service/     # Service 5: Deduplication
├── src/                       # Service 6: Ranking (Main)
├── summarization-service/     # Service 7: Summarization
├── personalization-service/   # Service 8: Personalization
├── notification-service/      # Service 9: Notification Decisioning
├── feedback-service/          # Service 10: Feedback & Human Loop
├── evaluation-service/        # Service 11: Evaluation Suite
├── mlops-orchestration-service/ # Service 12: MLOps Orchestration
├── storage-service/           # Service 13: Storage & Indexing
├── realtime-interface-service/ # Service 14: Realtime Interfaces
└── observability-service/     # Service 15: Observability & Runbooks
```

## 🎉 Success!

You now have a complete, production-ready news aggregation pipeline with:
- ✅ **16 microservices** running locally with one command
- ✅ **Complete monitoring** with Grafana and Prometheus
- ✅ **Automated testing** and health checks
- ✅ **Production deployment** via GitHub Actions
- ✅ **Single-command orchestration** for both dev and prod

**Start developing**: `make dev-up` 🚀