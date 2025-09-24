# Deployment Guide

This guide covers deploying the Ranking Microservice in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Production Deployment](#production-deployment)
6. [Monitoring Setup](#monitoring-setup)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU**: 2+ cores (4+ recommended for production)
- **Memory**: 4GB+ RAM (8GB+ recommended for production)
- **Storage**: 10GB+ available space
- **Network**: Stable internet connection

### Software Requirements

- **Python**: 3.11+
- **Redis**: 6.0+
- **Docker**: 20.10+ (for containerized deployment)
- **Kubernetes**: 1.20+ (for K8s deployment)
- **kubectl**: Latest version (for K8s deployment)

## Local Development

### 1. Clone Repository

```bash
git clone <repository-url>
cd AI-Hud-Challenge
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Redis

```bash
# Using Docker
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Or install Redis locally
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis-server

# macOS
brew install redis
brew services start redis

# Windows
# Download from https://redis.io/download
```

### 5. Configure Environment

```bash
cp config.env.example .env
# Edit .env with your configuration
```

### 6. Run Application

```bash
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

## Docker Deployment

### 1. Build Image

```bash
docker build -t ranking-service:latest .
```

### 2. Run Container

```bash
# Run with Docker Compose (recommended)
docker-compose up -d

# Or run manually
docker run -d \
  --name ranking-service \
  -p 8000:8000 \
  -p 8001:8001 \
  -e REDIS_URL=redis://host.docker.internal:6379 \
  ranking-service:latest
```

### 3. Verify Deployment

```bash
# Check container status
docker ps

# Check logs
docker logs ranking-service

# Health check
curl http://localhost:8000/health
```

### 4. Scale Service

```bash
# Scale to 3 instances
docker-compose up -d --scale ranking-service=3
```

## Kubernetes Deployment

### 1. Create Namespace

```bash
kubectl create namespace ranking-service
```

### 2. Deploy Redis

```bash
kubectl apply -f kubernetes/redis.yaml -n ranking-service
```

### 3. Deploy Application

```bash
kubectl apply -f kubernetes/deployment.yaml -n ranking-service
```

### 4. Verify Deployment

```bash
# Check pods
kubectl get pods -n ranking-service

# Check services
kubectl get services -n ranking-service

# Check logs
kubectl logs -f deployment/ranking-service -n ranking-service
```

### 5. Access Service

```bash
# Port forward for local access
kubectl port-forward service/ranking-service 8000:80 -n ranking-service

# Or get external IP
kubectl get service ranking-service -n ranking-service
```

## Production Deployment

### 1. Environment Setup

Create production environment file:

```bash
# .env.production
HOST=0.0.0.0
PORT=8000
WORKERS=4
LOG_LEVEL=info
REDIS_URL=redis://redis-cluster:6379
PROMETHEUS_PORT=8001
```

### 2. Redis Configuration

For production, use Redis Cluster or Redis Sentinel:

```yaml
# redis-cluster.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
data:
  redis.conf: |
    maxmemory 2gb
    maxmemory-policy allkeys-lru
    save 900 1
    save 300 10
    save 60 10000
```

### 3. Application Configuration

```yaml
# production-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ranking-service
spec:
  replicas: 5
  selector:
    matchLabels:
      app: ranking-service
  template:
    metadata:
      labels:
        app: ranking-service
    spec:
      containers:
      - name: ranking-service
        image: ranking-service:latest
        ports:
        - containerPort: 8000
        - containerPort: 8001
        env:
        - name: REDIS_URL
          value: "redis://redis-cluster:6379"
        - name: WORKERS
          value: "4"
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 4. Load Balancer

```yaml
# load-balancer.yaml
apiVersion: v1
kind: Service
metadata:
  name: ranking-service-lb
spec:
  type: LoadBalancer
  selector:
    app: ranking-service
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 8001
    targetPort: 8001
```

### 5. Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ranking-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ranking-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Monitoring Setup

### 1. Prometheus

```yaml
# prometheus.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'ranking-service'
      static_configs:
      - targets: ['ranking-service:8001']
      scrape_interval: 5s
```

### 2. Grafana

```yaml
# grafana.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "admin"
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
      volumes:
      - name: grafana-storage
        persistentVolumeClaim:
          claimName: grafana-pvc
```

### 3. Alerting Rules

```yaml
# alerts.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: alert-rules
data:
  ranking-alerts.yml: |
    groups:
    - name: ranking-service
      rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(ranking_response_time_seconds_bucket[5m])) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
      - alert: HighErrorRate
        expr: rate(ranking_errors_total[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
```

## Troubleshooting

### Common Issues

#### 1. Service Not Starting

**Problem**: Service fails to start

**Solutions**:
```bash
# Check logs
docker logs ranking-service
kubectl logs deployment/ranking-service

# Check Redis connection
redis-cli ping

# Check port availability
netstat -tulpn | grep :8000
```

#### 2. Redis Connection Issues

**Problem**: Cannot connect to Redis

**Solutions**:
```bash
# Check Redis status
redis-cli ping

# Check Redis configuration
redis-cli config get "*"

# Test connection from application
python -c "import redis; r = redis.Redis(host='localhost', port=6379); print(r.ping())"
```

#### 3. High Memory Usage

**Problem**: High memory consumption

**Solutions**:
```bash
# Check memory usage
docker stats ranking-service
kubectl top pods

# Optimize Redis memory
redis-cli config set maxmemory 1gb
redis-cli config set maxmemory-policy allkeys-lru

# Scale horizontally
kubectl scale deployment ranking-service --replicas=3
```

#### 4. Slow Response Times

**Problem**: High response times

**Solutions**:
```bash
# Check CPU usage
top
htop

# Check Redis performance
redis-cli --latency

# Optimize worker count
export WORKERS=4
```

#### 5. Health Check Failures

**Problem**: Health checks failing

**Solutions**:
```bash
# Check health endpoint
curl http://localhost:8000/health

# Check service logs
docker logs ranking-service

# Check resource limits
kubectl describe pod <pod-name>
```

### Performance Tuning

#### 1. Redis Optimization

```bash
# Increase memory limit
redis-cli config set maxmemory 2gb

# Optimize persistence
redis-cli config set save "900 1 300 10 60 10000"

# Enable compression
redis-cli config set rdbcompression yes
```

#### 2. Application Optimization

```bash
# Increase worker count
export WORKERS=4

# Enable GZIP compression
export GZIP_ENABLED=true

# Optimize feature cache
export FEATURE_CACHE_TTL=3600
```

#### 3. Kubernetes Optimization

```yaml
# Resource limits
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"

# Node affinity
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: node-type
          operator: In
          values:
          - compute-optimized
```

### Monitoring Commands

```bash
# Check service status
curl http://localhost:8000/health

# Get performance metrics
curl http://localhost:8000/metrics/performance

# Get cache statistics
curl http://localhost:8000/cache/stats

# Check Prometheus metrics
curl http://localhost:8001/metrics

# View logs
docker logs -f ranking-service
kubectl logs -f deployment/ranking-service
```

### Backup and Recovery

#### 1. Redis Backup

```bash
# Create backup
redis-cli BGSAVE

# Copy backup file
cp /var/lib/redis/dump.rdb /backup/redis-$(date +%Y%m%d).rdb
```

#### 2. Application Backup

```bash
# Backup configuration
kubectl get configmap ranking-config -o yaml > ranking-config-backup.yaml

# Backup secrets
kubectl get secret ranking-secrets -o yaml > ranking-secrets-backup.yaml
```

#### 3. Recovery

```bash
# Restore Redis
redis-cli FLUSHALL
redis-cli --pipe < /backup/redis-20240101.rdb

# Restore application
kubectl apply -f ranking-config-backup.yaml
kubectl apply -f ranking-secrets-backup.yaml
```

## Security Considerations

### 1. Network Security

- Use TLS/SSL for all communications
- Implement network policies in Kubernetes
- Use private networks for internal communication

### 2. Authentication

- Implement API key authentication
- Use JWT tokens for user sessions
- Implement rate limiting

### 3. Data Protection

- Encrypt sensitive data at rest
- Use secure communication protocols
- Implement data anonymization

### 4. Container Security

- Use non-root users in containers
- Scan images for vulnerabilities
- Keep base images updated

## Maintenance

### 1. Regular Updates

```bash
# Update application
docker pull ranking-service:latest
kubectl set image deployment/ranking-service ranking-service=ranking-service:latest

# Update dependencies
pip install -r requirements.txt --upgrade
```

### 2. Monitoring

- Set up alerts for critical metrics
- Monitor resource usage
- Track performance trends

### 3. Backup

- Regular Redis backups
- Configuration backups
- Database backups (if applicable)

### 4. Scaling

- Monitor load patterns
- Scale based on metrics
- Plan for peak loads

## Support

For deployment issues:

1. Check the troubleshooting section
2. Review logs and metrics
3. Check GitHub issues
4. Contact support team

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Redis Documentation](https://redis.io/documentation)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
