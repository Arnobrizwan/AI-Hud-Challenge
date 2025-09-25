#!/bin/bash
# scripts/wait-for-services.sh

services=(
  "http://localhost:8001/health"  # foundations-guards
  "http://localhost:8002/health"  # ingestion-normalization
  "http://localhost:8003/health"  # content-extraction
  "http://localhost:8004/health"  # enrichment
  "http://localhost:8005/health"  # deduplication
  "http://localhost:8006/health"  # ranking
  "http://localhost:8007/health"  # summarization
  "http://localhost:8008/health"  # personalization
  "http://localhost:8009/health"  # notification-decisioning
  "http://localhost:8010/health"  # feedback-human-loop
  "http://localhost:8011/health"  # evaluation-suite
  "http://localhost:8012/health"  # mlops-orchestration
  "http://localhost:8013/health"  # drift-abuse-safety
  "http://localhost:8014/health"  # storage-indexing
  "http://localhost:8015/health"  # realtime-interfaces
  "http://localhost:8016/health"  # observability-runbooks
  "http://localhost:8000/health"  # api-gateway
)

echo "⏳ Waiting for all services to be healthy..."

for service in "${services[@]}"; do
  echo "Checking $service..."
  while ! curl -f -s "$service" > /dev/null; do
    echo "  ⏳ Waiting for $service..."
    sleep 5
  done
  echo "  ✅ $service is healthy"
done

echo "🎉 All services are healthy and ready!"

