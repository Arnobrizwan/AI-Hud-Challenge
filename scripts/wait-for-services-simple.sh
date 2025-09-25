#!/bin/bash
# scripts/wait-for-services-simple.sh - Timeout-based service health check

set -e

# Configuration
MAX_WAIT_TIME=300  # 5 minutes max
CHECK_INTERVAL=5   # Check every 5 seconds
TIMEOUT=10         # HTTP timeout

services=(
  "http://localhost:8001/health"  # foundations-guards
  "http://localhost:8002/health"  # ingestion-normalization
  "http://localhost:8003/health"  # content-extraction
  "http://localhost:8000/health"  # api-gateway
)

echo "⏳ Waiting for core services to be healthy (max ${MAX_WAIT_TIME}s)..."

start_time=$(date +%s)

for service in "${services[@]}"; do
  echo "Checking $service..."
  service_start=$(date +%s)
  
  while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    service_elapsed=$((current_time - service_start))
    
    # Check if we've exceeded max wait time
    if [ $elapsed -gt $MAX_WAIT_TIME ]; then
      echo "❌ TIMEOUT: Maximum wait time of ${MAX_WAIT_TIME}s exceeded"
      echo "🔍 Checking what's running..."
      docker-compose -f docker-compose.dev-simple.yml ps
      exit 1
    fi
    
    # Check if service is healthy
    if curl -f -s --max-time $TIMEOUT "$service" > /dev/null 2>&1; then
      echo "  ✅ $service is healthy (${service_elapsed}s)"
      break
    else
      echo "  ⏳ Waiting for $service... (${service_elapsed}s elapsed)"
      sleep $CHECK_INTERVAL
    fi
  done
done

total_time=$(($(date +%s) - start_time))
echo "🎉 All core services are healthy and ready! (${total_time}s total)"
