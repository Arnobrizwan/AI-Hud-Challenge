#!/bin/bash
# scripts/build-all-services.sh

set -e

echo "🏗️ Building AI News Hub for Hugging Face Spaces..."

# Build main application
echo "🔨 Building main application..."
docker build -t ai-news-hub:latest .

echo "✅ Main application built successfully: ai-news-hub:latest"

# Optional: Build individual services for local development
if [[ "$BUILD_SERVICES" == "true" ]]; then
  echo "🔨 Building individual services..."
  
  SERVICES=(
    "safety-service"
    "ingestion-service" 
    "content-extraction-service"
    "content-enrichment-service"
    "deduplication-service"
    "src"
    "summarization-service"
    "personalization-service"
    "notification-service"
    "feedback-service"
    "evaluation-service"
    "mlops-orchestration-service"
    "storage-service"
    "realtime-interface-service"
    "observability-service"
  )

  for service in "${SERVICES[@]}"; do
    echo "🔨 Building $service..."
    docker build -t "ai-news-hub-$service:latest" "./$service"
    echo "✅ $service built successfully: ai-news-hub-$service:latest"
  done
fi

echo "🎉 Build completed successfully!"
echo "🚀 Ready for Hugging Face Spaces deployment!"