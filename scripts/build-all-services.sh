#!/bin/bash
# scripts/build-all-services.sh

set -e

echo "🏗️ Building all services for production..."

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

REGISTRY=${REGISTRY:-"gcr.io"}
PROJECT_ID=${PROJECT_ID:-"your-project-id"}
TAG=${TAG:-"latest"}

for service in "${SERVICES[@]}"; do
  echo "🔨 Building $service..."
  
  # Build the Docker image
  IMAGE_TAG="${REGISTRY}/${PROJECT_ID}/${service}:${TAG}"
  
  docker build -t "$IMAGE_TAG" "./$service"
  
  # Push to registry if not local
  if [[ "$REGISTRY" != "local" ]]; then
    echo "📤 Pushing $service to registry..."
    docker push "$IMAGE_TAG"
  fi
  
  echo "✅ $service built successfully: $IMAGE_TAG"
done

echo "🎉 All services built successfully!"

