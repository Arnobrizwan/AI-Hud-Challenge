#!/bin/bash

# Deploy All 16 AI Hub Challenge Services to Google Cloud Run
# This script builds and deploys all microservices

set -e

PROJECT_ID="news-hub-prod-2024"
REGION="us-central1"
REGISTRY="us-central1-docker.pkg.dev"

echo "🚀 Starting deployment of all 16 AI Hub Challenge services..."

# List of all services
SERVICES=(
    "safety-service"
    "ingestion-service"
    "content-extraction-service"
    "content-enrichment-service"
    "deduplication-service"
    "evaluation-service"
    "feedback-service"
    "mlops-orchestration-service"
    "notification-service"
    "observability-service"
    "personalization-service"
    "realtime-interface-service"
    "storage-service"
    "summarization-service"
)

# Function to build and push a service
deploy_service() {
    local service=$1
    echo "📦 Building and deploying $service..."
    
    # Build Docker image
    docker build --platform linux/amd64 -t $REGISTRY/$PROJECT_ID/$service/$service:latest ./$service
    
    # Push to registry
    docker push $REGISTRY/$PROJECT_ID/$service/$service:latest
    
    # Deploy to Cloud Run
    gcloud run deploy $service \
        --image $REGISTRY/$PROJECT_ID/$service/$service:latest \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --port 8000 \
        --memory 2Gi \
        --cpu 1 \
        --timeout 300 \
        --max-instances 5 \
        --set-env-vars="PYTHONPATH=/app"
    
    echo "✅ $service deployed successfully!"
}

# Deploy each service
for service in "${SERVICES[@]}"; do
    if [ -d "./$service" ]; then
        deploy_service $service
    else
        echo "⚠️  Service $service not found, skipping..."
    fi
done

echo "🎉 All services deployed successfully!"
echo ""
echo "📋 Service URLs:"
echo "=================="

# List all deployed services
gcloud run services list --region=$REGION --format="table(metadata.name,status.url)" --project=$PROJECT_ID
