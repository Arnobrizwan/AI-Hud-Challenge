#!/bin/bash
# scripts/deploy-to-gcp.sh

set -e

echo "🚀 Deploying News Hub Pipeline to GCP..."

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
REGION=${GCP_REGION:-"us-central1"}
REGISTRY="gcr.io"

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first."
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required GCP APIs..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and push all services
echo "🏗️ Building and pushing all services..."
./scripts/build-all-services.sh

# Deploy to Cloud Run
echo "🚀 Deploying services to Cloud Run..."

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
  echo "🚀 Deploying $service..."
  
  IMAGE_TAG="${REGISTRY}/${PROJECT_ID}/${service}:latest"
  
  gcloud run deploy $service \
    --image=$IMAGE_TAG \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --memory=1Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=100 \
    --concurrency=1000 \
    --timeout=900 \
    --set-env-vars="ENVIRONMENT=production"
    
  echo "✅ $service deployed successfully"
done

echo "🎉 All services deployed to GCP Cloud Run!"
echo "🌐 Services are available at:"
for service in "${SERVICES[@]}"; do
  SERVICE_URL=$(gcloud run services describe $service --region=$REGION --format="value(status.url)")
  echo "   • $service: $SERVICE_URL"
done
