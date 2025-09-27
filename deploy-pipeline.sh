#!/bin/bash

# AI Pipeline Deployment Script for Railway
# This script deploys the complete AI/ML pipeline system to Railway

set -e

echo "🚀 Deploying AI/ML Pipeline System to Railway..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    print_error "Railway CLI is not installed. Please install it first:"
    echo "npm install -g @railway/cli"
    exit 1
fi

# Check if user is logged in to Railway
if ! railway whoami &> /dev/null; then
    print_error "Not logged in to Railway. Please run 'railway login' first."
    exit 1
fi

print_status "Starting deployment process..."

# 1. Deploy AI Pipeline Service
print_status "Deploying AI Pipeline Service..."
cd ai-pipeline-service

# Create railway.json if it doesn't exist
if [ ! -f railway.json ]; then
    cat > railway.json << EOF
{
  "\$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "python main.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
EOF
fi

# Deploy the service
railway up --detach
PIPELINE_SERVICE_URL=$(railway domain)
print_success "AI Pipeline Service deployed at: $PIPELINE_SERVICE_URL"

cd ..

# 2. Deploy Dashboard Service
print_status "Deploying Dashboard Service..."
cd dashboard-service

# Create railway.json
cat > railway.json << EOF
{
  "\$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "python src/main.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
EOF

# Set environment variables
railway variables set PIPELINE_SERVICE_URL=$PIPELINE_SERVICE_URL
railway variables set NEWS_SERVICE_URL=http://localhost:8000

# Deploy the service
railway up --detach
DASHBOARD_SERVICE_URL=$(railway domain)
print_success "Dashboard Service deployed at: $DASHBOARD_SERVICE_URL"

cd ..

# 3. Deploy Dashboard Frontend
print_status "Deploying Dashboard Frontend..."
cd dashboard-frontend

# Create railway.json
cat > railway.json << EOF
{
  "\$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "nginx -g 'daemon off;'",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
EOF

# Set environment variables
railway variables set REACT_APP_API_URL=$DASHBOARD_SERVICE_URL

# Deploy the service
railway up --detach
DASHBOARD_FRONTEND_URL=$(railway domain)
print_success "Dashboard Frontend deployed at: $DASHBOARD_FRONTEND_URL"

cd ..

# 4. Create main deployment configuration
print_status "Creating main deployment configuration..."

cat > railway.json << EOF
{
  "\$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "python app.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
EOF

# 5. Deploy main application
print_status "Deploying main AI News Hub application..."
railway up --detach
MAIN_APP_URL=$(railway domain)
print_success "Main AI News Hub deployed at: $MAIN_APP_URL"

# 6. Summary
print_success "🎉 Deployment Complete!"
echo ""
echo "📊 Service URLs:"
echo "  • Main AI News Hub: $MAIN_APP_URL"
echo "  • AI Pipeline Service: $PIPELINE_SERVICE_URL"
echo "  • Dashboard Service: $DASHBOARD_SERVICE_URL"
echo "  • Dashboard Frontend: $DASHBOARD_FRONTEND_URL"
echo ""
echo "🔧 Next Steps:"
echo "  1. Access the dashboard at: $DASHBOARD_FRONTEND_URL"
echo "  2. Create your first pipeline"
echo "  3. Configure monitoring and alerts"
echo "  4. Set up your ML workflows"
echo ""
echo "📚 Documentation:"
echo "  • Read AI_PIPELINE_README.md for detailed usage"
echo "  • Check service health at /health endpoints"
echo "  • Monitor logs with 'railway logs'"
echo ""

print_status "Deployment completed successfully! 🚀"
