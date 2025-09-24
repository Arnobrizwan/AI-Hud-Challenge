#!/bin/bash

# Content Enrichment Service Deployment Script
# This script handles deployment to various environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
NAMESPACE="content-enrichment"
IMAGE_TAG="latest"
REGISTRY=""
DRY_RUN=false
SKIP_TESTS=false
SKIP_BUILD=false

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

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --environment ENV    Deployment environment (development, staging, production)"
    echo "  -n, --namespace NS       Kubernetes namespace (default: content-enrichment)"
    echo "  -t, --tag TAG           Docker image tag (default: latest)"
    echo "  -r, --registry REG      Docker registry URL"
    echo "  -d, --dry-run           Show what would be deployed without actually deploying"
    echo "  --skip-tests            Skip running tests before deployment"
    echo "  --skip-build            Skip building Docker image"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --environment development"
    echo "  $0 --environment production --tag v1.0.0 --registry gcr.io/my-project"
    echo "  $0 --dry-run --environment staging"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT. Must be one of: development, staging, production"
    exit 1
fi

print_status "Starting deployment for environment: $ENVIRONMENT"
print_status "Namespace: $NAMESPACE"
print_status "Image tag: $IMAGE_TAG"

# Set image name
if [[ -n "$REGISTRY" ]]; then
    IMAGE_NAME="${REGISTRY}/content-enrichment-service:${IMAGE_TAG}"
else
    IMAGE_NAME="content-enrichment-service:${IMAGE_TAG}"
fi

print_status "Image name: $IMAGE_NAME"

# Function to run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        print_warning "Skipping tests"
        return 0
    fi
    
    print_status "Running tests..."
    
    if ! python -m pytest src/tests/ -v --cov=src --cov-report=term-missing; then
        print_error "Tests failed. Deployment aborted."
        exit 1
    fi
    
    print_success "All tests passed"
}

# Function to build Docker image
build_image() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        print_warning "Skipping Docker build"
        return 0
    fi
    
    print_status "Building Docker image..."
    
    if ! docker build -t "$IMAGE_NAME" .; then
        print_error "Docker build failed"
        exit 1
    fi
    
    print_success "Docker image built successfully"
    
    # Push to registry if specified
    if [[ -n "$REGISTRY" ]]; then
        print_status "Pushing image to registry..."
        if ! docker push "$IMAGE_NAME"; then
            print_error "Failed to push image to registry"
            exit 1
        fi
        print_success "Image pushed to registry"
    fi
}

# Function to deploy to Kubernetes
deploy_k8s() {
    print_status "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        print_status "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
    
    # Apply Kubernetes manifests
    K8S_FILES=(
        "k8s/namespace.yaml"
        "k8s/configmap.yaml"
        "k8s/secret.yaml"
        "k8s/serviceaccount.yaml"
        "k8s/deployment.yaml"
        "k8s/ingress.yaml"
        "k8s/hpa.yaml"
    )
    
    for file in "${K8S_FILES[@]}"; do
        if [[ -f "$file" ]]; then
            print_status "Applying $file..."
            if [[ "$DRY_RUN" == "true" ]]; then
                kubectl apply --dry-run=client -f "$file" -n "$NAMESPACE"
            else
                kubectl apply -f "$file" -n "$NAMESPACE"
            fi
        else
            print_warning "File not found: $file"
        fi
    done
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_success "Dry run completed. No changes were made."
        return 0
    fi
    
    # Wait for deployment to be ready
    print_status "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/content-enrichment-service -n "$NAMESPACE"
    
    print_success "Deployment completed successfully"
}

# Function to deploy with Docker Compose
deploy_compose() {
    print_status "Deploying with Docker Compose..."
    
    # Set environment variables
    export ENVIRONMENT="$ENVIRONMENT"
    export IMAGE_TAG="$IMAGE_TAG"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "Dry run - showing what would be deployed"
        docker-compose config
    else
        # Stop existing containers
        docker-compose down
        
        # Start services
        docker-compose up -d
        
        # Wait for services to be healthy
        print_status "Waiting for services to be healthy..."
        sleep 30
        
        # Check service health
        if curl -f http://localhost:8001/health >/dev/null 2>&1; then
            print_success "Service is healthy"
        else
            print_error "Service health check failed"
            exit 1
        fi
    fi
}

# Function to run health checks
run_health_checks() {
    if [[ "$DRY_RUN" == "true" ]]; then
        print_warning "Skipping health checks in dry run mode"
        return 0
    fi
    
    print_status "Running health checks..."
    
    # Determine service URL based on environment
    case "$ENVIRONMENT" in
        development)
            SERVICE_URL="http://localhost:8001"
            ;;
        staging)
            SERVICE_URL="https://staging-content-enrichment.yourdomain.com"
            ;;
        production)
            SERVICE_URL="https://content-enrichment.yourdomain.com"
            ;;
    esac
    
    # Health check
    if curl -f "${SERVICE_URL}/health" >/dev/null 2>&1; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
        exit 1
    fi
    
    # Metrics check
    if curl -f "${SERVICE_URL}/metrics" >/dev/null 2>&1; then
        print_success "Metrics endpoint accessible"
    else
        print_warning "Metrics endpoint not accessible"
    fi
}

# Function to show deployment status
show_status() {
    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    print_status "Deployment status:"
    
    case "$ENVIRONMENT" in
        development)
            echo "  Service URL: http://localhost:8001"
            echo "  Health: http://localhost:8001/health"
            echo "  Metrics: http://localhost:8001/metrics"
            echo "  API Docs: http://localhost:8001/docs"
            ;;
        staging|production)
            echo "  Namespace: $NAMESPACE"
            kubectl get pods -n "$NAMESPACE" -l app=content-enrichment-service
            kubectl get services -n "$NAMESPACE"
            kubectl get ingress -n "$NAMESPACE"
            ;;
    esac
}

# Main deployment logic
main() {
    print_status "Content Enrichment Service Deployment"
    print_status "====================================="
    
    # Run tests
    run_tests
    
    # Build image
    build_image
    
    # Deploy based on environment
    case "$ENVIRONMENT" in
        development)
            deploy_compose
            ;;
        staging|production)
            deploy_k8s
            ;;
    esac
    
    # Run health checks
    run_health_checks
    
    # Show status
    show_status
    
    print_success "Deployment completed successfully!"
}

# Run main function
main
