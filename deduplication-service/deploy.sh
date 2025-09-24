#!/bin/bash

# Deduplication Service Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="deduplication-service"
NAMESPACE="deduplication-service"
DOCKER_IMAGE="deduplication-service:latest"
REGISTRY_URL="your-registry.com"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is installed (optional)
    if ! command -v helm &> /dev/null; then
        log_warn "helm is not installed - some features may not be available"
    fi
    
    log_info "Prerequisites check completed"
}

build_docker_image() {
    log_info "Building Docker image..."
    
    # Build the Docker image
    docker build -t ${DOCKER_IMAGE} .
    
    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
    
    # Tag for registry (if provided)
    if [ ! -z "$REGISTRY_URL" ]; then
        docker tag ${DOCKER_IMAGE} ${REGISTRY_URL}/${DOCKER_IMAGE}
        log_info "Image tagged for registry: ${REGISTRY_URL}/${DOCKER_IMAGE}"
    fi
}

push_docker_image() {
    if [ ! -z "$REGISTRY_URL" ]; then
        log_info "Pushing Docker image to registry..."
        docker push ${REGISTRY_URL}/${DOCKER_IMAGE}
        
        if [ $? -eq 0 ]; then
            log_info "Docker image pushed successfully"
        else
            log_error "Failed to push Docker image"
            exit 1
        fi
    else
        log_warn "No registry URL provided - skipping push"
    fi
}

create_namespace() {
    log_info "Creating namespace..."
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    log_info "Namespace created/verified"
}

apply_configurations() {
    log_info "Applying Kubernetes configurations..."
    
    # Apply namespace
    kubectl apply -f k8s/namespace.yaml
    
    # Apply configmap
    kubectl apply -f k8s/configmap.yaml
    
    # Apply secrets
    kubectl apply -f k8s/secret.yaml
    
    # Apply service account
    kubectl apply -f k8s/serviceaccount.yaml
    
    log_info "Configurations applied successfully"
}

deploy_service() {
    log_info "Deploying service..."
    
    # Update image in deployment if registry URL is provided
    if [ ! -z "$REGISTRY_URL" ]; then
        sed "s|image: ${DOCKER_IMAGE}|image: ${REGISTRY_URL}/${DOCKER_IMAGE}|g" k8s/deployment.yaml | kubectl apply -f -
    else
        kubectl apply -f k8s/deployment.yaml
    fi
    
    # Apply service
    kubectl apply -f k8s/service.yaml
    
    # Apply HPA
    kubectl apply -f k8s/hpa.yaml
    
    # Apply ingress
    kubectl apply -f k8s/ingress.yaml
    
    log_info "Service deployed successfully"
}

wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    kubectl wait --for=condition=available --timeout=300s deployment/${SERVICE_NAME} -n ${NAMESPACE}
    
    if [ $? -eq 0 ]; then
        log_info "Deployment is ready"
    else
        log_error "Deployment failed to become ready"
        exit 1
    fi
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pods
    kubectl get pods -n ${NAMESPACE} -l app=${SERVICE_NAME}
    
    # Check service
    kubectl get service -n ${NAMESPACE} ${SERVICE_NAME}
    
    # Check ingress
    kubectl get ingress -n ${NAMESPACE}
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    kubectl port-forward -n ${NAMESPACE} service/${SERVICE_NAME} 8000:8000 &
    PORT_FORWARD_PID=$!
    
    sleep 5
    
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "Health check passed"
    else
        log_warn "Health check failed - service may still be starting"
    fi
    
    kill $PORT_FORWARD_PID 2>/dev/null || true
    
    log_info "Deployment verification completed"
}

cleanup_old_deployments() {
    log_info "Cleaning up old deployments..."
    
    # Delete old replicasets
    kubectl delete replicaset -n ${NAMESPACE} -l app=${SERVICE_NAME} --ignore-not-found=true
    
    log_info "Cleanup completed"
}

show_status() {
    log_info "Deployment Status:"
    echo "=================="
    
    # Show pods
    echo "Pods:"
    kubectl get pods -n ${NAMESPACE} -l app=${SERVICE_NAME}
    echo ""
    
    # Show services
    echo "Services:"
    kubectl get services -n ${NAMESPACE}
    echo ""
    
    # Show ingress
    echo "Ingress:"
    kubectl get ingress -n ${NAMESPACE}
    echo ""
    
    # Show HPA
    echo "HPA:"
    kubectl get hpa -n ${NAMESPACE}
    echo ""
}

# Main deployment function
deploy() {
    log_info "Starting deployment of ${SERVICE_NAME}"
    echo "========================================"
    
    check_prerequisites
    build_docker_image
    push_docker_image
    create_namespace
    apply_configurations
    deploy_service
    wait_for_deployment
    verify_deployment
    cleanup_old_deployments
    show_status
    
    log_info "Deployment completed successfully!"
    echo ""
    echo "Service is available at:"
    echo "- Health: http://localhost:8000/health"
    echo "- API Docs: http://localhost:8000/docs"
    echo "- Metrics: http://localhost:8000/metrics"
    echo ""
    echo "To access the service:"
    echo "kubectl port-forward -n ${NAMESPACE} service/${SERVICE_NAME} 8000:8000"
}

# Rollback function
rollback() {
    log_info "Rolling back deployment..."
    
    # Get previous revision
    PREVIOUS_REVISION=$(kubectl rollout history deployment/${SERVICE_NAME} -n ${NAMESPACE} --no-headers | tail -2 | head -1 | awk '{print $1}')
    
    if [ -z "$PREVIOUS_REVISION" ]; then
        log_error "No previous revision found"
        exit 1
    fi
    
    kubectl rollout undo deployment/${SERVICE_NAME} -n ${NAMESPACE} --to-revision=${PREVIOUS_REVISION}
    
    log_info "Rollback completed"
}

# Scale function
scale() {
    local replicas=$1
    
    if [ -z "$replicas" ]; then
        log_error "Number of replicas not specified"
        echo "Usage: $0 scale <number_of_replicas>"
        exit 1
    fi
    
    log_info "Scaling deployment to ${replicas} replicas..."
    kubectl scale deployment/${SERVICE_NAME} -n ${NAMESPACE} --replicas=${replicas}
    
    wait_for_deployment
    log_info "Scaling completed"
}

# Delete function
delete() {
    log_info "Deleting deployment..."
    
    kubectl delete -f k8s/ --ignore-not-found=true
    kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
    
    log_info "Deployment deleted"
}

# Main script
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    rollback)
        rollback
        ;;
    scale)
        scale $2
        ;;
    delete)
        delete
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|scale|delete|status}"
        echo ""
        echo "Commands:"
        echo "  deploy     - Deploy the service (default)"
        echo "  rollback   - Rollback to previous version"
        echo "  scale N    - Scale to N replicas"
        echo "  delete     - Delete the deployment"
        echo "  status     - Show deployment status"
        exit 1
        ;;
esac
