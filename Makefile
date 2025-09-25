# Makefile for News Aggregation Pipeline
.PHONY: dev-up dev-down dev-restart dev-logs dev-test dev-clean build deploy build-fast run clean

# Environment variables
export COMPOSE_PROJECT_NAME=news-hub
export DOCKER_BUILDKIT=1

# Build variables for fast local builds
IMAGE_NAME ?= $(shell basename $$PWD | tr '[:upper:]' '[:lower:]')
PLATFORM   ?= linux/arm64
CACHE_DIR  ?= .docker-cache

# 🚀 SINGLE COMMAND TO RUN ALL 16 SERVICES LOCALLY
dev-up: 
	@echo "🚀 Starting News Aggregation Pipeline (16 Services)..."
	@docker-compose -f docker-compose.dev.yml up --build -d
	@echo "⏳ Waiting for services to be ready..."
	@./scripts/wait-for-services.sh
	@echo "✅ All services are running!"
	@echo "🌐 Access points:"
	@echo "   • API Gateway: http://localhost:8000"
	@echo "   • Admin UI: http://localhost:3000" 
	@echo "   • Grafana: http://localhost:3001"
	@echo "   • MLflow: http://localhost:5000"
	@echo "📊 Run 'make dev-logs' to see all service logs"

# 🚀 FAST DEVELOPMENT - CORE SERVICES ONLY (3-5 minutes)
dev-up-fast:
	@echo "🚀 Starting Core Services (Fast Development)..."
	@docker-compose -f docker-compose.dev-simple.yml up --build -d
	@echo "⏳ Waiting for core services to be ready..."
	@./scripts/wait-for-services-simple.sh
	@echo "✅ Core services are running!"
	@echo "🌐 Access points:"
	@echo "   • API Gateway: http://localhost:8000"
	@echo "   • Foundations: http://localhost:8001"
	@echo "   • Ingestion: http://localhost:8002"
	@echo "   • Extraction: http://localhost:8003"
	@echo "📊 Run 'make dev-logs-fast' to see service logs"

dev-down:
	@echo "🛑 Stopping all services..."
	@docker-compose -f docker-compose.dev.yml down -v
	@echo "✅ All services stopped and volumes removed"

dev-down-fast:
	@echo "🛑 Stopping core services..."
	@docker-compose -f docker-compose.dev-simple.yml down -v
	@echo "✅ Core services stopped and volumes removed"

dev-restart: dev-down dev-up
dev-restart-fast: dev-down-fast dev-up-fast

dev-logs:
	@docker-compose -f docker-compose.dev.yml logs -f

dev-logs-fast:
	@docker-compose -f docker-compose.dev-simple.yml logs -f

dev-test:
	@echo "🧪 Running comprehensive tests..."
	@docker-compose -f docker-compose.test.yml up --abort-on-container-exit
	@docker-compose -f docker-compose.test.yml down

dev-clean:
	@echo "🧹 Cleaning up Docker resources..."
	@docker system prune -f
	@docker volume prune -f

# Fast local builds (Apple Silicon optimized)
build:
	@echo "🏗️ Building optimized Docker image with BuildKit caching..."
	@DOCKER_BUILDKIT=1 docker build \
		--platform $(PLATFORM) \
		--progress=plain \
		-t $(IMAGE_NAME):dev \
		.

run:
	@echo "🚀 Running optimized container..."
	@docker run --rm -it -p 8080:8080 $(IMAGE_NAME):dev

clean:
	@echo "🧹 Cleaning Docker cache and images..."
	@rm -rf $(CACHE_DIR)
	@docker image prune -f

# Production build
build-prod:
	@echo "🏗️ Building production images..."
	@./scripts/build-all-services.sh

# Deploy to GCP
deploy:
	@echo "🚀 Deploying to production..."
	@./scripts/deploy-to-gcp.sh