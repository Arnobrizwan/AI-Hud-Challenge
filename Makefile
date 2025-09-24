# AI HUD Challenge - Makefile
# This Makefile provides common commands for development, testing, and deployment

.PHONY: help install test lint format security build deploy clean

# Default target
help: ## Show this help message
	@echo "AI HUD Challenge - Available Commands:"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development Setup
install: ## Install all dependencies
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@for service in */; do \
		if [ -f "$$service/requirements.txt" ]; then \
			echo "Installing dependencies for $$service"; \
			pip install -r "$$service/requirements.txt"; \
		fi; \
	done

install-dev: ## Install development dependencies
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install black isort flake8 mypy pytest pytest-cov pytest-xdist
	pip install safety bandit pip-audit

# Code Quality
lint: ## Run linting checks
	@echo "Running linting checks..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics
	mypy . --ignore-missing-imports --no-strict-optional

format: ## Format code with black and isort
	@echo "Formatting code..."
	black .
	isort .

format-check: ## Check code formatting
	@echo "Checking code formatting..."
	black --check --diff .
	isort --check-only --diff .

# Testing
test: ## Run all tests
	@echo "Running tests..."
	pytest -v --tb=short

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	pytest -v --tb=short -m "not integration and not slow"

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	pytest -v --tb=short -m integration

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	pytest --cov=src --cov-report=html --cov-report=xml --cov-report=term

test-service: ## Run tests for a specific service (usage: make test-service SERVICE=content-enrichment-service)
	@echo "Running tests for $(SERVICE)..."
	cd $(SERVICE) && pytest -v --tb=short

# Security
security: ## Run security checks
	@echo "Running security checks..."
	safety check
	bandit -r . -f json -o bandit-report.json
	pip-audit --format=json --output=pip-audit-report.json

security-deps: ## Check for vulnerable dependencies
	@echo "Checking for vulnerable dependencies..."
	safety check --json

security-code: ## Run code security analysis
	@echo "Running code security analysis..."
	bandit -r . -f json -o bandit-report.json

# Docker
build: ## Build all Docker images
	@echo "Building Docker images..."
	@for service in */; do \
		if [ -f "$$service/Dockerfile" ]; then \
			echo "Building $$service"; \
			docker build -t ai-hud-$$(basename $$service) $$service; \
		fi; \
	done

build-service: ## Build Docker image for specific service (usage: make build-service SERVICE=content-enrichment-service)
	@echo "Building Docker image for $(SERVICE)..."
	docker build -t ai-hud-$(SERVICE) $(SERVICE)

push: ## Push Docker images to registry
	@echo "Pushing Docker images..."
	@for service in */; do \
		if [ -f "$$service/Dockerfile" ]; then \
			echo "Pushing $$service"; \
			docker push ai-hud-$$(basename $$service); \
		fi; \
	done

# Docker Compose
up: ## Start all services with docker-compose
	@echo "Starting services with docker-compose..."
	docker-compose up -d

down: ## Stop all services
	@echo "Stopping services..."
	docker-compose down

logs: ## Show logs for all services
	@echo "Showing logs..."
	docker-compose logs -f

logs-service: ## Show logs for specific service (usage: make logs-service SERVICE=content-enrichment-service)
	@echo "Showing logs for $(SERVICE)..."
	docker-compose logs -f $(SERVICE)

# Database
db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	# Add migration commands here

db-seed: ## Seed database with test data
	@echo "Seeding database..."
	# Add seed commands here

db-reset: ## Reset database
	@echo "Resetting database..."
	# Add reset commands here

# Monitoring
monitor: ## Start monitoring stack (Prometheus, Grafana)
	@echo "Starting monitoring stack..."
	docker-compose -f docker-compose.monitoring.yml up -d

# Development
dev: ## Start development environment
	@echo "Starting development environment..."
	docker-compose -f docker-compose.dev.yml up -d

dev-logs: ## Show development logs
	@echo "Showing development logs..."
	docker-compose -f docker-compose.dev.yml logs -f

# Deployment
deploy-staging: ## Deploy to staging
	@echo "Deploying to staging..."
	# Add staging deployment commands here

deploy-production: ## Deploy to production
	@echo "Deploying to production..."
	# Add production deployment commands here

# Kubernetes
k8s-apply: ## Apply Kubernetes manifests
	@echo "Applying Kubernetes manifests..."
	kubectl apply -f k8s/

k8s-delete: ## Delete Kubernetes resources
	@echo "Deleting Kubernetes resources..."
	kubectl delete -f k8s/

k8s-status: ## Check Kubernetes status
	@echo "Checking Kubernetes status..."
	kubectl get pods
	kubectl get services
	kubectl get ingress

# Cleanup
clean: ## Clean up temporary files and containers
	@echo "Cleaning up..."
	docker system prune -f
	docker volume prune -f
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

clean-docker: ## Clean up Docker resources
	@echo "Cleaning up Docker resources..."
	docker system prune -af
	docker volume prune -f

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	# Add documentation generation commands here

docs-serve: ## Serve documentation locally
	@echo "Serving documentation..."
	# Add documentation serving commands here

# Performance
benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	# Add benchmark commands here

load-test: ## Run load tests
	@echo "Running load tests..."
	# Add load test commands here

# All-in-one commands
ci: format-check lint test security ## Run full CI pipeline locally
	@echo "CI pipeline completed successfully!"

dev-setup: install-dev up ## Set up development environment
	@echo "Development environment ready!"

# Service-specific commands
services: ## List all available services
	@echo "Available services:"
	@for service in */; do \
		if [ -f "$$service/requirements.txt" ]; then \
			echo "  - $$(basename $$service)"; \
		fi; \
	done

# Help for specific service
help-service: ## Show help for specific service (usage: make help-service SERVICE=content-enrichment-service)
	@echo "Help for $(SERVICE):"
	@if [ -f "$(SERVICE)/Makefile" ]; then \
		make -C $(SERVICE) help; \
	else \
		echo "No Makefile found for $(SERVICE)"; \
	fi
