# Safety Service Makefile
# Common tasks for development and deployment

.PHONY: help install test run clean docker-build docker-run docker-stop

# Default target
help:
	@echo "Safety Service - Available Commands:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run tests"
	@echo "  run         - Run the service locally"
	@echo "  clean       - Clean up temporary files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run with Docker Compose"
	@echo "  docker-stop - Stop Docker Compose services"
	@echo "  format      - Format code with black and isort"
	@echo "  lint        - Lint code with flake8 and mypy"

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	python -m pytest tests/ -v

# Run the service locally
run:
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
test-installation:
	python test_installation.py

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Lint code
lint:
	flake8 src/ tests/
	mypy src/

# Clean up temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Build Docker image
docker-build:
	docker build -t safety-service .

# Run with Docker Compose
docker-run:
	docker-compose up -d

# Stop Docker Compose services
docker-stop:
	docker-compose down

# View logs
logs:
	docker-compose logs -f safety-service

# Database operations
db-init:
	docker-compose exec postgres psql -U safety_user -d safety_db -f /docker-entrypoint-initdb.d/init.sql

# Development setup
dev-setup: install
	cp env.example .env
	@echo "Please edit .env file with your configuration"

# Production setup
prod-setup: install
	@echo "Setting up for production..."
	@echo "Please configure your production environment variables"

# Health check
health:
	curl -f http://localhost:8000/health || echo "Service is not running"

# API documentation
docs:
	@echo "API Documentation available at:"
	@echo "  Swagger UI: http://localhost:8000/docs"
	@echo "  ReDoc: http://localhost:8000/redoc"

# Monitoring
monitor:
	@echo "Monitoring dashboards:"
	@echo "  Grafana: http://localhost:3000 (admin/admin)"
	@echo "  Prometheus: http://localhost:9090"
