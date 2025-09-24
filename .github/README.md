# GitHub Actions CI/CD Pipeline

This directory contains GitHub Actions workflows for the AI HUD Challenge project.

## Workflows Overview

### 1. Main CI/CD Pipeline (`ci-cd.yml`)
The main pipeline that runs on every push and pull request:

- **Code Quality & Linting**: Black, isort, flake8, mypy
- **Unit Tests**: Tests for all microservices with Python 3.11 and 3.12
- **Integration Tests**: Full integration tests with Redis, PostgreSQL, Elasticsearch
- **Security Scanning**: Trivy, Bandit, dependency vulnerability checks
- **Docker Build**: Builds and pushes Docker images for all services
- **Deployment**: Staging and production deployments

### 2. Docker Build (`docker-build.yml`)
Specialized workflow for building and pushing Docker images:

- Builds images for all microservices
- Supports multi-platform builds (linux/amd64, linux/arm64)
- Generates Software Bill of Materials (SBOM)
- Pushes to GitHub Container Registry

### 3. Security Scan (`security.yml`)
Comprehensive security scanning:

- **Dependency Check**: Safety and pip-audit for vulnerable dependencies
- **Code Security**: Trivy filesystem scan, Bandit, Semgrep
- **Container Security**: Docker image vulnerability scanning
- **Secrets Detection**: TruffleHog for secret scanning
- **License Check**: License compliance verification

### 4. Test Suite (`test.yml`)
Dedicated testing workflow:

- **Unit Tests**: Matrix testing across Python versions and services
- **Integration Tests**: Full service integration with external dependencies
- **Performance Tests**: Load testing and benchmarking
- **Test Coverage**: Code coverage reporting and upload to Codecov

### 5. Deployment (`deploy.yml`)
Production deployment workflow:

- **Staging Deployment**: Automatic deployment on develop branch
- **Production Deployment**: Manual deployment on main branch
- **Rollback**: Automatic rollback on deployment failure
- **Health Checks**: Post-deployment verification

## Service Matrix

The following microservices are included in the CI/CD pipeline:

- `content-enrichment-service`
- `content-extraction-service`
- `deduplication-service`
- `evaluation-service`
- `feedback-service`
- `ingestion-service`
- `mlops-orchestration-service`
- `notification-service`
- `observability-service`
- `personalization-service`
- `safety-service`
- `storage-service`
- `summarization-service`

## Environment Variables

The following secrets need to be configured in GitHub:

### Required Secrets
- `GITHUB_TOKEN`: Automatically provided by GitHub
- `KUBE_CONFIG_STAGING`: Base64 encoded kubeconfig for staging
- `KUBE_CONFIG_PRODUCTION`: Base64 encoded kubeconfig for production

### Optional Secrets
- `CODECOV_TOKEN`: For code coverage reporting
- `SLACK_WEBHOOK_URL`: For deployment notifications
- `DOCKER_REGISTRY_USERNAME`: For custom Docker registry
- `DOCKER_REGISTRY_PASSWORD`: For custom Docker registry

## Workflow Triggers

### Automatic Triggers
- **Push to main/develop**: Full CI/CD pipeline
- **Pull Request**: Code quality, testing, security scanning
- **Weekly**: Security scanning (Monday 2 AM UTC)

### Manual Triggers
- **Workflow Dispatch**: Manual deployment to staging/production
- **Tag Push**: Production deployment with version tags

## Local Development

### Pre-commit Hooks
Install pre-commit hooks for local development:

```bash
pip install pre-commit
pre-commit install
```

### Running Tests Locally
```bash
# Run all tests
make test

# Run specific service tests
make test-service SERVICE=content-enrichment-service

# Run with coverage
make test-coverage
```

### Code Quality
```bash
# Format code
make format

# Run linting
make lint

# Run security checks
make security
```

## Monitoring and Observability

### GitHub Actions Metrics
- Workflow run times and success rates
- Test coverage trends
- Security scan results
- Deployment status

### External Integrations
- **Codecov**: Code coverage reporting
- **GitHub Security**: Vulnerability scanning results
- **Container Registry**: Docker image storage and scanning

## Troubleshooting

### Common Issues

1. **Test Failures**: Check service-specific requirements and dependencies
2. **Docker Build Failures**: Verify Dockerfile syntax and base images
3. **Security Scan Failures**: Review and fix security vulnerabilities
4. **Deployment Failures**: Check Kubernetes configuration and secrets

### Debugging

1. Check workflow logs in GitHub Actions
2. Review service-specific logs
3. Verify environment variables and secrets
4. Test locally with `make` commands

## Contributing

When adding new services or modifying existing ones:

1. Update the service matrix in workflows
2. Add service-specific tests
3. Update Docker configurations
4. Verify security scanning coverage
5. Test locally before pushing

## Best Practices

1. **Keep workflows fast**: Use caching and parallel execution
2. **Fail fast**: Run quick checks first (linting, formatting)
3. **Security first**: Always run security scans
4. **Test thoroughly**: Include unit, integration, and performance tests
5. **Monitor deployments**: Use health checks and rollback strategies
