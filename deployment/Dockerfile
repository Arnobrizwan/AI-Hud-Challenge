# Multi-stage build optimized for Cloud Run
# Stage 1: Build stage with all build dependencies
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_ENV=production
ARG APP_VERSION=1.0.0

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BUILD_ENV=${BUILD_ENV}

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production stage with minimal footprint
FROM python:3.11-slim as production

# Set production arguments
ARG BUILD_ENV=production
ARG APP_VERSION=1.0.0
ARG BUILD_DATE
ARG VCS_REF

# Add labels for container metadata
LABEL maintainer="your-team@company.com" \
      org.opencontainers.image.title="Foundations & Guards Service" \
      org.opencontainers.image.description="Production-ready microservice for news aggregation pipeline" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Your Company" \
      org.opencontainers.image.source="https://github.com/your-org/foundations-guards-service"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/home/appuser/.local/bin:$PATH" \
    APP_VERSION=${APP_VERSION} \
    BUILD_ENV=${BUILD_ENV} \
    PORT=8080 \
    HOST=0.0.0.0

# Install minimal system dependencies for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser

# Create app directory and set ownership
WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy Python packages from builder stage
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Switch to non-root user
USER appuser

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser requirements.txt .

# Create necessary directories
RUN mkdir -p logs tmp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health/live || exit 1

# Expose port (Cloud Run uses PORT env variable)
EXPOSE ${PORT}

# Use tini as entrypoint for proper signal handling
ENTRYPOINT ["tini", "--"]

# Default command
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]


# Stage 3: Development stage with development tools
FROM production as development

# Switch back to root to install dev dependencies
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Switch back to appuser
USER appuser

# Install development Python packages
RUN pip install --user \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-mock \
    black \
    isort \
    flake8 \
    mypy

# Override command for development
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload", "--log-level", "debug"]


# Stage 4: Testing stage for CI/CD
FROM development as testing

# Copy test files
COPY --chown=appuser:appuser src/tests/ ./src/tests/
COPY --chown=appuser:appuser pytest.ini .
COPY --chown=appuser:appuser .coveragerc .

# Set test environment
ENV ENVIRONMENT=testing \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# Run tests by default
CMD ["python", "-m", "pytest", "src/tests/", "-v", "--cov=src", "--cov-report=term-missing", "--cov-report=html:coverage_html"]
