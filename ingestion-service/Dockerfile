# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_INPUT=1
WORKDIR /app

FROM base AS deps
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# system deps for building wheels
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc g++ make curl git libffi-dev libssl-dev \
      libxml2-dev libxslt1-dev zlib1g-dev libjpeg-dev libpng-dev \
      libfreetype6-dev liblcms2-dev libwebp-dev libtiff5-dev \
      libopenjp2-7-dev libharfbuzz-dev libfribidi-dev libxcb1-dev \
    && rm -rf /var/lib/apt/lists/*
# uv (faster pip)
RUN python -m pip install -U pip && pip install -U uv
# copy only dependency manifests
ARG REQ_FILE=requirements.txt
COPY ${REQ_FILE} /app/requirements.txt
# create venv and install deps with cache
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install --upgrade pip wheel \
    && uv pip install -r /app/requirements.txt
# Install Playwright browsers
RUN playwright install chromium
RUN playwright install-deps chromium

FROM base AS runtime
# Install runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && apt-get install -y --no-install-recommends \
      libxml2 libxslt1.1 libjpeg62-turbo libpng16-16 libfreetype6 \
      liblcms2-2 libwebp6 libtiff5 libopenjp2-7 libharfbuzz0b \
      libfribidi0 libxcb1 libnss3 libatk1.0-0 libatk-bridge2.0-0 \
      libdrm2 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxss1 \
      libasound2 libatspi2.0-0 libgtk-3-0 libgdk-pixbuf2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
ENV PATH="/opt/venv/bin:$PATH"
# copy resolved env and deps from deps stage
COPY --from=deps /opt/venv /opt/venv
# only now copy the rest of the app
COPY src/ ./src/
COPY config/ ./config/
# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/tmp
# Set ownership
RUN chown -R appuser:appuser /app
# Switch to non-root user
USER appuser
# default port
EXPOSE 8000
# healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
# CMD
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
