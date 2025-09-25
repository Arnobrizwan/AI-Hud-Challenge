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
      wget gnupg unzip \
    && rm -rf /var/lib/apt/lists/*
# Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-sdk \
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

FROM base AS runtime
# Install runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && apt-get install -y --no-install-recommends \
      libnss3 libatk-bridge2.0-0 libdrm2 libxkbcommon0 \
      libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxss1 libasound2 \
    && rm -rf /var/lib/apt/lists/*
# Create non-root user
RUN useradd --create-home --shell /bin/bash app
ENV PATH="/opt/venv/bin:$PATH"
# copy resolved env and deps from deps stage
COPY --from=deps /opt/venv /opt/venv
# only now copy the rest of the app
COPY . /app
# Set ownership
RUN chown -R app:app /app
# Switch to non-root user
USER app
# default port
EXPOSE 8000
# healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
# CMD
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
