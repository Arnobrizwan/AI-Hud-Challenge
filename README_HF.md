---
title: AI News Hub
emoji: 📰
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# AI News Hub

An intelligent news aggregation and personalization platform.

## Features

- News Ingestion
- Content Extraction & Enrichment
- ML-powered Ranking
- Personalization
- Summarization
- Safety Checks

## API Endpoints

- `GET /health` - Health check
- `POST /news/ingest?url=<url>` - Ingest news
- `POST /news/rank?user_id=<id>&limit=<n>` - Rank news
- `POST /news/summarize?article_id=<id>` - Summarize article
- `GET /news/personalize?user_id=<id>&limit=<n>` - Get personalized news

## Technology Stack

- FastAPI, Python 3.11
- PyTorch, Transformers
- PostgreSQL, Redis
- Docker

## License

MIT
