# API Documentation

## Overview

The Ranking Microservice provides a comprehensive REST API for content ranking with personalization, A/B testing, and monitoring capabilities.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, implement API key or JWT authentication.

## Content Types

All requests and responses use `application/json` content type.

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

Error responses include a JSON object with error details:

```json
{
  "detail": "Error message",
  "status_code": 400
}
```

## Rate Limiting

Rate limiting is not currently implemented but should be added in production.

---

## Endpoints

### 1. Root Endpoint

**GET** `/`

Get service information.

**Response:**
```json
{
  "service": "Ranking Microservice",
  "version": "1.0.0",
  "status": "running",
  "description": "High-performance content ranking with ML and personalization"
}
```

### 2. Health Check

**GET** `/health`

Get service health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "checks": {
    "response_time": {
      "status": "healthy",
      "value": 45.2,
      "threshold": 100
    },
    "error_rate": {
      "status": "healthy",
      "value": 0.01,
      "threshold": 0.05
    },
    "cpu_usage": {
      "status": "healthy",
      "value": 65.0,
      "threshold": 80
    }
  }
}
```

### 3. Rank Content

**POST** `/rank`

Rank content based on request parameters.

**Request Body:**
```json
{
  "user_id": "string (required)",
  "query": "string (optional)",
  "content_types": ["article", "video", "podcast", "image"] (optional, default: ["article"]),
  "limit": "integer (optional, default: 20, min: 1, max: 100)",
  "offset": "integer (optional, default: 0, min: 0)",
  "location": {
    "lat": "number (optional)",
    "lng": "number (optional)"
  },
  "timezone": "string (optional)",
  "device_type": "string (optional)",
  "enable_personalization": "boolean (optional, default: true)",
  "personalization_weights": {
    "topic": "number (optional)",
    "source": "number (optional)",
    "cf": "number (optional)",
    "cb": "number (optional)",
    "time": "number (optional)",
    "geo": "number (optional)"
  },
  "topics": ["string"] (optional),
  "sources": ["string"] (optional),
  "date_range": {
    "start": "datetime (optional)",
    "end": "datetime (optional)"
  }
}
```

**Response:**
```json
{
  "articles": [
    {
      "article": {
        "id": "string",
        "title": "string",
        "content": "string (optional)",
        "summary": "string (optional)",
        "url": "string",
        "published_at": "datetime",
        "updated_at": "datetime (optional)",
        "content_type": "article|video|podcast|image",
        "word_count": "integer",
        "reading_time": "integer",
        "quality_score": "number (0-1)",
        "image_url": "string (optional)",
        "videos": ["string"] (optional),
        "sentiment": {
          "polarity": "number (-1 to 1)",
          "subjectivity": "number (0-1)"
        } (optional),
        "entities": [
          {
            "text": "string",
            "label": "string",
            "confidence": "number (0-1)"
          }
        ],
        "topics": [
          {
            "name": "string",
            "confidence": "number (0-1)",
            "category": "string (optional)"
          }
        ],
        "author": {
          "id": "string",
          "name": "string",
          "bio": "string (optional)",
          "authority_score": "number (optional)"
        } (optional),
        "source": {
          "id": "string",
          "name": "string",
          "domain": "string",
          "authority_score": "number (optional)",
          "reliability_score": "number (optional)",
          "popularity_score": "number (optional)"
        },
        "view_count": "integer",
        "like_count": "integer",
        "share_count": "integer",
        "comment_count": "integer",
        "language": "string (default: en)",
        "country": "string (optional)",
        "region": "string (optional)"
      },
      "rank": "integer",
      "score": "number (0-1)",
      "personalized_score": "number (0-1) (optional)",
      "explanation": "string (optional)",
      "feature_scores": {
        "relevance": "number (optional)",
        "freshness": "number (optional)",
        "authority": "number (optional)",
        "personalization": "number (optional)"
      } (optional)
    }
  ],
  "total_count": "integer",
  "algorithm_variant": "string",
  "processing_time_ms": "number",
  "features_computed": "integer",
  "cache_hit_rate": "number (0-1)"
}
```

### 4. Get Article

**GET** `/articles/{article_id}`

Get article by ID.

**Path Parameters:**
- `article_id` (string, required): Article identifier

**Response:**
```json
{
  "id": "string",
  "title": "string",
  "content": "string (optional)",
  "summary": "string (optional)",
  "url": "string",
  "published_at": "datetime",
  "updated_at": "datetime (optional)",
  "content_type": "article|video|podcast|image",
  "word_count": "integer",
  "reading_time": "integer",
  "quality_score": "number (0-1)",
  "image_url": "string (optional)",
  "videos": ["string"] (optional),
  "sentiment": {
    "polarity": "number (-1 to 1)",
    "subjectivity": "number (0-1)"
  } (optional),
  "entities": [
    {
      "text": "string",
      "label": "string",
      "confidence": "number (0-1)"
    }
  ],
  "topics": [
    {
      "name": "string",
      "confidence": "number (0-1)",
      "category": "string (optional)"
    }
  ],
  "author": {
    "id": "string",
    "name": "string",
    "bio": "string (optional)",
    "authority_score": "number (optional)"
  } (optional),
  "source": {
    "id": "string",
    "name": "string",
    "domain": "string",
    "authority_score": "number (optional)",
    "reliability_score": "number (optional)",
    "popularity_score": "number (optional)"
  },
  "view_count": "integer",
  "like_count": "integer",
  "share_count": "integer",
  "comment_count": "integer",
  "language": "string (default: en)",
  "country": "string (optional)",
  "region": "string (optional)"
}
```

### 5. Get User Profile

**GET** `/users/{user_id}/profile`

Get user profile for personalization.

**Path Parameters:**
- `user_id` (string, required): User identifier

**Response:**
```json
{
  "user_id": "string",
  "topic_preferences": {
    "string": "number (0-1)"
  },
  "source_preferences": {
    "string": "number (0-1)"
  },
  "reading_patterns": {
    "preferred_hours": ["integer"],
    "string": "any"
  },
  "content_preferences": {
    "preferred_length": "integer (optional)",
    "preferred_quality": "number (optional)",
    "string": "any"
  },
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### 6. Update User Profile

**PUT** `/users/{user_id}/profile`

Update user profile.

**Path Parameters:**
- `user_id` (string, required): User identifier

**Request Body:**
```json
{
  "user_id": "string",
  "topic_preferences": {
    "string": "number (0-1)"
  },
  "source_preferences": {
    "string": "number (0-1)"
  },
  "reading_patterns": {
    "preferred_hours": ["integer"],
    "string": "any"
  },
  "content_preferences": {
    "preferred_length": "integer (optional)",
    "preferred_quality": "number (optional)",
    "string": "any"
  },
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

**Response:**
```json
{
  "message": "Profile updated successfully"
}
```

### 7. Get Experiments

**GET** `/experiments`

Get all A/B test experiments.

**Response:**
```json
[
  {
    "experiment_id": "string",
    "name": "string",
    "is_active": "boolean",
    "start_date": "datetime",
    "end_date": "datetime (optional)",
    "variant_count": "integer",
    "stats": {
      "experiment_id": "string",
      "total_users": "integer",
      "variants": {
        "string": {
          "name": "string",
          "user_count": "integer",
          "avg_metrics": {
            "string": {
              "mean": "number",
              "count": "integer",
              "min": "number",
              "max": "number"
            }
          },
          "weight": "number (0-1)"
        }
      },
      "is_active": "boolean",
      "start_date": "datetime",
      "end_date": "datetime (optional)"
    }
  }
]
```

### 8. Get Experiment Stats

**GET** `/experiments/{experiment_id}/stats`

Get experiment statistics.

**Path Parameters:**
- `experiment_id` (string, required): Experiment identifier

**Response:**
```json
{
  "experiment_id": "string",
  "total_users": "integer",
  "variants": {
    "string": {
      "name": "string",
      "user_count": "integer",
      "avg_metrics": {
        "string": {
          "mean": "number",
          "count": "integer",
          "min": "number",
          "max": "number"
        }
      },
      "weight": "number (0-1)"
    }
  },
  "is_active": "boolean",
  "start_date": "datetime",
  "end_date": "datetime (optional)"
}
```

### 9. Create Experiment

**POST** `/experiments`

Create a new A/B test experiment.

**Request Body:**
```json
{
  "experiment_id": "string",
  "name": "string",
  "variants": [
    {
      "variant_id": "string",
      "name": "string",
      "weight": "number (0-1)",
      "config": {
        "string": "any"
      },
      "is_active": "boolean (default: true)"
    }
  ],
  "start_date": "datetime",
  "end_date": "datetime (optional)",
  "is_active": "boolean (default: true)"
}
```

**Response:**
```json
{
  "message": "Experiment created successfully"
}
```

### 10. Get Performance Metrics

**GET** `/metrics/performance`

Get performance metrics.

**Query Parameters:**
- `time_window` (integer, optional): Time window in minutes (default: 60, min: 1, max: 1440)

**Response:**
```json
{
  "total_requests": "integer",
  "avg_response_time_ms": "number",
  "p95_response_time_ms": "number",
  "p99_response_time_ms": "number",
  "avg_feature_time_ms": "number",
  "avg_ranking_time_ms": "number",
  "avg_cache_hit_rate": "number (0-1)",
  "error_rate": "number (0-1)",
  "avg_article_count": "number"
}
```

### 11. Get Algorithm Comparison

**GET** `/metrics/algorithm-comparison`

Get algorithm performance comparison.

**Response:**
```json
{
  "string": {
    "request_count": "integer",
    "avg_response_time_ms": "number",
    "p95_response_time_ms": "number",
    "avg_cache_hit_rate": "number (0-1)"
  }
}
```

### 12. Get System Metrics

**GET** `/metrics/system`

Get system metrics.

**Query Parameters:**
- `time_window` (integer, optional): Time window in minutes (default: 60, min: 1, max: 1440)

**Response:**
```json
{
  "avg_cpu_usage": "number",
  "max_cpu_usage": "number",
  "avg_memory_usage": "number",
  "max_memory_usage": "number",
  "avg_redis_connections": "number",
  "avg_active_requests": "number",
  "avg_queue_size": "number"
}
```

### 13. Get Cache Stats

**GET** `/cache/stats`

Get cache statistics.

**Response:**
```json
{
  "hit_count": "integer",
  "miss_count": "integer",
  "total_requests": "integer",
  "hit_rate": "number (0-1)",
  "redis_connected": "boolean"
}
```

### 14. Clear Cache

**POST** `/cache/clear`

Clear cache.

**Response:**
```json
{
  "message": "Cache cleared successfully"
}
```

---

## Examples

### Basic Ranking Request

```bash
curl -X POST "http://localhost:8000/rank" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "query": "artificial intelligence",
    "limit": 10,
    "enable_personalization": true
  }'
```

### Ranking with Filters

```bash
curl -X POST "http://localhost:8000/rank" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "query": "machine learning",
    "limit": 20,
    "content_types": ["article", "video"],
    "topics": ["technology", "AI"],
    "sources": ["source1", "source2"],
    "location": {
      "lat": 40.7128,
      "lng": -74.0060
    },
    "enable_personalization": true
  }'
```

### Update User Profile

```bash
curl -X PUT "http://localhost:8000/users/user123/profile" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "topic_preferences": {
      "technology": 0.9,
      "AI": 0.8,
      "science": 0.7
    },
    "source_preferences": {
      "source1": 0.8,
      "source2": 0.6
    },
    "reading_patterns": {
      "preferred_hours": [9, 10, 11, 14, 15, 16]
    },
    "content_preferences": {
      "preferred_length": 500,
      "preferred_quality": 0.8
    },
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  }'
```

### Create A/B Test Experiment

```bash
curl -X POST "http://localhost:8000/experiments" \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "ranking_algorithm_v5",
    "name": "New Ranking Algorithm",
    "variants": [
      {
        "variant_id": "ml_ranker_v2",
        "name": "ML Ranker V2",
        "weight": 0.5,
        "config": {
          "algorithm": "lightgbm",
          "model_version": "v4"
        }
      },
      {
        "variant_id": "hybrid_v2",
        "name": "Hybrid V2",
        "weight": 0.5,
        "config": {
          "algorithm": "hybrid",
          "ml_weight": 0.8,
          "heuristic_weight": 0.2
        }
      }
    ],
    "start_date": "2024-01-01T00:00:00Z",
    "is_active": true
  }'
```

### Get Performance Metrics

```bash
curl "http://localhost:8000/metrics/performance?time_window=60"
```

### Health Check

```bash
curl "http://localhost:8000/health"
```

---

## SDK Examples

### Python

```python
import requests

# Rank content
response = requests.post(
    "http://localhost:8000/rank",
    json={
        "user_id": "user123",
        "query": "artificial intelligence",
        "limit": 10,
        "enable_personalization": True
    }
)
results = response.json()

# Update user profile
profile_data = {
    "user_id": "user123",
    "topic_preferences": {"technology": 0.9, "AI": 0.8},
    "source_preferences": {"source1": 0.8},
    "reading_patterns": {"preferred_hours": [9, 10, 11]},
    "content_preferences": {"preferred_length": 500},
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
}
response = requests.put(
    "http://localhost:8000/users/user123/profile",
    json=profile_data
)
```

### JavaScript

```javascript
// Rank content
const response = await fetch('http://localhost:8000/rank', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    user_id: 'user123',
    query: 'artificial intelligence',
    limit: 10,
    enable_personalization: true
  })
});
const results = await response.json();

// Update user profile
const profileData = {
  user_id: 'user123',
  topic_preferences: { technology: 0.9, AI: 0.8 },
  source_preferences: { source1: 0.8 },
  reading_patterns: { preferred_hours: [9, 10, 11] },
  content_preferences: { preferred_length: 500 },
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z'
};
const profileResponse = await fetch('http://localhost:8000/users/user123/profile', {
  method: 'PUT',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(profileData)
});
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid request data |
| 404 | Not Found - Resource not found |
| 422 | Unprocessable Entity - Validation error |
| 500 | Internal Server Error - Server error |

## Rate Limits

Rate limiting is not currently implemented but should be added in production with appropriate limits per endpoint.

## Changelog

### Version 1.0.0
- Initial release
- Core ranking functionality
- Personalization system
- A/B testing framework
- Monitoring and metrics
- Docker deployment support
