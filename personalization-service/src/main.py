"""Main FastAPI application for the personalization service."""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import structlog
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi.responses import Response

from .config.settings import settings
from .database.redis_client import RedisClient
from .database.postgres_client import PostgreSQLClient
from .models.schemas import (
    PersonalizationRequest, PersonalizedResponse, UserInteraction,
    UserProfile, Recommendation, BanditRecommendation, ContentItem,
    DiversityParams, UserContext, ABExperiment, ModelMetrics
)
from .collaborative.collaborative_filter import CollaborativeFilter
from .content_based.content_based_filter import ContentBasedFilter
from .bandits.contextual_bandit import ContextualBandit
from .profiles.user_profile_manager import UserProfileManager
from .diversity.diversity_optimizer import DiversityOptimizer
from .cold_start.cold_start_handler import ColdStartHandler

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter(
    'personalization_requests_total',
    'Total number of personalization requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'personalization_request_duration_seconds',
    'Duration of personalization requests',
    ['method', 'endpoint']
)

ACTIVE_USERS = Gauge(
    'personalization_active_users',
    'Number of active users'
)

RECOMMENDATION_COUNT = Counter(
    'personalization_recommendations_total',
    'Total number of recommendations generated',
    ['algorithm']
)

# Global variables for services
redis_client: Optional[RedisClient] = None
postgres_client: Optional[PostgreSQLClient] = None
collaborative_filter: Optional[CollaborativeFilter] = None
content_based_filter: Optional[ContentBasedFilter] = None
contextual_bandit: Optional[ContextualBandit] = None
user_profile_manager: Optional[UserProfileManager] = None
diversity_optimizer: Optional[DiversityOptimizer] = None
cold_start_handler: Optional[ColdStartHandler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global redis_client, postgres_client, collaborative_filter, content_based_filter
    global contextual_bandit, user_profile_manager, diversity_optimizer, cold_start_handler
    
    # Startup
    logger.info("Starting personalization service")
    
    try:
        # Initialize database clients
        redis_client = RedisClient()
        postgres_client = PostgreSQLClient()
        
        await redis_client.connect()
        await postgres_client.connect()
        
        # Initialize services
        collaborative_filter = CollaborativeFilter(redis_client, postgres_client)
        content_based_filter = ContentBasedFilter(redis_client, postgres_client)
        contextual_bandit = ContextualBandit(redis_client, postgres_client)
        user_profile_manager = UserProfileManager(redis_client, postgres_client)
        diversity_optimizer = DiversityOptimizer()
        cold_start_handler = ColdStartHandler(redis_client, postgres_client)
        
        # Initialize models
        await collaborative_filter.initialize()
        await content_based_filter.initialize()
        await contextual_bandit.initialize()
        
        logger.info("Personalization service started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start personalization service: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down personalization service")
        
        if redis_client:
            await redis_client.disconnect()
        if postgres_client:
            await postgres_client.disconnect()
        
        logger.info("Personalization service shut down")


# Create FastAPI app
app = FastAPI(
    title="Personalization Service",
    description="Advanced personalization service with collaborative filtering, content-based filtering, and contextual bandits",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Dependency to get services
def get_services():
    """Get service instances."""
    return {
        'redis': redis_client,
        'postgres': postgres_client,
        'collaborative_filter': collaborative_filter,
        'content_based_filter': content_based_filter,
        'contextual_bandit': contextual_bandit,
        'user_profile_manager': user_profile_manager,
        'diversity_optimizer': diversity_optimizer,
        'cold_start_handler': cold_start_handler
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connections
        redis_health = await redis_client.get_health() if redis_client else {"status": "disconnected"}
        postgres_health = await postgres_client.get_health() if postgres_client else {"status": "disconnected"}
        
        # Check service status
        services_healthy = all([
            redis_health.get("status") == "connected",
            postgres_health.get("status") == "healthy"
        ])
        
        return {
            "status": "healthy" if services_healthy else "unhealthy",
            "timestamp": time.time(),
            "services": {
                "redis": redis_health,
                "postgres": postgres_health
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Main personalization endpoint
@app.post("/personalize", response_model=PersonalizedResponse)
async def personalize_content(
    request: PersonalizationRequest,
    background_tasks: BackgroundTasks,
    services: Dict[str, Any] = Depends(get_services)
):
    """Personalize content for a user."""
    start_time = time.time()
    
    try:
        # Get services
        user_profile_manager = services['user_profile_manager']
        collaborative_filter = services['collaborative_filter']
        content_based_filter = services['content_based_filter']
        contextual_bandit = services['contextual_bandit']
        diversity_optimizer = services['diversity_optimizer']
        cold_start_handler = services['cold_start_handler']
        
        # Get or create user profile
        user_profile = await user_profile_manager.get_or_create_profile(request.user_id)
        
        # Check if user is in cold start
        if user_profile.total_interactions < 5:
            # Use cold start handler
            recommendations = await cold_start_handler.handle_user_cold_start(
                request.user_id, request.context
            )
            algorithm_used = "cold_start"
        else:
            # Use hybrid personalization
            recommendations = await _hybrid_personalization(
                request, user_profile, services
            )
            algorithm_used = "hybrid"
        
        # Apply diversity optimization
        if request.diversity_params:
            recommendations = await diversity_optimizer.optimize(
                recommendations, user_profile, request.diversity_params
            )
        
        # Limit results
        recommendations = recommendations[:request.max_results]
        
        # Compute personalization strength
        personalization_strength = await _compute_personalization_strength(user_profile)
        
        # Generate explanation
        explanation = await _generate_explanation(recommendations, user_profile, algorithm_used)
        
        # Create response
        response = PersonalizedResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            algorithm_used=algorithm_used,
            personalization_strength=personalization_strength,
            explanation=explanation,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # Log recommendation
        background_tasks.add_task(
            _log_recommendation, request, response, services
        )
        
        # Update metrics
        REQUEST_COUNT.labels(method="POST", endpoint="/personalize", status="success").inc()
        REQUEST_DURATION.labels(method="POST", endpoint="/personalize").observe(time.time() - start_time)
        RECOMMENDATION_COUNT.labels(algorithm=algorithm_used).inc(len(recommendations))
        
        return response
        
    except Exception as e:
        logger.error(f"Personalization failed: {e}")
        REQUEST_COUNT.labels(method="POST", endpoint="/personalize", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


async def _hybrid_personalization(
    request: PersonalizationRequest,
    user_profile: UserProfile,
    services: Dict[str, Any]
) -> List[Recommendation]:
    """Perform hybrid personalization combining multiple algorithms."""
    collaborative_filter = services['collaborative_filter']
    content_based_filter = services['content_based_filter']
    contextual_bandit = services['contextual_bandit']
    
    # Get collaborative filtering scores
    cf_scores = await collaborative_filter.predict_batch(
        request.user_id, [c.id for c in request.candidates]
    )
    
    # Get content-based scores
    cb_scores = await content_based_filter.compute_similarities(
        user_profile, request.candidates
    )
    
    # Get bandit scores if context is available
    bandit_scores = []
    if request.context:
        bandit_recs = await contextual_bandit.select_content(
            request.context, request.candidates
        )
        bandit_scores = [rec.expected_reward for rec in bandit_recs]
    else:
        bandit_scores = [0.5] * len(request.candidates)
    
    # Combine scores with learned weights
    recommendations = []
    cf_weight = user_profile.collaborative_weight
    cb_weight = user_profile.content_weight
    bandit_weight = 1.0 - cf_weight - cb_weight
    
    for i, candidate in enumerate(request.candidates):
        combined_score = (
            cf_scores[i] * cf_weight +
            cb_scores[i] * cb_weight +
            bandit_scores[i] * bandit_weight
        )
        
        recommendations.append(Recommendation(
            item_id=candidate.id,
            score=combined_score,
            method='hybrid',
            features={
                'cf_score': cf_scores[i],
                'cb_score': cb_scores[i],
                'bandit_score': bandit_scores[i],
                'combined_score': combined_score
            },
            topics=candidate.topics,
            source=candidate.source
        ))
    
    return sorted(recommendations, key=lambda x: x.score, reverse=True)


async def _compute_personalization_strength(user_profile: UserProfile) -> float:
    """Compute personalization strength based on user profile."""
    # Base strength on interaction count and preference diversity
    interaction_strength = min(1.0, user_profile.total_interactions / 100.0)
    
    # Preference diversity
    topic_diversity = len(user_profile.topic_preferences) / 10.0  # Normalize by 10 topics
    source_diversity = len(user_profile.source_preferences) / 5.0  # Normalize by 5 sources
    
    diversity_strength = (topic_diversity + source_diversity) / 2.0
    
    # Combine strengths
    personalization_strength = (interaction_strength * 0.6 + diversity_strength * 0.4)
    
    return min(1.0, personalization_strength)


async def _generate_explanation(recommendations: List[Recommendation],
                              user_profile: UserProfile,
                              algorithm_used: str) -> str:
    """Generate explanation for recommendations."""
    if not recommendations:
        return "No recommendations available at this time."
    
    top_rec = recommendations[0]
    
    if algorithm_used == "cold_start":
        return f"Based on popular content and general preferences, we recommend '{top_rec.item_id}'."
    elif algorithm_used == "hybrid":
        return f"Based on your reading history and similar users, we recommend '{top_rec.item_id}' with a score of {top_rec.score:.2f}."
    else:
        return f"We recommend '{top_rec.item_id}' based on your preferences."


async def _log_recommendation(request: PersonalizationRequest,
                            response: PersonalizedResponse,
                            services: Dict[str, Any]):
    """Log recommendation for analytics."""
    try:
        postgres_client = services['postgres']
        
        query = """
        INSERT INTO recommendation_logs (
            user_id, request_id, algorithm_used, recommendations, 
            context, response_time_ms
        ) VALUES ($1, $2, $3, $4, $5, $6)
        """
        
        await postgres_client.execute(
            query,
            response.user_id,
            response.request_id,
            response.algorithm_used,
            [rec.dict() for rec in response.recommendations],
            request.context.dict() if request.context else {},
            response.processing_time_ms
        )
    except Exception as e:
        logger.error(f"Failed to log recommendation: {e}")


# User interaction endpoint
@app.post("/interaction")
async def record_interaction(
    interaction: UserInteraction,
    background_tasks: BackgroundTasks,
    services: Dict[str, Any] = Depends(get_services)
):
    """Record user interaction for learning."""
    try:
        postgres_client = services['postgres']
        user_profile_manager = services['user_profile_manager']
        
        # Save interaction
        query = """
        INSERT INTO user_interactions (
            user_id, item_id, interaction_type, rating, timestamp,
            context, session_id, device_type, location
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        
        await postgres_client.execute(
            query,
            interaction.user_id,
            interaction.item_id,
            interaction.interaction_type.value,
            interaction.rating,
            interaction.timestamp,
            interaction.context,
            interaction.session_id,
            interaction.device_type,
            interaction.location
        )
        
        # Update user profile
        background_tasks.add_task(
            user_profile_manager.update_profile_from_interaction,
            interaction.user_id,
            interaction
        )
        
        return {"status": "success", "message": "Interaction recorded"}
        
    except Exception as e:
        logger.error(f"Failed to record interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# User profile endpoint
@app.get("/profile/{user_id}", response_model=UserProfile)
async def get_user_profile(
    user_id: str,
    services: Dict[str, Any] = Depends(get_services)
):
    """Get user profile."""
    try:
        user_profile_manager = services['user_profile_manager']
        profile = await user_profile_manager.get_or_create_profile(user_id)
        return profile
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model metrics endpoint
@app.get("/metrics/models")
async def get_model_metrics(services: Dict[str, Any] = Depends(get_services)):
    """Get model performance metrics."""
    try:
        collaborative_filter = services['collaborative_filter']
        content_based_filter = services['content_based_filter']
        contextual_bandit = services['contextual_bandit']
        
        metrics = {
            'collaborative_filter': await collaborative_filter.get_model_metrics(),
            'content_based_filter': await content_based_filter.get_model_metrics(),
            'contextual_bandit': contextual_bandit.get_model_info()
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Retrain models endpoint
@app.post("/retrain")
async def retrain_models(
    background_tasks: BackgroundTasks,
    services: Dict[str, Any] = Depends(get_services)
):
    """Retrain all models."""
    try:
        collaborative_filter = services['collaborative_filter']
        content_based_filter = services['content_based_filter']
        contextual_bandit = services['contextual_bandit']
        
        # Retrain models in background
        background_tasks.add_task(collaborative_filter.retrain_model)
        background_tasks.add_task(content_based_filter.retrain_models)
        background_tasks.add_task(contextual_bandit.retrain_models)
        
        return {"status": "success", "message": "Models retraining started"}
    except Exception as e:
        logger.error(f"Failed to retrain models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )
