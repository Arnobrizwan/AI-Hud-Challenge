"""
Real-Time Interface Service for HUD Microservice
Provides WebSocket, GraphQL, SSE, and real-time collaboration features
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import redis.asyncio as redis

from realtime.websocket_manager import WebSocketManager
from realtime.graphql_executor import GraphQLRealtimeExecutor
from realtime.sse_manager import SSEManager
from realtime.collaboration_engine import CollaborationEngine
from realtime.analytics_streamer import AnalyticsStreamer
from realtime.redis_pubsub import RedisPubSub
from realtime.subscription_manager import SubscriptionManager
from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Global managers
websocket_manager: Optional[WebSocketManager] = None
graphql_executor: Optional[GraphQLRealtimeExecutor] = None
sse_manager: Optional[SSEManager] = None
collaboration_engine: Optional[CollaborationEngine] = None
analytics_streamer: Optional[AnalyticsStreamer] = None
redis_pubsub: Optional[RedisPubSub] = None
subscription_manager: Optional[SubscriptionManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global websocket_manager, graphql_executor, sse_manager
    global collaboration_engine, analytics_streamer, redis_pubsub, subscription_manager
    
    # Initialize Redis connection
    redis_client = redis.from_url(settings.REDIS_URL)
    
    # Initialize managers
    websocket_manager = WebSocketManager()
    graphql_executor = GraphQLRealtimeExecutor()
    sse_manager = SSEManager()
    collaboration_engine = CollaborationEngine()
    analytics_streamer = AnalyticsStreamer()
    redis_pubsub = RedisPubSub(redis_client)
    subscription_manager = SubscriptionManager(redis_client)
    
    # Start background tasks
    asyncio.create_task(redis_pubsub.start_listening())
    asyncio.create_task(websocket_manager.start_heartbeat_monitoring())
    
    logger.info("Real-time interface service started")
    
    yield
    
    # Cleanup
    await redis_client.close()
    logger.info("Real-time interface service stopped")

# Create FastAPI application
app = FastAPI(
    title="Real-Time Interface Service",
    description="High-performance real-time interfaces for HUD microservice",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")

# Pydantic models
class ConnectionRequest(BaseModel):
    user_id: str
    connection_type: str  # websocket, sse, graphql_subscription
    subscriptions: list[str] = []
    preferences: Dict[str, Any] = {}

class ConnectionResponse(BaseModel):
    connection_id: str
    session_id: str
    connection_type: str
    subscriptions: list[str]
    established_at: datetime

class RealtimeUpdate(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    target_users: list[str] = []
    target_groups: list[str] = []

# WebSocket endpoints
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time connections"""
    if not websocket_manager:
        await websocket.close(code=1011, reason="Service unavailable")
        return
    
    await websocket_manager.handle_websocket_connection(websocket, user_id)

@app.websocket("/ws/collaboration/{session_id}")
async def collaboration_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time collaboration"""
    if not collaboration_engine:
        await websocket.close(code=1011, reason="Service unavailable")
        return
    
    await collaboration_engine.handle_collaboration_websocket(websocket, session_id)

# Server-Sent Events endpoints
@app.get("/sse/{user_id}")
async def sse_endpoint(user_id: str, request: Request):
    """Server-Sent Events endpoint for real-time updates"""
    if not sse_manager:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    return await sse_manager.create_sse_endpoint(request, user_id)

@app.get("/sse/analytics/{user_id}")
async def analytics_sse_endpoint(user_id: str, request: Request):
    """SSE endpoint for real-time analytics dashboard"""
    if not analytics_streamer:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    return await analytics_streamer.create_analytics_sse_endpoint(request, user_id)

# GraphQL endpoint
@app.post("/graphql")
async def graphql_endpoint(request: Request):
    """GraphQL endpoint for queries and mutations"""
    if not graphql_executor:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    return await graphql_executor.execute_request(request)

@app.websocket("/graphql")
async def graphql_subscription_endpoint(websocket: WebSocket):
    """GraphQL subscription WebSocket endpoint"""
    if not graphql_executor:
        await websocket.close(code=1011, reason="Service unavailable")
        return
    
    await graphql_executor.handle_subscription_websocket(websocket)

# REST API endpoints
@app.post("/api/connect", response_model=ConnectionResponse)
async def initialize_connection(connection_request: ConnectionRequest):
    """Initialize real-time connection"""
    if not websocket_manager or not subscription_manager:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    # Create user session
    session = await subscription_manager.create_user_session(
        connection_request.user_id, 
        connection_request
    )
    
    # Set up appropriate connection
    if connection_request.connection_type == 'websocket':
        connection = await websocket_manager.create_connection(session)
    elif connection_request.connection_type == 'sse':
        connection = await sse_manager.create_connection(session)
    elif connection_request.connection_type == 'graphql_subscription':
        connection = await graphql_executor.create_subscription_connection(session)
    else:
        raise HTTPException(status_code=400, detail="Unsupported connection type")
    
    # Subscribe to relevant data streams
    await subscription_manager.subscribe_to_data_streams(
        connection, 
        connection_request.subscriptions
    )
    
    # Send initial data
    initial_data = await subscription_manager.get_initial_data(connection_request)
    await connection.send_initial_data(initial_data)
    
    return ConnectionResponse(
        connection_id=connection.id,
        session_id=session.id,
        connection_type=connection_request.connection_type,
        subscriptions=connection_request.subscriptions,
        established_at=datetime.utcnow()
    )

@app.post("/api/broadcast")
async def broadcast_update(update: RealtimeUpdate):
    """Broadcast real-time update to connected clients"""
    if not redis_pubsub:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    await redis_pubsub.publish_update(update)
    return {"status": "broadcast_sent", "timestamp": datetime.utcnow()}

@app.get("/api/connections/status")
async def get_connections_status():
    """Get status of active connections"""
    if not websocket_manager:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    return await websocket_manager.get_connections_status()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "services": {
            "websocket_manager": websocket_manager is not None,
            "graphql_executor": graphql_executor is not None,
            "sse_manager": sse_manager is not None,
            "collaboration_engine": collaboration_engine is not None,
            "analytics_streamer": analytics_streamer is not None,
            "redis_pubsub": redis_pubsub is not None,
            "subscription_manager": subscription_manager is not None
        }
    }

# Serve React frontend
@app.get("/{path:path}")
async def serve_frontend(path: str):
    """Serve React frontend for all unmatched routes"""
    try:
        with open("frontend/build/index.html", "r") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return {"message": "Frontend not built. Run 'npm run build' in frontend directory."}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
