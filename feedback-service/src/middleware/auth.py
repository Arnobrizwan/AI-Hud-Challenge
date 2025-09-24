"""
Authentication middleware
"""

import logging
from typing import Optional
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

security = HTTPBearer()

class AuthMiddleware:
    """Authentication middleware for API requests"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # Skip auth for health check and docs
            if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
                await self.app(scope, receive, send)
                return
            
            # Extract token from header
            authorization = request.headers.get("Authorization")
            if not authorization:
                raise HTTPException(status_code=401, detail="Authorization header required")
            
            # Validate token (simplified)
            if not self.validate_token(authorization):
                raise HTTPException(status_code=401, detail="Invalid token")
        
        await self.app(scope, receive, send)
    
    def validate_token(self, authorization: str) -> bool:
        """Validate authentication token"""
        
        try:
            # Extract token from "Bearer <token>" format
            if not authorization.startswith("Bearer "):
                return False
            
            token = authorization[7:]  # Remove "Bearer " prefix
            
            # Simple validation - in production, this would verify JWT or check database
            return len(token) > 10
            
        except Exception as e:
            logger.error("Error validating token", error=str(e))
            return False
