"""
Vercel entry point for the AI Hub Challenge application.
This file serves as the entry point for Vercel serverless functions.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try to import the Vercel-optimized version first
    from src.main_vercel import app
    print("Using Vercel-optimized main application")
except ImportError as e:
    print(f"Failed to import main_vercel: {e}")
    try:
        # Fallback to regular main
        from src.main import app
        print("Using regular main application")
    except ImportError as e2:
        print(f"Failed to import main: {e2}")
        # Create a minimal FastAPI app as final fallback
        from fastapi import FastAPI
        app = FastAPI(title="AI Hub Challenge - Fallback")
        
        @app.get("/")
        async def root():
            return {"message": "AI Hub Challenge API", "status": "running", "error": "Import failed"}
        
        @app.get("/health")
        async def health():
            return {"status": "unhealthy", "error": "Import failed"}

# Export the FastAPI app for Vercel
handler = app