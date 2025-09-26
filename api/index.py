"""
Vercel entry point for the AI Hub Challenge application.
This file serves as the entry point for Vercel serverless functions.
"""

from src.main import app

# Export the FastAPI app for Vercel
handler = app
