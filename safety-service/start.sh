#!/bin/bash

# Safety Service Startup Script
# This script starts the safety service with proper configuration

set -e

echo "Starting Safety Service..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp env.example .env
    echo "Please edit .env file with your configuration before running again."
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check if database is available
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Checking database connection..."
    # Add database connection check here
fi

# Run tests
echo "Running installation tests..."
python test_installation.py

# Start the service
echo "Starting Safety Service..."
if [ "$ENVIRONMENT" = "development" ]; then
    uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
else
    uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
fi
