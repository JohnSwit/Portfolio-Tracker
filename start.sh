#!/bin/bash

# Portfolio Analytics Tool - Quick Start Script

echo "==================================="
echo "Portfolio Analytics Tool - Startup"
echo "==================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if Node is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    exit 1
fi

# Backend setup
echo ""
echo "Starting backend..."
cd backend

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
if [ ! -f ".installed" ]; then
    echo "Installing backend dependencies..."
    pip install -r requirements.txt
    touch .installed
fi

# Start backend in background
echo "Starting FastAPI server..."
python main.py &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

cd ..

# Frontend setup
echo ""
echo "Starting frontend..."
cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start frontend
echo "Starting React development server..."
npm run dev &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

cd ..

echo ""
echo "==================================="
echo "Portfolio Analytics Tool is running!"
echo "==================================="
echo ""
echo "Backend API:  http://localhost:8000"
echo "API Docs:     http://localhost:8000/docs"
echo "Frontend UI:  http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for Ctrl+C
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit 0" INT
wait
